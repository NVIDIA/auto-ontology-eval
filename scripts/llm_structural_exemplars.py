"""Find structurally equivalent train exemplars with an LLM, two ways, and
score both against each other.

Run this file and nothing else. It carries the lexical retrieval helpers it
once imported, so producing the exemplar file is one command with no prior
step::

    python scripts/llm_structural_exemplars.py predict

which writes ``output/predicted_structural_exemplars.csv``, the file the eval
passes as ``--sql-examples``. The retriever that used to live in
``find_structural_exemplars.py``, and the skeleton study in
``train_structural_twins.py``, were removed once this file stopped importing
them; both are in git history, and the lexical CSV one of them wrote can still
be fed to ``compare`` through ``--method`` if you kept a copy.

``rerank`` -- judge the lexical shortlist
-----------------------------------------
Nothing here asks the LLM to read all of train. The train set is 0.82M tokens
of prompt text, so it does not fit in one context, and scoring dev against it
pairwise is 1,534 x 9,428 = 14.5M calls. So the LLM reranks the top
``--candidates`` the lexical index already returned.

That makes the comparison an A/B on *ranking* with recall held fixed -- both
methods see the identical pool, so a difference is a difference in judgement,
not in reach. It also inherits the pool's ceiling, and that ceiling binds hard:
measured on 200 questions, a pool of 30 contains an exactly-equivalent train
example for 13.6% of questions while one exists somewhere in train for 41.4%.
``compare`` prints both ceilings so a flat result can be read correctly.

``predict`` -- write the SQL first, then search all of train
------------------------------------------------------------
The reason the lexical retriever has to match on question wording is that
skeleton-space search needs a skeleton for the dev question, and its gold SQL
is exactly what is unavailable at inference time. So the LLM writes a
*hypothetical* query for the question instead, with invented table and column
names, and that prediction's skeleton is matched against all 9,428 train
skeletons.

Inventing identifiers is sound rather than sloppy here: ``skeleton()`` erases
every identifier and literal before anything is compared, so a wrong column
name costs nothing while a wrong clause structure costs everything. One LLM
call per question buys full coverage of train and removes the pool ceiling
rather than ranking better beneath it.

What the LLM is and is not shown
--------------------------------
Never shown, by either mode: the dev gold SQL. ``load_dev`` drops it on load,
as it does for the lexical path, so no method can see the answer it is being
measured against. Train gold SQL is fair game -- it is the exemplar being
retrieved -- and ``predict``'s few-shot examples are drawn from train only.

``rerank`` does read train SQL while the lexical index ranks on train
*questions* alone. That is strictly more information, not merely more
intelligence; ``--hide-train-sql`` runs the ablation that separates the two.

Batching
--------
Two independent multipliers, because the endpoint is an OpenAI-compatible
gateway with no offline batch API to submit to:

``--questions-per-request``  packs several dev questions into one call, so the
    system prompt and the HTTP round trip amortise across them. In ``rerank``
    the candidate blocks dominate the prompt, so this saves round trips rather
    than tokens, and past a handful of questions it trades accuracy for
    throughput -- the model has to keep several candidate lists apart. In
    ``predict`` the prompt is small and packing is nearly free.
``--workers``  requests in flight concurrently, which is where most of the
    wall-clock win comes from.

Every response is cached on disk by prompt hash, so a re-run, a crash or a
widened ``--limit`` re-costs only what it has not already asked.

Usage:
    # dump one prompt and exit, no API calls
    python scripts/llm_structural_exemplars.py rerank --limit 4 --dry-run

    python scripts/llm_structural_exemplars.py rerank --k 5 --candidates 30
    python scripts/llm_structural_exemplars.py predict --k 5
    python scripts/llm_structural_exemplars.py compare   # rerank vs predict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------
# Retrieval helpers, held here so this file is the only one to run.
#
# Two vocabularies meet below and must not be confused. The question side
# (SHAPE_PATTERNS through ExemplarIndex) reads English; the SQL side
# (SQL_* through skeleton) reads queries. Both originally defined constants
# named BACKTICKED and WORD, with different patterns -- ``[A-Za-z_][A-Za-z_'-]*``
# for a question word against ``[A-Za-z_]\w*`` for a SQL identifier -- so the
# SQL ones carry an SQL_ prefix. Dropping it would not raise; it would quietly
# change which exemplars are retrieved.
# --------------------------------------------------------------------------

# --- the question's demand, independent of its subject matter --------------

SHAPE_PATTERNS: list[tuple[str, str]] = [
    ("COUNT", r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b"),
    ("PERCENT", r"\bpercent(age)?\b|\b%\b|\bproportion\b"),
    ("RATIO", r"\bratio\b|\brate\b|\bper capita\b|\bdivided by\b"),
    ("AVERAGE", r"\baverage\b|\bmean\b|\bavg\b|\bon average\b"),
    ("SUM", r"\btotal\b(?!\s+number)|\bsum\b|\baltogether\b|\bcombined\b"),
    ("DIFFERENCE", r"\bdifference\b|\bhow much (more|less|higher|lower)\b|\bgap\b"),
    (
        "SUPERLATIVE",
        r"\b(highest|lowest|most|least|largest|smallest|biggest|"
        r"greatest|maximum|minimum|max|min|oldest|youngest|longest|"
        r"shortest|best|worst|top|earliest|latest)\b",
    ),
    ("TOP_N", r"\btop \d+\b|\bfirst \d+\b|\b\d+ (most|highest|lowest|largest)\b"),
    (
        "COMPARISON",
        r"\bmore than\b|\bless than\b|\bgreater than\b|\bfewer than\b|"
        r"\bat least\b|\bat most\b|\bover \d|\bunder \d|\babove\b|"
        r"\bbelow\b|\bexceed",
    ),
    ("BETWEEN", r"\bbetween\b.{0,40}\band\b"),
    (
        "LIST",
        r"^\s*(list|name|give|show|state|please list|identify|find)\b|"
        r"\bwhat are\b|\bwhich .*\bare\b",
    ),
    (
        "MULTI_ATTR",
        r"\band (also )?(its|their|his|her|the)\b|\balso (give|list|"
        r"state|show|mention|provide)\b|\brespectively\b",
    ),
    ("DISTINCT", r"\bdifferent\b|\bdistinct\b|\bunique\b"),
    ("PER_GROUP", r"\bfor each\b|\bper \w+\b|\bby each\b|\bin each\b|\bgroup"),
    ("YEAR", r"\b(19|20)\d{2}\b"),
    ("DATE_PART", r"\bmonth\b|\byear\b|\bday\b|\bdecade\b|\bquarter\b"),
    ("NAME_OF", r"\bname of\b|\bnames of\b|\bcalled\b|\btitle of\b|\bfull name\b"),
    ("ID_OF", r"\bid of\b|\bids of\b|\bidentifier\b|\bcode of\b|\bnumber of the\b"),
    ("YES_NO_FLAG", r"\bwhether\b|\bdoes\b|\bis it\b|\bany\b"),
    ("AMONG", r"\bamong\b|\bout of\b|\bof (all|these|those)\b"),
]

_COMPILED = [(tag, re.compile(rx, re.IGNORECASE)) for tag, rx in SHAPE_PATTERNS]

# BIRD evidence often spells out the arithmetic ("rate = `Free Meal Count` /
# `Enrollment`"). The operators are a strong signal for the SELECT list, but
# the backticked column names are schema vocabulary, not intent -- a column
# called "Free Meal Count" would otherwise register as a COUNT question. So
# identifiers are stripped before word matching and the operators are read
# separately.
BACKTICKED = re.compile(r"`[^`]*`|\[[^\]]*\]")

EVIDENCE_PATTERNS: list[tuple[str, str]] = [
    ("EV_DIV", r"/"),
    ("EV_PCT", r"\*\s*100|100\s*\*|\bpercent"),
    ("EV_SUB", r"\s-\s|\bsubtract\b|\bminus\b"),
    ("EV_COUNT", r"\bcount\s*\("),
    ("EV_SUM", r"\bsum\s*\("),
    ("EV_AVG", r"\bavg\s*\(|\baverage\s*\("),
    ("EV_MAXMIN", r"\bmax\s*\(|\bmin\s*\("),
    ("EV_CASE", r"\bcase\s+when\b|\biif\s*\("),
    ("EV_LITERAL_MAP", r"\brefers? to\b|\bmeans\b|\bis denoted\b|\bstands for\b"),
]

_EV_COMPILED = [(tag, re.compile(rx, re.IGNORECASE)) for tag, rx in EVIDENCE_PATTERNS]

# Tags that pin down SQL construction hardest, so a match on them is worth
# more than a match on, say, DATE_PART.
TAG_WEIGHT = defaultdict(
    lambda: 1.0,
    {
        "COUNT": 2.5,
        "SUPERLATIVE": 2.5,
        "PERCENT": 2.5,
        "MULTI_ATTR": 2.5,
        "RATIO": 2.0,
        "DIFFERENCE": 2.0,
        "AVERAGE": 2.0,
        "TOP_N": 2.0,
        "DISTINCT": 2.0,
        "PER_GROUP": 2.0,
        "NAME_OF": 1.5,
        "LIST": 1.5,
        "COMPARISON": 1.5,
        "SUM": 1.5,
        # Arithmetic spelled out in the evidence dictates the SELECT list, so
        # agreement on it matters more than agreement on phrasing.
        "EV_PCT": 2.5,
        "EV_CASE": 2.5,
        "EV_DIV": 2.0,
        "EV_SUB": 2.0,
        "EV_LITERAL_MAP": 1.5,
    },
)


def shape_tags(question: str, evidence: str = "") -> frozenset[str]:
    prose = f"{question} {BACKTICKED.sub(' ', evidence)}".strip()
    tags = {tag for tag, rx in _COMPILED if rx.search(prose)}
    if evidence:
        tags |= {tag for tag, rx in _EV_COMPILED if rx.search(evidence)}
    return frozenset(tags)


# --- the question with its domain vocabulary erased ------------------------

QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,60})[\"'“”‘’]")
NUMBER = re.compile(r"\b\d[\d,.]*\b")
WORD = re.compile(r"[A-Za-z_][A-Za-z_'-]*")
# Words that carry question structure and must survive the proper-noun filter
# even when a sentence starts with them.
STRUCTURAL = {
    "what", "which", "who", "whose", "when", "where", "how", "list", "name",
    "give", "show", "state", "please", "find", "identify", "the", "of", "in",
    "for", "and", "or", "is", "are", "was", "were", "has", "have", "had", "do",
    "does", "did", "with", "without", "among", "between", "more", "less",
    "than", "most", "least", "all", "each", "per", "total", "average",
    "number", "many", "much", "percentage", "percent", "ratio", "rate",
    "difference", "highest", "lowest", "top", "first", "last", "also", "its",
    "their", "that", "this", "these", "those", "there", "any", "only", "both",
}


def template(question: str) -> list[str]:
    """Question reduced to structural words, with domain terms as placeholders."""
    text = QUOTED.sub(" <val> ", question)
    text = NUMBER.sub(" <num> ", text)
    out: list[str] = []
    for token in re.findall(r"<val>|<num>|[A-Za-z_][A-Za-z_'-]*|\?", text):
        if token in ("<val>", "<num>", "?"):
            out.append(token)
            continue
        low = token.lower()
        if low in STRUCTURAL:
            out.append(low)
        elif token[0].isupper():
            # A capitalised word mid-sentence is a domain entity, not structure.
            out.append("<ent>")
        else:
            out.append(low)
    return out


def ngrams(tokens: list[str], n: int = 2) -> list[str]:
    grams = list(tokens)
    for size in range(2, n + 1):
        grams += [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
    return grams


# --- lexical retrieval over train ------------------------------------------


class ExemplarIndex:
    """Inverted index over train question templates plus shape tags."""

    def __init__(self, train: list[dict], ngram: int = 2) -> None:
        self.train = train
        self.ngram = ngram
        self.tags: list[frozenset[str]] = []
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.norms: list[float] = []

        docs: list[Counter] = []
        df: Counter = Counter()
        for ex in train:
            grams = Counter(ngrams(template(ex["question"]), ngram))
            docs.append(grams)
            df.update(grams.keys())
            self.tags.append(shape_tags(ex["question"], ex.get("evidence", "")))

        n_docs = len(train)
        self.idf = {g: math.log(1.0 + n_docs / (1 + c)) for g, c in df.items()}
        for idx, grams in enumerate(docs):
            weights = {g: (1 + math.log(c)) * self.idf[g] for g, c in grams.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            self.norms.append(norm)
            for g, w in weights.items():
                self.postings[g].append((idx, w / norm))

    def _tag_score(self, a: frozenset[str], b: frozenset[str]) -> float:
        """Weighted Jaccard: reward shared demands, punish unshared ones."""
        if not a and not b:
            return 0.0
        shared = sum(TAG_WEIGHT[t] for t in a & b)
        union = sum(TAG_WEIGHT[t] for t in a | b)
        return shared / union if union else 0.0

    def query(
        self, question: str, evidence: str, k: int, tag_weight: float = 2.0
    ) -> list[tuple[float, float, float, frozenset[str], dict]]:
        grams = Counter(ngrams(template(question), self.ngram))
        weights = {
            g: (1 + math.log(c)) * self.idf.get(g, 0.0)
            for g, c in grams.items()
            if g in self.idf
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for g, w in weights.items():
            wq = w / norm
            for idx, wd in self.postings[g]:
                scores[idx] += wq * wd

        q_tags = shape_tags(question, evidence)
        ranked = []
        for idx, lexical in scores.items():
            tag = self._tag_score(q_tags, self.tags[idx])
            ranked.append(
                (
                    lexical + tag_weight * tag,
                    lexical,
                    tag,
                    self.tags[idx],
                    self.train[idx],
                )
            )
        # Shape-only matches are still useful when wording diverges entirely.
        if not ranked:
            for idx, tags in enumerate(self.tags):
                tag = self._tag_score(q_tags, tags)
                if tag:
                    ranked.append((tag_weight * tag, 0.0, tag, tags, self.train[idx]))
        ranked.sort(key=lambda t: -t[0])
        return ranked[:k]


def load_dev(path: Path) -> list[dict]:
    """Dev questions with gold SQL stripped, so retrieval cannot see it."""
    raw = json.loads(path.read_text())
    return [
        {
            "question_id": int(r["question_id"]),
            "db_id": r.get("db_id") or r.get("db"),
            "question": r["question"],
            "evidence": r.get("evidence") or "",
            "difficulty": r.get("difficulty", ""),
        }
        for r in raw
    ]


# --- the SQL side: a query with every identifier and literal erased --------

SQL_KEYWORDS = {
    "select", "distinct", "from", "where", "group", "by", "order", "having",
    "limit", "join", "inner", "left", "outer", "on", "and", "or", "not", "in",
    "is", "null", "like", "between", "as", "asc", "desc", "case", "when",
    "then", "else", "end", "cast", "real", "integer", "union", "all", "exists",
    "count", "sum", "avg", "max", "min", "iif", "strftime", "substr", "round",
    "julianday", "date", "datetime", "length", "abs", "coalesce", "div",
}
SQL_ALIAS_PREFIX = re.compile(r"\b[Tt]\d+\s*\.|\b[a-z]{1,3}\d?\s*\.", re.ASCII)
SQL_AS_ALIAS = re.compile(r"\bas\s+[`\"\[]?\w+[`\"\]]?", re.IGNORECASE)
SQL_STRINGS = re.compile(r"'[^']*'")
SQL_BACKTICKED = re.compile(r"[`\"\[][^`\"\]]+[`\"\]]")
SQL_NUMBERS = re.compile(r"\b\d+(\.\d+)?\b")
SQL_WORD = re.compile(r"[A-Za-z_]\w*")


def skeleton(sql: object) -> str:
    s = " ".join(str(sql).split())
    s = SQL_STRINGS.sub(" L ", s)
    s = SQL_BACKTICKED.sub(" C ", s)
    s = SQL_NUMBERS.sub(" N ", s)
    # Aliases carry no structure; drop the declaration and the qualifier.
    s = SQL_AS_ALIAS.sub(" ", s)
    s = SQL_ALIAS_PREFIX.sub(" ", s)
    out: list[str] = []
    prev = ""
    for tok in re.findall(r"[A-Za-z_]\w*|[(),*=<>!+/-]|N|L|C", s):
        low = tok.lower()
        if low in SQL_KEYWORDS:
            out.append(low)
        elif tok in {"N", "L", "C"}:
            out.append(tok)
        elif SQL_WORD.fullmatch(tok):
            out.append("T" if prev in {"from", "join"} else "C")
        else:
            out.append(tok)
        if SQL_WORD.fullmatch(tok):
            prev = low
        elif tok not in {"(", ")"}:
            prev = ""
    # "as" only ever introduced an alias or a cast type; keep casts readable.
    return " ".join(t for t in out if t != "as")

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

# The definition of "structurally equivalent" is deliberately the definition
# the metric uses: same SQL skeleton, i.e. the same query once identifiers and
# literals are erased. Anything vaguer and the model falls back on topic
# similarity, which the lexical retriever already established is worthless
# here -- dev and train share no databases.
SYSTEM_PROMPT = """\
You match natural-language database questions to example questions that need \
the SAME SQL CONSTRUCTION.

Two questions are STRUCTURALLY EQUIVALENT when correct SQL for each has the \
same skeleton: the same query once every table name, column name, literal and \
alias is erased. Judge on these axes only:
  - aggregation: which of COUNT / SUM / AVG / MAX / MIN / none, and how many
  - projection arity: how many expressions the SELECT list returns
  - arithmetic in the projection: division, `* 100` for a percentage, \
subtraction, CAST to REAL
  - conditional aggregation: CASE WHEN or IIF inside an aggregate
  - join shape: single table, one join, a chain of three or more, a self-join
  - grouping: whether GROUP BY is needed, whether HAVING is needed
  - ordering and row limits: ORDER BY with LIMIT 1 for a superlative, LIMIT N \
for a top-N
  - nesting: a subquery in WHERE, a correlated subquery, a set operation
  - DISTINCT

Ignore completely:
  - subject matter, industry and vocabulary. The candidates come from \
different databases on purpose; a topic match is worth nothing, and a \
candidate about an unrelated domain is a perfectly good answer.
  - how many filter conditions the WHERE clause needs.
  - specific table and column names.

Score each candidate you return:
  5 - identical skeleton; the same SQL with identifiers swapped
  4 - same skeleton but for one clause (e.g. one extra join, or DISTINCT)
  3 - same aggregation and projection shape, but a different join or nesting \
shape
Return NOTHING for a candidate you would score 2 or 1. Returning an empty list \
for a question is correct when the pool holds no structural match; a padded \
list is worse than a short one.

Return at most {k} candidates per question, best first. Use each candidate \
number at most once, and only numbers from that question's own list.\
"""


def _shown(path: Path) -> Path:
    """Repo-relative for readability, absolute when the path is outside it."""
    resolved = path.resolve()
    return resolved.relative_to(ROOT) if ROOT in resolved.parents else resolved


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_question_block(
    question: dict, candidates: list[dict], show_sql: bool
) -> str:
    """One dev question and its numbered candidate pool, as prompt text."""
    lines = [
        f"### QUESTION {question['question_id']}",
        f"question: {_truncate(question['question'], 400)}",
    ]
    if question["evidence"]:
        lines.append(f"evidence: {_truncate(question['evidence'], 300)}")
    lines.append("candidates:")
    for number, candidate in enumerate(candidates, start=1):
        parts = [f"q: {_truncate(candidate['question'], 300)}"]
        if candidate.get("evidence"):
            parts.append(f"ev: {_truncate(candidate['evidence'], 200)}")
        if show_sql:
            parts.append(f"sql: {_truncate(candidate['SQL'], 450)}")
        lines.append(f"[{number}] " + " | ".join(parts))
    return "\n".join(lines)


def build_prompt(batch: list[tuple[dict, list[dict]]], k: int, show_sql: bool) -> str:
    blocks = [
        render_question_block(question, candidates, show_sql)
        for question, candidates in batch
    ]
    ids = ", ".join(str(question["question_id"]) for question, _ in batch)
    return (
        f"Rank candidates for {len(batch)} question(s): {ids}.\n"
        "Answer every question id, in the order given. Judge each question's "
        "candidates only against that question.\n\n" + "\n\n".join(blocks)
    )


# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------


def _schemas():
    """Built lazily: importing gsf triggers env-dependent client construction."""
    from gsf.utils.llm_invoke import StrictLLMOutputModel
    from pydantic import Field

    class Pick(StrictLLMOutputModel):
        candidate: int = Field(
            description="Candidate number from that question's list."
        )
        equivalence: int = Field(
            description="5, 4 or 3 as defined in the instructions."
        )
        reason: str = Field(
            description="Under 15 words, naming the shared construction."
        )

    class Answer(StrictLLMOutputModel):
        question_id: int = Field(description="The question id being answered.")
        target_shape: str = Field(
            description=(
                "Under 20 words: the SQL construction this question needs, "
                "e.g. 'count over one join with GROUP BY and HAVING'."
            )
        )
        picks: list[Pick] = Field(description="Best first. Empty when nothing matches.")

    class BatchAnswer(StrictLLMOutputModel):
        answers: list[Answer]

    return BatchAnswer


def _prediction_schema():
    from gsf.utils.llm_invoke import StrictLLMOutputModel
    from pydantic import Field

    class Prediction(StrictLLMOutputModel):
        question_id: int = Field(description="The question id being answered.")
        sql: str = Field(
            description=(
                "One SQLite query answering that question, with invented table "
                "and column names. Only its structure is used."
            )
        )

    class BatchPrediction(StrictLLMOutputModel):
        predictions: list[Prediction]

    return BatchPrediction


# --------------------------------------------------------------------------
# Predicting the target skeleton, then searching train in skeleton space
# --------------------------------------------------------------------------

# The conventions below are not stylistic preferences; each is a measured
# property of the 9,428 train queries a prediction gets matched against. A
# prediction spelling a join `JOIN` rather than `INNER JOIN` is structurally
# right and still scores badly, because the corpus writes it the other way
# 99.9% of the time -- so the corpus's habits have to be stated outright.
HYDE_SYSTEM_PROMPT = """\
You write the SQL a question needs, without being given the schema.

Invent plausible table and column names. Only the SHAPE of your query is used: \
every table name, column name and literal is erased before it is compared to \
anything. A wrong column name costs nothing; a wrong clause structure costs \
everything. Spend your effort on which aggregates, joins, grouping, ordering \
and nesting the question requires.

Match the conventions of the corpus you are matched against, as measured over \
its 9,428 queries:
  - write joins as `INNER JOIN <table> ON <col> = <col>`. A bare JOIN or a \
comma join appears in 0.1% of the corpus, so never use one.
  - wrap a division of integers in `CAST(... AS REAL)`; 71% of the corpus's \
divisions do.
  - write a percentage as `CAST(SUM(CASE WHEN <cond> THEN 1 ELSE 0 END) AS \
REAL) * 100 / COUNT(<col>)`.
  - write a ratio of two subsets as `CAST(SUM(CASE WHEN <a> THEN 1 ELSE 0 END) \
AS REAL) / SUM(CASE WHEN <b> THEN 1 ELSE 0 END)`.
  - answer a superlative with `ORDER BY <col> DESC LIMIT 1`, not a subquery \
against `MAX(...)`.
  - answer "the Nth highest" with `ORDER BY <col> DESC LIMIT <N-1>, 1`.
  - filter on a year with `strftime('%Y', <col>) = '2020'`.
  - project exactly the columns the question asks for, in the order asked, and \
no others.
  - use GROUP BY only when the question wants a value per group, not to find a \
single maximum.

Answer every question id you are given, and return nothing but the queries.\
"""

# One worked example per construction the corpus actually uses. Picking the
# *most common* skeletons instead would spend every shot on the two-table
# lookup that is 5% of train by itself, teaching nothing about the shapes the
# generator gets wrong.
SHOT_PATTERNS: list[tuple[str, str]] = [
    (
        "lookup over one join",
        r"^select (?!count|sum|avg|max|min|distinct)[^(]+ from \w+ as \w+ inner join",
    ),
    ("count over one join", r"^select count\s*\(.*inner join"),
    ("superlative", r"order by .+ desc limit 1\s*$"),
    ("percentage of a subset", r"case when.+as real\s*\)\s*\*\s*100"),
    ("ratio of two subsets", r"sum\s*\(\s*case when.+as real\s*\)\s*/"),
    ("group by with having", r"group by .+ having"),
    ("subquery in where", r"where[^()]*\(\s*select"),
    ("distinct over a chain of joins", r"^select distinct .+inner join .+inner join"),
    ("date part filter", r"strftime"),
    ("average over a join", r"^select avg\s*\(.*inner join"),
]


def few_shot_examples(train: list[dict], limit: int) -> list[tuple[str, dict]]:
    """First train example matching each construction, for the prompt's shots.

    Deterministic -- first match in train order -- so the prompt is stable
    across runs and the response cache stays valid.
    """
    shots: list[tuple[str, dict]] = []
    for label, pattern in SHOT_PATTERNS[:limit]:
        matcher = re.compile(pattern, re.IGNORECASE)
        for example in train:
            sql = " ".join(example["SQL"].split())
            # A long example crowds out the questions and teaches the same
            # shape as a short one.
            if len(sql) < 260 and matcher.search(sql):
                shots.append((label, example))
                break
    return shots


def build_hyde_prompt(batch: list[dict], shots: list[tuple[str, dict]]) -> str:
    lines = ["Worked examples from the corpus:"]
    for label, example in shots:
        lines.append(f"\n# {label}")
        lines.append(f"question: {_truncate(example['question'], 200)}")
        if example.get("evidence"):
            lines.append(f"evidence: {_truncate(example['evidence'], 160)}")
        lines.append(f"SQL: {_truncate(example['SQL'], 260)}")
    ids = ", ".join(str(question["question_id"]) for question in batch)
    lines.append(f"\n\nNow write SQL for each of these {len(batch)} questions: {ids}")
    for question in batch:
        lines.append(f"\n### QUESTION {question['question_id']}")
        lines.append(f"question: {_truncate(question['question'], 400)}")
        if question["evidence"]:
            lines.append(f"evidence: {_truncate(question['evidence'], 300)}")
    return "\n".join(lines)


def _skeleton_grams(skel: str, n: int = 3) -> list[str]:
    """Unigrams plus n-grams; the unigrams keep recall for short skeletons."""
    tokens = skel.split()
    grams = list(tokens)
    for size in range(2, n + 1):
        grams += [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
    return grams


class SkeletonIndex:
    """Rank train examples by SQL-skeleton similarity to a predicted skeleton.

    Aligning one prediction against all 4,172 distinct train skeletons with
    ``SequenceMatcher`` is the whole runtime, so a TF-IDF pass over skeleton
    token n-grams shortlists a few hundred and only those get aligned. The
    shortlist decides recall, not order -- it is wide enough that the ranking
    is settled by the alignment.
    """

    def __init__(self, train: list[dict], shortlist: int = 400) -> None:
        self.shortlist = shortlist
        # One representative per distinct skeleton. Duplicates demonstrate the
        # same construction, so returning k of them is k wasted exemplars.
        # Deduplicating cannot lower best-of-k: the representative kept scores
        # exactly what its duplicates would have.
        self.examples: list[dict] = []
        self.skeletons: list[str] = []
        seen: set[str] = set()
        for example in train:
            skel = skeleton(example["SQL"])
            if skel not in seen:
                seen.add(skel)
                self.skeletons.append(skel)
                self.examples.append(example)

        frequency: Counter = Counter()
        documents: list[Counter] = []
        for skel in self.skeletons:
            grams = Counter(_skeleton_grams(skel))
            documents.append(grams)
            frequency.update(grams.keys())
        total = len(documents)
        self.idf = {
            gram: math.log(1.0 + total / (1 + count))
            for gram, count in frequency.items()
        }
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, grams in enumerate(documents):
            weights = {
                gram: (1 + math.log(count)) * self.idf[gram]
                for gram, count in grams.items()
            }
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for gram, weight in weights.items():
                self.postings[gram].append((index, weight / norm))

    def query(self, predicted: str, k: int) -> list[tuple[float, dict]]:
        grams = Counter(_skeleton_grams(predicted))
        weights = {
            gram: (1 + math.log(count)) * self.idf.get(gram, 0.0)
            for gram, count in grams.items()
            if gram in self.idf
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for gram, weight in weights.items():
            query_weight = weight / norm
            for index, document_weight in self.postings[gram]:
                scores[index] += query_weight * document_weight
        shortlist = sorted(scores, key=lambda i: (-scores[i], i))[: self.shortlist]
        if not shortlist:
            shortlist = list(range(len(self.skeletons)))
        ranked = sorted(
            (
                (SequenceMatcher(None, predicted, self.skeletons[i]).ratio(), i)
                for i in shortlist
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [(ratio, self.examples[i]) for ratio, i in ranked[:k]]


# --------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------


class ResponseCache:
    """Append-only JSONL keyed by prompt hash, so re-runs are near-free."""

    def __init__(self, path: Path, model: str) -> None:
        self.path = path
        self.model = model
        self.lock = threading.Lock()
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.entries[record["key"]] = record["response"]

    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{prompt}".encode()).hexdigest()[:32]

    def get(self, prompt: str) -> dict | None:
        return self.entries.get(self.key(prompt))

    def put(self, prompt: str, response: dict) -> None:
        key = self.key(prompt)
        with self.lock:
            self.entries[key] = response
            with self.path.open("a") as handle:
                handle.write(
                    json.dumps({"key": key, "model": self.model, "response": response})
                    + "\n"
                )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def dispatch(
    requests: list[tuple[str, set[int]]],
    schema,
    system_prompt: str,
    field: str,
    args: argparse.Namespace,
) -> tuple[dict[int, dict], Counter]:
    """Send every prompt concurrently and collect answers by question id.

    Each request carries the ids it asked about, because a model answering a
    question from a neighbouring batch has confused two candidate lists; that
    answer is dropped rather than attributed to the wrong question. One failed
    request must not cost the other hundreds, so failures are counted and
    skipped.
    """
    from gsf.utils.llm_invoke import get_llm_client, invoke_with_structured_output
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm_client(model=args.model or None, max_tokens=args.max_tokens)
    model_name = args.model or getattr(llm, "model_name", "") or str(llm)
    cache = ResponseCache(args.cache, model_name)
    print(f"model: {model_name}   cached responses: {len(cache.entries)}")

    system = SystemMessage(content=system_prompt)
    counts: Counter = Counter()
    tally = threading.Lock()

    def ask(prompt: str) -> dict | None:
        hit = cache.get(prompt)
        if hit is not None:
            with tally:
                counts["cached"] += 1
            return hit
        result = invoke_with_structured_output(
            llm, [system, HumanMessage(content=prompt)], schema
        )
        with tally:
            counts["failed" if result is None else "sent"] += 1
        if result is None:
            return None
        response = result.model_dump()
        cache.put(prompt, response)
        return response

    started = time.monotonic()
    answers: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ask, prompt): asked for prompt, asked in requests}
        for done, future in enumerate(as_completed(futures), start=1):
            asked = futures[future]
            try:
                response = future.result()
            except Exception as error:
                print(f"  request failed: {type(error).__name__}: {error}")
                response = None
            for answer in (response or {}).get(field, []):
                qid = answer.get("question_id")
                if qid in asked:
                    answers[qid] = answer
                else:
                    counts["misattributed"] += 1
            if done % 25 == 0 or done == len(futures):
                print(
                    f"  {done}/{len(futures)} requests, "
                    f"{len(answers)} questions answered, "
                    f"{time.monotonic() - started:.0f}s"
                )
    counts["elapsed"] = round(time.monotonic() - started)
    return answers, counts


def batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def report_dispatch(counts: Counter, path: Path, rows: int) -> None:
    print(
        f"\nwrote {_shown(path)} ({rows} rows) in {counts['elapsed']}s"
        f"  [sent {counts['sent']}, cached {counts['cached']}, "
        f"failed {counts['failed']}]"
    )
    if counts["misattributed"] or counts["bad_candidate_number"]:
        print(
            f"  dropped: {counts['misattributed']} cross-batch answers, "
            f"{counts['bad_candidate_number']} invalid candidate numbers"
        )


# --------------------------------------------------------------------------
# rerank: the LLM judges the lexical retriever's shortlist
# --------------------------------------------------------------------------


def candidate_pools(
    dev: list[dict], train: list[dict], args: argparse.Namespace
) -> list[tuple[dict, list[dict]]]:
    """The lexical retriever's top-C per question -- the pool both methods see."""
    index = ExemplarIndex(train, ngram=args.ngram)
    print(f"index: {len(index.postings):,} template n-grams")
    pools = []
    for question in dev:
        hits = index.query(
            question["question"],
            question["evidence"],
            args.candidates,
            args.tag_weight,
        )
        pools.append((question, [example for *_, example in hits]))
    return pools


def run(args: argparse.Namespace) -> None:
    dev = load_dev(args.input)
    if args.limit:
        dev = dev[: args.limit]
    train = json.loads(args.train.read_text())
    print(f"dev questions: {len(dev)}   train examples: {len(train)}")

    pools = candidate_pools(dev, train, args)
    batches = batched(pools, args.questions_per_request)
    requests = [
        (
            build_prompt(batch, args.k, not args.hide_train_sql),
            {question["question_id"] for question, _ in batch},
        )
        for batch in batches
    ]
    system_prompt = SYSTEM_PROMPT.format(k=args.k)
    print(
        f"{len(requests)} requests "
        f"({args.questions_per_request} questions each, "
        f"{args.candidates} candidates each), {args.workers} workers"
    )

    if args.dry_run:
        print("\n=== system prompt ===")
        print(system_prompt)
        print("\n=== first user prompt ===")
        print(requests[0][0])
        print(f"\n(dry run: {len(requests)} prompts built, no requests sent)")
        return

    answers, counts = dispatch(requests, _schemas(), system_prompt, "answers", args)

    rows = []
    for question, candidates in pools:
        answer = answers.get(question["question_id"])
        tags = shape_tags(question["question"], question["evidence"])
        picks = (answer or {}).get("picks", [])[: args.k]
        seen: set[int] = set()
        rank = 0
        for pick in picks:
            number = pick.get("candidate", 0)
            if not (1 <= number <= len(candidates)) or number in seen:
                counts["bad_candidate_number"] += 1
                continue
            seen.add(number)
            rank += 1
            example = candidates[number - 1]
            rows.append(
                {
                    "question_id": question["question_id"],
                    "db_id": question["db_id"],
                    "difficulty": question["difficulty"],
                    "question": question["question"],
                    "shape_tags": ",".join(sorted(tags)),
                    "rank": rank,
                    "equivalence": pick.get("equivalence"),
                    "reason": _truncate(pick.get("reason", ""), 200),
                    "target_shape": _truncate(
                        (answer or {}).get("target_shape", ""), 200
                    ),
                    # Where this pick sat in the lexical ranking. The whole
                    # agreement analysis reads off this column.
                    "candidate_rank": number,
                    "pool_size": len(candidates),
                    "answered": answer is not None,
                    "train_db": example["db_id"],
                    "train_question": example["question"].strip(),
                    "train_evidence": (example.get("evidence") or "").strip(),
                    "train_sql": " ".join(example["SQL"].split()),
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    report_dispatch(counts, args.out, len(frame))

    unanswered = len(pools) - len(answers)
    abstained = sum(1 for answer in answers.values() if not answer.get("picks"))
    print(f"  questions with no response   : {unanswered}")
    print(f"  questions the model abstained: {abstained}")
    if not frame.empty:
        top = frame[frame["rank"] == 1]
        print(f"  mean equivalence at rank 1   : {top.equivalence.mean():.2f}")
        print(
            f"  mean lexical rank of a pick  : {frame.candidate_rank.mean():.1f} "
            f"of {args.candidates}"
        )


# --------------------------------------------------------------------------
# predict: the LLM writes the SQL, then train is searched in skeleton space
# --------------------------------------------------------------------------


def predict(args: argparse.Namespace) -> None:
    dev = load_dev(args.input)
    if args.limit:
        dev = dev[: args.limit]
    train = json.loads(args.train.read_text())
    shots = few_shot_examples(train, args.shots)
    print(
        f"dev questions: {len(dev)}   train examples: {len(train)}   "
        f"shots: {len(shots)}"
    )

    batches = batched(dev, args.questions_per_request)
    requests = [
        (
            build_hyde_prompt(batch, shots),
            {question["question_id"] for question in batch},
        )
        for batch in batches
    ]
    print(
        f"{len(requests)} requests "
        f"({args.questions_per_request} questions each), {args.workers} workers"
    )

    if args.dry_run:
        print("\n=== system prompt ===")
        print(HYDE_SYSTEM_PROMPT)
        print("\n=== first user prompt ===")
        print(requests[0][0])
        print(f"\n(dry run: {len(requests)} prompts built, no requests sent)")
        return

    answers, counts = dispatch(
        requests, _prediction_schema(), HYDE_SYSTEM_PROMPT, "predictions", args
    )

    # Built after the requests, so a failed run costs nothing but the wait.
    index = SkeletonIndex(train, shortlist=args.shortlist)
    print(f"skeleton index: {len(index.skeletons):,} distinct train skeletons")

    rows = []
    for question in dev:
        answer = answers.get(question["question_id"])
        predicted_sql = " ".join(str((answer or {}).get("sql") or "").split())
        if not predicted_sql:
            continue
        predicted = skeleton(predicted_sql)
        tags = shape_tags(question["question"], question["evidence"])
        for rank, (ratio, example) in enumerate(
            index.query(predicted, args.k), start=1
        ):
            rows.append(
                {
                    "question_id": question["question_id"],
                    "db_id": question["db_id"],
                    "difficulty": question["difficulty"],
                    "question": question["question"],
                    "shape_tags": ",".join(sorted(tags)),
                    "rank": rank,
                    # How well the exemplar matches the *prediction*. This is
                    # the method's own confidence, and unlike the rerank
                    # equivalence score it is not self-reported.
                    "skeleton_match": round(ratio, 4),
                    # The column `--sql-examples-min-score` gates on, so this
                    # CSV can be fed to eval_chatbot unchanged. Duplicated
                    # rather than renamed because the gate reads `score` from
                    # every exemplar source, while `skeleton_match` says what
                    # the number actually measures for this one.
                    "score": round(ratio, 4),
                    "predicted_sql": predicted_sql,
                    "predicted_skeleton": predicted,
                    "train_db": example["db_id"],
                    "train_question": example["question"].strip(),
                    "train_evidence": (example.get("evidence") or "").strip(),
                    "train_sql": " ".join(example["SQL"].split()),
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    report_dispatch(counts, args.out, len(frame))
    print(f"  questions with no prediction : {len(dev) - len(answers)}")
    if not frame.empty:
        top = frame[frame["rank"] == 1]
        print(f"  mean skeleton match at rank 1: {top.skeleton_match.mean():.3f}")
        print(
            f"  predictions matching a train skeleton exactly: "
            f"{(top.skeleton_match == 1.0).sum()}/{len(top)}"
        )


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------


def exemplar_key(row: pd.Series) -> str:
    """Identify a train example across both CSVs; train.json has no id field."""
    raw = f"{row.train_db}\x00{row.train_question}\x00{row.train_sql}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def ranked_keys(frame: pd.DataFrame) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for qid, group in frame.groupby("question_id"):
        ordered = group.sort_values("rank")
        out[int(qid)] = [exemplar_key(row) for _, row in ordered.iterrows()]
    return out


def ranked_sql(frame: pd.DataFrame) -> dict[int, list[str]]:
    return {
        int(qid): list(group.sort_values("rank").train_sql)
        for qid, group in frame.groupby("question_id")
    }


def paired_bootstrap(
    left: list[float], right: list[float], rounds: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (left - right) with a percentile CI, resampling questions.

    Differences between two rankings over the same pool are small by
    construction, so a bare mean invites reading noise as a result.
    """
    diffs = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(rounds):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (
        sum(diffs) / n,
        means[int(0.025 * rounds)],
        means[int(0.975 * rounds)],
    )


def load_methods(specs: list[str]) -> dict[str, pd.DataFrame]:
    """Parse repeated ``--method label=path.csv`` into frames, order preserved.

    The first method is the baseline every other one is measured against.
    """
    methods: dict[str, pd.DataFrame] = {}
    for spec in specs:
        label, _, raw = spec.partition("=")
        if not raw:
            raise SystemExit(f"--method wants label=path.csv, got {spec!r}")
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"--method {label}: {path} does not exist")
        methods[label] = pd.read_csv(path)
    return methods


def compare(args: argparse.Namespace) -> None:
    methods = load_methods(args.method)
    gold = {int(r["question_id"]): r["SQL"] for r in json.loads(args.input.read_text())}
    train = json.loads(args.train.read_text())

    keys = {label: ranked_keys(frame) for label, frame in methods.items()}
    sqls = {label: ranked_sql(frame) for label, frame in methods.items()}
    labels = list(methods)
    baseline = labels[0]

    # Every method must have ranked the question, or the columns are not
    # comparable row by row: an LLM run may be a --limit slice, and a question
    # a method abstained on has no rows at all.
    qids = sorted(set.intersection(*(set(k) for k in keys.values())))
    print("questions ranked, by method:")
    for label in labels:
        print(f"  {label:<20} {len(keys[label])}")
    print(f"  {'-> compared on':<20} {len(qids)}")
    if not qids:
        return
    n = len(qids)
    k = int(min(frame["rank"].max() for frame in methods.values()))

    # ---------------- agreement, no gold involved ----------------
    print("\n" + "=" * 78)
    print("1. AGREEMENT  — do the methods choose the same exemplars?")
    print("=" * 78)
    print(f"\n  mean overlap of top-{k} sets, and identical rank-1 rate:\n")
    print("  " + " " * 20 + "".join(f"{label:>20}" for label in labels))
    for row in labels:
        cells = []
        for column in labels:
            if row == column:
                cells.append(f"{'—':>20}")
                continue
            overlap = (
                sum(
                    len(set(keys[row][q][:k]) & set(keys[column][q][:k])) / k
                    for q in qids
                )
                / n
            )
            same = sum(int(keys[row][q][:1] == keys[column][q][:1]) for q in qids) / n
            cells.append(f"{f'{overlap:.2f} / {100 * same:.0f}%':>20}")
        print(f"  {row:<20}" + "".join(cells))
    print("\n  (each cell is: set overlap / share with the same rank-1 exemplar)")

    for label in labels:
        frame = methods[label]
        if "candidate_rank" not in frame.columns:
            continue
        picked = frame[frame.question_id.isin(qids)]
        pool = int(picked.pool_size.max())
        print(f"\n  where {label}'s picks sat in the lexical ranking (pool of {pool}):")
        for text, lo, hi in [
            ("lexical top 5", 1, 5),
            ("ranks 6-15", 6, 15),
            (f"ranks 16-{pool}", 16, pool),
        ]:
            share = picked.candidate_rank.between(lo, hi).mean()
            print(f"    {text:<16}: {100 * share:5.1f}% of picks")

    # ---------------- quality against dev gold ----------------
    print("\n" + "=" * 78)
    print("2. QUALITY  — whose exemplars are structurally closest to dev gold?")
    print("   (gold is read only here, after every ranking is fixed)")
    print("=" * 78)

    skel_cache: dict[str, str] = {}

    def skel(sql: str) -> str:
        if sql not in skel_cache:
            skel_cache[sql] = skeleton(sql)
        return skel_cache[sql]

    def similarity(gold_skeleton: str, candidates: list[str]) -> tuple[float, float]:
        sims = [
            SequenceMatcher(None, gold_skeleton, skel(s)).ratio() for s in candidates
        ]
        return (sims[0] if sims else 0.0), (max(sims) if sims else 0.0)

    train_skeletons = [skel(example["SQL"]) for example in train]
    rng = random.Random(0)
    ceilings = compute_ceilings(args, train, qids, gold, skel) if args.ceilings else {}

    series = ["random"] + labels + list(ceilings)
    metrics = {name: {"top1": [], "bestk": []} for name in series}
    for qid in qids:
        gold_skeleton = skel(gold[qid])
        for label in labels:
            top1, bestk = similarity(gold_skeleton, sqls[label][qid][:k])
            metrics[label]["top1"].append(top1)
            metrics[label]["bestk"].append(bestk)
        drawn = [train_skeletons[rng.randrange(len(train_skeletons))] for _ in range(k)]
        sims = [SequenceMatcher(None, gold_skeleton, s).ratio() for s in drawn]
        metrics["random"]["top1"].append(sims[0])
        metrics["random"]["bestk"].append(max(sims))
        for name, best in ceilings.items():
            metrics[name]["top1"].append(best[qid])
            metrics[name]["bestk"].append(best[qid])

    print(
        f"\n  {'method':<30}{'mean sim @1':>13}{'exact @1':>11}"
        f"{'mean sim best-of-k':>21}{'exact best-of-k':>18}"
    )
    for name in series:
        top1, bestk = metrics[name]["top1"], metrics[name]["bestk"]
        exact1 = 100 * sum(1 for s in top1 if s == 1.0) / n
        exactk = 100 * sum(1 for s in bestk if s == 1.0) / n
        print(
            f"  {name:<30}{sum(top1) / n:>13.3f}{f'{exact1:.1f}%':>11}"
            f"{sum(bestk) / n:>21.3f}{f'{exactk:.1f}%':>18}"
        )

    if ceilings:
        print(
            "\n  The pool ceiling bounds only the methods that rerank the pool.\n"
            "  The train ceiling bounds everything, and the gap between them is\n"
            "  what searching all of train instead of a shortlist can recover."
        )

    print(f"\n  paired difference against {baseline}, best-of-k similarity:")
    for label in labels[1:]:
        mean, lo, hi = paired_bootstrap(
            metrics[label]["bestk"], metrics[baseline]["bestk"]
        )
        verdict = (
            "no difference"
            if lo <= 0 <= hi
            else (f"{label} better" if mean > 0 else f"{baseline} better")
        )
        print(
            f"    {label:<24}{mean:+.4f}  (95% CI {lo:+.4f} … {hi:+.4f})  → {verdict}"
        )

    # ---------------- head to head ----------------
    print("\n" + "=" * 78)
    print(f"3. HEAD TO HEAD  — vs {baseline}, on rank-1 disagreements only")
    print("=" * 78)
    for label in labels[1:]:
        disagreed = [
            i
            for i, qid in enumerate(qids)
            if keys[label][qid][:1] != keys[baseline][qid][:1]
        ]
        wins = sum(
            1
            for i in disagreed
            if metrics[label]["top1"][i] > metrics[baseline]["top1"][i]
        )
        losses = sum(
            1
            for i in disagreed
            if metrics[label]["top1"][i] < metrics[baseline]["top1"][i]
        )
        print(
            f"  {label:<24}{len(disagreed):>4} disagreements  "
            f"{wins} won / {losses} lost / {len(disagreed) - wins - losses} tied"
        )

    # ---------------- slices ----------------
    print("\n" + "=" * 78)
    print("4. WHERE THE DIFFERENCE LIVES")
    print("=" * 78)
    slice_frame = pd.DataFrame(
        {label: metrics[label]["bestk"] for label in labels},
        index=pd.Index(qids, name="question_id"),
    )
    meta = (
        methods[baseline][methods[baseline]["rank"] == 1]
        .set_index("question_id")[["difficulty", "shape_tags"]]
        .reindex(qids)
    )
    slice_frame = slice_frame.join(meta)

    print("\n  mean best-of-k similarity by difficulty:")
    print(
        slice_frame.groupby("difficulty")[labels]
        .agg(["size", "mean"])
        .round(3)
        .to_string()
    )

    print(
        f"\n  by shape tag (tags on 30+ questions, biggest gain over {baseline} first):"
    )
    tag_rows = []
    tags = {
        tag
        for raw in slice_frame.shape_tags.fillna("")
        for tag in str(raw).split(",")
        if tag
    }
    for tag in tags:
        mask = slice_frame.shape_tags.fillna("").str.contains(rf"\b{tag}\b", regex=True)
        if mask.sum() < 30:
            continue
        row = {"tag": tag, "questions": int(mask.sum())}
        row |= {label: slice_frame.loc[mask, label].mean() for label in labels}
        row["best gain"] = max(row[label] for label in labels[1:]) - row[baseline]
        tag_rows.append(row)
    if tag_rows:
        print(
            pd.DataFrame(tag_rows)
            .sort_values("best gain", ascending=False)
            .round(3)
            .to_string(index=False)
        )

    for label in labels:
        frame = methods[label]
        # Self-reported equivalence (rerank) and skeleton match against the
        # prediction (predict) are each a method's own confidence. Whether
        # they track exemplar quality decides if they can be used to gate.
        for column in ("equivalence", "skeleton_match"):
            if column not in frame.columns:
                continue
            top = frame[frame["rank"] == 1].set_index("question_id")
            confidence = top[column].reindex(qids)
            if column == "skeleton_match":
                confidence = confidence.round(1)
            print(f"\n  does {label}'s {column} predict exemplar quality?")
            print(
                slice_frame.assign(confidence=list(confidence))
                .groupby("confidence")[labels]
                .agg(["size", "mean"])
                .round(3)
                .to_string()
            )

    # ------- does generate-then-retrieve only help where help is not needed? -------
    for label in labels:
        frame = methods[label]
        if "predicted_skeleton" not in frame.columns:
            continue
        print("\n" + "=" * 78)
        print(f"5. {label}: SPLIT BY WHETHER ITS PREDICTION WAS RIGHT")
        print("=" * 78)
        print(
            "  This method predicts the target structure with the same model that\n"
            "  will consume the exemplar, so its two halves mean different things.\n"
            "  Where the prediction is already correct the exemplar only confirms\n"
            "  it. The half that matters is where the prediction was wrong, because\n"
            "  those are the questions an exemplar exists to fix -- and there the\n"
            "  method is retrieving a demonstration of its own mistake."
        )
        predicted = (
            frame[frame["rank"] == 1]
            .set_index("question_id")["predicted_skeleton"]
            .reindex(qids)
        )
        split = slice_frame.assign(
            prediction_correct=[
                skel(gold[qid]) == str(shape) for qid, shape in zip(qids, predicted)
            ]
        )
        print()
        print(
            split.groupby("prediction_correct")[labels]
            .agg(["size", "mean"])
            .round(3)
            .to_string()
        )

    out = args.out or ROOT / "output/structural_exemplars_comparison.csv"
    slice_frame.to_csv(out)
    print(f"\nwrote {_shown(out)} ({len(slice_frame)} questions)")


def compute_ceilings(
    args: argparse.Namespace,
    train: list[dict],
    qids: list[int],
    gold: dict[int, str],
    skel,
) -> dict[str, dict[int, float]]:
    """Best exemplar reachable, in the lexical pool and in all of train.

    Both are computed by searching with the dev question's *gold* skeleton,
    which no method may do -- that is what makes them ceilings rather than
    methods. They answer different questions: the pool ceiling says what a
    perfect reranker would score, and the train ceiling says what a perfect
    retriever would score, so the gap between them is the cost of shortlisting.
    """
    dev = {question["question_id"]: question for question in load_dev(args.input)}
    index = ExemplarIndex(train, ngram=args.ngram)
    skeleton_index = SkeletonIndex(train, shortlist=args.shortlist)
    pool_best: dict[int, float] = {}
    train_best: dict[int, float] = {}
    for qid in qids:
        question = dev.get(qid)
        if question is None:
            continue
        gold_skeleton = skel(gold[qid])
        hits = index.query(
            question["question"], question["evidence"], args.candidates, args.tag_weight
        )
        pool_best[qid] = max(
            (
                SequenceMatcher(None, gold_skeleton, skel(example["SQL"])).ratio()
                for *_, example in hits
            ),
            default=0.0,
        )
        # Searching train *by the gold skeleton* -- the shortlist is the same
        # approximation the predict method lives with, so this is a tight
        # bound on it rather than an absolute maximum over all 9,428.
        found = skeleton_index.query(gold_skeleton, 1)
        train_best[qid] = found[0][0] if found else 0.0
    return {
        f"ceiling: best in pool of {args.candidates}": pool_best,
        "ceiling: best in all of train": train_best,
    }


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def shared(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--input", type=Path, default=ROOT / "datasets/bird/evaluation.json"
        )
        sub.add_argument(
            "--train", type=Path, default=ROOT / "datasets/bird/train/train.json"
        )
        sub.add_argument(
            "--candidates",
            type=int,
            default=30,
            help="Lexical pool size per question, which `rerank` judges.",
        )
        sub.add_argument("--ngram", type=int, default=3)
        sub.add_argument("--tag-weight", type=float, default=1.0)
        sub.add_argument(
            "--shortlist",
            type=int,
            default=400,
            help="Skeletons aligned per question in skeleton-space search.",
        )

    def llm_args(sub: argparse.ArgumentParser, stem: str) -> None:
        sub.add_argument("--out", type=Path, default=ROOT / f"output/{stem}.csv")
        sub.add_argument(
            "--cache", type=Path, default=ROOT / f"output/{stem}_cache.jsonl"
        )
        sub.add_argument(
            "--k", type=int, default=5, help="Exemplars kept per question."
        )
        sub.add_argument("--questions-per-request", type=int, default=4)
        sub.add_argument("--workers", type=int, default=8)
        sub.add_argument("--model", default="", help="Override REASONING_MODEL.")
        sub.add_argument("--max-tokens", type=int, default=8192)
        sub.add_argument("--limit", type=int, default=0, help="First N dev questions.")
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the first prompt and exit; sends nothing.",
        )

    rerank_parser = subparsers.add_parser(
        "rerank", help="LLM reranks the lexical retriever's shortlist."
    )
    shared(rerank_parser)
    llm_args(rerank_parser, "llm_structural_exemplars")
    rerank_parser.add_argument(
        "--hide-train-sql",
        action="store_true",
        help="Ablation: judge on question text alone, as the lexical retriever does.",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="LLM writes the SQL, then all of train is searched."
    )
    shared(predict_parser)
    llm_args(predict_parser, "predicted_structural_exemplars")
    predict_parser.add_argument(
        "--shots",
        type=int,
        default=len(SHOT_PATTERNS),
        help="Worked examples in the prompt, one construction each.",
    )

    compare_parser = subparsers.add_parser("compare", help="Score every mapping.")
    shared(compare_parser)
    compare_parser.add_argument(
        "--method",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeatable. The first is the baseline the rest are measured against.",
    )
    compare_parser.add_argument("--out", type=Path, default=None)
    compare_parser.add_argument(
        "--no-ceilings",
        dest="ceilings",
        action="store_false",
        help="Skip the pool and train ceilings (each rebuilds an index).",
    )

    args = parser.parse_args()
    if args.command == "compare" and not args.method:
        # The lexical arm is no longer a default: its producer was removed, so
        # the CSV cannot be regenerated and defaulting to a path most checkouts
        # do not have would fail the command for everyone. Pass it by hand --
        # --method lexical=<path> first -- to keep it as the baseline.
        args.method = [
            f"llm_rerank={ROOT / 'output/llm_structural_exemplars.csv'}",
            f"llm_predict={ROOT / 'output/predicted_structural_exemplars.csv'}",
        ]
    {"rerank": run, "predict": predict, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
