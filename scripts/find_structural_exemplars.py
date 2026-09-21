"""Retrieve structurally equivalent train examples for every dev question.

Retrieval reads only what is available at inference time: the question text,
the evidence string, and the database name. It never reads the dev gold SQL --
``evaluation.json`` carries a ``SQL`` field and this module drops it on load so
it cannot leak into the ranking. Train gold SQL is fair game; that is the
exemplar being retrieved.

Matching runs on two signals:

*Shape tags* -- what the question asks for, as opposed to what it asks about.
"How many X" implies a COUNT, "highest X" implies a superlative, "X and Y"
implies two projected columns. These map to SQL construction far more reliably
than any word in the sentence.

*Template similarity* -- the question with its domain words removed. Proper
nouns, quoted values and numbers become placeholders, so "schools in Alameda
County" and "stores in Remulade city" collapse to nearly the same template.
Keeping the domain words would be actively harmful: dev and train share no
databases, so a rare noun like "Alameda" can only ever pull in noise.

Measured on BIRD dev against train (``--validate``), the k=5 candidate set
contains a query whose SQL skeleton exactly matches dev gold for 10.8% of
questions, against 0.5% for a random train example, at mean skeleton
similarity 0.84 vs 0.59. Defaults come from a sweep over ngram, tag weight and
k; note that the tag weight barely moves the result, because most questions
share the common tags and the soft overlap is therefore weakly discriminative.
Raising k is what helps.

Usage:
    python scripts/find_structural_exemplars.py --k 5
    python scripts/find_structural_exemplars.py --validate   # scores the
        # retrieval against dev gold *after* the fact, for measurement only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Shape tags: the question's demand, independent of its subject matter.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Template: the question with its domain vocabulary erased.
# --------------------------------------------------------------------------

QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,60})[\"'“”‘’]")
NUMBER = re.compile(r"\b\d[\d,.]*\b")
WORD = re.compile(r"[A-Za-z_][A-Za-z_'-]*")
# Words that carry question structure and must survive the proper-noun filter
# even when a sentence starts with them.
STRUCTURAL = {
    "what",
    "which",
    "who",
    "whose",
    "when",
    "where",
    "how",
    "list",
    "name",
    "give",
    "show",
    "state",
    "please",
    "find",
    "identify",
    "the",
    "of",
    "in",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "with",
    "without",
    "among",
    "between",
    "more",
    "less",
    "than",
    "most",
    "least",
    "all",
    "each",
    "per",
    "total",
    "average",
    "number",
    "many",
    "much",
    "percentage",
    "percent",
    "ratio",
    "rate",
    "difference",
    "highest",
    "lowest",
    "top",
    "first",
    "last",
    "also",
    "its",
    "their",
    "that",
    "this",
    "these",
    "those",
    "there",
    "any",
    "only",
    "both",
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


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=ROOT / "datasets/bird/evaluation.json"
    )
    ap.add_argument(
        "--train", type=Path, default=ROOT / "datasets/bird/train/train.json"
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "output/structural_exemplars.csv"
    )
    ap.add_argument("--k", type=int, default=5, help="Exemplars per question.")
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--tag-weight", type=float, default=1.0)
    ap.add_argument(
        "--validate",
        action="store_true",
        help="After retrieving, score exemplar SQL skeletons against dev gold. "
        "Measurement only -- gold is read after every ranking is fixed.",
    )
    args = ap.parse_args()

    dev = load_dev(args.input)
    train = json.loads(args.train.read_text())
    print(f"dev questions: {len(dev)}   train examples: {len(train)}")

    index = ExemplarIndex(train, ngram=args.ngram)
    print(f"index: {len(index.postings):,} template n-grams")

    rows = []
    for q in dev:
        hits = index.query(q["question"], q["evidence"], args.k, args.tag_weight)
        q_tags = shape_tags(q["question"], q["evidence"])
        for rank, (score, lexical, tag, ex_tags, ex) in enumerate(hits, start=1):
            rows.append(
                {
                    "question_id": q["question_id"],
                    "db_id": q["db_id"],
                    "difficulty": q["difficulty"],
                    "question": q["question"],
                    "shape_tags": ",".join(sorted(q_tags)),
                    "rank": rank,
                    "score": round(score, 4),
                    "lexical": round(lexical, 4),
                    "tag_overlap": round(tag, 4),
                    "shared_tags": ",".join(sorted(q_tags & ex_tags)),
                    "train_db": ex["db_id"],
                    "train_question": ex["question"].strip(),
                    "train_evidence": (ex.get("evidence") or "").strip(),
                    "train_sql": " ".join(ex["SQL"].split()),
                }
            )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    shown = args.out if ROOT not in args.out.parents else args.out.relative_to(ROOT)
    print(f"wrote {shown} ({len(df)} rows, k={args.k})")

    top = df[df["rank"] == 1]
    print(
        f"\nrank-1 exemplar: mean score {top.score.mean():.3f}, "
        f"mean tag overlap {top.tag_overlap.mean():.3f}, "
        f"mean lexical {top.lexical.mean():.3f}"
    )
    print(
        f"questions whose rank-1 exemplar shares no shape tag: "
        f"{(top.shared_tags.fillna('').str.len() == 0).sum()}"
    )
    print(f"distinct train databases used: {top.train_db.nunique()}")

    if args.validate:
        validate(args, df)


def validate(args: argparse.Namespace, df: pd.DataFrame) -> None:
    """Compare retrieved exemplar structure against dev gold, after the fact."""
    import random
    from difflib import SequenceMatcher

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_structural_twins import skeleton

    gold = {int(r["question_id"]): r["SQL"] for r in json.loads(args.input.read_text())}
    train = json.loads(args.train.read_text())
    train_skel = [skeleton(ex["SQL"]) for ex in train]

    rng = random.Random(0)
    retrieved, baseline, exact_r, exact_b = [], [], 0, 0
    best_of_k, exact_k = [], 0
    for qid, grp in df.groupby("question_id"):
        g_skel = skeleton(gold[qid])
        sims = [
            SequenceMatcher(None, g_skel, skeleton(sql)).ratio()
            for sql in grp.sort_values("rank").train_sql
        ]
        retrieved.append(sims[0])
        exact_r += int(sims[0] == 1.0)
        best_of_k.append(max(sims))
        exact_k += int(max(sims) == 1.0)
        rnd = train_skel[rng.randrange(len(train_skel))]
        r = SequenceMatcher(None, g_skel, rnd).ratio()
        baseline.append(r)
        exact_b += int(r == 1.0)

    n = len(retrieved)
    print("\n=== validation (gold read only here, after ranking) ===")
    print(f"  mean skeleton similarity, rank-1 exemplar : {sum(retrieved) / n:.3f}")
    print(f"  mean skeleton similarity, best of k       : {sum(best_of_k) / n:.3f}")
    print(f"  mean skeleton similarity, random baseline : {sum(baseline) / n:.3f}")
    print(
        f"  exact skeleton match, rank-1              : {exact_r} / {n} ({100 * exact_r / n:.1f}%)"
    )
    print(
        f"  exact skeleton match, best of k           : {exact_k} / {n} ({100 * exact_k / n:.1f}%)"
    )
    print(
        f"  exact skeleton match, random baseline     : {exact_b} / {n} ({100 * exact_b / n:.1f}%)"
    )


if __name__ == "__main__":
    main()
