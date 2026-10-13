#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything the two retrieval paths share, so retrieval is the only variable.

Two pipelines are compared on the ViDoRe v3 HR corpus:

    benchmark/retrieve_generate.py   query → chunk vectors (LanceDB) → chunks
    attr_retrieve_generate.py        query → attribute vectors → linked chunks

They answer the same questions from different evidence.  For the comparison to
mean anything, *only* the retrieval step may differ — so the generator, prompt,
token budget, reasoning cap, rate limiters, page-key arithmetic, graded qrels,
scoring tiers and report format all live here, on one path, used by both.

This is not merely tidiness.  When the two halves were separate files that
imported across, their judge timeout and token budget drifted apart unnoticed,
which is exactly the kind of difference that invalidates a comparison.

A retriever is anything with:

    name         str, also the label in reports
    description  str, one line for the run header
    setup()      open connections, build indexes; called once per run
    search(q)    -> (context_chunks, ranked_page_keys)

Tier 3, the LLM judge, is ``judge.py``: a separate pass over the answers on
disk, because the judge model has its own quota and is routinely spent while
generation is healthy.
"""

from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

MODULE_DIR = Path(__file__).parent
REPO_ROOT = MODULE_DIR.parent

load_dotenv(REPO_ROOT / ".env")

# Ray advertises Apple Silicon as a GPU, and NeMo Retriever then picks the
# CUDA-only operator variants instead of the hosted NIM endpoints. Must be set
# before anything imports nemo_retriever, which is why it is here: both
# pipelines import this module first.
os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# --- corpus -------------------------------------------------------------------
DATA_DIR = MODULE_DIR / "data" / "vidore_v3_hr"
PDF_DIR = DATA_DIR / "pdfs"
QUERIES_CSV = DATA_DIR / "queries_en.csv"
QRELS_CSV = DATA_DIR / "qrels" / "qrels_en.csv"

LANGUAGE = "english"
KS = (1, 3, 5, 10)

# Each pipeline writes its own directory, so a run can never overwrite the
# numbers it is being compared against. Both paths are named here rather than
# in the pipelines, so neither has to import the other just to find the report
# it compares itself to — the baseline must not drag in Postgres, and the
# attribute side must not drag in LanceDB.
CHUNK_RESULTS = MODULE_DIR / "benchmark" / "results"
ATTRIBUTE_RESULTS = MODULE_DIR / "results"


# --- models -------------------------------------------------------------------
API_BASE = "https://integrate.api.nvidia.com/v1"

# The published baseline generated with llama-3.3-nemotron-super-49b-v1.5 and
# judged with Mixtral-8x22B; both now return 410 from the hosted API.
GEN_MODEL = "nvidia/nemotron-3-super-120b-a12b"
GEN_MAX_TOKENS = 4096

# How much the generator may reason before answering. Unbounded, it spends the
# whole budget thinking on hard queries and returns nothing ("thinking
# truncated"); capping it keeps the reasoning and leaves room for the answer.
# Measured on a query that truncates with reasoning unbounded:
#
#   default  unbounded           4096 tokens, 47s, empty answer
#   low      kept but capped      502 tokens,  5s, answers
#   off      no reasoning block    19 tokens,  1s, answers
#
# The endpoint accepts several other spellings of this and then ignores them,
# so they look like they work and change nothing: a flat `reasoning_effort`, a
# `chat_template_kwargs.thinking_budget`, a "/no_think" suffix, a "detailed
# thinking off" system line, and asking for brief reasoning in the prompt.
# `max_thinking_tokens` is rejected with a 400.
REASONING_MODES = {
    "default": {},
    "off": {"chat_template_kwargs": {"thinking": False}},
    "low": {"reasoning": {"effort": "low"}},
    "medium": {"reasoning": {"effort": "medium"}},
    "high": {"reasoning": {"effort": "high"}},
}
GEN_REASONING = "low"

# The judge runs on its own endpoint and credential, not the generator's. The
# free integrate.api endpoint has no judge worth using: of everything `/v1/models`
# lists, almost nothing serves a request, and the one model that did — scoring a
# correct / partial / wrong / refusal answer 1.0 / 0.25 / 0.0 / 0.0 —
# exhausted its allowance after ~850 requests and then 429'd everything.
#
#   nvidia/nemotron-3.5-lightning-30b-a3b   usable, ~100s a verdict, then 429s
#   openai/gpt-oss-20b                      scores partial same as wrong
#   z-ai/glm-5.3                            429 on a single cold request
#   z-ai/glm-5.3-flash, google/gemma-4-31b-it, deepseek-ai/deepseek-v4.1-flash
#                                           time out
#   mistralai/mistral-large-2-instruct      404, listed but no endpoint
#
# inference-api.nvidia.com, reached with the sk- key, serves Claude. Probed on
# answers the Nemotron judge had already rated across its whole range, Opus
# agrees at both extremes, awards partial credit in between, and returns a
# verdict in 3.9s against ~100s. It is also a different family from the
# generator, so it drops the self-preference bias the Nemotron judge carried.
#
# Changing any of these re-judges every saved answer without regenerating one,
# because a row needs a verdict unless it already has one from this exact model.
JUDGE_API_BASE = os.environ.get(
    "JUDGE_BASE_URL", os.environ.get("DEFAULT_MODELS_ENDPOINT", API_BASE)
)
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY", os.environ.get("DEFAULT_MODELS_API_KEY", "")
)
# litellm routes by prefix; this endpoint is OpenAI-compatible, so `openai/`
# sends the rest of the string through as the model name.
JUDGE_MODEL = "openai/aws/anthropic/bedrock-claude-opus-4-8"
# Claude answers with the rating directly instead of thinking its way there, so
# the 32768 the Nemotron judge needed to avoid truncating mid-thought is waste.
JUDGE_MAX_TOKENS = 1024
JUDGE_TIMEOUT = 120.0
JUDGE_RETRIES = 2


# --- retrieval budgets --------------------------------------------------------
# Chunks retrieved per query; several can land on the same page.
CHUNK_K = 30
# Chunks handed to the generator, matching the QA_TOP_K default.
CONTEXT_K = 5

# Attributes matched per query. Each carries roughly five to six linked chunks,
# so this lands near the 30-chunk budget rather than handing the generator a
# wider net than chunk search had.
ATTRIBUTE_K = 10
# Distinct chunks the attribute page ranking is derived from, matching CHUNK_K.
# Chunk search's 30 chunks collapse to ~25 pages, so budgeting in chunks keeps
# both paths level; budgeting in pages would quietly give attributes a longer
# ranking off the same evidence.
CHUNK_CAP = CHUNK_K

# NeMo's extractor numbers pages from one; the qrels number them from zero.
# Both retrievers apply this, and omitting it is not a small error: it drops
# recall@5 from ~0.5 to ~0.06, since almost every page key then misses by one.
PAGE_OFFSET = -1


# --- pacing -------------------------------------------------------------------
# The hosted endpoint caps requests per minute per model, and the buckets are
# independent, so generation and judging never contend. Measured ceilings:
# nemotron-3-super tops out near 8/min and gets *worse* when pushed harder.
#
# Judging costs two requests per query (AnswerAccuracy rates each answer twice
# with the roles swapped), so the judge supplies JUDGE_RPM / 2 queries a minute.
#
# Pacing client-side keeps every request inside the budget instead of spending
# the budget on rejections; workers simply block on their model's limiter.
GEN_RPM = 7
# The judge is on a different endpoint and key from the generator, so its budget
# is unrelated to GEN_RPM. 24 was paced for the free endpoint, where it was never
# the constraint: verdicts took ~100s, so eight workers could only produce twelve
# a minute anyway. A 3.9s verdict makes the limiter the only thing holding the
# judge back, hence the raise. Lower it if refusals appear.
JUDGE_RPM = 60
MAX_WORKERS = 8

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 5

# Transient API failures are retried on the next run instead of being cached as
# a result. "thinking_truncated" is retried too: it used to be a dead end, since
# retrying at temperature 0 under the same budget truncates identically, but
# with GEN_REASONING capped the retry now has room to reach an answer.
RETRYABLE_ERRORS = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "APIStatusError",
)


class RateLimiter:
    """Spaces requests to one model evenly across all worker threads."""

    def __init__(self, per_minute: float):
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self, requests: int = 1) -> None:
        """Block until *requests* slots are available, then reserve them."""
        with self._lock:
            start = max(time.monotonic(), self._next)
            self._next = start + self._interval * requests
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


# One limiter per model, shared by every worker in the process. Two processes
# would each pace to the full budget and together exceed it, so the stages are
# meant to be run one at a time.
GEN_LIMITER = RateLimiter(GEN_RPM)
JUDGE_LIMITER = RateLimiter(JUDGE_RPM)


def require_api_key() -> None:
    if not NVIDIA_API_KEY:
        raise SystemExit(
            "NVIDIA_API_KEY is not set. Add it to .env — get a free key at "
            "https://build.nvidia.com"
        )


def require_judge_key() -> None:
    """The judge has its own credential, so generation can run without it."""
    if not JUDGE_API_KEY:
        raise SystemExit(
            "No judge API key. Set JUDGE_API_KEY (or DEFAULT_MODELS_API_KEY) in "
            f".env — {JUDGE_MODEL} is served by {JUDGE_API_BASE}"
        )


def make_client():
    from openai import OpenAI

    return OpenAI(
        base_url=API_BASE, api_key=NVIDIA_API_KEY, timeout=180.0, max_retries=3
    )


# ── dataset ──────────────────────────────────────────────────────────────────


def page_key(doc_id: str, page: int) -> str:
    """Identifier for one page of one document, page numbered from zero."""
    return f"{doc_id}#{page}"


def load_queries(limit: int | None = None) -> list[dict]:
    """The English queries that carry a reference answer, by ``query_id``.

    Read from the csv the dataset ships rather than the queries parquet: the
    two carry the same rows, and the csv needs no pyarrow.  Queries without a
    reference answer cannot be scored on tiers 2 and 3, so they are dropped.
    """
    with QUERIES_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    answered = [
        r
        for r in rows
        if (r.get("answer") or "").strip()
        and (r.get("language") or LANGUAGE) == LANGUAGE
    ]
    answered.sort(key=lambda r: int(r["query_id"]))
    if limit:
        answered = answered[:limit]
    return [
        {"query_id": int(r["query_id"]), "query": r["query"], "answer": r["answer"]}
        for r in answered
    ]


def load_qrels() -> dict[int, dict[str, int]]:
    """Graded gold pages per query: ``{query_id: {page_key: relevance}}``."""
    qrels: dict[int, dict[str, int]] = {}
    with QRELS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # page_number is the corpus page_number_in_doc, already zero-based.
            key = page_key(row["doc_id"], int(row["page_number"]))
            qrels.setdefault(int(row["query_id"]), {})[key] = int(row["score"])
    return qrels


# ── results on disk ──────────────────────────────────────────────────────────


class Results:
    """The three files one pipeline writes, and the rows already in them."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.retrieval_json = directory / "retrieval.json"
        self.answers_jsonl = directory / "answers.jsonl"
        self.report_json = directory / "report.json"

    def answers(self) -> dict[int, dict]:
        """Saved rows by query id; a later row supersedes an earlier one."""
        if not self.answers_jsonl.exists():
            return {}
        with self.answers_jsonl.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return {row["query_id"]: row for row in rows}

    def retrieval_cache(self) -> dict[str, dict]:
        """Searches already paid for, keyed by query text.

        Cached because every query costs an embedding call against a
        rate-limited endpoint, and a resumed run must not pay twice. Answers
        from an earlier run already carry the chunks they used, so they seed
        the cache too.
        """
        cached: dict[str, dict] = {}
        if self.retrieval_json.exists():
            cached = {
                r["query"]: r for r in json.loads(self.retrieval_json.read_text())
            }
        for row in self.answers().values():
            cached.setdefault(
                row["query"],
                {k: row[k] for k in ("query_id", "query", "answer", "chunks", "pages")},
            )
        return cached

    def save_retrieval(self, cached: dict[str, dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_json.write_text(json.dumps(list(cached.values())))


def is_retryable(row: dict) -> bool:
    error = row.get("gen_error") or ""
    return any(error.startswith(name) for name in RETRYABLE_ERRORS) or error in (
        "retries_exhausted",
        "thinking_truncated",
    )


def is_answered(row: dict) -> bool:
    """Whether generation is finished with this query, judged or not."""
    if is_retryable(row):
        return False
    return bool(row.get("gen_error")) or bool(row.get("generated"))


# ── generate ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the information provided in the context below. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be concise and factual."
)
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate(client, record: dict) -> tuple[str, str | None]:
    """Grounded answer for one query, as ``(answer, error)``."""
    context = "\n\n---\n\n".join(record["chunks"]) or "(no context retrieved)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, query=record["query"]),
        },
    ]
    body = REASONING_MODES[GEN_REASONING]
    extra = {"extra_body": body} if body else {}

    for attempt in range(MAX_ATTEMPTS):
        GEN_LIMITER.acquire()
        try:
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=GEN_MAX_TOKENS,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 — one bad query must not end the run
            error = f"{type(exc).__name__}: {exc}"[:200]
            if (
                type(exc).__name__ not in RETRYABLE_ERRORS
                or attempt == MAX_ATTEMPTS - 1
            ):
                return "", error
            time.sleep(BACKOFF_BASE_SECONDS * 2**attempt + random.random())
            continue

        answer = (response.choices[0].message.content or "").strip()
        # Reasoning models can spend the whole budget thinking and return
        # nothing. Retryable now, because GEN_REASONING caps the thinking.
        return answer, None if answer else "thinking_truncated"

    return "", "retries_exhausted"


def answer_one(record: dict, client, retriever, cache: dict[str, dict]) -> dict:
    """The whole per-query pipeline: retrieve, generate, score tiers 1 and 2.

    Retrieval lives in here rather than in a pass of its own so the first
    answers appear immediately.  It costs nothing extra: every worker then
    blocks on the 7/min generation limiter, which paces the embedding calls
    far below anything the endpoint objects to.
    """
    from nemo_retriever.tools.evaluation.scoring import (
        answer_in_context,
        classify_failure,
        token_f1,
    )

    if not record.get("chunks"):
        hit = cache.get(record["query"])
        if hit:
            record = {**record, "chunks": hit["chunks"], "pages": hit["pages"]}
        else:
            chunks, pages = retriever.search(record["query"])
            record = {**record, "chunks": chunks, "pages": pages}

    answer, gen_error = generate(client, record)
    in_context = answer_in_context(record["answer"], record["chunks"])
    return {
        **record,
        "generated": answer,
        "gen_error": gen_error,
        # Tier 3 is judge.py's. Left null rather than absent so judge.py can
        # tell "not yet judged" from "judged and refused".
        "judge_score": None,
        "judge_error": None,
        "judge_model": None,
        "answer_in_context": in_context,
        "token_f1": token_f1(record["answer"], answer)["f1"],
        "failure_mode": classify_failure(
            ref_in_chunks=in_context,
            judge_score=None,
            gen_error=gen_error,
            candidate=answer,
        ),
    }


def _running(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, width: int = 5) -> str:
    return f"{value:{width}.3f}" if value is not None else " " * (width - 3) + "n/a"


def run_pipeline(queries: list[dict], retriever, results: Results) -> list[dict]:
    """Retrieve and answer every query, reporting as each one lands."""
    saved = results.answers()
    done = {qid: row for qid, row in saved.items() if is_answered(row)}
    todo = [r for r in queries if r["query_id"] not in done]
    cache = results.retrieval_cache()
    print(
        f"  {len(done)} already answered, {len(todo)} to run "
        f"({len(cache)} searches cached)"
    )
    if not todo:
        return [done[r["query_id"]] for r in queries if r["query_id"] in done]

    # Only connect if a search is actually needed. Regenerating an answer whose
    # retrieval is already cached must not require the store to be up: the
    # chunks are on disk, and the one thing being re-paid for is generation.
    if any(r["query"] not in cache for r in todo):
        retriever.setup()
    client = make_client()

    header = f"  {'#':>9}  {'query':>5}  {'ctx':^5}{'f1':>7}   {'running f1':>10}"
    print(f"\n{header}\n  {'-' * (len(header) - 2)}")

    results.dir.mkdir(parents=True, exist_ok=True)
    scored: dict[int, dict] = dict(done)
    landed: list[dict] = []
    with results.answers_jsonl.open("a") as handle, ThreadPoolExecutor(
        MAX_WORKERS
    ) as pool:
        futures = [pool.submit(answer_one, r, client, retriever, cache) for r in todo]
        # Completion order, not submission order, so one slow query cannot hold
        # finished work unwritten.
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            scored[row["query_id"]] = row
            landed.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()

            cache[row["query"]] = {
                k: row[k] for k in ("query_id", "query", "answer", "chunks", "pages")
            }
            if i % 25 == 0 or i == len(todo):
                results.save_retrieval(cache)

            print(
                f"  {i:4d}/{len(todo):<4d}  {row['query_id']:>5}  "
                f"{'Y' if row['answer_in_context'] else '.':^5}"
                f"{row['token_f1']:7.3f}   "
                f"{_fmt(_running(landed, 'token_f1'), 10)}"
                + (f"   {row['gen_error']}" if row["gen_error"] else "")
            )

    return [scored[r["query_id"]] for r in queries if r["query_id"] in scored]


# ── score ────────────────────────────────────────────────────────────────────


def build_report(records: list[dict], qrels: dict[int, dict[str, int]]) -> dict:
    from nemo_retriever.tools.recall.beir import compute_beir_metrics

    # compute_beir_metrics ranks by descending score, so turn rank into a score.
    run = {
        str(r["query_id"]): {
            p: float(len(r["pages"]) - i) for i, p in enumerate(r["pages"])
        }
        for r in records
    }
    gold = {str(r["query_id"]): qrels.get(r["query_id"], {}) for r in records}

    # Only verdicts from the judge this report is labelled with. Averaging every
    # score on disk would report a previous judge's numbers under the current
    # judge's name, which is how a stale 0.585 from a since-retired model
    # survived a judge swap and read as if it were the new one's.
    judged = [
        r["judge_score"]
        for r in records
        if r.get("judge_score") is not None and r.get("judge_model") == JUDGE_MODEL
    ]
    stale = sum(
        1
        for r in records
        if r.get("judge_score") is not None and r.get("judge_model") != JUDGE_MODEL
    )
    return {
        "queries": len(records),
        "generator": GEN_MODEL,
        "judge": JUDGE_MODEL,
        # Answers still carrying a verdict from a different judge. Run judge.py
        # to replace them; until then they are excluded from the mean above.
        "stale_verdicts": stale,
        "retrieval": compute_beir_metrics(gold, run, ks=KS),
        "answer_in_context_rate": sum(r["answer_in_context"] for r in records)
        / len(records),
        "mean_token_f1": sum(r["token_f1"] for r in records) / len(records),
        "mean_judge_score": (sum(judged) / len(judged)) if judged else None,
        "judged": len(judged),
        "judge_score_distribution": dict(sorted(Counter(judged).items())),
        "failure_modes": dict(
            Counter(r["failure_mode"] for r in records).most_common()
        ),
    }


def print_report(report: dict, label: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"ViDoRe v3 HR — {report['queries']} English queries, {label} search")
    print(f"  generator: {report['generator']}")
    print(f"  judge:     {report['judge']}")
    print(f"{'=' * 64}")

    print("\nRetrieval (page level, against the shipped qrels)")
    for name, value in report["retrieval"].items():
        print(f"  {name:<12} {value:.4f}")

    print("\nTier 1 — retrieval quality")
    print(f"  answer-in-context rate   {report['answer_in_context_rate']:.1%}")

    print("\nTier 2 — programmatic answer quality")
    print(f"  mean token F1            {report['mean_token_f1']:.4f}")

    print("\nTier 3 — LLM judge (ragas AnswerAccuracy, 0-1)")
    mean = report["mean_judge_score"]
    print(
        f"  mean judge score         {mean:.4f}  (scored {report['judged']})"
        if mean is not None
        else "  mean judge score         n/a  (run judge.py)"
    )
    if report.get("stale_verdicts"):
        print(
            f"  {report['stale_verdicts']} answer(s) still hold a verdict from a "
            "different judge and are excluded; run judge.py to replace them."
        )
    for score, count in report["judge_score_distribution"].items():
        print(f"    {score:<4} {count}")

    print("\nFailure modes")
    for mode, count in report["failure_modes"].items():
        print(f"  {mode:<26} {count}")
    print()


def print_comparison(
    this_label: str, this_report: dict, other_label: str, other_path: Path
) -> None:
    """Print this run beside the other pipeline's, where one exists."""
    if not other_path.exists():
        print(f"No {other_label} report at {other_path} — nothing to compare against.")
        return

    other = json.loads(other_path.read_text())
    rows: list[tuple[str, float | None, float | None]] = []
    for name, value in this_report.get("retrieval", {}).items():
        rows.append((name, value, other.get("retrieval", {}).get(name)))
    for name in ("answer_in_context_rate", "mean_token_f1", "mean_judge_score"):
        rows.append((name, this_report.get(name), other.get(name)))

    width = max(len(name) for name, _, _ in rows)
    print(f"\n{'=' * (width + 34)}")
    print(f"{this_label} search  vs  {other_label} search")
    print(
        f"queries: {this_report.get('queries')} {this_label}, "
        f"{other.get('queries')} {other_label}"
    )
    print("=" * (width + 34))
    print(
        f"{'metric'.ljust(width)}   {this_label[:9]:>9} {other_label[:11]:>11} {'delta':>9}"
    )
    for name, mine, theirs in rows:
        # A metric one side has measured is still worth showing: judging the two
        # sides is two separate passes, so a missing value on one is routine and
        # must not blank out the other. Only the delta needs both.
        left = f"{mine:9.4f}" if mine is not None else f"{'n/a':>9}"
        right = f"{theirs:11.4f}" if theirs is not None else f"{'n/a':>11}"
        delta = (
            f"{mine - theirs:+9.4f}"
            if mine is not None and theirs is not None
            else f"{'':>9}"
        )
        print(f"{name.ljust(width)}   {left} {right} {delta}")
    print()


def report_and_compare(
    records: list[dict],
    qrels: dict[int, dict[str, int]],
    results: Results,
    label: str,
    other_label: str,
    other_report: Path,
) -> dict:
    """Build, print, save and compare one pipeline's report."""
    print("\nSCORE")
    report = build_report(records, qrels)
    results.dir.mkdir(parents=True, exist_ok=True)
    results.report_json.write_text(json.dumps(report, indent=2))
    print_report(report, label)
    print(f"Report: {results.report_json}")
    print(f"Per-query detail: {results.answers_jsonl}")
    print_comparison(label, report, other_label, other_report)
    return report
