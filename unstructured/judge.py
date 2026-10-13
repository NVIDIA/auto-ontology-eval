#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 — the LLM judge, as its own pass over answers already on disk.

The two pipelines retrieve and generate; this scores what they produced.  It is
a separate stage because it fails independently: the judge model has its own
quota, and on this endpoint it is routinely spent while generation is perfectly
healthy.  Keeping them apart means a dead judge costs nothing already paid for,
and a judge can be swapped without regenerating a single answer.

Judges either side, or both, since the two pipelines write the same fields:

    chunks      benchmark/results/answers.jsonl   benchmark/retrieve_generate.py
    attributes  results/answers.jsonl             attr_retrieve_generate.py

Only rows that need a verdict are sent.  "Need" is keyed on the judge *model*,
not merely on a score being absent, so changing ``pipeline.JUDGE_MODEL``
re-scores everything rather than leaving a report that silently mixes two
judges.  A row whose generation never produced an answer is skipped: there is
nothing to rate.

Usage
-----
    python -m unstructured.judge --dry-run
    python -m unstructured.judge --only attributes
    python -m unstructured.judge

Verdicts are appended, so an interrupted pass resumes where it stopped. Rebuild
each report afterwards with that pipeline's ``--report-only``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Loads the .env, sets the Ray override, and carries every setting this stage
# uses. Nothing here imports either retrieve/generate pipeline: this stage meets
# them only through the answer files on disk.
from unstructured import pipeline

JUDGE_MODEL = pipeline.JUDGE_MODEL

#: Which answers file belongs to which pipeline, and the command that rebuilds
#: that pipeline's report once verdicts have landed.
TARGETS = {
    "chunks": (
        pipeline.CHUNK_RESULTS,
        "python -m unstructured.benchmark.retrieve_generate --report-only",
    ),
    "attributes": (
        pipeline.ATTRIBUTE_RESULTS,
        "python -m unstructured.attr_retrieve_generate --report-only",
    ),
}


def load(path: Path) -> dict[int, dict]:
    """Saved rows by query id; a later row for a query supersedes an earlier one."""
    if not path.exists():
        return {}
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["query_id"]: row for row in rows}


def needs_verdict(row: dict) -> bool:
    if not row.get("generated") or row.get("gen_error"):
        return False
    return not (
        row.get("judge_score") is not None and row.get("judge_model") == JUDGE_MODEL
    )


def build_judge():
    from nemo_retriever.models.llm.clients import LLMJudge

    return LLMJudge.from_kwargs(
        model=JUDGE_MODEL,
        api_base=pipeline.JUDGE_API_BASE,
        api_key=pipeline.JUDGE_API_KEY,
        temperature=0.1,
        max_tokens=pipeline.JUDGE_MAX_TOKENS,
        timeout=pipeline.JUDGE_TIMEOUT,
        num_retries=pipeline.JUDGE_RETRIES,
    )


def judge_one(judge, row: dict) -> dict:
    from nemo_retriever.tools.evaluation.scoring import classify_failure

    # AnswerAccuracy spends two requests per verdict: it rates the answer twice,
    # with reference and candidate swapped, and averages the two.
    pipeline.JUDGE_LIMITER.acquire(2)
    verdict = judge.judge(row["query"], row["answer"], row["generated"])
    return {
        **row,
        "judge_score": verdict.score,
        "judge_error": verdict.error,
        "judge_model": JUDGE_MODEL,
        "failure_mode": classify_failure(
            ref_in_chunks=row["answer_in_context"],
            judge_score=verdict.score,
            gen_error=row.get("gen_error"),
            candidate=row["generated"],
        ),
    }


def process(
    name: str,
    directory: Path,
    report_command: str,
    *,
    dry_run: bool,
    limit: int | None,
) -> None:
    path = pipeline.Results(directory).answers_jsonl
    saved = load(path)
    if not saved:
        print(f"\n{name}: no answers at {path} — generate them first, skipping.")
        return

    todo = [r for r in saved.values() if needs_verdict(r)]
    unjudgeable = sum(1 for r in saved.values() if r.get("gen_error"))
    done = len(saved) - len(todo) - unjudgeable
    print(
        f"\n{name}: {len(saved)} answer(s) — {done} judged by {JUDGE_MODEL}, "
        f"{len(todo)} to judge"
        + (f", {unjudgeable} unjudgeable (no answer was generated)" if unjudgeable else "")
    )
    if limit:
        todo = todo[:limit]
        print(f"  limited to {len(todo)}")
    if not todo or dry_run:
        if dry_run and todo:
            print(f"  dry run — would judge {len(todo)}, nothing sent")
        return

    judge = build_judge()
    scored = 0
    with path.open("a") as handle, ThreadPoolExecutor(pipeline.MAX_WORKERS) as pool:
        futures = [pool.submit(judge_one, judge, r) for r in todo]
        # Completion order, not submission order, so one slow verdict cannot
        # hold finished work unwritten.
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            scored += row["judge_score"] is not None
            print(
                f"  {i:3d}/{len(todo)}  q{row['query_id']:<5} "
                f"score={row['judge_score'] if row['judge_score'] is not None else 'none':<5} "
                f"({scored} scored, {i - scored} refused)"
            )

    print(f"{name}: {scored} of {len(todo)} verdict(s) returned.")
    if scored == 0:
        print(
            "  Nothing came back. The judge model's quota is spent; the answers "
            "are safe on disk, so re-run this when it recovers."
        )
    else:
        print(f"  Rebuild the report: {report_command}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=sorted(TARGETS), help="Judge one side instead of both."
    )
    parser.add_argument(
        "--limit", type=int, help="Judge only the first N queries needing a verdict."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be judged without calling the model.",
    )
    args = parser.parse_args()

    pipeline.require_judge_key()
    targets = {args.only: TARGETS[args.only]} if args.only else TARGETS
    print(f"Judge: {JUDGE_MODEL}")
    print(f"  endpoint={pipeline.JUDGE_API_BASE}")
    print(
        f"  max_tokens={pipeline.JUDGE_MAX_TOKENS} timeout={pipeline.JUDGE_TIMEOUT}s "
        f"retries={pipeline.JUDGE_RETRIES} workers={pipeline.MAX_WORKERS} "
        f"rpm={pipeline.JUDGE_RPM}"
    )
    for name, (directory, report_command) in targets.items():
        process(
            name,
            directory,
            report_command,
            dry_run=args.dry_run,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
