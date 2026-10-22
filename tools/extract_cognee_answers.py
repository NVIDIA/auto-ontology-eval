# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distil cognee's published BEAM artifacts into the reference we join against.

Their four run artifacts are 2MB each unpacked and almost entirely
``retrieval_context`` -- 85,000 to 109,000 characters per question. This keeps
the answer, the score, the judge's reason and the rubric, and records the
context only as a length.

Run it from the repo root with the four ``.json.gz`` files in one directory:

    uv run --no-sync python tools/extract_cognee_answers.py ~/Downloads

    curl -O https://raw.githubusercontent.com/topoteretes/cognee/main/cognee/\\
    eval_framework/beam/report_artifacts/100k_fixed/beam_existing_ingestion_\\
    metrics_conv1_hybrid_completion_20_20_qa_v1_run0.json.gz

The output is only trustworthy if it still adds up to what they published, so
that is asserted rather than assumed: every per-ability mean, the overall and
the tau are recomputed from the kept rows and checked against their
``hybrid_completion_20_20_qa_v1_cross_run_summary.json``. A mismatch anywhere
and nothing is written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
VENDORED = HERE / "conversation" / "beam_benchmark" / "upstream" / "cognee_report"
SUMMARY = VENDORED / "hybrid_completion_20_20_qa_v1_cross_run_summary.json"
OUTPUT = VENDORED / "cognee_answers_conv2.json"

STEM = "beam_existing_ingestion_metrics_conv1_hybrid_completion_20_20_qa_v1_run"
RUNS = 4

#: Their figures are published to full float precision, and the point of the
#: check is to catch a dropped or duplicated row rather than to police
#: arithmetic, so this is tight.
TOLERANCE = 1e-9


def _context_chars(context: Any) -> int:
    """Length of a retrieval context, whether it is one string or several."""
    if isinstance(context, str):
        return len(context)
    return sum(len(str(part)) for part in context)


def distil(directory: Path) -> dict[str, Any]:
    """Read their four artifacts into one question-keyed document."""
    questions: dict[str, dict[str, Any]] = {}
    for run in range(RUNS):
        path = directory / f"{STEM}{run}.json.gz"
        if not path.exists():
            path = directory / f"{STEM}{run}.json"
        if not path.exists():
            raise SystemExit(f"missing run {run}: looked for {path}")

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:  # type: ignore[operator]
            rows = json.load(handle)

        for row in rows:
            entry = questions.setdefault(
                row["question"].strip(),
                {
                    "conversation_id": str(row["conversation_id"]),
                    "question_idx": row["question_idx"],
                    "ability": row["question_type"],
                    "golden_answer": row["golden_answer"],
                    "rubric": row["rubric"],
                    "runs": [],
                },
            )
            entry["runs"].append(
                {
                    "run": run,
                    "answer": row["answer"],
                    "score": row["metrics"]["beam_rubric"]["score"],
                    "reason": row["metrics"]["beam_rubric"]["reason"],
                    "tau": row["metrics"]["kendall_tau"]["score"],
                    "context_chars": _context_chars(row["retrieval_context"]),
                }
            )

    for entry in questions.values():
        entry["runs"].sort(key=lambda r: r["run"])
    return {
        "source": (
            "https://github.com/topoteretes/cognee/tree/main/cognee/"
            "eval_framework/beam/report_artifacts/100k_fixed"
        ),
        "retriever": "hybrid_completion_20_20_qa_v1",
        "scale": "100K",
        "note": (
            "Distilled by tools/extract_cognee_answers.py. retrieval_context is "
            "kept as a length only; fetch the artifacts above for the text."
        ),
        "questions": questions,
    }


def verify(questions: dict[str, dict[str, Any]]) -> list[str]:
    """Every published figure this document should reproduce, and whether it does."""
    published = json.loads(SUMMARY.read_text())
    complaints: list[str] = []

    expected_rows = published["question_count"] * published["num_runs"]
    actual_rows = sum(len(q["runs"]) for q in questions.values())
    if actual_rows != expected_rows:
        complaints.append(f"{actual_rows} rows, expected {expected_rows}")

    by_ability: dict[str, list[float]] = defaultdict(list)
    for entry in questions.values():
        for run in entry["runs"]:
            by_ability[entry["ability"]].append(run["score"])

    for ability, stats in published["by_question_type"].items():
        want = stats["beam_rubric"]["mean"]
        if ability not in by_ability:
            complaints.append(f"{ability}: absent, expected mean {want}")
            continue
        got = statistics.mean(by_ability[ability])
        if abs(got - want) > TOLERANCE:
            complaints.append(f"{ability}: {got!r} != published {want!r}")

    # Theirs is the mean of per-run overalls, each a flat mean over questions.
    per_run = [
        statistics.mean(
            run["score"]
            for entry in questions.values()
            for run in entry["runs"]
            if run["run"] == index
        )
        for index in range(published["num_runs"])
    ]
    want_overall = published["overall"]["beam_rubric"]["mean"]
    if abs(statistics.mean(per_run) - want_overall) > TOLERANCE:
        complaints.append(
            f"overall: {statistics.mean(per_run)!r} != published {want_overall!r}"
        )

    taus = [
        run["tau"]
        for entry in questions.values()
        for run in entry["runs"]
        if run["tau"] is not None
    ]
    want_tau = published["overall"]["kendall_tau"]["mean"]
    if not taus:
        complaints.append("no tau values kept")
    elif abs(statistics.mean(taus) - want_tau) > TOLERANCE:
        complaints.append(f"tau: {statistics.mean(taus)!r} != published {want_tau!r}")

    return complaints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="where the four beam_existing_ingestion_metrics_*.json.gz files are",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    document = distil(args.directory)
    complaints = verify(document["questions"])
    if complaints:
        print("Does not reproduce their published figures:", file=sys.stderr)
        for complaint in complaints:
            print(f"  {complaint}", file=sys.stderr)
        return 1

    args.output.write_text(json.dumps(document, indent=2))
    questions = document["questions"]
    print(
        f"Wrote {args.output.relative_to(HERE)}: {len(questions)} questions, "
        f"{sum(len(q['runs']) for q in questions.values())} answers, "
        f"{args.output.stat().st_size / 1024:.0f} KB."
    )
    print("Reproduces all ten ability means, the 0.794 overall and tau 0.515.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
