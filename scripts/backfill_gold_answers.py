#!/usr/bin/env python3
"""Populate BEAVER ground-truth answers and re-score answer metrics.

BEAVER ships gold SQL but no materialized answers, so ``answer_raw`` is empty
for every question. That silently inflates scores: ``score_answer`` treats
"no expected numbers and no actual numbers" as a match and forces similarity to
1.0, so a question the agent answered with an empty result set scores a perfect
1.0 against an empty ground truth.

The eval already executes the gold SQL (``score_sql`` stores the rows in
``expected_sql_result``), so evaluated rows are repaired from the CSV itself.
The dataset JSON is filled by executing each gold query once against the live
database, which makes future runs correct at the source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from ontology_sql_eval.retrieval.scoring import score_answer, stringify_db_result

load_dotenv()
csv.field_size_limit(sys.maxsize)


def repair_csv(csv_path: Path, answers_by_qid: dict[str, str]) -> None:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print(f"  {csv_path}: no rows")
        return

    repaired = 0
    flipped = 0
    for row in rows:
        if str(row.get("expected_answer_raw") or "").strip():
            continue

        # Prefer the gold rows the eval already executed for this question.
        expected = str(row.get("expected_sql_result") or "").strip()
        if not expected:
            expected = answers_by_qid.get(str(row.get("question_id", "")), "")
        if not expected:
            continue

        before = str(row.get("answer_text_similarity", ""))
        row["expected_answer_raw"] = expected
        row.update(score_answer(expected, str(row.get("returned_answer") or "")))
        repaired += 1
        if before in {"1.0", "1"} and str(row["answer_text_similarity"]) not in {
            "1.0",
            "1",
        }:
            flipped += 1

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {csv_path.name}: repaired {repaired} row(s); {flipped} false 1.0 corrected")


def materialize_dataset(dataset_path: Path) -> dict[str, str]:
    """Execute each gold query once so ``answer_raw`` is correct at the source."""
    from gsf.connectors.registry import get_connectors

    questions = json.loads(dataset_path.read_text())
    connectors = {
        name: c
        for c in get_connectors()
        if (name := getattr(c, "database_name", None))
    }

    answers: dict[str, str] = {}
    filled = 0
    failed = 0
    for item in questions:
        qid = str(item.get("question_id", ""))
        gold_sql = str(item.get("SQL") or "").strip()
        if str(item.get("answer_raw") or "").strip():
            answers[qid] = str(item["answer_raw"])
            continue
        connector = connectors.get(str(item.get("db_id") or ""))
        if connector is None or not gold_sql:
            failed += 1
            continue
        try:
            rendered = stringify_db_result(connector.execute(gold_sql))
        except Exception as exc:  # noqa: BLE001 - one bad query must not stop the pass
            print(f"    {qid}: gold SQL failed — {type(exc).__name__}: {exc}")
            failed += 1
            continue
        item["answer_raw"] = rendered
        item["answer"] = rendered
        answers[qid] = rendered
        filled += 1

    dataset_path.write_text(json.dumps(questions, indent=2))
    print(f"  {dataset_path.name}: materialized {filled}, could not fill {failed}")
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill BEAVER gold answers.")
    parser.add_argument("--csv", type=Path, action="append", default=[])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Only repair CSVs, using answers already present in the dataset.",
    )
    args = parser.parse_args()

    if args.skip_dataset:
        questions = json.loads(args.dataset.read_text())
        answers = {
            str(q.get("question_id", "")): str(q.get("answer_raw") or "")
            for q in questions
        }
    else:
        print("Materializing gold answers from the live database…")
        answers = materialize_dataset(args.dataset)

    for csv_path in args.csv:
        if csv_path.exists():
            repair_csv(csv_path, answers)
        else:
            print(f"  skip (missing): {csv_path}")


if __name__ == "__main__":
    main()
