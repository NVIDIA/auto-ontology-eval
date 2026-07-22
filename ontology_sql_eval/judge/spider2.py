# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score predicted SQL with the official Spider2-Snow evaluation suite.

The retrieval evaluator writes one CSV row per question. The upstream
Spider2-Snow evaluator instead expects a directory containing
``<instance_id>.sql`` files. This adapter performs that conversion, runs the
vendored official evaluator unchanged, and adds ``spider2_snow_score`` to a
copy of the input CSV.

Example::

    uv run python -m ontology_sql_eval.judge.spider2 \
        --input input/spider2_gpt-5.5.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "third_party" / "Spider2" / "spider2-snow" / "evaluation_suite"
GOLD_DIR = SUITE_DIR / "gold"
EVAL_SCRIPT = SUITE_DIR / "evaluate.py"
METADATA_JSONL = REPO_ROOT / "third_party" / "Spider2" / "spider2-snow" / "spider2-snow.jsonl"
DEFAULT_INPUT = REPO_ROOT / "input" / "spider2_gpt-5.5.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "spider2_gpt-5.5_scores.csv"


def _gold_ids() -> set[str]:
    config_path = GOLD_DIR / "spider2snow_eval.jsonl"
    return {
        str(json.loads(line)["instance_id"])
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _load_predictions(input_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not input_path.is_file():
        raise SystemExit(f"Spider2 eval CSV not found: {input_path}")

    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"question_id", "returned_sql"}
        missing = required - set(fieldnames)
        if missing:
            raise SystemExit(
                f"{input_path} is missing required column(s): {sorted(missing)}"
            )
        rows = list(reader)

    question_ids = [str(row["question_id"]).strip() for row in rows]
    duplicates = sorted({qid for qid in question_ids if question_ids.count(qid) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate question_id values: {duplicates}")

    unknown = sorted(set(question_ids) - _gold_ids())
    if unknown:
        raise SystemExit(
            f"{len(unknown)} question_id(s) are not in Spider2-Snow gold config: "
            f"{unknown[:10]}"
        )
    if not rows:
        raise SystemExit(f"No prediction rows found in {input_path}")
    return rows, fieldnames


def _validate_suite() -> None:
    required = [
        EVAL_SCRIPT,
        METADATA_JSONL,
        GOLD_DIR / "spider2snow_eval.jsonl",
        GOLD_DIR / "exec_result",
        SUITE_DIR / "snowflake_credential.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Official Spider2-Snow evaluation files are missing:\n"
            f"{paths}\nRun scripts/seed_spider2_snow.py first."
        )


def run_official_spider2_judge(
    input_path: Path,
    output_path: Path,
    *,
    max_workers: int = 20,
    timeout: int = 60,
) -> tuple[int, int]:
    """Run official execution evaluation and return ``(correct, total)``."""
    _validate_suite()
    rows, fieldnames = _load_predictions(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spider2-snow-judge-") as tmp:
        run_root = Path(tmp)
        predictions_dir = run_root / "predictions"
        predictions_dir.mkdir()

        for row in rows:
            question_id = str(row["question_id"]).strip()
            sql = str(row.get("returned_sql") or "").strip()
            (predictions_dir / f"{question_id}.sql").write_text(
                sql + ("\n" if sql else ""),
                encoding="utf-8",
            )

        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--mode",
            "sql",
            "--result_dir",
            str(predictions_dir),
            "--gold_dir",
            str(GOLD_DIR),
            "--max_workers",
            str(max_workers),
            "--timeout",
            str(timeout),
            "--temp_dir",
            str(run_root / "temp"),
        ]
        logger.info(
            "Running official Spider2-Snow evaluator for %d predictions", len(rows)
        )
        subprocess.run(command, cwd=SUITE_DIR, check=True)

        # The official suite writes IDs scoring 1 to a sibling CSV named after
        # result_dir (e.g. predictions.csv).
        correct_path = predictions_dir.with_suffix(".csv")
        if not correct_path.is_file():
            raise RuntimeError(
                f"Official evaluator did not produce expected file: {correct_path}"
            )
        with correct_path.open(encoding="utf-8", newline="") as handle:
            correct_ids = {
                str(row["output"]).strip()
                for row in csv.DictReader(handle)
                if row.get("output")
            }

    score_field = "spider2_snow_score"
    output_fields = fieldnames + ([score_field] if score_field not in fieldnames else [])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            row[score_field] = (
                "1" if str(row["question_id"]).strip() in correct_ids else "0"
            )
            writer.writerow(row)

    correct = len(correct_ids)
    total = len(rows)
    logger.info(
        "Official Spider2-Snow score: %.4f (%d/%d). Wrote %s",
        correct / total,
        correct,
        total,
        output_path,
    )
    return correct, total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    correct, total = run_official_spider2_judge(
        args.input.resolve(),
        args.output.resolve(),
        max_workers=args.max_workers,
        timeout=args.timeout,
    )
    print(f"Official Spider2-Snow score: {correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
