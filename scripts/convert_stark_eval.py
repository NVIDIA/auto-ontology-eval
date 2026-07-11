# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert STaRK QA CSV exports into ``datasets/amazon/evaluation.json``.

The human-generated STaRK-Amazon eval set ships as a CSV with columns
``id``, ``query``, ``answer_ids``, and ``answer_ids_source``. This script
maps those rows into the evaluation JSON format consumed by
:mod:`ontology_sql_eval.retrieval.eval_chatbot`.

Gold answers use ``answer_ids_source`` (human-curated product IDs). STaRK has
no ground-truth SQL, so the ``SQL`` field is left empty and the gold product
IDs are recorded in ``answer_raw`` (and ``evidence``) for scoring.

Usage::

    uv run python scripts/convert_stark_eval.py \\
        --input datasets/amazon/stark_qa_human_generated_eval.csv \\
        --output datasets/amazon/evaluation.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT = _REPO_ROOT / "datasets" / "amazon" / "stark_qa_human_generated_eval.csv"
_DEFAULT_OUTPUT = _REPO_ROOT / "datasets" / "amazon" / "evaluation.json"


def _parse_id_list(raw: str) -> list[int]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse ID list: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list of IDs, got {type(parsed).__name__}: {raw!r}")
    return [int(item) for item in parsed]


def _gold_answer_raw(product_ids: list[int]) -> str:
    if not product_ids:
        return ""
    lines = ["product_id", *[str(pid) for pid in product_ids]]
    return "\n".join(lines)


def convert(input_path: Path, output_path: Path) -> int:
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_id = int(row["id"])
            question = row["query"].strip()
            answer_ids = _parse_id_list(row.get("answer_ids", ""))
            gold_ids = _parse_id_list(row.get("answer_ids_source", ""))

            rows.append(
                {
                    "question_id": question_id,
                    "db_id": "amazon",
                    "question": question,
                    "evidence": json.dumps(
                        {
                            "answer_ids": answer_ids,
                            "answer_ids_source": gold_ids,
                            "split": "human_generated_eval",
                        },
                        ensure_ascii=False,
                    ),
                    "SQL": "",
                    "difficulty": "human_generated",
                    "answer_raw": _gold_answer_raw(gold_ids),
                    "answer": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info(
        "Wrote %d questions to %s from %s",
        len(rows),
        output_path,
        input_path,
    )
    return len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert STaRK QA CSV to datasets/amazon/evaluation.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"STaRK QA CSV input (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Evaluation JSON output (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
