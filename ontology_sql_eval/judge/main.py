# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for ontology-sql-eval."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ontology_sql_eval.judge.runner import run_directory

_DEFAULT_INPUT_DIR = Path("input")
_DEFAULT_OUTPUT_DIR = Path("output")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ontology-sql-eval",
        description=(
            "Re-score Text-to-SQL evaluation CSVs with LLM-based logic/semantic scoring. "
            "Every CSV in the input folder is scored into '<name>_scores.csv' in the output folder."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help=f"Folder to scan for input CSVs (default: {_DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Folder to write scored CSVs (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent scoring workers (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    run_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
