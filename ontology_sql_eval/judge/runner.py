# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive LLM scoring across an evaluation CSV.

Reads a CSV produced by an eval run (must contain ``question``,
``expected_sql``, ``returned_sql``, and optionally ``returned_answer``),
scores each row with the LLM, and writes a new CSV with the additional
LLM score columns appended.
"""

from __future__ import annotations

import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ontology_sql_eval.judge.config import Settings, load_settings
from ontology_sql_eval.judge.models import LLM_SCORE_FIELDS, SqlScore
from ontology_sql_eval.judge.scorer import score_sql

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


def _score_row(settings: Settings, row: dict[str, str]) -> SqlScore | None:
    return score_sql(
        settings,
        question=row.get("question", ""),
        sql_code=row.get("returned_sql", ""),
        ground_truth_sql=row.get("expected_sql", ""),
        sql_result_preview=row.get("returned_answer", ""),
    )


def _empty_score_fields() -> dict[str, str]:
    return {field: "" for field in LLM_SCORE_FIELDS}


def run(
    input_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    workers: int = 1,
) -> None:
    """Score every row of ``input_path`` and write results to ``output_path``."""
    settings = settings or load_settings()

    with input_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    new_fields = [f for f in LLM_SCORE_FIELDS if f not in original_fields]
    out_fields = original_fields + new_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    scored = 0

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = pool.map(lambda row: _score_row(settings, row), rows)
        else:
            results = (_score_row(settings, row) for row in rows)

        for i, (row, score) in enumerate(zip(rows, results)):
            question = row.get("question", "")
            logger.info(
                "[%d/%d] Scored q%s: %s",
                i + 1,
                total,
                row.get("question_id", i + 1),
                question[:80],
            )

            out_row = dict(row)
            if score is not None:
                out_row.update(score.as_csv_row())
                scored += 1
            else:
                for field, value in _empty_score_fields().items():
                    out_row.setdefault(field, value)

            writer.writerow(out_row)
            f.flush()

    logger.info("Scored %d/%d rows. Output: %s", scored, total, output_path)
    try:
        import pandas as pd

        df = pd.read_csv(output_path)

        def _avg(col: str) -> float:
            if col not in df.columns:
                return float("nan")
            series = pd.Series(pd.to_numeric(df[col], errors="coerce"))
            return float(series.mean())

        sep = "=" * 50
        logger.info("%s", sep)
        logger.info("  LLM SCORING SUMMARY  (%d questions)", len(df))
        logger.info("%s", sep)
        logger.info("  Logic match            : %.4f", _avg("llm_logic_match"))
        logger.info("  Semantic match         : %.4f", _avg("llm_semantic_match"))
        logger.info("  Final weighted score   : %.4f", _avg("llm_final_weighted_score"))
        logger.info("  SQL vs ground truth    : %.4f", _avg("llm_sql_vs_ground_truth"))
        logger.info("%s", sep)
    except Exception as exc:  # noqa: BLE001 - summary is best-effort
        logger.warning("Could not compute summary: %s", exc)


def run_directory(
    input_dir: Path,
    output_dir: Path,
    settings: Settings | None = None,
    workers: int = 1,
) -> None:
    """Score every ``*.csv`` in ``input_dir``, writing ``<name>_scores.csv`` to ``output_dir``."""
    settings = settings or load_settings()

    csv_files = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        logger.warning("No CSV files found in %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        output_path = output_dir / f"{csv_path.stem}_scores.csv"
        logger.info("Processing %s -> %s", csv_path, output_path)
        run(csv_path, output_path, settings=settings, workers=workers)
