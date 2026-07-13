# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD official EX + VES evaluation on eval CSV output.

Ports the execution logic from Alibaba DAMO-ConvAI BIRD ``evaluation.py`` and
``evaluation_ves.py``:

- **EX (Execution Accuracy):** ``set(predicted_rows) == set(gold_rows)`` per question.
- **VES (Valid Efficiency Score):** timing ratio on EX-passing questions only.

See: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird#evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVAL_JSON = _REPO_ROOT / "datasets" / "bird" / "evaluation.json"
_DEFAULT_DB_ROOT = _REPO_ROOT / "datasets" / "bird"

BIRD_SCORE_FIELDS = [
    "bird_ex_match",
    "bird_ves_ratio",
    "bird_pred_error",
    "bird_gold_error",
]


@dataclass(frozen=True)
class BirdRow:
    """One eval CSV row prepared for BIRD scoring."""

    sql_idx: int
    question_id: int
    difficulty: str
    predicted_sql: str
    ground_truth_sql: str
    db_path: Path


@dataclass(frozen=True)
class ExResult:
    sql_idx: int
    res: int
    pred_error: str = ""
    gold_error: str = ""


@dataclass(frozen=True)
class VesResult:
    sql_idx: int
    time_ratio: float


def _load_evaluation_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def question_db_map(evaluation_json: Path) -> dict[int, str]:
    """Map ``question_id`` → ``db_id`` from BIRD evaluation JSON."""
    return {
        int(item["question_id"]): str(item["db_id"])
        for item in _load_evaluation_json(evaluation_json)
        if item.get("question_id") is not None and item.get("db_id")
    }


def sqlite_path(db_root: Path, db_id: str) -> Path:
    return db_root / db_id / f"{db_id}.sqlite"


# ---------------------------------------------------------------------------
# EX — ported from bird/llm/src/evaluation.py
# ---------------------------------------------------------------------------


def execute_sql(predicted_sql: str, ground_truth_sql: str, db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        return 1 if set(predicted_res) == set(ground_truth_res) else 0
    finally:
        conn.close()


def _execute_ex_model(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: str,
    sql_idx: int,
    meta_time_out: float,
) -> ExResult:
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth_sql, db_path),
        )
        return ExResult(sql_idx=sql_idx, res=int(res))
    except FunctionTimedOut:
        return ExResult(sql_idx=sql_idx, res=0, pred_error="timeout")
    except Exception as exc:
        return ExResult(sql_idx=sql_idx, res=0, pred_error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# VES — ported from bird/llm/src/evaluation_ves.py
# ---------------------------------------------------------------------------


def _execute_sql_timing(sql: str, db_path: str) -> float:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        start = time.time()
        cursor.execute(sql)
        cursor.fetchall()
        return time.time() - start
    finally:
        conn.close()


def clean_abnormal(values: list[float]) -> list[float]:
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return [float(x) for x in arr if x < mean + 3 * std and x > mean - 3 * std]


def iterated_execute_sql(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: str,
    iterate_num: int,
) -> float:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        if set(predicted_res) != set(ground_truth_res):
            return 0.0

        diff_list: list[float] = []
        for _ in range(iterate_num):
            predicted_time = _execute_sql_timing(predicted_sql, db_path)
            ground_truth_time = _execute_sql_timing(ground_truth_sql, db_path)
            if predicted_time > 0:
                diff_list.append(ground_truth_time / predicted_time)
        processed = clean_abnormal(diff_list)
        if not processed:
            return 0.0
        return sum(processed) / len(processed)
    finally:
        conn.close()


def _execute_ves_model(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: str,
    sql_idx: int,
    iterate_num: int,
    meta_time_out: float,
) -> VesResult:
    try:
        time_ratio = func_timeout(
            meta_time_out * iterate_num,
            iterated_execute_sql,
            args=(predicted_sql, ground_truth_sql, db_path, iterate_num),
        )
        return VesResult(sql_idx=sql_idx, time_ratio=float(time_ratio))
    except FunctionTimedOut:
        return VesResult(sql_idx=sql_idx, time_ratio=0.0)
    except Exception:
        return VesResult(sql_idx=sql_idx, time_ratio=0.0)


# ---------------------------------------------------------------------------
# Aggregation — official difficulty tables
# ---------------------------------------------------------------------------


def compute_acc_by_diff(
    exec_results: list[ExResult],
    difficulties: list[str],
) -> tuple[float, float, float, float, list[int]]:
    num_queries = len(exec_results)
    simple_results: list[ExResult] = []
    moderate_results: list[ExResult] = []
    challenging_results: list[ExResult] = []

    for result, difficulty in zip(exec_results, difficulties):
        if difficulty == "simple":
            simple_results.append(result)
        elif difficulty == "moderate":
            moderate_results.append(result)
        elif difficulty == "challenging":
            challenging_results.append(result)

    def _acc(results: list[ExResult]) -> float:
        if not results:
            return 0.0
        return sum(r.res for r in results) / len(results) * 100

    simple_acc = _acc(simple_results)
    moderate_acc = _acc(moderate_results)
    challenging_acc = _acc(challenging_results)
    all_acc = sum(r.res for r in exec_results) / num_queries * 100 if num_queries else 0.0
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return simple_acc, moderate_acc, challenging_acc, all_acc, count_lists


def compute_ves(ves_results: list[VesResult]) -> float:
    if not ves_results:
        return 0.0
    total = sum(math.sqrt(r.time_ratio) * 100 for r in ves_results if r.time_ratio != 0)
    return total / len(ves_results)


def compute_ves_by_diff(
    ves_results: list[VesResult],
    difficulties: list[str],
) -> tuple[float, float, float, float, list[int]]:
    simple_results: list[VesResult] = []
    moderate_results: list[VesResult] = []
    challenging_results: list[VesResult] = []

    for result, difficulty in zip(ves_results, difficulties):
        if difficulty == "simple":
            simple_results.append(result)
        elif difficulty == "moderate":
            moderate_results.append(result)
        elif difficulty == "challenging":
            challenging_results.append(result)

    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        len(ves_results),
    ]
    return (
        compute_ves(simple_results),
        compute_ves(moderate_results),
        compute_ves(challenging_results),
        compute_ves(ves_results),
        count_lists,
    )


_EX_DESCRIPTION = (
    "EX (Execution Accuracy): run predicted and gold SQL on each question's BIRD "
    "SQLite DB. Score 1 when the result row sets match (order-independent), "
    "0 otherwise. Reported as accuracy % by difficulty."
)
_VES_DESCRIPTION = (
    "VES (Valid Efficiency Score): for EX-passing questions only, compare "
    "predicted vs gold SQL execution time over many runs. "
    "sqrt(gold_time / pred_time) * 100, averaged by difficulty. "
    "Failed EX questions count as 0."
)


def _format_score_table(
    title: str,
    description: str,
    metric_label: str,
    score_lists: list[float],
    count_lists: list[int],
    *,
    suffix: str = "",
) -> str:
    columns = ["simple", "moderate", "challenging", "total"]
    rows = [
        ("count", [str(n) for n in count_lists]),
        (metric_label, [f"{v:.2f}{suffix}" for v in score_lists]),
    ]

    col_width = max(12, max(len(c) for c in columns))
    label_width = max(len(metric_label), len("count")) + 2

    def _row(label: str, values: list[str]) -> str:
        cells = "".join(f"{v:>{col_width}}" for v in values)
        return f"  {label:<{label_width}}{cells}"

    header = _row("", columns)
    divider = "  " + "-" * (label_width + col_width * len(columns) - 2)
    body = "\n".join(_row(label, values) for label, values in rows)
    desc = f"  {description}\n"
    return f"\n{title}\n{desc}{divider}\n{header}\n{divider}\n{body}\n"


def _print_ex_table(score_lists: list[float], count_lists: list[int]) -> None:
    print(
        _format_score_table(
            "Execution Accuracy (EX)",
            _EX_DESCRIPTION,
            "accuracy %",
            score_lists,
            count_lists,
        ),
        flush=True,
    )


def _print_ves_table(score_lists: list[float], count_lists: list[int]) -> None:
    print(
        _format_score_table(
            "Valid Efficiency Score (VES)",
            _VES_DESCRIPTION,
            "ves",
            score_lists,
            count_lists,
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# CSV driver
# ---------------------------------------------------------------------------


def _prepare_rows(
    input_path: Path,
    evaluation_json: Path,
    db_root: Path,
) -> list[BirdRow]:
    q_to_db = question_db_map(evaluation_json)
    rows: list[BirdRow] = []

    with input_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for sql_idx, row in enumerate(reader):
            question_id = int(row.get("question_id") or sql_idx)
            db_id = q_to_db.get(question_id)
            if not db_id:
                raise KeyError(
                    f"question_id {question_id} not found in {evaluation_json}"
                )
            db_path = sqlite_path(db_root, db_id)
            if not db_path.exists():
                raise FileNotFoundError(f"SQLite database not found: {db_path}")

            rows.append(
                BirdRow(
                    sql_idx=sql_idx,
                    question_id=question_id,
                    difficulty=str(row.get("difficulty") or ""),
                    predicted_sql=str(row.get("returned_sql") or "").strip(),
                    ground_truth_sql=str(row.get("expected_sql") or "").strip(),
                    db_path=db_path,
                )
            )
    return rows


def _run_ex_parallel(
    bird_rows: list[BirdRow],
    *,
    num_cpus: int,
    meta_time_out: float,
) -> list[ExResult]:
    worker_args = [
        (
            row.predicted_sql,
            row.ground_truth_sql,
            str(row.db_path),
            row.sql_idx,
            meta_time_out,
        )
        for row in bird_rows
    ]

    if num_cpus <= 1:
        return [_execute_ex_model(*args) for args in worker_args]

    with mp.Pool(processes=num_cpus) as pool:
        results = pool.starmap(_execute_ex_model, worker_args)
    return sorted(results, key=lambda r: r.sql_idx)


def _run_ves_parallel(
    bird_rows: list[BirdRow],
    ex_results: list[ExResult],
    *,
    num_cpus: int,
    meta_time_out: float,
    iterate_num: int,
) -> list[VesResult]:
    ex_by_idx = {r.sql_idx: r for r in ex_results}
    ves_rows = [
        row for row in bird_rows if ex_by_idx.get(row.sql_idx, ExResult(row.sql_idx, 0)).res
    ]

    worker_args = [
        (
            row.predicted_sql,
            row.ground_truth_sql,
            str(row.db_path),
            row.sql_idx,
            iterate_num,
            meta_time_out,
        )
        for row in ves_rows
    ]

    if not worker_args:
        return []

    if num_cpus <= 1:
        results = [_execute_ves_model(*args) for args in worker_args]
    else:
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.starmap(_execute_ves_model, worker_args)
    return sorted(results, key=lambda r: r.sql_idx)


def run(
    input_path: Path,
    output_path: Path | None = None,
    *,
    evaluation_json: Path = _DEFAULT_EVAL_JSON,
    db_root: Path = _DEFAULT_DB_ROOT,
    num_cpus: int = 1,
    meta_time_out: float = 30.0,
    iterate_num: int = 100,
    skip_ves: bool = False,
) -> None:
    """Score ``input_path`` with official BIRD EX (+ VES) and optionally write CSV."""
    bird_rows = _prepare_rows(input_path, evaluation_json, db_root)
    if not bird_rows:
        logger.warning("No rows to score in %s", input_path)
        return

    logger.info("Scoring %d questions (EX%s)", len(bird_rows), "" if skip_ves else " + VES")
    print(f"\nBIRD official evaluation — {len(bird_rows)} questions", flush=True)
    ex_results = _run_ex_parallel(bird_rows, num_cpus=num_cpus, meta_time_out=meta_time_out)
    difficulties = [row.difficulty for row in bird_rows]

    simple_acc, moderate_acc, challenging_acc, all_acc, ex_counts = compute_acc_by_diff(
        ex_results, difficulties
    )
    _print_ex_table(
        [simple_acc, moderate_acc, challenging_acc, all_acc],
        ex_counts,
    )

    ves_by_idx: dict[int, VesResult] = {}
    if not skip_ves:
        ex_pass = sum(r.res for r in ex_results)
        logger.info(
            "Running VES on %d / %d EX-passing questions (iterate_num=%d, timeout=%.1fs each)",
            ex_pass,
            len(bird_rows),
            iterate_num,
            meta_time_out * iterate_num,
        )
        ves_results = _run_ves_parallel(
            bird_rows,
            ex_results,
            num_cpus=num_cpus,
            meta_time_out=meta_time_out,
            iterate_num=iterate_num,
        )
        ves_by_idx = {r.sql_idx: r for r in ves_results}
        ves_aligned = [
            ves_by_idx.get(i, VesResult(sql_idx=i, time_ratio=0.0))
            for i in range(len(bird_rows))
        ]
        simple_ves, moderate_ves, challenging_ves, all_ves, ves_counts = compute_ves_by_diff(
            ves_aligned, difficulties
        )
        _print_ves_table(
            [simple_ves, moderate_ves, challenging_ves, all_ves],
            ves_counts,
        )

    if output_path is None:
        return

    ex_by_idx = {r.sql_idx: r for r in ex_results}
    with input_path.open(encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        original_fields = list(reader.fieldnames or [])
        in_rows = list(reader)

    out_fields = original_fields + [f for f in BIRD_SCORE_FIELDS if f not in original_fields]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for sql_idx, row in enumerate(in_rows):
            ex = ex_by_idx.get(sql_idx, ExResult(sql_idx=sql_idx, res=0))
            ves = ves_by_idx.get(sql_idx, VesResult(sql_idx=sql_idx, time_ratio=0.0))
            row["bird_ex_match"] = ex.res
            row["bird_ves_ratio"] = round(ves.time_ratio, 6) if ves.time_ratio else 0
            row["bird_pred_error"] = ex.pred_error
            row["bird_gold_error"] = ex.gold_error
            writer.writerow(row)

    logger.info("Wrote BIRD scores to %s", output_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BIRD official EX (+ VES) evaluation on eval CSV output.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Eval CSV from input/ (must contain question_id, expected_sql, returned_sql).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: output/<input_stem>_bird_scores.csv).",
    )
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=_DEFAULT_EVAL_JSON,
        help=f"BIRD evaluation JSON for question_id → db_id (default: {_DEFAULT_EVAL_JSON}).",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        default=_DEFAULT_DB_ROOT,
        help=f"Root folder containing <db_id>/<db_id>.sqlite (default: {_DEFAULT_DB_ROOT}).",
    )
    parser.add_argument("--num-cpus", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument(
        "--meta-time-out",
        type=float,
        default=30.0,
        help="Per-question EX timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--iterate-num",
        type=int,
        default=100,
        help="VES timing iterations per EX-passing question (default: 100).",
    )
    parser.add_argument(
        "--skip-ves",
        action="store_true",
        help="Run EX only (skip VES timing).",
    )
    parser.add_argument(
        "--no-output-csv",
        action="store_true",
        help="Print summary only; do not write scored CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    output_path = None if args.no_output_csv else (
        args.output or Path("output") / f"{args.input.stem}_bird_scores.csv"
    )
    run(
        input_path=args.input,
        output_path=output_path,
        evaluation_json=args.evaluation_json,
        db_root=args.db_root,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        iterate_num=args.iterate_num,
        skip_ves=args.skip_ves,
    )


if __name__ == "__main__":
    main()
