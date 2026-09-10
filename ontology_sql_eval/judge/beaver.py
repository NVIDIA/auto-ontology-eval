# SPDX-FileCopyrightText: Copyright (c) 2024 MIT Data Systems Group
# SPDX-License-Identifier: MIT
#
# Portions of this file are ported from the BEAVER benchmark's official
# evaluation scripts (``eval/utils/ex_acc.py`` and ``eval/evaluate_ex_acc.py``)
# in https://github.com/beaverbench/beaver, used under the MIT License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAVER official execution accuracy (ex_acc) on eval CSV output.

Ports the execution + comparison logic from the official BEAVER evaluator so
scores are comparable to the published leaderboard:

- Execute predicted and gold SQL on the BEAVER MySQL database with a
  10-second query timeout (errors/timeouts count as an empty result).
- Normalize every value to a stripped string.
- Score 1 iff the *sets* of result rows match (row order and duplicates
  ignored, column names ignored, column count must match).
- Report accuracy both including and excluding empty-gold-result questions.

Usage::

    python -m ontology_sql_eval.judge.beaver --input input/beaverbench_<model>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVAL_JSON = _REPO_ROOT / "datasets" / "beaverbench" / "evaluation.json"

# Official BEAVER evaluator constants (eval/utils/ex_acc.py).
QUERY_TIMEOUT = 10
CONNECTION_TIMEOUT = 10

BEAVER_SCORE_FIELDS = [
    "beaver_ex_match",
    "beaver_ex_message",
    "beaver_gold_empty",
    "beaver_pred_error",
    "beaver_gold_error",
]


@dataclass(frozen=True)
class BeaverRow:
    """One eval CSV row prepared for BEAVER scoring."""

    sql_idx: int
    question_id: str
    difficulty: str
    predicted_sql: str
    ground_truth_sql: str
    database: str


@dataclass(frozen=True)
class ExResult:
    sql_idx: int
    match: bool = False
    message: str = ""
    gold_empty: bool = True
    pred_error: str = ""
    gold_error: str = ""


@dataclass(frozen=True)
class MySQLCreds:
    host: str
    port: int
    user: str
    password: str = field(repr=False, default="")

    def as_kwargs(self, database: str) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": database,
        }


def creds_from_url(url: str) -> MySQLCreds:
    """Parse a ``mysql://user:pass@host:port[/db]`` URL into credentials."""
    parsed = urlparse(url)
    if parsed.scheme not in ("mysql", "mysql+mysqlconnector"):
        raise ValueError(f"Not a MySQL URL: {url!r}")
    return MySQLCreds(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=unquote(parsed.username or "root"),
        password=unquote(parsed.password or ""),
    )


def creds_from_env() -> MySQLCreds:
    """First ``mysql://`` entry in ``CONNECTION_STRINGS``."""
    for url in os.environ.get("CONNECTION_STRINGS", "").split(","):
        url = url.strip()
        if url.startswith("mysql"):
            return creds_from_url(url)
    raise ValueError(
        "No mysql:// URL in CONNECTION_STRINGS; pass --mysql-url explicitly."
    )


# ---------------------------------------------------------------------------
# Execution — ported from beaver/eval/utils/ex_acc.py
# ---------------------------------------------------------------------------


def _execute_query_thread(
    sql: str, mysql_kwargs: dict[str, Any], result_holder: dict[str, Any]
) -> None:
    import mysql.connector

    conn = None
    cursor = None
    try:
        # use_pure: the C extension segfaults when abandoned timeout threads
        # race live ones; the pure-Python driver is thread-safe.
        conn = mysql.connector.connect(
            **mysql_kwargs, connection_timeout=CONNECTION_TIMEOUT, use_pure=True
        )
        cursor = conn.cursor()
        cursor.execute(sql)
        if cursor.description is None:
            rows: list[tuple] = []
            columns: list[str] = []
        else:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        # 0 rows -> empty DataFrame (NOT an error), matching the official evaluator.
        result_holder["df"] = pd.DataFrame(rows, columns=columns)
        result_holder["completed"] = True
    except Exception as exc:  # noqa: BLE001 - mirror official broad handling
        result_holder["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except Exception:  # noqa: BLE001
            pass


def execute_sql_with_timeout(
    sql: str, mysql_kwargs: dict[str, Any], timeout: float = QUERY_TIMEOUT
) -> tuple[pd.DataFrame | None, str | None]:
    """Execute *sql*, returning ``(dataframe, error)`` with a threaded timeout."""
    result_holder: dict[str, Any] = {"df": None, "error": None, "completed": False}
    t = threading.Thread(
        target=_execute_query_thread, args=(sql, mysql_kwargs, result_holder)
    )
    t.daemon = True
    t.start()

    start_time = time.time()
    while t.is_alive():
        t.join(timeout=0.5)
        if time.time() - start_time > timeout:
            return None, "Timeout exceeded"

    if result_holder["completed"]:
        return result_holder["df"], None
    return None, result_holder["error"]


# ---------------------------------------------------------------------------
# Comparison — ported from beaver/eval/utils/ex_acc.py
# ---------------------------------------------------------------------------


def normalize_dataframe_values(df: pd.DataFrame | None) -> list[list[str]]:
    """Every value to a stripped string; rows and columns keep their order."""
    if df is None or df.empty:
        return []
    df = df.astype(str)
    try:
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    except AttributeError:
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df.values.tolist()


def compare_results(
    pred_df: pd.DataFrame | None, gold_df: pd.DataFrame | None
) -> tuple[bool, str]:
    """Official BEAVER comparison: deduped row-set equality on stringified values."""
    pred_empty = pred_df is None or pred_df.empty
    gold_empty = gold_df is None or gold_df.empty

    if pred_empty and gold_empty:
        return True, "Both empty"
    if pred_empty or gold_empty:
        return False, "One is empty, other is not"

    pred_rows = normalize_dataframe_values(pred_df)
    gold_rows = normalize_dataframe_values(gold_df)
    gold_rows_set = set(tuple(r) for r in gold_rows)

    n_cols_pred = len(pred_rows[0]) if pred_rows else 0
    n_cols_gold = len(gold_rows[0]) if gold_rows else 0
    if n_cols_pred != n_cols_gold:
        return (
            False,
            f"Column count mismatch: pred {n_cols_pred} vs gold {n_cols_gold}",
        )

    pred_rows_set = set(tuple(r) for r in pred_rows)
    if pred_rows_set == gold_rows_set:
        return True, "Match (values match, ignoring column names)"
    return False, "Values mismatch"


# ---------------------------------------------------------------------------
# CSV driver
# ---------------------------------------------------------------------------


def question_db_map(evaluation_json: Path) -> dict[str, str]:
    """Map ``question_id`` → ``db_id`` from the BEAVER evaluation JSON."""
    with evaluation_json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {evaluation_json}")
    return {
        str(item["question_id"]): str(item["db_id"])
        for item in data
        if item.get("question_id") is not None and item.get("db_id")
    }


def _prepare_rows(input_path: Path, evaluation_json: Path) -> list[BeaverRow]:
    q_to_db = question_db_map(evaluation_json)
    rows: list[BeaverRow] = []
    with input_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for sql_idx, row in enumerate(reader):
            question_id = str(row.get("question_id") or sql_idx)
            database = q_to_db.get(question_id)
            if not database:
                raise KeyError(
                    f"question_id {question_id} not found in {evaluation_json}"
                )
            rows.append(
                BeaverRow(
                    sql_idx=sql_idx,
                    question_id=question_id,
                    difficulty=str(row.get("difficulty") or ""),
                    predicted_sql=str(row.get("returned_sql") or "").strip(),
                    ground_truth_sql=str(row.get("expected_sql") or "").strip(),
                    database=database,
                )
            )
    return rows


def score_row(
    row: BeaverRow, creds: MySQLCreds, timeout: float = QUERY_TIMEOUT
) -> ExResult:
    """Score one row with official semantics (errors → empty result)."""
    kwargs = creds.as_kwargs(row.database)

    gold_df, gold_err = execute_sql_with_timeout(
        row.ground_truth_sql, kwargs, timeout=timeout
    )
    if gold_df is None:
        gold_df = pd.DataFrame()

    pred_df: pd.DataFrame | None = None
    pred_err: str | None = None
    if row.predicted_sql:
        pred_df, pred_err = execute_sql_with_timeout(
            row.predicted_sql, kwargs, timeout=timeout
        )
    if pred_df is None:
        pred_df = pd.DataFrame()

    match, message = compare_results(pred_df, gold_df)
    return ExResult(
        sql_idx=row.sql_idx,
        match=match,
        message=message,
        gold_empty=gold_df.empty,
        pred_error=pred_err or "",
        gold_error=gold_err or "",
    )


def _accuracy(results: list[ExResult]) -> float:
    if not results:
        return 0.0
    return sum(r.match for r in results) / len(results) * 100


def _print_summary(rows: list[BeaverRow], results: list[ExResult]) -> None:
    by_idx = {r.sql_idx: r for r in results}
    aligned = [by_idx[row.sql_idx] for row in rows]

    nonempty = [r for r in aligned if not r.gold_empty]
    gold_errors = sum(1 for r in aligned if r.gold_error)
    pred_errors = sum(1 for r in aligned if r.pred_error)

    print(f"\nBEAVER official evaluation (ex_acc) — {len(rows)} questions")
    print(
        "  Deduped row-set equality on stringified values; "
        f"{QUERY_TIMEOUT}s query timeout; errors count as empty results.\n"
    )
    print(f"  accuracy (incl. empty-gold) : {_accuracy(aligned):6.2f}%")
    print(
        f"  accuracy (excl. empty-gold) : {_accuracy(nonempty):6.2f}%"
        f"   ({len(nonempty)} non-empty-gold questions)"
    )
    print(f"  gold SQL errors/timeouts    : {gold_errors}")
    print(f"  predicted SQL errors/timeouts: {pred_errors}\n")

    difficulties = sorted({row.difficulty for row in rows})
    if len(difficulties) > 1:
        width = max(len(d) for d in difficulties)
        print("  By difficulty:")
        for diff in difficulties:
            subset = [by_idx[row.sql_idx] for row in rows if row.difficulty == diff]
            print(
                f"    {diff:<{width}}  {_accuracy(subset):6.2f}%  "
                f"({sum(r.match for r in subset)}/{len(subset)})"
            )
        print(flush=True)


def run(
    input_path: Path,
    output_path: Path | None = None,
    *,
    evaluation_json: Path = _DEFAULT_EVAL_JSON,
    creds: MySQLCreds | None = None,
    num_workers: int = 4,
    timeout: float = QUERY_TIMEOUT,
) -> list[ExResult]:
    """Score ``input_path`` with official BEAVER ex_acc and optionally write CSV."""
    creds = creds or creds_from_env()
    rows = _prepare_rows(input_path, evaluation_json)
    if not rows:
        logger.warning("No rows to score in %s", input_path)
        return []

    logger.info("Scoring %d questions with %d workers", len(rows), num_workers)
    if num_workers <= 1:
        results = [score_row(row, creds, timeout=timeout) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(
                pool.map(lambda r: score_row(r, creds, timeout=timeout), rows)
            )

    _print_summary(rows, results)

    if output_path is not None:
        by_idx = {r.sql_idx: r for r in results}
        with input_path.open(encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            original_fields = list(reader.fieldnames or [])
            in_rows = list(reader)

        out_fields = original_fields + [
            f for f in BEAVER_SCORE_FIELDS if f not in original_fields
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            for sql_idx, row in enumerate(in_rows):
                res = by_idx.get(sql_idx, ExResult(sql_idx=sql_idx))
                row["beaver_ex_match"] = int(res.match)
                row["beaver_ex_message"] = res.message
                row["beaver_gold_empty"] = int(res.gold_empty)
                row["beaver_pred_error"] = res.pred_error
                row["beaver_gold_error"] = res.gold_error
                writer.writerow(row)
        logger.info("Wrote BEAVER scores to %s", output_path)

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BEAVER official execution accuracy on eval CSV output.",
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
        help="Output CSV path (default: output/<input_stem>_beaver_scores.csv).",
    )
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=_DEFAULT_EVAL_JSON,
        help=f"BEAVER evaluation JSON for question_id → db_id (default: {_DEFAULT_EVAL_JSON}).",
    )
    parser.add_argument(
        "--mysql-url",
        type=str,
        default=None,
        help="mysql://user:pass@host:port URL (default: first mysql:// in CONNECTION_STRINGS).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Parallel workers (default: 4)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=QUERY_TIMEOUT,
        help=f"Per-query timeout in seconds (official default: {QUERY_TIMEOUT}).",
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
    logging.getLogger("mysql.connector").setLevel(logging.WARNING)
    from dotenv import load_dotenv

    load_dotenv()
    args = _parse_args(argv)
    creds = creds_from_url(args.mysql_url) if args.mysql_url else creds_from_env()
    output_path = (
        None
        if args.no_output_csv
        else (args.output or Path("output") / f"{args.input.stem}_beaver_scores.csv")
    )
    run(
        input_path=args.input,
        output_path=output_path,
        evaluation_json=args.evaluation_json,
        creds=creds,
        num_workers=args.num_workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
