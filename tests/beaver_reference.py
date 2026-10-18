# SPDX-FileCopyrightText: Copyright (c) 2024 MIT Data Systems Group
# SPDX-License-Identifier: MIT
#
# Portions of this file are ported from the BEAVER benchmark's official
# evaluation script (``eval/utils/ex_acc.py``) in
# https://github.com/beaverbench/beaver, used under the MIT License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAVER's official execution accuracy (ex_acc) rule, as a reference.

Ports the execution and comparison logic of the official BEAVER evaluator:

- Execute SQL on the BEAVER MySQL database with a 10-second query timeout
  (errors/timeouts count as an empty result).
- Normalize every value to a stripped string.
- Score 1 iff the *sets* of result rows match (row order and duplicates
  ignored, column names ignored, column count must match).

Only ``tests/test_gym_beaver_match.py`` imports it, which is why it lives
under ``tests/``. The benchmark scores with ``ontology_sql_eval.gym.exec_match``,
which carries its own pandas-free copy of the rule for the control arm's venv;
the test checks the two agree.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd

# Official BEAVER evaluator constants (eval/utils/ex_acc.py).
QUERY_TIMEOUT = 10
CONNECTION_TIMEOUT = 10


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
