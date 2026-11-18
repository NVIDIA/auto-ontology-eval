# SPDX-FileCopyrightText: Copyright (c) 2024 bird_sql
# SPDX-License-Identifier: MIT
#
# Portions of this file are ported from the Spider2-lite official evaluator
# (``spider2-lite/evaluation_suite/evaluate.py``) in
# https://github.com/xlang-ai/Spider2, used under the MIT License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spider2-lite's official scoring of one local (SQLite) question, as a reference.

``compare_multi_pandas_table`` and ``compare_pandas_table`` are upstream's,
unchanged. ``score_sqlite`` strings together what upstream's
``evaluate_single_sql_instance`` does for a ``local*`` question: run the SQL
with ``pd.read_sql_query`` in 500-row chunks, write the chunks to CSV, read the
CSV back, and compare it with the gold CSVs through the single-file or
multi-file path. Upstream's module is not imported because it pulls in
BigQuery and replaces ``sys.stdout`` at import.

Only ``tests/gym/test_spider2_match.py`` imports this, which is why it lives
under ``tests/``. The benchmark scores with ``ontology_sql_eval.gym.spider2``,
the pandas-free copy; the test checks the two agree.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


def compare_multi_pandas_table(pred: pd.DataFrame, multi_gold, multi_condition_cols=None, multi_ignore_order=False) -> int:
    if not multi_gold:
        return 0

    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0


def compare_pandas_table(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols=None, ignore_order: bool = False) -> int:
    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]

        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))

        if len(v1) != len(v2):
            return False

        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold

    pred_cols = pred
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1

    for gold_vector in t_gold_list:
        if not any(vectors_match(gold_vector, pred_vector, ignore_order_=ignore_order) for pred_vector in t_pred_list):
            score = 0
            break

    return score


def prediction_frame(db_path: Path, query: str) -> pd.DataFrame:
    """Upstream ``get_sqlite_result`` with a save_dir, then ``pd.read_csv``."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.csv"
        conn = sqlite3.connect(db_path)
        memory_conn = sqlite3.connect(":memory:")
        conn.backup(memory_conn)
        try:
            for i, chunk in enumerate(pd.read_sql_query(query, memory_conn, chunksize=500)):
                mode = "a" if i > 0 else "w"
                header = i == 0
                chunk.to_csv(out, mode=mode, header=header, index=False)
        finally:
            memory_conn.close()
            conn.close()
        return pd.read_csv(out)


def score_sqlite(
    db_path: Path,
    query: str,
    gold_paths: list[Path],
    single: bool,
    condition_cols=None,
    ignore_order: bool = False,
) -> int:
    """Upstream's score for one ``local*`` question; any exception scores 0."""
    try:
        pred_pd = prediction_frame(db_path, query)
        if single:
            return compare_pandas_table(pred_pd, pd.read_csv(gold_paths[0]), condition_cols, ignore_order)
        gold_pds = [pd.read_csv(path) for path in gold_paths]
        return compare_multi_pandas_table(pred_pd, gold_pds, condition_cols, ignore_order)
    except Exception:
        return 0
