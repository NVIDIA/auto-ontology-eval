# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and load Spider2-lite gold execution-result CSVs.

When upstream gold ``*.sql`` files are absent (common for many ``local*``
instances), the official benchmark still scores by comparing predicted query
results to pre-computed gold CSVs under
``third_party/Spider2/spider2-lite/evaluation_suite/gold/exec_result/``.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXEC_RESULT_DIR = (
    _REPO_ROOT
    / "third_party"
    / "Spider2"
    / "spider2-lite"
    / "evaluation_suite"
    / "gold"
    / "exec_result"
)

_SPIDER2_INSTANCE_RE = re.compile(r"^(local|bq|ga)\d", re.IGNORECASE)
_ANSWER_RAW_ROW_LIMIT = 200


def exec_result_dir() -> Path:
    return _EXEC_RESULT_DIR


def is_spider2_instance_id(instance_id: str) -> bool:
    """Return True for Spider2-lite instance ids (``local009``, ``bq001``, …)."""
    return bool(_SPIDER2_INSTANCE_RE.match((instance_id or "").strip()))


def resolve_gold_exec_result_paths(
    instance_id: str,
    gold_result_dir: Path | None = None,
) -> list[Path]:
    """Return gold ``exec_result`` CSV path(s) for *instance_id*.

    Mirrors ``evaluate.py::resolve_gold_paths``: prefer ``<id>.csv``, else
    every ``<id>_*.csv`` variant.
    """
    gold_result_dir = gold_result_dir or _EXEC_RESULT_DIR
    if not gold_result_dir.is_dir():
        return []

    base_path = gold_result_dir / f"{instance_id}.csv"
    if base_path.exists():
        return [base_path]

    pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    return sorted(
        gold_result_dir / name
        for name in gold_result_dir.iterdir()
        if pattern.match(name.name)
    )


def load_gold_exec_result_dfs(
    instance_id: str,
    gold_result_dir: Path | None = None,
) -> list[pd.DataFrame]:
    """Load every gold ``exec_result`` CSV for *instance_id* (may be empty)."""
    dfs: list[pd.DataFrame] = []
    for path in resolve_gold_exec_result_paths(instance_id, gold_result_dir):
        try:
            dfs.append(pd.read_csv(path))
        except Exception:
            continue
    return dfs


def gold_dfs_to_answer_raw(dfs: list[pd.DataFrame]) -> str:
    """Serialize gold DataFrame(s) to CSV text for ``answer_raw`` / debugging."""
    if not dfs:
        return ""
    if len(dfs) == 1:
        return dfs[0].head(_ANSWER_RAW_ROW_LIMIT).to_csv(index=False)
    parts = []
    for i, df in enumerate(dfs):
        parts.append(
            f"--- gold variant {i + 1} ---\n{df.head(_ANSWER_RAW_ROW_LIMIT).to_csv(index=False)}"
        )
    return "\n".join(parts)


def gold_dfs_to_expected_result(dfs: list[pd.DataFrame]) -> str:
    """Format gold DataFrame(s) for the eval CSV ``expected_sql_result`` column."""
    return gold_dfs_to_answer_raw(dfs)
