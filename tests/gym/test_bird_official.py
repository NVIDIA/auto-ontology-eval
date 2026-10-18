# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD is scored the way its official ``evaluation_ex.py`` scores it.

The prediction and the gold query share one time limit, and a gold query that
fails scores 0 with the task kept, where every other dataset excludes it.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from ontology_sql_eval.gym.exec_match import execute_and_compare

GOLD, PRED = "gold", "pred"


class FakeExecutor:
    """Returns canned rows after a canned delay, honouring the per-call limit."""

    def __init__(self, results: dict, timeout_s: float = 1.0) -> None:
        self.results = results  # sql -> (seconds, rows or None)
        self.timeout_s = timeout_s
        self.calls: list[tuple[str, float]] = []

    async def execute(
        self, db_id: str, sql: str, *, timeout_s: Optional[float] = None
    ) -> tuple[Optional[list], str]:
        limit = self.timeout_s if timeout_s is None else timeout_s
        self.calls.append((sql, limit))
        seconds, rows = self.results[sql]
        if seconds > limit:
            await asyncio.sleep(limit)
            return None, "timeout"
        await asyncio.sleep(seconds)
        return (rows, "") if rows is not None else (None, "error")


def run(executor: FakeExecutor, dataset: str, pred: Optional[str] = PRED):
    return asyncio.run(execute_and_compare(executor, "db", GOLD, pred, dataset))


def test_match_uses_set_equality():
    ex = FakeExecutor({PRED: (0, [(1,), (2,), (2,)]), GOLD: (0, [(2,), (1,)])})
    assert run(ex, "bird") == (True, None, "nonempty_gold")


def test_prediction_runs_first_and_gold_gets_what_is_left():
    ex = FakeExecutor({PRED: (0.6, [(1,)]), GOLD: (0.6, [(1,)])}, timeout_s=1.0)
    assert run(ex, "bird") == (False, "gold_sql_scored", "failed_gold")
    assert [sql for sql, _ in ex.calls] == [PRED, GOLD]
    assert ex.calls[0][1] == 1.0 and ex.calls[1][1] < 0.5


def test_same_queries_score_on_a_dataset_with_separate_limits():
    ex = FakeExecutor({PRED: (0.6, [(1,)]), GOLD: (0.6, [(1,)])}, timeout_s=1.0)
    assert run(ex, "fdabench") == (True, None, "nonempty_gold")


@pytest.mark.parametrize("gold", [(0, None), (5, [(1,)])], ids=["error", "timeout"])
def test_failed_gold_is_scored_not_excluded(gold):
    ex = FakeExecutor({PRED: (0, [(1,)]), GOLD: gold})
    assert run(ex, "bird") == (False, "gold_sql_scored", "failed_gold")


def test_failed_gold_still_excludes_on_other_datasets():
    ex = FakeExecutor({PRED: (0, [(1,)]), GOLD: (0, None)})
    assert run(ex, "fdabench") == (False, "gold_sql_error", None)


def test_failed_prediction_is_a_miss_with_gold_still_tagged():
    ex = FakeExecutor({PRED: (0, None), GOLD: (0, [])})
    assert run(ex, "bird") == (False, "pred_sql_error", "empty_gold")


def test_missing_prediction_is_a_miss():
    ex = FakeExecutor({GOLD: (0, [(1,)])})
    assert run(ex, "bird", pred=None) == (False, "pred_sql_missing", "nonempty_gold")
