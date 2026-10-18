# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Gym verifier's BEAVER rule must agree with the official BEAVER evaluator.

``ontology_sql_eval.gym.exec_match.beaver_result_match`` reimplements the rule on
raw rows instead of importing ``tests.beaver_reference.compare_results``, because
that port pulls in pandas and the schema-only server's venv deliberately does not
have it. This test is what stops the two drifting apart.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pandas as pd
import pytest

from ontology_sql_eval.gym.exec_match import (
    beaver_result_match,
    execute_and_compare,
    matcher_for,
    result_sets_match,
    timeout_for,
)
from tests.beaver_reference import compare_results

CASES = [
    ("identical", [(1, "a")], [(1, "a")]),
    ("row order", [(1, "a"), (2, "b")], [(2, "b"), (1, "a")]),
    ("duplicate rows", [(1,)], [(1,), (1,)]),
    ("decimal trailing zeros", [(Decimal("150.250"),)], [(Decimal("150.25"),)]),
    ("decimal vs float", [(Decimal("150.25"),)], [(150.25,)]),
    ("int vs str", [(1,)], [("1",)]),
    ("whitespace", [(" a",)], [("a",)]),
    ("column count", [(1,)], [(1, "x")]),
    ("both empty", [], []),
    ("gold empty only", [], [(1,)]),
    ("pred empty only", [(1,)], []),
    ("multi row multi col", [(1, "a"), (2, "b")], [(2, "b"), (1, "a")]),
]


@pytest.mark.parametrize("label,gold,pred", CASES, ids=[c[0] for c in CASES])
def test_matches_official_beaver_evaluator(label, gold, pred):
    official = compare_results(
        pd.DataFrame(pred) if pred else None,
        pd.DataFrame(gold) if gold else None,
    )[0]
    assert beaver_result_match(gold, pred) is official, (
        f"{label}: ours={beaver_result_match(gold, pred)} official={official}"
    )


def test_beaverbench_routes_to_the_beaver_rule():
    assert matcher_for("beaverbench") is beaver_result_match
    # Everything else keeps BIRD's rule.
    for dataset in ("bird", "fdabench", "wideworldimporters", ""):
        assert matcher_for(dataset) is result_sets_match


def test_the_two_rules_genuinely_differ():
    """Guards against someone 'simplifying' one into the other."""
    gold, pred = [(Decimal("150.250"),)], [(Decimal("150.25"),)]
    assert result_sets_match(gold, pred) is True
    assert beaver_result_match(gold, pred) is False


def test_beaverbench_uses_the_official_timeout():
    assert timeout_for("beaverbench", 30.0) == 10.0
    for dataset in ("bird", "fdabench", "wideworldimporters", ""):
        assert timeout_for(dataset, 30.0) == 30.0


class _FakeExecutor:
    """Returns canned results keyed by SQL text; ``None`` rows mean failure."""

    def __init__(self, results):
        self._results = results

    async def execute(self, db_id, sql):
        return self._results[sql]


# pred is an executor result, or None for "no prediction at all".
SCORING_CASES = [
    # (label, dataset, gold, pred, expected (match, error_tag, gold_result))
    (
        "nonempty match",
        "beaverbench",
        ([(1,)], ""),
        ([(1,)], ""),
        (True, None, "nonempty_gold"),
    ),
    (
        "empty gold, empty pred",
        "beaverbench",
        ([], ""),
        ([], ""),
        (True, None, "empty_gold"),
    ),
    # BEAVER's official evaluator scores every query that produces no rows -- an
    # error, a timeout, or no prediction -- as an empty result, gold included.
    (
        "empty gold, failed pred",
        "beaverbench",
        ([], ""),
        (None, "error"),
        (True, None, "empty_gold"),
    ),
    ("empty gold, no pred", "beaverbench", ([], ""), None, (True, None, "empty_gold")),
    (
        "nonempty gold, no pred",
        "beaverbench",
        ([(1,)], ""),
        None,
        (False, None, "nonempty_gold"),
    ),
    (
        "failed gold, empty pred",
        "beaverbench",
        (None, "timeout"),
        ([], ""),
        (True, None, "failed_gold"),
    ),
    (
        "failed gold, failed pred",
        "beaverbench",
        (None, "error"),
        (None, "timeout"),
        (True, None, "failed_gold"),
    ),
    (
        "failed gold, rows pred",
        "beaverbench",
        (None, "timeout"),
        ([(1,)], ""),
        (False, None, "failed_gold"),
    ),
    # Every other dataset keeps the stricter handling.
    (
        "bird failed pred",
        "bird",
        ([], ""),
        (None, "timeout"),
        (False, "pred_sql_timeout", "empty_gold"),
    ),
    ("bird no pred", "bird", ([], ""), None, (False, "pred_sql_missing", "empty_gold")),
    (
        "bird failed gold",
        "bird",
        (None, "timeout"),
        ([], ""),
        (False, "gold_sql_timeout", None),
    ),
    (
        "bird failed gold, no pred",
        "bird",
        (None, "error"),
        None,
        (False, "gold_sql_error", None),
    ),
]


@pytest.mark.parametrize(
    "label,dataset,gold,pred,expected",
    SCORING_CASES,
    ids=[c[0] for c in SCORING_CASES],
)
def test_execute_and_compare(label, dataset, gold, pred, expected):
    executor = _FakeExecutor({"gold": gold, "pred": pred})
    pred_sql = None if pred is None else "pred"
    got = asyncio.run(execute_and_compare(executor, "db", "gold", pred_sql, dataset))
    assert got == expected


def test_failed_queries_score_like_the_official_evaluator():
    """A failed query is an empty DataFrame officially; ours must agree."""
    for gold, pred in (([], []), ([(1,)], []), ([], [(1,)])):
        official = compare_results(
            pd.DataFrame(pred) if pred else None,
            pd.DataFrame(gold) if gold else None,
        )[0]
        executor = _FakeExecutor(
            {
                "gold": (gold, "") if gold else (None, "error"),
                "pred": (pred, "") if pred else (None, "timeout"),
            }
        )
        match, _, _ = asyncio.run(
            execute_and_compare(executor, "db", "gold", "pred", "beaverbench")
        )
        assert match is official, (gold, pred)
