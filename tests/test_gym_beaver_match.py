# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Gym verifier's BEAVER rule must agree with the official BEAVER evaluator.

``ontology_sql_eval.gym.exec_match.beaver_result_match`` reimplements the rule on
raw rows instead of importing ``judge.beaver.compare_results``, because that
module pulls in pandas and the schema-only server's venv deliberately does not
have it. This test is what stops the two drifting apart.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from ontology_sql_eval.gym.exec_match import (
    beaver_result_match,
    matcher_for,
    result_sets_match,
)
from ontology_sql_eval.judge.beaver import compare_results

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
