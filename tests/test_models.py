# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM score model and its CSV mapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluation.judge.models import LLM_SCORE_FIELDS, SqlScore


def _valid_kwargs() -> dict:
    return {
        "logic_match": 0.8,
        "logic_issues": "missing date filter",
        "semantic_match": 0.6,
        "final_weighted_score": 0.7,
        "sql_compared_to_ground_truth_score": 0.9,
        "is_valid_sql": True,
        "is_sql_returns_data": False,
    }


def test_valid_construction() -> None:
    score = SqlScore(**_valid_kwargs())
    assert score.logic_match == 0.8
    assert score.is_valid_sql is True


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_score_bounds_rejected(value: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["logic_match"] = value
    with pytest.raises(ValidationError):
        SqlScore(**kwargs)


def test_extra_fields_forbidden() -> None:
    kwargs = _valid_kwargs()
    kwargs["unexpected"] = 1
    with pytest.raises(ValidationError):
        SqlScore(**kwargs)


def test_as_csv_row_keys_match_fields() -> None:
    row = SqlScore(**_valid_kwargs()).as_csv_row()
    assert set(row.keys()) == set(LLM_SCORE_FIELDS)


def test_as_csv_row_values() -> None:
    row = SqlScore(**_valid_kwargs()).as_csv_row()
    assert row["llm_logic_match"] == 0.8
    assert row["llm_semantic_match"] == 0.6
    assert row["llm_final_weighted_score"] == 0.7
    assert row["llm_sql_vs_ground_truth"] == 0.9
    assert row["llm_is_valid_sql"] is True
    assert row["llm_is_sql_returns_data"] is False
    assert row["llm_logic_issues"] == "missing date filter"
