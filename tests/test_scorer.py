# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-row LLM SQL scoring (LLM call mocked)."""

from __future__ import annotations

import pytest

from evaluation.judge import scorer
from evaluation.judge.config import Settings
from evaluation.judge.models import SqlScore

_SETTINGS = Settings(model_name="m", base_url="https://x/v1", api_key="k")


def _score() -> SqlScore:
    return SqlScore(
        logic_match=1.0,
        logic_issues="",
        semantic_match=1.0,
        final_weighted_score=1.0,
        sql_compared_to_ground_truth_score=1.0,
        is_valid_sql=True,
        is_sql_returns_data=True,
    )


class _FakeStructured:
    def __init__(self, result: object) -> None:
        self._result = result

    def invoke(self, prompt: str) -> object:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeLLM:
    def __init__(self, result: object) -> None:
        self._result = result

    def with_structured_output(self, schema: object) -> _FakeStructured:
        return _FakeStructured(self._result)


def _patch_llm(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    monkeypatch.setattr(scorer, "_build_llm", lambda settings: _FakeLLM(result))


def _call() -> SqlScore | None:
    return scorer.score_sql(
        _SETTINGS,
        question="how many customers?",
        sql_code="select count(*) from customers",
        ground_truth_sql="select count(*) from customers",
        sql_result_preview="",
    )


def test_empty_sql_returns_none() -> None:
    result = scorer.score_sql(
        _SETTINGS,
        question="q",
        sql_code="",
        ground_truth_sql="select 1",
        sql_result_preview="",
    )
    assert result is None


def test_returns_score_object(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _score()
    _patch_llm(monkeypatch, expected)
    assert _call() is expected


def test_dict_result_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(monkeypatch, _score().model_dump())
    result = _call()
    assert isinstance(result, SqlScore)
    assert result.logic_match == 1.0


def test_llm_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(monkeypatch, RuntimeError("boom"))
    assert _call() is None


def test_unparseable_result_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(monkeypatch, {"not": "a score"})
    assert _call() is None
