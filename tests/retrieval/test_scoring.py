# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ontology_sql_eval.retrieval.scoring import score_answer


def test_score_answer_empty_actual_on_text_only_expected() -> None:
    """Empty/failed agent response on a text-only answer must score 0.0, not 1.0."""
    expected = "| state |\n|---|\n| California |"
    res = score_answer(expected, "")
    assert res["answer_text_similarity"] == 0.0
    assert res["answer_numbers_match"] == 0


def test_score_answer_wrong_text_only_answer_uses_difflib() -> None:
    """Wrong text-only answer must fall through to difflib rather than getting 1.0."""
    expected = "| state |\n|---|\n| California |"
    actual = '[{"state": "New York"}]'
    res = score_answer(expected, actual)
    assert res["answer_text_similarity"] < 1.0
    assert res["answer_numbers_match"] == 0


def test_score_answer_structural_mismatch_not_overridden_by_nums_match() -> None:
    """When both sides parse into DataFrames and fail structural match, matching numbers alone must not force 1.0."""
    expected = "| name | score |\n|---|---|\n| Alice | 10 |\n| Bob | 20 |"
    actual = '[{"name": "Alice", "score": 20}, {"name": "Bob", "score": 10}]'
    res = score_answer(expected, actual)
    assert res["answer_text_similarity"] < 1.0
    assert res["answer_numbers_match"] == 1


def test_score_answer_numeric_fallback_when_unstructured() -> None:
    """When expected_raw is unstructured text (not a DataFrame), matching numeric multiset still scores 1.0."""
    expected = "The total count is 712."
    actual = "712"
    res = score_answer(expected, actual)
    assert res["answer_text_similarity"] == 1.0
    assert res["answer_numbers_match"] == 1