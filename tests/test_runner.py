# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the batch/directory CSV scoring runner (scorer mocked)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluation.judge import runner
from evaluation.judge.config import Settings
from evaluation.judge.models import LLM_SCORE_FIELDS, SqlScore

_SETTINGS = Settings(model_name="m", base_url="https://x/v1", api_key="k")


def _fake_score() -> SqlScore:
    return SqlScore(
        logic_match=0.5,
        logic_issues="issue",
        semantic_match=0.5,
        final_weighted_score=0.5,
        sql_compared_to_ground_truth_score=0.5,
        is_valid_sql=True,
        is_sql_returns_data=True,
    )


@pytest.fixture(autouse=True)
def _patch_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the LLM scorer: score non-empty SQL, skip empty (returns None)."""

    def fake(settings, question, sql_code, ground_truth_sql, sql_result_preview):
        return _fake_score() if sql_code else None

    monkeypatch.setattr(runner, "score_sql", fake)


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_appends_llm_columns_and_preserves_originals(tmp_path: Path) -> None:
    fields = ["question_id", "question", "expected_sql", "returned_sql"]
    src = tmp_path / "in.csv"
    _write_csv(
        src,
        [
            {
                "question_id": "1",
                "question": "how many?",
                "expected_sql": "select count(*) from t",
                "returned_sql": "select count(*) from t",
            }
        ],
        fields,
    )
    out = tmp_path / "out.csv"

    runner.run(src, out, settings=_SETTINGS)

    rows = _read_csv(out)
    assert len(rows) == 1
    row = rows[0]
    # Original columns preserved.
    assert row["question"] == "how many?"
    assert row["expected_sql"] == "select count(*) from t"
    # LLM columns appended.
    for field in LLM_SCORE_FIELDS:
        assert field in row
    assert row["llm_logic_match"] == "0.5"
    assert row["llm_logic_issues"] == "issue"


def test_run_blank_scores_for_empty_returned_sql(tmp_path: Path) -> None:
    fields = ["question", "expected_sql", "returned_sql"]
    src = tmp_path / "in.csv"
    _write_csv(
        src,
        [{"question": "q", "expected_sql": "select 1", "returned_sql": ""}],
        fields,
    )
    out = tmp_path / "out.csv"

    runner.run(src, out, settings=_SETTINGS)

    row = _read_csv(out)[0]
    for field in LLM_SCORE_FIELDS:
        assert row[field] == ""


def test_run_directory_names_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fields = ["question", "expected_sql", "returned_sql"]
    _write_csv(
        input_dir / "nemotron.csv",
        [{"question": "q", "expected_sql": "select 1", "returned_sql": "select 1"}],
        fields,
    )

    runner.run_directory(input_dir, output_dir, settings=_SETTINGS)

    assert (output_dir / "nemotron_scores.csv").exists()


def test_run_directory_no_csvs_is_noop(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    runner.run_directory(input_dir, output_dir, settings=_SETTINGS)

    assert not output_dir.exists() or not any(output_dir.iterdir())
