# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``compare_cognee.py`` only puts a run beside 0.794 if it was measured the same way."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import compare_cognee as cc

ROOT = Path(__file__).resolve().parent.parent
_GPT5 = "azure/openai/gpt-5"


def _row(ability: str, index: int, scores: list[float], **extra: Any) -> dict[str, Any]:
    return {
        "conversation_id": "2",
        "ability": ability,
        "question_index": str(index),
        "verdicts": [{"score": s, "reason": "ok", "nugget": "n"} for s in scores],
        "judge_prompt": "cognee",
        "judge_model": _GPT5,
        **extra,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_nugget_mean_is_the_mean_of_verdicts() -> None:
    assert cc._nugget_mean(_row("a", 0, [1.0, 0.5, 0.0])) == 0.5


def test_round_files_refuse_a_run_stored_both_ways(tmp_path: Path) -> None:
    _write(tmp_path / "100K" / "agent.jsonl", [_row("a", 0, [1.0])])
    _write(tmp_path / "100K" / "agent_r1.jsonl", [_row("a", 0, [1.0])])
    with pytest.raises(SystemExit, match="count answers twice"):
        cc.round_files("100K", "agent", 1, tmp_path)


def test_round_files_refuse_a_missing_round(tmp_path: Path) -> None:
    _write(tmp_path / "100K" / "agent_r1.jsonl", [_row("a", 0, [1.0])])
    with pytest.raises(SystemExit, match="agent_r2.jsonl"):
        cc.round_files("100K", "agent", 2, tmp_path)


def test_round_files_pair_rounds_with_cognee_runs(tmp_path: Path) -> None:
    for i in (1, 2):
        _write(tmp_path / "100K" / f"agent_r{i}.jsonl", [_row("a", 0, [1.0])])
    assert [i for i, _ in cc.round_files("100K", "agent", 2, tmp_path)] == [0, 1]


def test_older_rows_take_their_judge_from_the_round_summary(tmp_path: Path) -> None:
    row = _row("a", 0, [1.0])
    del row["judge_prompt"], row["judge_model"]
    _write(tmp_path / "100K" / "agent_r1.jsonl", [row])
    (tmp_path / "100K" / "agent_r1_summary.json").write_text(
        json.dumps({"judge_prompt": "cognee", "judge_model": "m"})
    )
    [[read]] = cc.read_rounds("100K", "agent", 1, tmp_path)
    assert (read["judge_prompt"], read["judge_model"]) == ("cognee", "m")


@pytest.mark.parametrize(
    "judge",
    [
        {"judge_prompt": "beam", "judge_model": _GPT5},
        {"judge_prompt": "cognee", "judge_model": "azure/openai/gpt-4.1-mini"},
    ],
)
def test_verdicts_from_another_judge_are_refused(judge: dict[str, str]) -> None:
    rows = [_row("a", 0, [1.0]), {**_row("a", 1, [1.0]), **judge}]
    with pytest.raises(SystemExit, match="re-judge"):
        cc.require_cognee_judge(rows, _GPT5)


def test_cognee_judge_is_accepted() -> None:
    cc.require_cognee_judge([_row("a", 0, [1.0])], _GPT5)


def test_rounds_must_hold_every_question_once() -> None:
    full = [_row("a", 0, [1.0]), _row("a", 1, [1.0]), _row("b", 0, [1.0])]
    cc.require_complete_rounds([full, full], {"a", "b"}, 3)

    with pytest.raises(SystemExit, match="round 2 .*2 rows"):
        cc.require_complete_rounds([full, full[:2]], {"a", "b"}, 3)
    with pytest.raises(SystemExit, match="duplicate"):
        cc.require_complete_rounds([[full[0], full[0], full[2]]], {"a", "b"}, 3)


def test_overall_is_cognee_flat_mean_not_mean_of_ability_means() -> None:
    # Ability a has three questions at 1.0, ability b one at 0.0.
    one_round = [[1.0, 1.0, 1.0, 0.0]]
    assert cc.cognee_overall(one_round) == 0.75
    # The mean of ability means would be 0.5.
    assert cc.cognee_overall([[1.0], [0.0]]) == 0.5


def test_judge_failures_are_counted() -> None:
    row = _row("a", 0, [1.0])
    row["verdicts"].append({"score": 0.0, "reason": "ERROR: 503", "nugget": "n"})
    assert cc.judge_errors([row, _row("a", 1, [1.0])]) == 1


def _extractor() -> Any:
    spec = importlib.util.spec_from_file_location(
        "extract_cognee_answers", ROOT / "tools" / "extract_cognee_answers.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vendored_answers() -> dict[str, dict[str, Any]]:
    path = (
        ROOT
        / "conversation"
        / "beam_benchmark"
        / "cognee_results"
        / "cognee_answers_conv2.json"
    )
    return json.loads(path.read_text())["questions"]


def test_vendored_cognee_answers_reproduce_their_published_figures() -> None:
    assert _extractor().verify(_vendored_answers()) == []


def test_verify_notices_a_changed_score() -> None:
    questions = copy.deepcopy(_vendored_answers())
    first = next(iter(questions.values()))
    first["runs"][0]["score"] = 1.0 - first["runs"][0]["score"]
    assert _extractor().verify(questions)
