# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""compare_arms.py must not report while a dispatched question has no result.

run_gym_arm.sh records a failed /run in a sidecar and keeps going, so a failed
question is simply absent from the rollouts. Without this gate a question both
arms lost would vanish from the comparison and the report would look complete.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_arms", Path(__file__).resolve().parents[1] / "scripts" / "compare_arms.py"
)
compare_arms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_arms)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _arm(directory: Path, done: list[str], dispatched: list[str]) -> Path:
    directory.mkdir()
    out = directory / "beaverbench.jsonl"
    _write(out, [{"task_id": t, "reward": 1.0, "failure_reason": "none"} for t in done])
    _write(
        directory / "beaverbench_materialized_inputs.jsonl",
        [{"task_id": t} for t in dispatched],
    )
    return out


def test_unfinished_lists_dispatched_questions_without_a_row(tmp_path: Path) -> None:
    out = _arm(tmp_path / "arm", done=["a", "b"], dispatched=["a", "b", "c"])

    assert compare_arms.unfinished(out, compare_arms.load(out)) == {"c"}


def test_a_run_without_materialized_inputs_has_nothing_unfinished(
    tmp_path: Path,
) -> None:
    out = tmp_path / "beaverbench.jsonl"
    _write(out, [{"task_id": "a", "reward": 1.0}])

    assert compare_arms.unfinished(out, compare_arms.load(out)) == set()


def test_a_question_both_arms_lost_blocks_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = _arm(tmp_path / "control", done=["a", "b"], dispatched=["a", "b", "c"])
    treatment = _arm(
        tmp_path / "treatment", done=["a", "b"], dispatched=["a", "b", "c"]
    )
    argv = ["compare_arms.py", "--control", str(control), "--treatment", str(treatment)]
    monkeypatch.setattr(sys, "argv", argv)

    assert compare_arms.main() == 1

    monkeypatch.setattr(sys, "argv", argv + ["--ignore-health"])
    assert compare_arms.main() == 0
