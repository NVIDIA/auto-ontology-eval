# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gym/compare.py must not report while a dispatched question has no result.

run_gym_arm.sh records a failed /run in a sidecar and keeps going, so a failed
question is simply absent from the rollouts. Without this gate a question both
arms lost would vanish from the comparison and the report would look complete.
"""

from __future__ import annotations

import json
from pathlib import Path

from ontology_sql_eval.gym import compare


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

    assert compare.unfinished(out, compare.load(out)) == {"c"}


def test_a_run_without_materialized_inputs_has_nothing_unfinished(
    tmp_path: Path,
) -> None:
    out = tmp_path / "beaverbench.jsonl"
    _write(out, [{"task_id": "a", "reward": 1.0}])

    assert compare.unfinished(out, compare.load(out)) == set()


def test_a_question_both_arms_lost_blocks_the_report(tmp_path: Path) -> None:
    control = _arm(tmp_path / "control", done=["a", "b"], dispatched=["a", "b", "c"])
    treatment = _arm(
        tmp_path / "treatment", done=["a", "b"], dispatched=["a", "b", "c"]
    )
    argv = ["--control", str(control), "--treatment", str(treatment)]

    assert compare.main(argv) == 1
    assert compare.main(argv + ["--ignore-health"]) == 0


def test_finished_arms_report(tmp_path: Path) -> None:
    control = _arm(tmp_path / "control", done=["a", "b"], dispatched=["a", "b"])
    treatment = _arm(tmp_path / "treatment", done=["a", "b"], dispatched=["a", "b"])

    assert compare.compare(control, treatment) == 0
