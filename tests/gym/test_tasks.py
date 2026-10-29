# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAVER's evidence can be left out of the tasks, and only BEAVER's.

The seed fills each BEAVER question's evidence from the benchmark's domain
knowledge, and both arms see it: the schema-only arm in its prompt, the
ontology arm from the task's ``evidence`` field. The BEAVER leaderboard's
Setting 0 gives no hints. BIRD's evidence is part of its standard setting, so
the option is refused there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import main
from ontology_sql_eval.gym.tasks import DATASETS, build_records
from scripts import build_gym_tasks

QUESTION = {
    "question_id": "dw_1",
    "db_id": "dw",
    "question": "How many rooms does building 7 have?",
    "evidence": "rooms: FCLT_ROOMS.ROOM_COUNT",
    "SQL": "SELECT 1",
    "difficulty": "base",
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "datasets" / "beaverbench" / "evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([QUESTION]))
    return tmp_path


def _prompt(record: dict[str, Any]) -> str:
    return record["responses_create_params"]["input"][-1]["content"]


def test_beaver_tasks_never_carry_evidence_on_this_branch(root: Path) -> None:
    # The BEAVER leaderboard run gives no hints (Setting 0), flag or not.
    [record] = build_records(DATASETS["beaverbench"], root, "auto_ontology")

    assert record["evidence"] == ""
    assert "Evidence" not in _prompt(record)


def test_no_evidence_leaves_it_out_of_the_prompt_and_the_task(root: Path) -> None:
    [record] = build_records(
        DATASETS["beaverbench"], root, "auto_ontology", evidence=False
    )

    assert record["evidence"] == ""
    assert "Evidence" not in _prompt(record)
    assert record["task_id"] == "beaverbench-dw_1"


def test_main_offers_no_evidence_for_beaver_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = main._parse_args(["--database-name", "beaverbench", "--no-evidence"])
    assert args.no_evidence

    with pytest.raises(SystemExit):
        main._parse_args(["--database-name", "bird", "--no-evidence"])
    assert "only for beaverbench, not bird" in capsys.readouterr().err


def test_build_gym_tasks_offers_no_evidence_for_beaver_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["build_gym_tasks.py", "--dataset", "bird", "--arm", "auto_ontology"]
    monkeypatch.setattr(sys, "argv", [*argv, "--no-evidence"])

    with pytest.raises(SystemExit):
        build_gym_tasks.main()
    assert "only for beaverbench, not bird" in capsys.readouterr().err
