# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The installed Auto Ontology accepts the request the SQL arm sends."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from auto_ontology_eval.gym.agent_request import build_request

# Run in a fresh interpreter: tests/bird_interact/test_server.py replaces
# ``auto_ontology`` in ``sys.modules`` with stubs for the rest of the session.
_BUILD_STATE = textwrap.dedent(
    """
    from types import SimpleNamespace

    from auto_ontology.retrieval.text_to_sql.main import _build_state
    from auto_ontology.retrieval.text_to_sql.state import AgentRequest
    from auto_ontology_eval.gym.agent_request import build_request

    request = build_request(
        question="How many rows?",
        evidence="rows refers to table t",
        connector=SimpleNamespace(database_name="db", dialect="sqlite"),
        sql_examples=[{"question": "q", "sql": "SELECT 1"}],
    )
    state = _build_state(AgentRequest(**request))

    assert state["evidence"] == "rows refers to table t"
    assert state["calculation_only"] is True
    assert state["shorten_answer"] is True
    assert state["sql_examples"] == [{"question": "q", "sql": "SELECT 1"}]
    assert state["path_state"]["target_db"] == "db"
    """
)


def test_the_installed_library_builds_state_from_the_arm_request() -> None:
    """Every question fails if the pinned library rejects this request.

    The arm turns each agent error into ``UNKNOWN_ERROR``, so a request the
    library refuses scores 0% rather than crashing. CI installs the library
    from ``uv.lock``, so this runs against the pinned commit.
    """
    result = subprocess.run(
        [sys.executable, "-c", _BUILD_STATE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_empty_extra_prompts_are_left_off() -> None:
    request = build_request(
        question="q", evidence="", connector=object(), sql_examples=[]
    )
    assert "sql_examples" not in request
    assert "value_anchors" not in request
