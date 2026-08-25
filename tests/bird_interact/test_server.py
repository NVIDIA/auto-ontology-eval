# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked tests for the BIRD-Interact adapter server.

All GSF and Bird HTTP calls are mocked — no external services needed.
"""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Inject mock modules for gsf.retrieval.interactive BEFORE any import of
# server.py (which has top-level imports from that package).
# ---------------------------------------------------------------------------


class AskUserAction:
    """Minimal stand-in for gsf.retrieval.interactive.AskUserAction."""

    def __init__(self, question: str) -> None:
        self.question = question


class SubmitSQLAction:
    """Minimal stand-in for gsf.retrieval.interactive.SubmitSQLAction."""

    def __init__(self, sql: str) -> None:
        self.sql = sql


def _ensure_gsf_mocks() -> None:
    """Populate sys.modules with lightweight stubs if gsf is absent."""
    if "gsf" in sys.modules:
        return  # already present (real package or previously injected)

    gsf_mod = types.ModuleType("gsf")
    gsf_retrieval = types.ModuleType("gsf.retrieval")
    gsf_interactive = types.ModuleType("gsf.retrieval.interactive")

    # Action types used by server.py
    gsf_interactive.AskUserAction = AskUserAction  # type: ignore[attr-defined]
    gsf_interactive.SubmitSQLAction = SubmitSQLAction  # type: ignore[attr-defined]

    # Callable stubs (tests override these per-test via patch)
    gsf_interactive.create_session = MagicMock()  # type: ignore[attr-defined]
    gsf_interactive.step = MagicMock()  # type: ignore[attr-defined]
    gsf_interactive.apply_user_answer = MagicMock()  # type: ignore[attr-defined]
    gsf_interactive.apply_submit_result = MagicMock()  # type: ignore[attr-defined]

    # Sub-module alias so `from gsf.retrieval.interactive.types import ...` works
    gsf_interactive_types = types.ModuleType("gsf.retrieval.interactive.types")
    gsf_interactive_types.AskUserAction = AskUserAction  # type: ignore[attr-defined]
    gsf_interactive_types.SubmitSQLAction = SubmitSQLAction  # type: ignore[attr-defined]

    gsf_utils = types.ModuleType("gsf.utils")
    gsf_utils.get_data_objects_retriever = MagicMock()  # type: ignore[attr-defined]
    gsf_utils.get_semantic_objects_retriever = MagicMock()  # type: ignore[attr-defined]

    gsf_mod.retrieval = gsf_retrieval  # type: ignore[attr-defined]
    gsf_retrieval.interactive = gsf_interactive  # type: ignore[attr-defined]
    gsf_mod.utils = gsf_utils  # type: ignore[attr-defined]

    sys.modules["gsf"] = gsf_mod
    sys.modules["gsf.retrieval"] = gsf_retrieval
    sys.modules["gsf.retrieval.interactive"] = gsf_interactive
    sys.modules["gsf.retrieval.interactive.types"] = gsf_interactive_types
    sys.modules["gsf.utils"] = gsf_utils


_ensure_gsf_mocks()

# ---------------------------------------------------------------------------
# Now it is safe to import the server module.
# ---------------------------------------------------------------------------

import ontology_sql_eval.bird_interact.server as server_mod  # noqa: E402
from ontology_sql_eval.bird_interact.server import app  # noqa: E402
from ontology_sql_eval.bird_interact import session as session_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app):  # type: ignore[no-untyped-def]
    """Replace the real lifespan so no GSF initialisation occurs."""
    yield


def _make_client(
    mock_sess: MagicMock,
    step_side_effect,
    ask_user_return: str = "2023",
    submit_sql_return: dict | None = None,
) -> tuple[TestClient, MagicMock, MagicMock]:
    """Build a patched TestClient and return (client, mock_step, mock_submit)."""
    if submit_sql_return is None:
        submit_sql_return = {"reward": 1.0, "phase_completed": 1, "message": "correct"}

    mock_step = MagicMock(side_effect=step_side_effect)
    mock_ask = AsyncMock(return_value=ask_user_return)
    mock_submit = AsyncMock(return_value=submit_sql_return)

    ctx = (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(server_mod, "gsf_step", mock_step),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch("ontology_sql_eval.bird_interact.bird_interact_http.ask_user", mock_ask),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql", mock_submit
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    )
    return ctx, mock_step, mock_submit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Wipe the session store before each test to prevent state leakage."""
    session_mod._sessions.clear()
    yield
    session_mod._sessions.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health():
    mock_sess = MagicMock()
    ctx, _, _ = _make_client(mock_sess, [])
    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {}),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["adk_available"] is True
    assert data["status"] == "healthy"


def test_init_session():
    mock_sess = MagicMock()
    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            resp = client.post(
                "/init_session",
                json={
                    "task_id": "task-001",
                    "mode": "c-interact",
                    "state": {"db_name": "alien", "db_schema": "CREATE TABLE ..."},
                    "reset": True,
                },
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task-001"
    assert "session_id" in data
    assert data["adk_available"] is True


def test_run_session_submit_only():
    """step() immediately returns SubmitSQLAction → phase1_completed, total_reward=1.0."""
    mock_sess = MagicMock()
    submit_result = {"reward": 1.0, "phase_completed": 1, "message": "correct"}

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(
            server_mod, "gsf_step", MagicMock(return_value=SubmitSQLAction("SELECT 1"))
        ),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=submit_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            # First: init the session
            client.post(
                "/init_session",
                json={"task_id": "task-002", "state": {"db_name": "alien"}},
            )
            resp = client.post(
                "/run_session",
                json={"task_id": "task-002", "message": "Find all aliens"},
            )

    assert resp.status_code == 200
    state = resp.json()["state"]
    assert state["phase1_completed"] is True
    assert state["total_reward"] == 1.0


def test_run_session_corrects_known_p1snap_collision():
    """A Phase 1 submission that ADK crashed on with the known p1snap
    name-collision signature must be reported as the pass it actually was
    (see known_issues.py), not the reward=0.0 the crash produced.
    """
    mock_sess = MagicMock()
    mock_sess.phase1_completed = False
    mock_sess.phase1_submit_attempts = 0
    submit_result = {
        "reward": 0.0,
        "phase_completed": None,
        "message": (
            "Error: Command ['createdb', '-h', '127.0.0.1', '-p', '5433', "
            "'-U', 'root', "
            "'labor_certification_applications__labor_certification_applicati"
            "ons_2__p1snap', '--template', "
            "'labor_certification_applications__labor_certification_applicati"
            "ons_2'] returned non-zero exit status 1."
        ),
    }

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(
            server_mod, "gsf_step", MagicMock(return_value=SubmitSQLAction("SELECT 1"))
        ),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=submit_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={
                    "task_id": "task-003",
                    "state": {"db_name": "labor_certification_applications"},
                },
            )
            resp = client.post(
                "/run_session",
                json={"task_id": "task-003", "message": "Find something"},
            )

    assert resp.status_code == 200
    state = resp.json()["state"]
    # First submit attempt on this session → the "first try" reward (0.7).
    assert state["total_reward"] == 0.7
    assert state["phase1_completed"] is True


def test_run_session_ask_then_submit():
    """step() returns AskUserAction first, then SubmitSQLAction; dialogue_history has 2 entries."""
    mock_sess = MagicMock()
    submit_result = {"reward": 0.5, "phase_completed": 1, "message": "ok"}

    step_mock = MagicMock(
        side_effect=[
            AskUserAction("Which year?"),
            SubmitSQLAction("SELECT 1"),
        ]
    )

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(server_mod, "gsf_step", step_mock),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.ask_user",
            AsyncMock(return_value="2023"),
        ),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=submit_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={"task_id": "task-003", "state": {"db_name": "alien"}},
            )
            resp = client.post(
                "/run_session", json={"task_id": "task-003", "message": "How many?"}
            )

    assert resp.status_code == 200
    state = resp.json()["state"]
    assert len(state["dialogue_history"]) == 2
    assert state["dialogue_history"][0] == {"role": "agent", "content": "Which year?"}
    assert state["dialogue_history"][1] == {"role": "user", "content": "2023"}
    assert state["total_reward"] == 0.5


def test_one_submit_per_turn():
    """_submitted_this_phase is True after a single SubmitSQLAction in a turn."""
    mock_sess = MagicMock()
    submit_result = {"reward": 0.5, "phase_completed": 1, "message": "ok"}

    step_mock = MagicMock(
        side_effect=[
            AskUserAction("Which year?"),
            SubmitSQLAction("SELECT 1"),
        ]
    )

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(server_mod, "gsf_step", step_mock),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.ask_user",
            AsyncMock(return_value="2023"),
        ),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=submit_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={"task_id": "task-004", "state": {"db_name": "alien"}},
            )
            resp = client.post(
                "/run_session", json={"task_id": "task-004", "message": "Count?"}
            )

    assert resp.status_code == 200
    assert resp.json()["state"]["_submitted_this_phase"] is True


def test_exec_err_flg_preserved():
    """_last_submit_raw contains the [exec_err_flg] prefix when submit returns an error message."""
    mock_sess = MagicMock()
    submit_result = {
        "message": "[exec_err_flg] syntax error near SELECT",
        "reward": 0.0,
    }

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(
            server_mod,
            "gsf_step",
            MagicMock(return_value=SubmitSQLAction("SELECT bad")),
        ),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=submit_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={"task_id": "task-005", "state": {"db_name": "alien"}},
            )
            resp = client.post(
                "/run_session", json={"task_id": "task-005", "message": "broken query"}
            )

    assert resp.status_code == 200
    state = resp.json()["state"]
    assert "[exec_err_flg]" in state["_last_submit_raw"]
    assert state["total_reward"] == 0.0


def test_missing_session_404():
    """run_session for an unknown task_id returns 404."""
    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {}),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            resp = client.post(
                "/run_session",
                json={"task_id": "no-such-task", "message": "hello"},
            )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _format_external_kg
# ---------------------------------------------------------------------------

from ontology_sql_eval.bird_interact.server import _format_external_kg  # noqa: E402


def test_format_empty_array():
    assert _format_external_kg("[]") == ""


def test_format_malformed_json():
    assert _format_external_kg("not json") == ""


def test_format_single_item():
    raw = '[{"knowledge": "PPR", "description": "Panel Performance Ratio.", "definition": "PPR = A/B * 100%"}]'
    result = _format_external_kg(raw)
    assert "PPR" in result
    assert "Panel Performance Ratio" in result
    assert "PPR = A/B * 100%" in result


def test_format_item_missing_optional_fields():
    raw = '[{"knowledge": "ROI"}]'
    result = _format_external_kg(raw)
    assert "ROI" in result
    assert "Description" not in result
    assert "Definition" not in result


def test_format_multiple_items():
    raw = '[{"knowledge": "A", "description": "desc A"}, {"knowledge": "B", "description": "desc B"}]'
    result = _format_external_kg(raw)
    assert "- A" in result
    assert "- B" in result


def test_init_session_formats_external_kg():
    """init_session with raw knowledge JSON stores formatted text on the GSF session."""
    mock_sess = MagicMock()
    create_mock = MagicMock(return_value=mock_sess)
    raw_kg = '[{"knowledge": "PPR", "description": "Panel ratio.", "definition": "PPR = A/B"}]'

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", create_mock),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={
                    "task_id": "task-kg",
                    "state": {"db_name": "alien", "external_kg": raw_kg},
                },
            )

    passed_kg = create_mock.call_args.kwargs["external_kg"]
    assert "PPR" in passed_kg
    assert "Panel ratio" in passed_kg
    assert raw_kg not in passed_kg  # raw JSON must NOT be passed through
