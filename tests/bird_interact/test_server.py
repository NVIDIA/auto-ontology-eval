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
# Inject mock modules for auto_ontology.retrieval.interactive BEFORE any import of
# server.py (which has top-level imports from that package).
# ---------------------------------------------------------------------------


class AskUserAction:
    """Minimal stand-in for auto_ontology.retrieval.interactive.AskUserAction."""

    def __init__(self, question: str) -> None:
        self.question = question


class SubmitSQLAction:
    """Minimal stand-in for auto_ontology.retrieval.interactive.SubmitSQLAction."""

    def __init__(self, sql: str) -> None:
        self.sql = sql


class TurnType:
    """Minimal stand-in for auto_ontology.retrieval.interactive.TurnType."""

    INITIAL = "initial"
    DEBUG = "debug"
    FOLLOW_UP = "follow_up"


def _ensure_gsf_mocks() -> None:
    """Populate sys.modules with lightweight stubs if gsf is absent."""
    if "auto_ontology" in sys.modules:
        return  # already present (real package or previously injected)

    gsf_mod = types.ModuleType("auto_ontology")
    gsf_retrieval = types.ModuleType("auto_ontology.retrieval")
    gsf_interactive = types.ModuleType("auto_ontology.retrieval.interactive")

    # Action types used by server.py
    gsf_interactive.AskUserAction = AskUserAction  # type: ignore[attr-defined]
    gsf_interactive.SubmitSQLAction = SubmitSQLAction  # type: ignore[attr-defined]
    gsf_interactive.TurnType = TurnType  # type: ignore[attr-defined]

    # Callable stubs (tests override these per-test via patch)
    gsf_interactive.create_session = MagicMock()  # type: ignore[attr-defined]
    gsf_interactive.step = MagicMock()  # type: ignore[attr-defined]
    gsf_interactive.apply_user_answer = MagicMock()  # type: ignore[attr-defined]
    gsf_interactive.apply_submit_result = MagicMock()  # type: ignore[attr-defined]

    # Sub-module alias so `from auto_ontology.retrieval.interactive.types import ...` works
    gsf_interactive_types = types.ModuleType("auto_ontology.retrieval.interactive.types")
    gsf_interactive_types.AskUserAction = AskUserAction  # type: ignore[attr-defined]
    gsf_interactive_types.SubmitSQLAction = SubmitSQLAction  # type: ignore[attr-defined]

    gsf_utils = types.ModuleType("auto_ontology.utils")
    gsf_utils.get_data_objects_retriever = MagicMock()  # type: ignore[attr-defined]
    gsf_utils.get_semantic_objects_retriever = MagicMock()  # type: ignore[attr-defined]

    # Reached via server.py -> ontology_sql_eval.env, which falls back to
    # GSF's .env for anything this repo's own .env does not define.
    gsf_env = types.ModuleType("auto_ontology.env")
    gsf_env.load_env = MagicMock()  # type: ignore[attr-defined]

    gsf_mod.retrieval = gsf_retrieval  # type: ignore[attr-defined]
    gsf_retrieval.interactive = gsf_interactive  # type: ignore[attr-defined]
    gsf_mod.utils = gsf_utils  # type: ignore[attr-defined]
    gsf_mod.env = gsf_env  # type: ignore[attr-defined]

    sys.modules["auto_ontology"] = gsf_mod
    sys.modules["auto_ontology.retrieval"] = gsf_retrieval
    sys.modules["auto_ontology.retrieval.interactive"] = gsf_interactive
    sys.modules["auto_ontology.retrieval.interactive.types"] = gsf_interactive_types
    sys.modules["auto_ontology.utils"] = gsf_utils
    sys.modules["auto_ontology.env"] = gsf_env


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


def test_run_session_treats_null_reward_as_zero():
    mock_sess = MagicMock()
    submit_result = {"reward": None, "phase_completed": 1, "message": "no score"}

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(
            server_mod, "gsf_step", MagicMock(return_value=SubmitSQLAction("SELECT 1"))
        ),
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
                json={"task_id": "task-null-reward", "state": {"db_name": "alien"}},
            )
            resp = client.post(
                "/run_session",
                json={"task_id": "task-null-reward", "message": "Find all aliens"},
            )

    assert resp.status_code == 200
    assert resp.json()["state"]["total_reward"] == 0.0


def test_run_session_fails_loud_on_known_p1snap_collision():
    """The ADK p1snap name-collision signature should be unreachable now that
    scripts/start_bird_services.sh mandatorily patches ADK before it starts
    (see known_issues.py) — seeing it means ADK was started some other way,
    or the patch stopped applying, so the run should fail loud, not silently
    reinterpret the crash as a pass.
    """
    mock_sess = MagicMock()
    mock_sess.phase1_completed = False
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

    assert resp.status_code == 500


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


def test_next_turn_type_threaded_as_debug_with_error():
    """A failing exec-error submit on turn N must make turn N+1's run_session
    call gsf_step with turn_type=DEBUG and the pre-extracted error text,
    instead of leaving gsf to re-derive it by pattern-matching the next
    orchestrator message."""
    mock_sess = MagicMock()
    first_result = {
        "message": '[exec_err_flg] column "bad_column" does not exist',
        "reward": 0.0,
        "phase_completed": None,
    }
    step_mock = MagicMock(return_value=SubmitSQLAction("SELECT bad_column"))

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(server_mod, "gsf_step", step_mock),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=first_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={"task_id": "task-006", "state": {"db_name": "alien"}},
            )
            # Turn 1: fresh session — first-ever call is classified INITIAL,
            # with the question extracted from this same message.
            client.post(
                "/run_session",
                json={
                    "task_id": "task-006",
                    "message": "User Query:\nFind all aliens\n\n",
                },
            )
            first_call_kwargs = step_mock.call_args.kwargs
            assert first_call_kwargs["turn_type"] == server_mod.TurnType.INITIAL
            assert first_call_kwargs["initial_question"] == "Find all aliens"
            assert first_call_kwargs["debug_error"] is None

            # Turn 2: the orchestrator's debug-retry message arrives — the
            # adapter already knows (from turn 1's submit result) that this is
            # a DEBUG/exec-error turn, without needing to parse this message.
            client.post(
                "/run_session",
                json={
                    "task_id": "task-006",
                    "message": "Your SQL is not executable: ignored anyway",
                },
            )
            second_call_kwargs = step_mock.call_args.kwargs

    assert second_call_kwargs["turn_type"] == server_mod.TurnType.DEBUG
    assert second_call_kwargs["debug_error"] == 'column "bad_column" does not exist'


def test_next_turn_type_threaded_as_follow_up():
    """A phase-1-completing submit must make the next run_session call
    gsf_step with turn_type=FOLLOW_UP."""
    mock_sess = MagicMock()
    first_result = {"message": "correct", "reward": 1.0, "phase_completed": 1}
    step_mock = MagicMock(return_value=SubmitSQLAction("SELECT 1"))

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(server_mod, "gsf_step", step_mock),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(return_value=first_result),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={"task_id": "task-007", "state": {"db_name": "alien"}},
            )
            client.post(
                "/run_session",
                json={"task_id": "task-007", "message": "Find all aliens"},
            )
            client.post(
                "/run_session",
                json={
                    "task_id": "task-007",
                    "message": (
                        "Phase 1 is complete. Here is a follow-up "
                        "question:\n\nShow totals.\n\nGenerate the SQL."
                    ),
                },
            )
            second_call_kwargs = step_mock.call_args.kwargs

    assert second_call_kwargs["turn_type"] == server_mod.TurnType.FOLLOW_UP
    assert second_call_kwargs["debug_error"] is None
    assert second_call_kwargs["follow_up_question"] == "Show totals."


def test_first_turn_stays_retryable_after_transient_failure():
    """A 502 on the first run_session must leave the session retryable: the
    retry is still the INITIAL turn, not a 400 'no pending turn'."""
    mock_sess = MagicMock()
    step_mock = MagicMock(return_value=SubmitSQLAction("SELECT 1"))
    submit_mock = AsyncMock(
        side_effect=[
            RuntimeError("connection reset"),
            {"message": "correct", "reward": 1.0, "phase_completed": 1},
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
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            submit_mock,
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={"task_id": "task-008", "state": {"db_name": "alien"}},
            )
            payload = {
                "task_id": "task-008",
                "message": "User Query:\nFind all aliens\n\n",
            }
            failed = client.post("/run_session", json=payload)
            retried = client.post("/run_session", json=payload)
            retry_kwargs = step_mock.call_args.kwargs

    assert failed.status_code == 502
    assert retried.status_code == 200
    assert retry_kwargs["turn_type"] == server_mod.TurnType.INITIAL
    assert retry_kwargs["initial_question"] == "Find all aliens"


def test_run_session_caps_a_never_ending_clarification_loop():
    """A coordinator that only ever asks must not keep the request open: the
    loop stops at max_turn + _TURN_CAP_MARGIN instead of asking forever."""
    mock_sess = MagicMock()
    max_turn = 3
    step_mock = MagicMock(return_value=AskUserAction("Which year?"))
    ask_mock = AsyncMock(return_value="2023")
    submit_mock = AsyncMock()

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(server_mod, "gsf_create_session", return_value=mock_sess),
        patch.object(server_mod, "gsf_step", step_mock),
        patch.object(server_mod, "gsf_apply_user_answer", MagicMock()),
        patch.object(server_mod, "gsf_apply_submit_result", MagicMock()),
        patch("ontology_sql_eval.bird_interact.bird_interact_http.ask_user", ask_mock),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql", submit_mock
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    ):
        with TestClient(app) as client:
            client.post(
                "/init_session",
                json={
                    "task_id": "task-009",
                    "state": {"db_name": "alien", "max_turn": max_turn},
                },
            )
            resp = client.post(
                "/run_session",
                json={"task_id": "task-009", "message": "User Query:\nHow many?\n\n"},
            )

    expected_turns = max_turn + server_mod._TURN_CAP_MARGIN
    assert resp.status_code == 200
    assert ask_mock.await_count == expected_turns
    assert step_mock.call_count == expected_turns
    submit_mock.assert_not_awaited()
    # The turn is reported as unsubmitted rather than faking a submission.
    assert resp.json()["state"]["_submitted_this_phase"] is False


def test_load_kb_children_rejects_unsafe_db_name():
    """db_name reaches a filesystem path straight from the request body, so
    anything outside the real name shape is refused and left uncached."""
    server_mod._kb_children_cache.clear()
    try:
        for unsafe in ("../../../etc", "foo/../bar", "a/b", "x\x00y"):
            assert server_mod._load_kb_children(unsafe) == {}
            assert unsafe not in server_mod._kb_children_cache

        # An empty name is routine (state with no db_name) and also yields {}.
        assert server_mod._load_kb_children("") == {}

        # A legitimate name is still accepted and memoised -- {} here only
        # because no *_kb.jsonl exists for it in the test checkout.
        assert server_mod._load_kb_children("labor_certification_applications") == {}
        assert "labor_certification_applications" in server_mod._kb_children_cache
    finally:
        server_mod._kb_children_cache.clear()


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
