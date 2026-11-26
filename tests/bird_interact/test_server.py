# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked tests for the BIRD-Interact adapter server.

All Auto Ontology and Bird HTTP calls are mocked — no external services needed.
"""

from __future__ import annotations

import sys
import types
from contextlib import ExitStack, asynccontextmanager
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


def _ensure_auto_ontology_mocks() -> None:
    """Populate sys.modules with lightweight stubs if Auto Ontology is absent."""
    if "auto_ontology" in sys.modules:
        return  # already present (real package or previously injected)

    auto_ontology_mod = types.ModuleType("auto_ontology")
    auto_ontology_retrieval = types.ModuleType("auto_ontology.retrieval")
    auto_ontology_interactive = types.ModuleType("auto_ontology.retrieval.interactive")

    # Action types used by server.py
    auto_ontology_interactive.AskUserAction = AskUserAction  # type: ignore[attr-defined]
    auto_ontology_interactive.SubmitSQLAction = SubmitSQLAction  # type: ignore[attr-defined]
    auto_ontology_interactive.TurnType = TurnType  # type: ignore[attr-defined]

    # Callable stubs (tests override these per-test via patch)
    auto_ontology_interactive.create_session = MagicMock()  # type: ignore[attr-defined]
    auto_ontology_interactive.step = MagicMock()  # type: ignore[attr-defined]
    auto_ontology_interactive.apply_user_answer = MagicMock()  # type: ignore[attr-defined]
    auto_ontology_interactive.apply_submit_result = MagicMock()  # type: ignore[attr-defined]

    # Sub-module alias so `from auto_ontology.retrieval.interactive.types import ...` works
    auto_ontology_interactive_types = types.ModuleType(
        "auto_ontology.retrieval.interactive.types"
    )
    auto_ontology_interactive_types.AskUserAction = AskUserAction  # type: ignore[attr-defined]
    auto_ontology_interactive_types.SubmitSQLAction = SubmitSQLAction  # type: ignore[attr-defined]

    auto_ontology_utils = types.ModuleType("auto_ontology.utils")
    auto_ontology_utils.get_data_objects_retriever = MagicMock()  # type: ignore[attr-defined]
    auto_ontology_utils.get_semantic_objects_retriever = MagicMock()  # type: ignore[attr-defined]

    # Reached via server.py -> ontology_sql_eval.env, which falls back to
    # Auto Ontology's .env for anything this repo's own .env does not define.
    auto_ontology_env = types.ModuleType("auto_ontology.env")
    auto_ontology_env.load_env = MagicMock()  # type: ignore[attr-defined]

    auto_ontology_mod.retrieval = auto_ontology_retrieval  # type: ignore[attr-defined]
    auto_ontology_retrieval.interactive = auto_ontology_interactive  # type: ignore[attr-defined]
    auto_ontology_mod.utils = auto_ontology_utils  # type: ignore[attr-defined]
    auto_ontology_mod.env = auto_ontology_env  # type: ignore[attr-defined]

    sys.modules["auto_ontology"] = auto_ontology_mod
    sys.modules["auto_ontology.retrieval"] = auto_ontology_retrieval
    sys.modules["auto_ontology.retrieval.interactive"] = auto_ontology_interactive
    sys.modules["auto_ontology.retrieval.interactive.types"] = (
        auto_ontology_interactive_types
    )
    sys.modules["auto_ontology.utils"] = auto_ontology_utils
    sys.modules["auto_ontology.env"] = auto_ontology_env


_ensure_auto_ontology_mocks()

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
    """Replace the real lifespan so no Auto Ontology initialisation occurs."""
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", mock_step),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(
            server_mod,
            "auto_ontology_step",
            MagicMock(return_value=SubmitSQLAction("SELECT 1")),
        ),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(
            server_mod,
            "auto_ontology_step",
            MagicMock(return_value=SubmitSQLAction("SELECT 1")),
        ),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
    scripts/bird_interact/start_bird_services.sh mandatorily patches ADK before it starts
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(
            server_mod,
            "auto_ontology_step",
            MagicMock(return_value=SubmitSQLAction("SELECT 1")),
        ),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", step_mock),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", step_mock),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(
            server_mod,
            "auto_ontology_step",
            MagicMock(return_value=SubmitSQLAction("SELECT bad")),
        ),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
    call auto_ontology_step with turn_type=DEBUG and the pre-extracted error text,
    instead of leaving Auto Ontology to re-derive it by pattern-matching the next
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", step_mock),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
    auto_ontology_step with turn_type=FOLLOW_UP."""
    mock_sess = MagicMock()
    first_result = {"message": "correct", "reward": 1.0, "phase_completed": 1}
    step_mock = MagicMock(return_value=SubmitSQLAction("SELECT 1"))

    with (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", step_mock),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", step_mock),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", step_mock),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
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

        # A legitimate name is accepted and memoised whether or not the optional
        # local Full dataset is installed in the test checkout.
        children = server_mod._load_kb_children("labor_certification_applications")
        assert isinstance(children, dict)
        assert "labor_certification_applications" in server_mod._kb_children_cache
        assert (
            server_mod._kb_children_cache["labor_certification_applications"]
            is children
        )
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


# ---------------------------------------------------------------------------
# a-Interact
# ---------------------------------------------------------------------------


def test_budget_follows_reference_charging_rules():
    from ontology_sql_eval.bird_interact.budget import Budget

    budget = Budget.start(5.0)
    assert budget.charge("ask_user") and budget.remaining == 3.0
    assert budget.affordable_asks() == 0  # 2 more for an ask would leave < 3 to submit
    # A submit that spends the balance to zero is the last action.
    assert budget.charge("submit_sql") and budget.exhausted

    budget = Budget.start(1.5)
    assert not budget.charge("ask_user")
    assert budget.remaining == 1.5
    # An unaffordable submit is still allowed, as the free final exit.
    assert budget.charge("submit_sql") and budget.exhausted


def _ainteract_ctx(
    mock_sess, steps, submits, phase_transition=None, ask_answer="the 2023 ones"
):
    return (
        patch.object(server_mod, "_DATA_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_SEMANTIC_RETRIEVER", MagicMock()),
        patch.object(server_mod, "_CONNECTORS", {"alien": [MagicMock()]}),
        patch.object(
            server_mod, "auto_ontology_create_session", return_value=mock_sess
        ),
        patch.object(server_mod, "auto_ontology_step", MagicMock(side_effect=steps)),
        patch.object(server_mod, "auto_ontology_apply_user_answer", MagicMock()),
        patch.object(server_mod, "auto_ontology_apply_submit_result", MagicMock()),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.get_schema",
            AsyncMock(return_value="CREATE TABLE aliens (id int)"),
        ),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.get_all_knowledge",
            AsyncMock(return_value="[]"),
        ),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.ask_user",
            AsyncMock(return_value=ask_answer),
        ),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.submit_sql",
            AsyncMock(side_effect=submits),
        ),
        patch(
            "ontology_sql_eval.bird_interact.bird_interact_http.phase_transition",
            phase_transition or AsyncMock(),
        ),
        patch.object(app.router, "lifespan_context", _noop_lifespan),
    )


def _run_ainteract(ctx, budget: float, task_id: str = "a-001") -> dict:
    with ExitStack() as stack:
        for patcher in ctx:
            stack.enter_context(patcher)
        client = stack.enter_context(TestClient(app))
        init = client.post(
            "/init_session",
            json={
                "task_id": task_id,
                "mode": "a-interact",
                "state": {
                    "db_name": "alien",
                    "user_query": "Find recent aliens",
                    "initial_budget": budget,
                },
            },
        )
        assert init.status_code == 200
        resp = client.post(
            "/run_session",
            json={
                "task_id": task_id,
                "mode": "a-interact",
                "message": "User Query:\nFind recent aliens\n\n...",
            },
        )
    assert resp.status_code == 200
    return resp.json()["state"]


def test_ainteract_runs_both_phases_in_one_call():
    mock_sess = MagicMock(clarify_history=[])
    steps = [
        AskUserAction("Which year?"),
        SubmitSQLAction("SELECT 1"),
        SubmitSQLAction("SELECT 2"),
        SubmitSQLAction("SELECT 3"),
    ]
    submits = [
        {
            "passed": False,
            "reward": 0.0,
            "message": "SQL failed Phase 1. [exec_err_flg] boom",
        },
        {
            "passed": True,
            "reward": 0.7,
            "phase_completed": 1,
            "has_follow_up": True,
            "follow_up_query": "Now by planet?",
            "message": "Phase 1 correct!",
        },
        {
            "passed": True,
            "reward": 0.3,
            "phase_completed": 2,
            "message": "Phase 2 correct!",
        },
    ]
    phase_transition = AsyncMock()
    ctx = _ainteract_ctx(mock_sess, steps, submits, phase_transition)
    step_mock = ctx[4].new

    state = _run_ainteract(ctx, budget=20.0)

    # 3 (schema + column meanings + KB) + 2 (ask) + 3 * 3 (submits)
    assert state["budget_remaining"] == 20.0 - 14.0
    assert state["phase1_completed"] and state["phase2_completed"]
    assert state["total_reward"] == 1.0
    assert state["stop_reason"] == "done"
    phase_transition.assert_awaited_once_with("a-001")

    calls = step_mock.call_args_list
    assert calls[0].kwargs["turn_type"] == TurnType.INITIAL
    assert calls[0].kwargs["initial_question"] == "Find recent aliens"
    assert calls[2].kwargs["turn_type"] == TurnType.DEBUG
    assert calls[2].kwargs["debug_error"] == "boom"
    assert calls[3].kwargs["turn_type"] == TurnType.FOLLOW_UP
    follow_up_question = calls[3].kwargs["follow_up_question"]
    assert follow_up_question.startswith("Now by planet?\n\nExecution requirements:")
    assert "Phase 1 SQL as authoritative" in follow_up_question
    assert "exactly the requested columns" in follow_up_question
    assert [t["phase"] for t in state["tool_trajectory"]] == [1, 1, 1, 2]


def test_ainteract_stops_when_a_submit_spends_the_budget():
    mock_sess = MagicMock(clarify_history=[])
    steps = [SubmitSQLAction("SELECT 1"), SubmitSQLAction("SELECT 2")]
    wrong = {"passed": False, "reward": 0.0, "message": "Your SQL is not correct."}
    ctx = _ainteract_ctx(mock_sess, steps, [wrong, wrong])

    # 6 - 3 for schema + column meanings + KB leaves exactly one submission.
    state = _run_ainteract(ctx, budget=6.0)

    assert state["stop_reason"] == "budget_exhausted"
    assert len(state["tool_trajectory"]) == 1
    # Only one submission's worth left after the setup purchases, so no asks.
    assert mock_sess.max_clarify_turns == 0


def test_ainteract_stops_instead_of_resubmitting_a_failed_sql():
    mock_sess = MagicMock(clarify_history=[])
    steps = [SubmitSQLAction("SELECT 1"), SubmitSQLAction("SELECT 1")]
    wrong = {"passed": False, "reward": 0.0, "message": "Your SQL is not correct."}
    ctx = _ainteract_ctx(mock_sess, steps, [wrong])

    state = _run_ainteract(ctx, budget=20.0)

    assert state["stop_reason"] == "repeated_sql"
    assert state["budget_remaining"] == 20.0 - 3.0 - 3.0


def test_ainteract_recovers_last_generated_sql_instead_of_empty_paid_submit():
    mock_sess = MagicMock(
        clarify_history=[], path_state={"sql_code": "SELECT recovered"}
    )
    steps = [SubmitSQLAction("")]
    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": False,
        "message": "Phase 1 correct!",
    }
    ctx = _ainteract_ctx(mock_sess, steps, [passed])
    submit = ctx[10].new

    state = _run_ainteract(ctx, budget=14.0, task_id="a-recover-empty-sql")

    submit.assert_awaited_once_with("a-recover-empty-sql", "SELECT recovered")
    assert state["stop_reason"] == "done"
    assert state["tool_trajectory"][0]["input"]["sql"] == "SELECT recovered"


def test_ainteract_phase1_still_asks_when_the_balance_covers_it():
    """An ask is sent while the balance still covers the ask and one submission.

    Budget 13, of which schema, column meanings and KB take 3. The ask is still sent.
    """
    mock_sess = MagicMock(clarify_history=[])
    steps = [
        AskUserAction("Which year?"),
        SubmitSQLAction("SELECT 1"),
        SubmitSQLAction("SELECT 2"),
    ]
    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": True,
        "follow_up_query": "Now by planet?",
        "message": "Phase 1 correct!",
    }
    follow = {
        "passed": True,
        "reward": 0.3,
        "phase_completed": 2,
        "message": "Phase 2 correct!",
    }
    ctx = _ainteract_ctx(mock_sess, steps, [passed, follow])
    ask = ctx[9].new

    state = _run_ainteract(ctx, budget=13.0, task_id="a-ask-below-floor")

    ask.assert_awaited()
    assert state["phase1_completed"] and state["phase2_completed"]
    assert state["stop_reason"] == "done"
    # 3 setup + 2 ask + 3 phase 1 + 3 phase 2.
    assert state["budget_remaining"] == 2.0
    assert [t["tool"] for t in state["tool_trajectory"]] == [
        "ask_user",
        "submit_sql",
        "submit_sql",
    ]


def test_ainteract_stops_asking_after_the_user_simulator_failure_fallback():
    mock_sess = MagicMock(clarify_history=[])
    steps = [
        AskUserAction("Which year?"),
        AskUserAction("Which year again?"),
        SubmitSQLAction("SELECT 1"),
    ]
    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": False,
        "message": "Phase 1 correct!",
    }
    ctx = _ainteract_ctx(
        mock_sess,
        steps,
        [passed],
        ask_answer="I'm not sure I understand your question.",
    )
    ask = ctx[9].new

    state = _run_ainteract(ctx, budget=20.0, task_id="a-sim-down")

    assert ask.await_count == 1
    assert state["phase1_completed"] is True
    assert state["stop_reason"] == "done"
    assert [t["tool"] for t in state["tool_trajectory"]] == ["ask_user", "submit_sql"]
    # 3 setup + 2 for the one ask that failed + 3 for the submit.
    assert state["budget_remaining"] == 12.0
    assert mock_sess.max_clarify_turns == 0


def test_ainteract_phase1_debug_submit_can_spend_the_last_coins():
    mock_sess = MagicMock(clarify_history=[])
    steps = [SubmitSQLAction("SELECT 1"), SubmitSQLAction("SELECT 2")]
    wrong = {"passed": False, "reward": 0.0, "message": "Your SQL is not correct."}
    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": False,
        "message": "Phase 1 correct!",
    }
    ctx = _ainteract_ctx(mock_sess, steps, [wrong, passed])

    # 8 - 2 leaves 6. The first submission leaves 3. The debug submission spends
    # what is left and is the one that passes; it is not withheld.
    state = _run_ainteract(ctx, budget=8.0, task_id="a-debug-spends-floor")

    assert state["stop_reason"] == "done"
    assert state["phase1_completed"] is True
    assert state["budget_remaining"] == -1.0
    assert [t["input"]["sql"] for t in state["tool_trajectory"]] == [
        "SELECT 1",
        "SELECT 2",
    ]


def test_phase1_holds_four_coins_and_phase2_can_spend_them():
    """Phase-1 queries stop at 4 coins; the submit may spend them; phase 2 may too."""
    mock_sess = MagicMock(clarify_history=[])
    captured = {}

    def create_session(**kwargs):
        captured.update(kwargs)
        return mock_sess

    def step(*_args, **kwargs):
        from ontology_sql_eval.bird_interact.budget import BudgetExhausted

        conn = captured["connectors"][0]
        turn_type = kwargs.get("turn_type")
        if turn_type == server_mod.TurnType.FOLLOW_UP:
            conn.execute("SELECT phase2")
            return SubmitSQLAction("SELECT 2")
        conn.execute("SELECT phase1a")
        try:
            conn.execute("SELECT phase1b")
        except BudgetExhausted:
            pass
        return SubmitSQLAction("SELECT 1")

    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": True,
        "follow_up_query": "Now by planet?",
        "message": "Phase 1 correct!",
    }
    follow = {
        "passed": True,
        "reward": 0.3,
        "phase_completed": 2,
        "message": "Phase 2 correct!",
    }
    ctx = list(_ainteract_ctx(mock_sess, step, [passed, follow]))
    ctx[3] = patch.object(
        server_mod, "auto_ontology_create_session", side_effect=create_session
    )

    # 8 - 3 for setup leaves 5. The first query leaves the floor of 4.
    state = _run_ainteract(ctx, budget=8.0, task_id="a-floor-4")

    queries = [t for t in state["tool_trajectory"] if t["tool"] == "execute_sql"]
    submits = [t for t in state["tool_trajectory"] if t["tool"] == "submit_sql"]
    assert [(q["phase"], q["charged"], q["budget_after"]) for q in queries] == [
        (1, True, 4.0),
        (1, False, 4.0),
        (2, True, 0.0),
    ]
    # Phase-1 submit spends the held coins from 4 down to 1. Phase 2 spends the
    # last coin on its query, then the submit ends the wallet.
    assert [s["budget_after"] for s in submits] == [1.0, -1.0]
    assert state["phase1_completed"] and state["phase2_completed"]


def test_phase1_ask_stops_when_four_coins_remain():
    mock_sess = MagicMock(clarify_history=[])
    steps = [
        AskUserAction("Which year?"),
        AskUserAction("Which year again?"),
        SubmitSQLAction("SELECT 1"),
    ]
    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": False,
        "message": "Phase 1 correct!",
    }
    ctx = _ainteract_ctx(mock_sess, steps, [passed])

    # 9 - 3 leaves 6, so one ask lands exactly on the floor of 4. The next ask
    # would leave 2 and is refused.
    state = _run_ainteract(ctx, budget=9.0, task_id="a-ask-floor")

    asks = [t for t in state["tool_trajectory"] if t["tool"] == "ask_user"]
    assert [t["budget_after"] for t in asks] == [4.0]
    assert state["stop_reason"] == "ask_unaffordable"
    assert state["phase1_completed"] is False


def test_metered_connector_charges_queries_and_refuses_unaffordable_ones():
    from ontology_sql_eval.bird_interact.budget import (
        Budget,
        BudgetExhausted,
        MeteredConnector,
    )

    base = MagicMock(database_name="alien")
    seen = []
    conn = MeteredConnector(
        base, Budget.start(1.5), lambda sql, charged: seen.append((sql, charged))
    )

    conn.execute("SELECT 1", timeout_s=5)
    base.execute.assert_called_once_with("SELECT 1", timeout_s=5)
    with pytest.raises(BudgetExhausted):
        conn.execute("SELECT 2")
    assert base.execute.call_count == 1
    assert seen == [("SELECT 1", True), ("SELECT 2", False)]
    # Everything but execute() passes through uncharged.
    assert conn.database_name == "alien"


def _run_sql(sql, connector, *, record_numeric_kinds=False):
    """Stand-in for the text-to-SQL runner: any execution error becomes text."""

    class _Response:
        def __init__(self, error: str, statement: str) -> None:
            self.error = error
            self.statement = statement

    try:
        return connector.execute(sql)
    except Exception as e:
        return _Response(str(e), sql)


def test_budget_refusal_is_not_turned_into_a_sql_error():
    from ontology_sql_eval.bird_interact.budget import (
        BudgetExhausted,
        _patch_run_sql_to_reraise_budget,
    )

    module = types.ModuleType("fake_sql_execution")
    module._run_sql = _run_sql
    _patch_run_sql_to_reraise_budget(module)

    connector = MagicMock()
    connector.execute.side_effect = BudgetExhausted("query floor")
    with pytest.raises(BudgetExhausted):
        module._run_sql("SELECT 1", connector)

    connector.execute.side_effect = RuntimeError("column missing")
    response = module._run_sql("SELECT 1", connector)
    assert response.error and "column missing" in response.error


def test_ainteract_charges_auto_ontology_queries_as_execute_sql():
    mock_sess = MagicMock(clarify_history=[])
    captured = {}

    def create_session(**kwargs):
        captured.update(kwargs)
        return mock_sess

    def step(*_args, **_kwargs):
        captured["connectors"][0].execute("SELECT DISTINCT planet FROM aliens")
        return SubmitSQLAction("SELECT 1")

    passed = {
        "passed": True,
        "reward": 0.7,
        "phase_completed": 1,
        "has_follow_up": False,
        "message": "Phase 1 correct!",
    }
    ctx = list(_ainteract_ctx(mock_sess, step, [passed]))
    ctx[3] = patch.object(
        server_mod, "auto_ontology_create_session", side_effect=create_session
    )

    state = _run_ainteract(ctx, budget=20.0)

    # 3 (schema + column meanings + KB) + 1 (query) + 3 (submit)
    assert state["budget_remaining"] == 13.0
    query = state["tool_trajectory"][0]
    assert query["tool"] == "execute_sql"
    assert query["phase"] == 1 and query["charged"] is True
