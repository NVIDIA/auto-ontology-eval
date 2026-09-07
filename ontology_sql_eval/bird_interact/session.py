"""Per-task adapter session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterSession:
    task_id: str
    mode: str
    session_id: str
    # Frozen copy of Bird's init_session state (returned verbatim in run_session responses)
    bird_state: dict = field(default_factory=dict)
    # Phase tracking
    phase1_completed: bool = False
    phase2_completed: bool = False
    task_done: bool = False
    total_reward: float = 0.0
    _submitted_this_phase: bool = False
    _last_submit_raw: str = ""
    # Turn-type classification for the *next* /run_session call, derived from
    # the previous submit_sql response's structured phase_completed field (and,
    # for a still-failing submission, the [exec_err_flg] marker in its message)
    # right when that response arrives — rather than re-deriving it later by
    # pattern-matching the orchestrator's own next message. See
    # gsf.retrieval.interactive.coordinator.step()'s turn_type/debug_error params.
    _next_turn_type: str | None = None  # "debug" | "follow_up" | None (= initial)
    _next_debug_error: str | None = None  # set only for the exec-error DEBUG case
    # True once this session's first /run_session call has been handled. Used
    # (instead of inferring from _next_turn_type, which is ambiguous between
    # "still initial" and "task already done") to know unambiguously when the
    # incoming message is the c-interact protocol's very first "User Query:
    # ..." message, so it can be classified as INITIAL up front rather than
    # via gsf's text-pattern fallback.
    _seen_first_run_session: bool = False
    # Conversation history (returned to orchestrator)
    dialogue_history: list = field(default_factory=list)
    tool_trajectory: list = field(default_factory=list)
    # GSF interactive session handle
    gsf_session: Any = None


# Module-level session store
_sessions: dict[str, AdapterSession] = {}


def get_session(task_id: str) -> AdapterSession:
    if task_id not in _sessions:
        raise KeyError(f"No session for task_id={task_id!r}")
    return _sessions[task_id]


def put_session(sess: AdapterSession) -> None:
    _sessions[sess.task_id] = sess


def clear_session(task_id: str) -> None:
    _sessions.pop(task_id, None)
