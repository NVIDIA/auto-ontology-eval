"""Per-task adapter session state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


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
