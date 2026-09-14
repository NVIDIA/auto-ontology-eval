"""FastAPI service on port 6000 — Bird system agent interface for c-Interact."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load GSF/.env into this process before importing any gsf.* module — several
# (e.g. gsf.retrieval.interactive.entity_resolution) read feature-flag env
# vars (INTERACTIVE, DB_PROBE_*, BIRD_INTERACT, ...) at import time via
# module-level constants, so this must run first or those flags silently
# fall back to defaults. Every other GSF entrypoint (gsf/server/__main__.py,
# ingestion_service, semantic, alembic) already does this; this adapter is
# the one process that runs gsf.retrieval code without it, so GSF/.env was
# previously never read here.
#
# BIRD_INTERACT itself is this repo's own master flag (ontology_sql_eval.env),
# not GSF's — GSF just provides the flags it cascades onto.
from ontology_sql_eval.env import load_env

load_env()

from gsf.retrieval.interactive import (  # noqa: E402
    AskUserAction,
    SubmitSQLAction,
    TurnType,
    create_session as gsf_create_session,
    step as gsf_step,
    apply_user_answer as gsf_apply_user_answer,
    apply_submit_result as gsf_apply_submit_result,
)
from . import bird_interact_http as bird_http  # noqa: E402
from .known_issues import is_p1snap_collision  # noqa: E402
from .session import AdapterSession, get_session, put_session  # noqa: E402

# ── Logging setup ─────────────────────────────────────────────────────────────
# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# Our adapter logs at INFO; silence uvicorn access noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("gsf.retrieval.interactive.clarify").setLevel(logging.DEBUG)
logging.getLogger("gsf.retrieval.interactive.kb_coverage").setLevel(logging.DEBUG)
logging.getLogger("gsf.retrieval.interactive.entity_resolution").setLevel(logging.DEBUG)
logging.getLogger("gsf.retrieval.interactive.completeness").setLevel(logging.DEBUG)
logging.getLogger("gsf.retrieval.interactive.coordinator").setLevel(logging.INFO)
logging.getLogger("gsf.retrieval.data_access").setLevel(logging.INFO)
logging.getLogger("gsf.retrieval.text_to_sql").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
# ─────────────────────────────────────────────────────────────────────────────

_DATA_RETRIEVER = None
_SEMANTIC_RETRIEVER = None
_CONNECTORS: dict[str, list] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DATA_RETRIEVER, _SEMANTIC_RETRIEVER, _CONNECTORS
    from gsf.utils import get_data_objects_retriever, get_semantic_objects_retriever
    from ontology_sql_eval.ingestion.ingest import database_name_for, create_connector

    logger.info("[startup] loading retrievers...")
    _DATA_RETRIEVER = get_data_objects_retriever()
    _SEMANTIC_RETRIEVER = get_semantic_objects_retriever()

    connection_strings = [
        cs.strip()
        for cs in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if cs.strip()
    ]
    for cs in connection_strings:
        db_name = database_name_for(cs)
        try:
            _CONNECTORS[db_name] = [create_connector(cs)]
        except Exception as e:
            logger.warning("[startup] failed connector for %s: %s", cs, e)

    logger.info("[startup] ready — %d connectors loaded", len(_CONNECTORS))
    yield


app = FastAPI(title="BIRD-Interact GSF Adapter", lifespan=lifespan)


# ── c-interact message-text extraction ──────────────────────────────────────
# These parse the two pieces of question text the c-interact orchestrator
# only ever sends embedded in free text (orchestrator/cinteract.py in the
# reference ADK: line 136 for the initial "User Query:" message, line ~157
# for the Phase 2 follow-up message) — there is no structured field for
# either at the point they're needed, so this is the earliest point in the
# adapter (not gsf) they can be pulled out once and threaded through.


def _extract_initial_question(message: str) -> str:
    """Extract the user question from the Phase 1 orchestrator message."""
    match = re.search(r"User Query:\s*\n(.*?)(?:\n\n|$)", message, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_followup_question(message: str) -> str:
    """Extract the follow-up question from the Phase 2 orchestrator message."""
    match = re.search(
        r"follow-up question:\s*\n\n(.*?)(?:\n\nGenerate|$)", message, re.DOTALL
    )
    return match.group(1).strip() if match else ""


_MAX_KG_CHILDREN = 5

# Root of the (symlinked) BIRD-Interact dataset checkout. children_knowledge is a
# dataset field the upstream db_environment /knowledge endpoint does not expose to
# agents (see KNOWLEDGE_VISIBLE_FIELDS in db_environment/server.py, which we leave
# untouched), so we read it straight from the local *_kb.jsonl files ourselves
# instead of asking upstream to relay it.
_ADK_DIR = Path(
    os.environ.get(
        "BIRD_INTERACT_ADK_DIR",
        str(
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "BIRD-Interact"
            / "BIRD-Interact-ADK"
        ),
    )
)

_kb_children_cache: dict[str, dict[int, list[int]]] = {}


def _load_kb_children(db_name: str) -> dict[int, list[int]]:
    """Read {db_name}_kb.jsonl directly and return {id: children_knowledge} for entries that declare it.

    This mirrors upstream's own kb.jsonl loading (db_environment/server.py
    _load_db_data) but only ever reads the field locally — it never depends on
    upstream relaying children_knowledge over the /knowledge HTTP endpoint.
    """
    if db_name in _kb_children_cache:
        return _kb_children_cache[db_name]
    result: dict[int, list[int]] = {}
    for dataset in ("bird-interact-lite", "bird-interact-full"):
        kb_path = _ADK_DIR / dataset / db_name / f"{db_name}_kb.jsonl"
        if not kb_path.exists():
            continue
        try:
            with open(kb_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    children = entry.get("children_knowledge")
                    if entry.get("id") is not None and isinstance(children, list):
                        result[entry["id"]] = children
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[kb] failed to read %s: %s", kb_path, e)
        break
    _kb_children_cache[db_name] = result
    return result


def _format_external_kb(
    raw_json: str, db_name: str = ""
) -> tuple[str, dict[str, list[str]]]:
    """Parse Bird's knowledge JSON array into a formatted bullet list and a children map.

    Returns (formatted_kb, children_map) where children_map maps each parent entry
    name to the full formatted texts of its declared children_knowledge entries.
    """
    try:
        items = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return "", {}
    if not items:
        return "", {}

    id_to_name: dict[int, str] = {}
    id_to_text: dict[int, str] = {}
    for item in items:
        item_id = item.get("id")
        name = item.get("knowledge", "")
        desc = item.get("description", "")
        defn = item.get("definition", "")
        parts = [f"- {name}"] if name else []
        if desc:
            parts.append(f"  Description: {desc}")
        if defn:
            parts.append(f"  Definition: {defn}")
        if parts and item_id is not None:
            id_to_name[item_id] = name
            id_to_text[item_id] = "\n".join(parts)

    formatted_kb = "\n".join(id_to_text.values())

    kb_children = _load_kb_children(db_name) if db_name else {}
    children_map: dict[str, list[str]] = {}
    for item in items:
        item_id = item.get("id")
        name = item.get("knowledge", "")
        raw_children = kb_children.get(item_id, -1)
        if not name or raw_children == -1 or not isinstance(raw_children, list):
            continue
        child_texts = [
            id_to_text[cid]
            for cid in raw_children[:_MAX_KG_CHILDREN]
            if cid in id_to_text
        ]
        if child_texts:
            children_map[name] = child_texts

    return formatted_kb, children_map


# ── Request models ────────────────────────────────────────────────────────────


class InitSessionRequest(BaseModel):
    task_id: str
    mode: str = "c-interact"
    state: Dict[str, Any] = {}
    reset: bool = True


class RunSessionRequest(BaseModel):
    task_id: str
    mode: str = "c-interact"
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "system_agent", "adk_available": True}


@app.post("/init_session")
async def init_session(req: InitSessionRequest):
    db_name = req.state.get("db_name", "")
    db_schema = req.state.get("db_schema", "")
    external_kb, external_kb_children_map = _format_external_kb(
        req.state.get("external_kb", "[]"), db_name
    )
    session_id = uuid.uuid4().hex

    connectors = _CONNECTORS.get(db_name, [])
    if not connectors:
        logger.warning("[session] no connector for db=%r", db_name)

    max_turn = int(req.state.get("max_turn", 5))
    gsf_sess = gsf_create_session(
        session_id=session_id,
        task_id=req.task_id,
        db_name=db_name,
        db_schema=db_schema,
        external_kb=external_kb,
        question="",
        data_retriever=_DATA_RETRIEVER,
        semantic_retriever=_SEMANTIC_RETRIEVER,
        connectors=connectors,
        max_clarify_turns=max_turn,
        external_kb_children_map=external_kb_children_map,
    )

    sess = AdapterSession(
        task_id=req.task_id,
        mode=req.mode,
        session_id=session_id,
        bird_state=req.state,
        gsf_session=gsf_sess,
    )
    put_session(sess)
    logger.info("[session] %s | init db=%s max_turn=%d", req.task_id, db_name, max_turn)

    return {
        "task_id": req.task_id,
        "mode": req.mode,
        "session_id": session_id,
        "adk_available": True,
    }


@app.post("/run_session")
async def run_session(req: RunSessionRequest):
    try:
        sess = get_session(req.task_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No session for task_id={req.task_id!r}. Call /init_session first.",
        )

    sess._submitted_this_phase = False
    import asyncio

    # Classify this request's message once, up front, and reuse the result for
    # every gsf_step call in this request's loop (mirrors how req.message
    # itself is reused unchanged across ask_user round-trips within the
    # loop). The classification itself comes from state the adapter already
    # holds — never by asking gsf to pattern-match req.message:
    #   - first-ever call for this session → INITIAL, with the question text
    #     extracted from req.message here (the c-interact orchestrator has no
    #     structured field for it — see _extract_initial_question).
    #   - otherwise, whatever the *previous* submit_sql response already told
    #     us about the next turn (see the submit branch below) — DEBUG (with
    #     its pre-extracted error text) or FOLLOW_UP (with its question text
    #     extracted from req.message here, same reasoning as INITIAL).
    debug_error_hint: str | None = None
    initial_question_hint: str | None = None
    follow_up_question_hint: str | None = None
    if not sess._seen_first_run_session:
        turn_type_hint = TurnType.INITIAL
        initial_question_hint = _extract_initial_question(req.message)
    elif sess._next_turn_type == "debug":
        turn_type_hint = TurnType.DEBUG
        debug_error_hint = sess._next_debug_error
    elif sess._next_turn_type == "follow_up":
        turn_type_hint = TurnType.FOLLOW_UP
        follow_up_question_hint = _extract_followup_question(req.message)
    else:
        # No pending turn to classify — either the task already finished
        # (phase_completed reached 2) or this session was never submitted
        # to. Either way this is an unexpected extra call; fail loudly
        # instead of guessing a turn_type for gsf.
        raise HTTPException(
            status_code=400,
            detail=(
                f"run_session called for task_id={req.task_id!r} with no "
                "pending turn — the task is already complete or no "
                "submit_sql call has been made yet for this session."
            ),
        )

    turn = 0
    t_start = time.time()

    # Timing and entity tracking for structured logging
    t_loop_start = time.time()
    t_last_ask_user: float | None = None
    t_submit: float | None = None
    unresolved_start = list(getattr(sess.gsf_session, "persistent_unresolved", []))

    while True:
        turn += 1
        t0 = time.time()
        action = await asyncio.to_thread(
            gsf_step,
            sess.gsf_session,
            turn_type=turn_type_hint,
            debug_error=debug_error_hint,
            initial_question=initial_question_hint,
            follow_up_question=follow_up_question_hint,
        )
        elapsed = time.time() - t0

        if isinstance(action, AskUserAction):
            logger.info(
                "[turn %d | %.1fs] DECISION: ASK\n"
                "  ┌─ \033[1;35mquestion:\033[0m \033[35m%s\033[0m",
                turn,
                elapsed,
                action.question,
            )
            try:
                answer = await bird_http.ask_user(req.task_id, action.question)
                logger.info(
                    "  └─ \033[1;35muser answer:\033[0m \033[35m%s\033[0m", answer
                )
            except Exception as e:
                logger.error("[turn %d] ask_user failed: %r", turn, e)
                raise HTTPException(
                    status_code=502, detail=f"ask_user error: {type(e).__name__}: {e}"
                )

            t_last_ask_user = time.time()
            sess.dialogue_history.append({"role": "agent", "content": action.question})
            sess.dialogue_history.append({"role": "user", "content": answer})
            sess.tool_trajectory.append(
                {
                    "tool": "ask_user",
                    "input": {"question": action.question},
                    "output": {"answer": answer},
                }
            )
            await asyncio.to_thread(gsf_apply_user_answer, sess.gsf_session, answer)
            continue

        if isinstance(action, SubmitSQLAction):
            _gsf = sess.gsf_session
            _hist_len = len(getattr(_gsf, "clarify_history", []))
            _max_turns = getattr(_gsf, "max_clarify_turns", "?")
            logger.info(
                "[turn %d | %.1fs] DECISION: SUBMIT SQL  (history len=%s/%s)\n"
                "  ┌─ \033[1msql:\033[0m \033[1;35m%s\033[0m",
                turn,
                elapsed,
                _hist_len,
                _max_turns,
                action.sql,
            )
            try:
                result = await bird_http.submit_sql(req.task_id, action.sql)
                reward = result.get("reward", 0)
                phase = result.get("phase_completed")
                msg = result.get("message", "")
                logger.info(
                    "  └─ result: reward=%.2f  phase_completed=%s  msg=%s",
                    reward,
                    phase,
                    msg,
                )
            except Exception as e:
                logger.error("[turn %d] submit_sql failed: %s", turn, e)
                raise HTTPException(status_code=502, detail=f"submit_sql error: {e}")

            # This ADK crash signature (see known_issues.py) should be
            # unreachable: scripts/start_bird_services.sh mandatorily patches
            # ADK's create_task_db() before ADK's services ever start, so a
            # run reaching here has an unpatched/bypassed ADK setup — a setup
            # bug worth stopping the run for, not a score to silently correct.
            if is_p1snap_collision(msg):
                logger.error(
                    "[turn %d] ADK p1snap name-collision signature seen despite "
                    "the mandatory patch (see known_issues.py) — ADK services were "
                    "likely started without scripts/start_bird_services.sh, or the "
                    "patch stopped applying after an ADK update.",
                    turn,
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "ADK p1snap name-collision bug detected; this should be "
                        "impossible with the patch applied — see known_issues.py"
                    ),
                )

            t_submit = time.time()
            sess._last_submit_raw = result.get("message", "")
            sess.total_reward += float(result.get("reward", 0.0))
            sess.tool_trajectory.append(
                {
                    "tool": "submit_sql",
                    "input": {"sql": action.sql},
                    "output": result,
                }
            )

            phase_completed = result.get("phase_completed")
            if phase_completed == 1:
                sess.phase1_completed = True
            if phase_completed == 2:
                sess.phase2_completed = True
                sess.task_done = True

            # Classify what the *next* run_session's message will be from this
            # submit response itself, instead of leaving it to be re-derived
            # later by pattern-matching the orchestrator's own next message
            # (see the turn_type_hint/debug_error_hint block above the loop).
            if phase_completed == 1:
                sess._next_turn_type = "follow_up"
                sess._next_debug_error = None
            elif phase_completed == 2:
                # Task is done — no further run_session is expected for this
                # task, but reset defensively in case one arrives anyway.
                sess._next_turn_type = None
                sess._next_debug_error = None
            else:
                sess._next_turn_type = "debug"
                raw_msg = result.get("message", "")
                if "[exec_err_flg]" in raw_msg:
                    sess._next_debug_error = raw_msg.split("[exec_err_flg]", 1)[
                        -1
                    ].strip()
                else:
                    sess._next_debug_error = None

            gsf_apply_submit_result(sess.gsf_session, result)
            sess._submitted_this_phase = True
            break

        logger.warning(
            "[turn %d] unexpected action type %s — aborting", turn, type(action)
        )
        break

    # Only once the turn has actually run to completion. Setting it before the
    # loop would make a transient failure (the 502s above) unrecoverable: on
    # the first call _next_turn_type is still unset, so the orchestrator's
    # retry of the same turn would fall through to the "no pending turn" 400
    # instead of being re-classified as INITIAL.
    sess._seen_first_run_session = True

    logger.info(
        "[done] %d turns in %.1fs | reward=%.2f",
        turn,
        time.time() - t_start,
        sess.total_reward,
    )

    # Compute timing breakdown: clarification = time until last ask_user answer;
    # sql_gen = time from last ask_user (or loop start) until submit_sql.
    t_clarify_end = t_last_ask_user or t_loop_start
    timing_clarification_secs = t_clarify_end - t_loop_start
    timing_sql_gen_secs = (t_submit - t_clarify_end) if t_submit is not None else None

    unresolved_end = list(getattr(sess.gsf_session, "persistent_unresolved", []))
    resolved_entities = list(getattr(sess.gsf_session, "resolved_persistent", set()))
    initial_extracted_entities = list(
        getattr(sess.gsf_session, "initial_extracted_entities", [])
    )
    extracted_entities = list(
        getattr(sess.gsf_session, "path_state", {}).get("entities") or []
    )

    out_state = {
        **sess.bird_state,
        "phase1_completed": sess.phase1_completed,
        "phase2_completed": sess.phase2_completed,
        "total_reward": sess.total_reward,
        "dialogue_history": sess.dialogue_history,
        "tool_trajectory": sess.tool_trajectory,
        "_last_submit_raw": sess._last_submit_raw,
        "_submitted_this_phase": sess._submitted_this_phase,
        "task_done": sess.task_done,
        "adk_events": [],
        # Structured logging fields for the overnight runner
        "timing_clarification_secs": timing_clarification_secs,
        "timing_sql_gen_secs": timing_sql_gen_secs,
        "unresolved_entities_start": unresolved_start,
        "unresolved_entities_end": unresolved_end,
        "resolved_entities": resolved_entities,
        "initial_extracted_entities": initial_extracted_entities,
        "extracted_entities": extracted_entities,
    }
    return {
        "task_id": req.task_id,
        "mode": req.mode,
        "session_id": sess.session_id,
        "state": out_state,
        "adk_available": True,
    }
