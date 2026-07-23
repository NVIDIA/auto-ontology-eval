"""FastAPI service on port 6000 — Bird system agent interface for c-Interact."""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gsf.retrieval.interactive import (
    AskUserAction,
    SubmitSQLAction,
    create_session as gsf_create_session,
    step as gsf_step,
    apply_user_answer as gsf_apply_user_answer,
    apply_submit_result as gsf_apply_submit_result,
)
from . import bird_http
from .session import AdapterSession, get_session, put_session

logger = logging.getLogger(__name__)

_DATA_RETRIEVER = None
_SEMANTIC_RETRIEVER = None
_CONNECTORS: dict[str, list] = {}   # db_name → [SQLDatabase]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DATA_RETRIEVER, _SEMANTIC_RETRIEVER, _CONNECTORS
    from gsf.utils import get_data_objects_retriever, get_semantic_objects_retriever
    from ontology_sql_eval.ingestion.ingest import database_name_for, create_connector

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
            logger.info("Loaded connector for %s", db_name)
        except Exception as e:
            logger.warning("Failed to load connector for %s: %s", cs, e)

    logger.info("Adapter startup complete: %d connectors loaded", len(_CONNECTORS))
    yield


app = FastAPI(title="BIRD-Interact GSF Adapter", lifespan=lifespan)


# ── Request models ───────────────────────────────────────────────────────────

class InitSessionRequest(BaseModel):
    task_id: str
    mode: str = "c-interact"
    state: Dict[str, Any] = {}
    reset: bool = True


class RunSessionRequest(BaseModel):
    task_id: str
    mode: str = "c-interact"
    message: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "system_agent", "adk_available": True}


@app.post("/init_session")
async def init_session(req: InitSessionRequest):
    db_name = req.state.get("db_name", "")
    db_schema = req.state.get("db_schema", "")
    external_kg = req.state.get("external_kg", "[]")
    session_id = uuid.uuid4().hex

    connectors = _CONNECTORS.get(db_name, [])
    if not connectors:
        logger.warning("No connector found for db_name=%r", db_name)

    gsf_sess = gsf_create_session(
        session_id=session_id,
        task_id=req.task_id,
        db_name=db_name,
        db_schema=db_schema,
        external_kg=external_kg,
        question="",            # extracted from first run_session message
        data_retriever=_DATA_RETRIEVER,
        semantic_retriever=_SEMANTIC_RETRIEVER,
        connectors=connectors,
    )

    sess = AdapterSession(
        task_id=req.task_id,
        mode=req.mode,
        session_id=session_id,
        bird_state=req.state,
        gsf_session=gsf_sess,
    )
    put_session(sess)

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

    while True:
        # gsf_step and gsf_apply_user_answer make blocking HTTP calls to the LLM;
        # run them in a thread so they don't block FastAPI's async event loop.
        action = await asyncio.to_thread(gsf_step, sess.gsf_session, req.message)

        if isinstance(action, AskUserAction):
            try:
                answer = await bird_http.ask_user(req.task_id, action.question)
            except Exception as e:
                logger.error("ask_user failed for %s: %r", req.task_id, e)
                raise HTTPException(status_code=502, detail=f"ask_user error: {type(e).__name__}: {e}")

            sess.dialogue_history.append({"role": "agent", "content": action.question})
            sess.dialogue_history.append({"role": "user",  "content": answer})
            sess.tool_trajectory.append({
                "tool": "ask_user",
                "input": {"question": action.question},
                "output": {"answer": answer},
            })
            await asyncio.to_thread(gsf_apply_user_answer, sess.gsf_session, answer)
            continue

        if isinstance(action, SubmitSQLAction):
            try:
                result = await bird_http.submit_sql(req.task_id, action.sql)
            except Exception as e:
                logger.error("submit_sql failed for %s: %s", req.task_id, e)
                raise HTTPException(status_code=502, detail=f"submit_sql error: {e}")

            raw_msg = result.get("message", "")
            sess._last_submit_raw = raw_msg
            sess.total_reward += float(result.get("reward", 0.0))
            sess.tool_trajectory.append({
                "tool": "submit_sql",
                "input": {"sql": action.sql},
                "output": result,
            })

            phase_completed = result.get("phase_completed")
            if phase_completed == 1:
                sess.phase1_completed = True
            if phase_completed == 2:
                sess.phase2_completed = True
                sess.task_done = True

            gsf_apply_submit_result(sess.gsf_session, result)
            sess._submitted_this_phase = True
            break

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
    }
    return {
        "task_id": req.task_id,
        "mode": req.mode,
        "session_id": sess.session_id,
        "state": out_state,
        "adk_available": True,
    }
