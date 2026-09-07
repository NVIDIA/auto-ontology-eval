#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-task BIRD-Interact evaluator — mirrors the real c-Interact orchestrator flow.

Starts the GSF adapter on :6003 (avoiding any service already on :6000),
drives a single task through the full pipeline:
  init → phase1 [→ phase1 debug] → phase2 [→ phase2 debug]

This script replicates the same debug-retry logic used by the official orchestrator:
  - If phase N SQL is executable but wrong results → "Your SQL is not correct..."
  - If phase N SQL throws an execution error ([exec_err_flg] in submit raw) → "Your SQL is not executable: {error}..."

Prerequisites — the Bird services must already be running:
  :6001  user_simulator
  :6002  db_environment
Start them with scripts/start_bird_services.sh (not ADK's own
scripts/start_services.sh directly) — it applies our required ADK patches
first (see known_issues.py).

Usage:
    python scripts/eval_bird_interact.py
    python scripts/eval_bird_interact.py --data /path/to/bird_interact_data_with_gt.jsonl
    python scripts/eval_bird_interact.py --instance-id alien_1
    python scripts/eval_bird_interact.py --db alien
    python scripts/eval_bird_interact.py --random
    python scripts/eval_bird_interact.py --random --category query --difficulty challenging
    python scripts/eval_bird_interact.py --agent-port 6003
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx

ONTOLOGY_DIR   = Path(__file__).resolve().parents[1]
GSF_DIR        = ONTOLOGY_DIR.parent / "GSF"
_default_adk   = ONTOLOGY_DIR / "third_party" / "BIRD-Interact" / "BIRD-Interact-ADK"
_ADK_DIR       = Path(os.environ.get("BIRD_INTERACT_ADK_DIR", str(_default_adk)))


def _read_adk_env() -> dict[str, str]:
    env: dict[str, str] = {}
    dotenv = _ADK_DIR / ".env"
    if not dotenv.exists():
        return env
    for raw in dotenv.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


_DATASET     = _read_adk_env().get("DATASET", "lite")
DEFAULT_DATA = _ADK_DIR.parent / f"bird-interact-{_DATASET}" / "bird_interact_data_with_gt.jsonl"

USER_SIM_URL = "http://127.0.0.1:6001"
DB_ENV_URL   = "http://127.0.0.1:6002"

PATIENCE = 3


def load_env(dotenv_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not dotenv_path.exists():
        return env
    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def _print_trajectory(state: dict) -> None:
    dialogue = state.get("dialogue_history", [])
    trajectory = state.get("tool_trajectory", [])
    if dialogue:
        print("  Dialogue:")
        for turn in dialogue:
            role = turn.get("role", "?").upper()
            print(f"    [{role}] {turn.get('content', '')[:200]}")
    sql_calls = [t for t in trajectory if t.get("tool") == "submit_sql"]
    if sql_calls:
        print("  SQL submitted:")
        for call in sql_calls:
            sql = call.get("input", {}).get("sql", "(empty)")
            result = call.get("output", {})
            print(f"    {sql[:400]}")
            print(f"    → reward={result.get('reward')}  phase_completed={result.get('phase_completed')}  msg={result.get('message','')[:100]}")
    elif not dialogue:
        print("  (no dialogue or SQL in trajectory)")


def check_health(url: str, label: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=3.0)
            if r.status_code == 200:
                print(f"  [OK] {label}")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  [FAIL] {label} not healthy after {timeout}s", file=sys.stderr)
    return False


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    r = httpx.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_task(
    data_path: Path,
    index: int,
    db_filter: str | None,
    random_pick: bool = False,
    instance_id: str | None = None,
    category_filter: str | None = None,
    difficulty_filter: str | None = None,
    output_type_filter: str | None = None,
) -> tuple[int, dict]:
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    tasks = [json.loads(line) for line in data_path.open() if line.strip()]
    if instance_id:
        matches = [(i, t) for i, t in enumerate(tasks) if t.get("instance_id") == instance_id]
        if not matches:
            print(f"No task found with instance_id={instance_id!r}", file=sys.stderr)
            sys.exit(1)
        return matches[0]
    pool = [(i, t) for i, t in enumerate(tasks) if t.get("selected_database") == db_filter] if db_filter else list(enumerate(tasks))
    if category_filter:
        pool = [(i, t) for i, t in pool if (t.get("category") or "").lower() == category_filter.lower()]
    if difficulty_filter:
        pool = [(i, t) for i, t in pool if (t.get("difficulty_tier") or "").lower() == difficulty_filter.lower()]
    if output_type_filter:
        pool = [(i, t) for i, t in pool if (t.get("output_type") or "").lower() == output_type_filter.lower()]
    if not pool:
        parts = []
        if db_filter:
            parts.append(f"db={db_filter!r}")
        if category_filter:
            parts.append(f"category={category_filter!r}")
        if difficulty_filter:
            parts.append(f"difficulty_tier={difficulty_filter!r}")
        if output_type_filter:
            parts.append(f"output_type={output_type_filter!r}")
        print(f"No tasks found for {', '.join(parts)}", file=sys.stderr)
        sys.exit(1)
    if random_pick:
        idx, task = random.choice(pool)
        return idx, task
    if db_filter:
        return pool[0]
    if index >= len(tasks):
        print(f"Task index {index} out of range (file has {len(tasks)} tasks)", file=sys.stderr)
        sys.exit(1)
    return index, tasks[index]


def _build_debug_message(last_submit_raw: str) -> str:
    """Mirror the official orchestrator's debug message construction."""
    if "[exec_err_flg]" in last_submit_raw:
        actual_error = last_submit_raw.split("[exec_err_flg]", 1)[1].strip()
        return f"Your SQL is not executable: {actual_error}\nPlease fix and call submit_sql."
    return "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."


def start_gsf_adapter(port: int) -> subprocess.Popen:
    dotenv_vars = load_env(ONTOLOGY_DIR / ".env")
    existing_pp = os.environ.get("PYTHONPATH", "")
    pp_parts = [str(ONTOLOGY_DIR), str(GSF_DIR)]
    if existing_pp:
        pp_parts.append(existing_pp)
    env = {**os.environ, **dotenv_vars, "PYTHONPATH": ":".join(pp_parts)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "ontology_sql_eval.bird_interact.server:app",
         "--host", "127.0.0.1",
         "--port", str(port),
         "--log-level", "info"],
        cwd=str(ONTOLOGY_DIR),
        env=env,
    )
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-task BIRD-Interact evaluator (mirrors real orchestrator)")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--task-index", type=int, default=0, help="0-based task index")
    parser.add_argument("--instance-id", default=None, help="Pick task by instance_id (e.g. alien_1)")
    parser.add_argument("--db", default=None, help="Pick first task for this database name")
    parser.add_argument("--random", action="store_true", help="Pick a random task")
    parser.add_argument("--category", default=None, choices=["query", "management"], type=str.lower,
                        help="Restrict selection to 'query' or 'management' tasks")
    parser.add_argument("--difficulty", default=None, choices=["simple", "moderate", "challenging"], type=str.lower,
                        help="Restrict selection by difficulty_tier")
    parser.add_argument("--output-type", default=None, choices=["scalar", "table"], type=str.lower,
                        help="Restrict selection by output_type ('scalar' = single-value result, 'table' = multi-row result)")
    parser.add_argument("--agent-port", type=int, default=6003, help="Port to start the GSF adapter on")
    parser.add_argument("--timeout", type=int, default=10, help="Health-check timeout per service (s)")
    parser.add_argument("--quit-after-phase", type=int, default=None, choices=[1, 2],
                        help="Stop cleanly after this phase (1 = skip phase 2 even if p1 passes)")
    args = parser.parse_args()

    agent_url = f"http://127.0.0.1:{args.agent_port}"

    # ── 1. Check :6001 and :6002 ──────────────────────────────────────────────
    print("Checking Bird services...")
    ok = all([
        check_health(USER_SIM_URL, "user_sim :6001", args.timeout),
        check_health(DB_ENV_URL,   "db_env   :6002", args.timeout),
    ])
    if not ok:
        sys.exit(1)

    # ── 2. Start GSF adapter ──────────────────────────────────────────────────
    print(f"\nStarting GSF adapter on :{args.agent_port} ...")
    adapter_proc = start_gsf_adapter(args.agent_port)
    try:
        if not check_health(agent_url, f"GSF adapter :{args.agent_port}", timeout=60):
            adapter_proc.terminate()
            sys.exit(1)

        # ── 3. Load task ──────────────────────────────────────────────────────
        idx, task = load_task(
            Path(args.data), args.task_index, args.db,
            random_pick=args.random,
            instance_id=args.instance_id,
            category_filter=args.category,
            difficulty_filter=args.difficulty,
            output_type_filter=args.output_type,
        )
        task_id   = task["instance_id"]
        db_name   = task["selected_database"]
        amb_query = task.get("amb_user_query") or task.get("query", "")

        print(f"\nTask:       {task_id}  (index {idx})")
        print(f"Database:   {db_name}")
        print(f"Category:   {task.get('category', '?')}  |  Difficulty: {task.get('difficulty_tier', '?')}  |  Output: {task.get('output_type', '?')}")
        print(f"Query:      {amb_query}")

        # ── 4. init_task on :6001 and :6002 ──────────────────────────────────
        print("\n-- init_task on :6001 and :6002 --")
        payload = {"task_id": task_id, "task_data": {**task, "_interact_mode": "c-interact"}}
        post(f"{DB_ENV_URL}/init_task",   payload)
        post(f"{USER_SIM_URL}/init_task", payload)
        print("  done")

        # ── 5. Get schema and knowledge from :6002 ────────────────────────────
        print("-- fetching schema + knowledge from :6002 --")
        db_schema   = post(f"{DB_ENV_URL}/schema",    {"task_id": task_id}).get("schema", "")
        external_kb = post(f"{DB_ENV_URL}/knowledge", {"task_id": task_id}).get("knowledge", "[]")
        print(f"  schema: {db_schema[:60] if db_schema else '(empty)'}")
        print(f"  knowledge items: {len(json.loads(external_kb)) if external_kb and external_kb != '[]' else 0}")

        # ── 6. Compute clarification budget ───────────────────────────────────
        n_critical  = len(task.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
        n_knowledge = len(task.get("knowledge_ambiguity", []))
        max_turn    = n_critical + n_knowledge + PATIENCE

        # ── 7. init_session on GSF adapter ────────────────────────────────────
        print(f"-- init_session on :{args.agent_port} --")
        init_resp = post(f"{agent_url}/init_session", {
            "task_id": task_id,
            "mode": "c-interact",
            "state": {
                "task_id": task_id,
                "mode": "c-interact",
                "db_name": db_name,
                "db_schema": db_schema,
                "external_kb": external_kb,
                "max_turn": max_turn,
                "phase_max_turns": max_turn * 3,
                "model_turns": 0,
                "tool_trajectory": [],
                "dialogue_history": [],
            },
            "reset": True,
        })
        print(f"  session_id: {init_resp.get('session_id', '?')}")

        # ── 8. Phase 1 ────────────────────────────────────────────────────────
        phase1_msg = (
            f"User Query:\n{amb_query}\n\n"
            f"You have {max_turn} clarification turns. "
            f"Ask questions with ask_user to resolve ambiguities, "
            f"then call submit_sql with your final PostgreSQL query."
        )
        print(f"\n-- Phase 1 (max {max_turn} clarification turns) --")
        resp  = post(f"{agent_url}/run_session",
                     {"task_id": task_id, "mode": "c-interact", "message": phase1_msg},
                     timeout=1800.0)
        state = resp.get("state", {})
        p1_pass = state.get("phase1_completed", False)
        reward  = state.get("total_reward", 0.0)
        print(f"  phase1_completed: {p1_pass}  |  reward: {reward}")
        if state.get("_last_submit_raw"):
            print(f"  submit feedback: {state['_last_submit_raw'][:200]}")
        _print_trajectory(state)

        # ── 9. Phase 1 Debug ─────────────────────────────────────────────────
        p1_debug_ran = False
        if not p1_pass and state.get("_submitted_this_phase"):
            debug_msg = _build_debug_message(state.get("_last_submit_raw", ""))
            print("\n-- Phase 1 Debug --")
            print(f"  prompt: {debug_msg[:120]}")
            resp  = post(f"{agent_url}/run_session",
                         {"task_id": task_id, "mode": "c-interact", "message": debug_msg},
                         timeout=1800.0)
            state = resp.get("state", {})
            p1_pass = state.get("phase1_completed", False)
            reward  = state.get("total_reward", 0.0)
            p1_debug_ran = True
            print(f"  phase1_completed: {p1_pass}  |  reward: {reward}")
            if state.get("_last_submit_raw"):
                print(f"  submit feedback: {state['_last_submit_raw'][:200]}")
            _print_trajectory(state)

        # ── 10. Phase 2 ───────────────────────────────────────────────────────
        follow_up    = task.get("follow_up") or {}
        has_follow_up = bool(follow_up.get("sol_sql"))
        p2_pass = None
        p2_debug_ran = False

        if args.quit_after_phase == 1:
            print("\n(--quit-after-phase 1: skipping phase 2)")
        elif p1_pass and has_follow_up:
            follow_up_query = follow_up.get("query", "")
            print("\n-- phase_transition on :6001 --")
            post(f"{USER_SIM_URL}/phase_transition", {"task_id": task_id}, timeout=30.0)

            fu_msg = (
                f"Phase 1 is complete. Here is a follow-up question:\n\n{follow_up_query}\n\n"
                f"Generate the PostgreSQL query and call submit_sql."
            )
            print("-- Phase 2 --")
            resp  = post(f"{agent_url}/run_session",
                         {"task_id": task_id, "mode": "c-interact", "message": fu_msg},
                         timeout=1800.0)
            state = resp.get("state", {})
            p2_pass = state.get("phase2_completed", False)
            reward  = state.get("total_reward", 0.0)
            print(f"  phase2_completed: {p2_pass}  |  reward: {reward}")
            if state.get("_last_submit_raw"):
                print(f"  submit feedback: {state['_last_submit_raw'][:200]}")
            _print_trajectory(state)

            # ── 11. Phase 2 Debug ─────────────────────────────────────────────
            if not p2_pass and state.get("_submitted_this_phase"):
                debug_msg = _build_debug_message(state.get("_last_submit_raw", ""))
                print("\n-- Phase 2 Debug --")
                print(f"  prompt: {debug_msg[:120]}")
                resp  = post(f"{agent_url}/run_session",
                             {"task_id": task_id, "mode": "c-interact", "message": debug_msg},
                             timeout=1800.0)
                state = resp.get("state", {})
                p2_pass = state.get("phase2_completed", False)
                reward  = state.get("total_reward", 0.0)
                p2_debug_ran = True
                print(f"  phase2_completed: {p2_pass}  |  reward: {reward}")
                if state.get("_last_submit_raw"):
                    print(f"  submit feedback: {state['_last_submit_raw'][:200]}")
                _print_trajectory(state)

        elif p1_pass:
            print("\n(task has no follow_up — skipping phase 2)")
        else:
            print("\n(phase 1 failed — skipping phase 2)")

        # ── 12. Cleanup ───────────────────────────────────────────────────────
        try:
            post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
        except Exception:
            pass

    finally:
        print("\nStopping GSF adapter...")
        adapter_proc.terminate()
        try:
            adapter_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            adapter_proc.kill()

    # ── 13. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Eval complete")
    print(f"{'='*50}")
    print(f"Task:          {task_id}  (db: {db_name})")
    print(f"Phase 1:       {'PASS' if p1_pass else 'FAIL'}{' (via debug)' if p1_debug_ran and p1_pass else ''}")
    if p2_pass is not None:
        print(f"Phase 2:       {'PASS' if p2_pass else 'FAIL'}{' (via debug)' if p2_debug_ran and p2_pass else ''}")
    elif not has_follow_up:
        print("Phase 2:       N/A (no follow-up)")
    else:
        print("Phase 2:       skipped (phase 1 failed)")
    print(f"Total reward:  {reward:.2f}")

    passed = p1_pass and (p2_pass is None or p2_pass or not has_follow_up)
    if passed:
        print("\nEVAL PASSED")
        sys.exit(0)
    else:
        print("\nEVAL: one or more phases did not complete", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
