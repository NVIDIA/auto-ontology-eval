#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal live smoke test for the BIRD-Interact three-port setup.

Starts the Auto Ontology adapter on :6003 (avoiding any service already on :6000),
drives a single task through the full pipeline (init → phase1 → phase2), then
shuts the adapter down.

Prerequisites — the Bird services must already be running:
  :6001  user_simulator
  :6002  db_environment
Start them with scripts/bird_interact/start_bird_services.sh (not ADK's own
scripts/start_services.sh directly) — it applies our required ADK patches
first (see known_issues.py).

Usage:
    python scripts/bird_interact/smoke_bird_interact.py
    python scripts/bird_interact/smoke_bird_interact.py --data /path/to/bird_interact_data_with_gt.jsonl
    python scripts/bird_interact/smoke_bird_interact.py --task-index 3
    python scripts/bird_interact/smoke_bird_interact.py --db alien       # pick first task for that DB
    python scripts/bird_interact/smoke_bird_interact.py --random --category query
    python scripts/bird_interact/smoke_bird_interact.py --agent-port 6003

Management-category tasks are not supported by this toolchain (query-category
only — see run_all_bird_interact.py). load_task() always excludes them from
the selection pool and warns; explicitly requesting --category management, or
an --instance-id/--db that resolves to one, warns and exits instead of running.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


from ontology_sql_eval.bird_interact.harness import (
    check_health,
    load_task,
    post,
    print_trajectory,
    start_auto_ontology_adapter,
)

ONTOLOGY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    ONTOLOGY_DIR / "datasets" / "bird_interact" / "bird_interact_data_with_gt.jsonl"
)
USER_SIM_URL = "http://127.0.0.1:6001"
DB_ENV_URL = "http://127.0.0.1:6002"

PATIENCE = 3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live smoke test for BIRD-Interact + Auto Ontology adapter"
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="0-based index into the filtered selection pool",
    )
    parser.add_argument(
        "--instance-id", default=None, help="Pick task by instance_id (e.g. alien_1)"
    )
    parser.add_argument(
        "--db", default=None, help="Restrict selection to this database name"
    )
    parser.add_argument("--random", action="store_true", help="Pick a random task")
    parser.add_argument(
        "--category",
        default=None,
        choices=["query", "management"],
        type=str.lower,
        help="Restrict selection to 'query' or 'management' tasks (case-insensitive)",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=["simple", "moderate", "challenging"],
        type=str.lower,
        help="Restrict selection by difficulty_tier: 'simple', 'moderate', or 'challenging' (case-insensitive)",
    )
    parser.add_argument(
        "--agent-port",
        type=int,
        default=6003,
        help="Port to start the Auto Ontology adapter on",
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="Health-check timeout per service (s)"
    )
    args = parser.parse_args()

    agent_url = f"http://127.0.0.1:{args.agent_port}"

    # ── 1. Check :6001 and :6002 ──────────────────────────────────────────────
    print("Checking Bird services...")
    ok = all(
        [
            check_health(USER_SIM_URL, "user_sim :6001", args.timeout),
            check_health(DB_ENV_URL, "db_env   :6002", args.timeout),
        ]
    )
    if not ok:
        sys.exit(1)

    # ── 2. Start Auto Ontology adapter ────────────────────────────────────────
    print(f"\nStarting Auto Ontology adapter on :{args.agent_port} ...")
    adapter_proc = start_auto_ontology_adapter(args.agent_port)
    try:
        if not check_health(
            agent_url, f"Auto Ontology adapter :{args.agent_port}", timeout=60
        ):
            adapter_proc.terminate()
            sys.exit(1)

        # ── 3. Load task ──────────────────────────────────────────────────────
        idx, task = load_task(
            Path(args.data),
            args.task_index,
            args.db,
            random_pick=args.random,
            instance_id=args.instance_id,
            category_filter=args.category,
            difficulty_filter=args.difficulty,
        )
        task_id = task["instance_id"]
        db_name = task["selected_database"]
        amb_query = task.get("amb_user_query") or task.get("query", "")

        print(f"\nTask:     {task_id}  (index {idx})")
        print(f"Database: {db_name}")
        print(f"Query:    {amb_query}")

        # ── 4. init_task on :6001 and :6002 ──────────────────────────────────
        print("\n-- init_task on :6001 and :6002 --")
        payload = {
            "task_id": task_id,
            "task_data": {**task, "_interact_mode": "c-interact"},
        }
        post(f"{DB_ENV_URL}/init_task", payload)
        post(f"{USER_SIM_URL}/init_task", payload)
        print("  done")

        # ── 5. Get schema and knowledge from :6002 (best-effort) ─────────────
        print("-- fetching schema + knowledge from :6002 --")
        db_schema = post(f"{DB_ENV_URL}/schema", {"task_id": task_id}).get("schema", "")
        external_kb = post(f"{DB_ENV_URL}/knowledge", {"task_id": task_id}).get(
            "knowledge", "[]"
        )
        print(f"  schema: {db_schema[:60] if db_schema else '(empty)'}")
        print(
            f"  knowledge items: {len(json.loads(external_kb)) if external_kb and external_kb != '[]' else 0}"
        )

        # ── 6. Compute clarification budget ───────────────────────────────────
        n_critical = len(
            task.get("user_query_ambiguity", {}).get("critical_ambiguity", [])
        )
        n_knowledge = len(task.get("knowledge_ambiguity", []))
        max_turn = n_critical + n_knowledge + PATIENCE

        # ── 7. init_session on Auto Ontology adapter ──────────────────────────
        print(f"-- init_session on :{args.agent_port} --")
        init_resp = post(
            f"{agent_url}/init_session",
            {
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
            },
        )
        print(f"  session_id: {init_resp.get('session_id', '?')}")

        # ── 8. Phase 1 ────────────────────────────────────────────────────────
        phase1_msg = (
            f"User Query:\n{amb_query}\n\n"
            f"You have {max_turn} clarification turns. "
            f"Ask questions with ask_user to resolve ambiguities, "
            f"then call submit_sql with your final PostgreSQL query."
        )
        print(f"\n-- Phase 1 (max {max_turn} clarification turns) --")
        resp = post(
            f"{agent_url}/run_session",
            {"task_id": task_id, "mode": "c-interact", "message": phase1_msg},
            timeout=1800.0,
        )
        state = resp.get("state", {})
        p1_pass = state.get("phase1_completed", False)
        reward = state.get("total_reward", 0.0)
        print(f"  phase1_completed: {p1_pass}  |  reward: {reward}")
        if state.get("_last_submit_raw"):
            print(f"  submit feedback: {state['_last_submit_raw'][:200]}")
        print_trajectory(state)

        # ── 9. Phase 2 ────────────────────────────────────────────────────────
        follow_up = task.get("follow_up") or {}
        has_follow_up = bool(follow_up.get("sol_sql"))
        p2_pass = None

        if p1_pass and has_follow_up:
            follow_up_query = follow_up.get("query", "")
            print("\n-- phase_transition on :6001 --")
            post(f"{USER_SIM_URL}/phase_transition", {"task_id": task_id}, timeout=30.0)

            fu_msg = (
                f"Phase 1 is complete. Here is a follow-up question:\n\n{follow_up_query}\n\n"
                f"Generate the PostgreSQL query and call submit_sql."
            )
            print("-- Phase 2 --")
            resp = post(
                f"{agent_url}/run_session",
                {"task_id": task_id, "mode": "c-interact", "message": fu_msg},
                timeout=1800.0,
            )
            state = resp.get("state", {})
            p2_pass = state.get("phase2_completed", False)
            reward = state.get("total_reward", 0.0)
            print(f"  phase2_completed: {p2_pass}  |  reward: {reward}")
            if state.get("_last_submit_raw"):
                print(f"  submit feedback: {state['_last_submit_raw'][:200]}")
            print_trajectory(state)

        elif p1_pass:
            print("\n(task has no follow_up — skipping phase 2)")
        else:
            print("\n(phase 1 failed — skipping phase 2)")

        # ── 10. Cleanup ───────────────────────────────────────────────────────
        try:
            post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
        except Exception:
            pass

    finally:
        print("\nStopping Auto Ontology adapter...")
        adapter_proc.terminate()
        try:
            adapter_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            adapter_proc.kill()

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("Smoke test complete")
    print(f"{'=' * 50}")
    print(f"Task:          {task_id}  (db: {db_name})")
    print(f"Phase 1 pass:  {p1_pass}")
    if p2_pass is not None:
        print(f"Phase 2 pass:  {p2_pass}")
    else:
        print(
            f"Phase 2:       {'N/A (no follow-up)' if has_follow_up is False else 'skipped (p1 failed)'}"
        )
    print(f"Total reward:  {reward:.2f}")

    passed = p1_pass and (p2_pass is None or p2_pass or not has_follow_up)
    if passed:
        print("\nSMOKE TEST PASSED")
        sys.exit(0)
    else:
        print("\nSMOKE TEST: one or more phases did not complete", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
