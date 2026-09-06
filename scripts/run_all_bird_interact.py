#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overnight multi-task BIRD-Interact evaluator for query-category tasks.

Runs all (or a filtered subset of) query-category tasks sequentially through the
full c-Interact pipeline: init → phase1 [→ phase1 debug] → phase2 [→ phase2 debug].

Mirrors eval_bird_interact.py's per-task flow exactly, but:
  - Loops over multiple tasks automatically
  - Starts the GSF adapter ONCE and reuses it across all tasks
  - Writes rich per-task JSONL (run-time data only; join dataset fields by instance_id)
  - Writes a lightweight CSV incrementally (safe on crash)
  - Catches timeouts and errors per task and continues to the next
  - Exposes per-phase timing and entity resolution data (requires server.py additions)

Prerequisites — the Bird services must already be running:
  :6001  user_simulator
  :6002  db_environment

Usage:
    caffeinate -i uv run python scripts/run_all_bird_interact.py
    caffeinate -i uv run python scripts/run_all_bird_interact.py --output overnight_run_1
    caffeinate -i uv run python scripts/run_all_bird_interact.py --limit 10 --shuffle
    caffeinate -i uv run python scripts/run_all_bird_interact.py --difficulty challenging
    caffeinate -i uv run python scripts/run_all_bird_interact.py --phase-timeout 600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ONTOLOGY_DIR = Path(__file__).resolve().parents[1]
GSF_DIR      = ONTOLOGY_DIR.parent / "GSF"
DEFAULT_DATA = ONTOLOGY_DIR / "datasets" / "bird_interact" / "bird_interact_data_with_gt.jsonl"
RESULTS_DIR  = ONTOLOGY_DIR / "results"

USER_SIM_URL = "http://127.0.0.1:6001"
DB_ENV_URL   = "http://127.0.0.1:6002"

PATIENCE = 3

CSV_COLUMNS = [
    "instance_id", "dataset", "database", "max_turn", "turns_used",
    "phase1_passed", "phase1_debug_ran", "phase2_passed", "phase2_debug_ran",
    "total_reward", "exec_error_p1", "exec_error_p2", "error",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_dataset_label(data_path: Path) -> str:
    """Infer 'lite' vs 'full' from the dataset path (e.g. .../bird_interact_full/...)."""
    return "full" if "full" in str(data_path).lower() else "lite"

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


def start_gsf_adapter(port: int, gsf_dir: Path, log_path: Path | None = None) -> subprocess.Popen:
    dotenv_vars = load_env(ONTOLOGY_DIR / ".env")
    existing_pp = os.environ.get("PYTHONPATH", "")
    pp_parts = [str(ONTOLOGY_DIR), str(gsf_dir)]
    if existing_pp:
        pp_parts.append(existing_pp)
    env = {**os.environ, **dotenv_vars, "PYTHONPATH": ":".join(pp_parts)}
    log_file = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "ontology_sql_eval.bird_interact.server:app",
         "--host", "127.0.0.1",
         "--port", str(port),
         "--log-level", "info"],
        cwd=str(ONTOLOGY_DIR),
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    return proc


def _extract_sql(trajectory: list[dict]) -> str | None:
    """Return the SQL from the last submit_sql call in a tool trajectory."""
    calls = [t for t in trajectory if t.get("tool") == "submit_sql"]
    return calls[-1]["input"]["sql"] if calls else None


def _exec_error(raw: str | None) -> str | None:
    """Return the exec error text if [exec_err_flg] is present, else None."""
    if raw and "[exec_err_flg]" in raw:
        return raw.split("[exec_err_flg]", 1)[1].strip()
    return None


def _build_debug_message(last_submit_raw: str) -> str:
    """Mirror the official orchestrator's debug message construction."""
    if "[exec_err_flg]" in last_submit_raw:
        actual_error = last_submit_raw.split("[exec_err_flg]", 1)[1].strip()
        return f"Your SQL is not executable: {actual_error}\nPlease fix and call submit_sql."
    return "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."


def _load_tasks(
    data_path: Path,
    difficulty_filter: str | None,
    limit: int | None,
    shuffle: bool,
    include_ids: set[str] | None = None,
    exclude_ids: set[str] | None = None,
    force_include_ids: set[str] | None = None,
) -> list[dict]:
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    tasks = [json.loads(line) for line in data_path.open() if line.strip()]
    # Hardcoded: query category only (management contradicts product goals)
    tasks = [t for t in tasks if (t.get("category") or "").lower() == "query"]
    if difficulty_filter:
        tasks = [t for t in tasks if (t.get("difficulty_tier") or "").lower() == difficulty_filter.lower()]
    if include_ids:
        tasks = [t for t in tasks if t["instance_id"] in include_ids]
    if exclude_ids:
        tasks = [t for t in tasks if t["instance_id"] not in exclude_ids]

    if force_include_ids:
        # Guarantee these instance_ids are in the final set, then fill the
        # rest of --limit (shuffled if requested) from the remaining pool.
        forced = [t for t in tasks if t["instance_id"] in force_include_ids]
        found_ids = {t["instance_id"] for t in forced}
        missing = force_include_ids - found_ids
        if missing:
            print(f"WARNING: --force-include ids not found in filtered pool: {sorted(missing)}",
                  file=sys.stderr)
        rest = [t for t in tasks if t["instance_id"] not in found_ids]
        if shuffle:
            random.shuffle(rest)
        if limit:
            fill_n = max(limit - len(forced), 0)
            rest = rest[:fill_n]
        tasks = forced + rest
        if shuffle:
            random.shuffle(tasks)
        return tasks

    if shuffle:
        random.shuffle(tasks)
    if limit:
        tasks = tasks[:limit]
    return tasks


def _append_jsonl(record: dict, path: Path) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _append_csv(record: dict, path: Path, write_header: bool) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(
    task: dict,
    agent_url: str,
    phase_timeout: float,
    dataset_label: str,
    _current_phase: list | None = None,
) -> dict[str, Any]:
    """_current_phase is a single-element list the caller can read to find which
    phase was active when a TimeoutException fires."""
    """Run one task through the full c-Interact pipeline and return a result record."""
    task_id   = task["instance_id"]
    db_name   = task["selected_database"]
    amb_query = task.get("amb_user_query") or task.get("query", "")
    follow_up = task.get("follow_up") or {}
    has_follow_up = bool(follow_up.get("sol_sql"))

    n_critical  = len(task.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    n_knowledge = len(task.get("knowledge_ambiguity", []))
    max_turn    = n_critical + n_knowledge + PATIENCE

    # Base record — filled in as we go; nulls indicate not-reached
    record: dict[str, Any] = {
        "instance_id":             task_id,
        "dataset":                 dataset_label,
        "database":                db_name,
        "task_start_timestamp":    time.time(),
        "max_turn":                max_turn,
        "turns_used":              None,
        "phase1_passed":           False,
        "phase1_debug_ran":        False,
        "phase2_passed":           None,
        "phase2_debug_ran":        False,
        "total_reward":            0.0,
        "p1_sql":                  None,
        "p1_debug_sql":            None,
        "p2_sql":                  None,
        "p2_debug_sql":            None,
        "exec_error_p1":           None,
        "exec_error_p2":           None,
        "dialogue_history":        [],
        "initial_extracted_entities": [],
        "extracted_entities":      [],
        "unresolved_entities_start": [],
        "unresolved_entities_end": [],
        "resolved_entities":       [],
        "timing": {
            "phase1_clarification_secs": None,
            "phase1_sql_gen_secs":       None,
            "phase1_debug_total_secs":   None,
            "phase2_sql_gen_secs":       None,
            "phase2_debug_total_secs":   None,
        },
        "error": None,
    }

    # ── init_task on :6001 and :6002 ─────────────────────────────────────────
    payload = {"task_id": task_id, "task_data": {**task, "_interact_mode": "c-interact"}}
    post(f"{DB_ENV_URL}/init_task",   payload)
    post(f"{USER_SIM_URL}/init_task", payload)

    # ── fetch schema + knowledge from :6002 ──────────────────────────────────
    db_schema   = post(f"{DB_ENV_URL}/schema",    {"task_id": task_id}).get("schema", "")
    external_kg = post(f"{DB_ENV_URL}/knowledge", {"task_id": task_id}).get("knowledge", "[]")

    # ── init_session on GSF adapter ──────────────────────────────────────────
    post(f"{agent_url}/init_session", {
        "task_id": task_id,
        "mode": "c-interact",
        "state": {
            "task_id":         task_id,
            "mode":            "c-interact",
            "db_name":         db_name,
            "db_schema":       db_schema,
            "external_kg":     external_kg,
            "max_turn":        max_turn,
            "phase_max_turns": max_turn * 3,
            "model_turns":     0,
            "tool_trajectory": [],
            "dialogue_history": [],
        },
        "reset": True,
    })

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if _current_phase is not None:
        _current_phase[0] = "phase1"
    phase1_msg = (
        f"User Query:\n{amb_query}\n\n"
        f"You have {max_turn} clarification turns. "
        f"Ask questions with ask_user to resolve ambiguities, "
        f"then call submit_sql with your final PostgreSQL query."
    )
    t0 = time.time()
    resp  = post(f"{agent_url}/run_session",
                 {"task_id": task_id, "mode": "c-interact", "message": phase1_msg},
                 timeout=phase_timeout)
    state = resp.get("state", {})

    p1_pass = state.get("phase1_completed", False)
    record["phase1_passed"]           = p1_pass
    record["total_reward"]            = state.get("total_reward", 0.0)
    record["dialogue_history"]        = state.get("dialogue_history", [])
    record["initial_extracted_entities"] = state.get("initial_extracted_entities", [])
    record["extracted_entities"]        = state.get("extracted_entities", [])
    record["unresolved_entities_start"] = state.get("unresolved_entities_start", [])
    record["unresolved_entities_end"]   = state.get("unresolved_entities_end", [])
    record["resolved_entities"]         = state.get("resolved_entities", [])
    record["p1_sql"]                  = _extract_sql(state.get("tool_trajectory", []))
    record["exec_error_p1"]           = _exec_error(state.get("_last_submit_raw"))
    record["turns_used"]              = len(state.get("dialogue_history", [])) // 2
    record["timing"]["phase1_clarification_secs"] = state.get("timing_clarification_secs")
    record["timing"]["phase1_sql_gen_secs"]        = state.get("timing_sql_gen_secs")

    # ── Phase 1 Debug ─────────────────────────────────────────────────────────
    if _current_phase is not None:
        _current_phase[0] = "phase1_debug"
    if not p1_pass and state.get("_submitted_this_phase"):
        debug_msg = _build_debug_message(state.get("_last_submit_raw", ""))
        t0 = time.time()
        resp  = post(f"{agent_url}/run_session",
                     {"task_id": task_id, "mode": "c-interact", "message": debug_msg},
                     timeout=phase_timeout)
        state = resp.get("state", {})
        p1_pass = state.get("phase1_completed", False)
        record["phase1_passed"]        = p1_pass
        record["phase1_debug_ran"]     = True
        record["total_reward"]         = state.get("total_reward", 0.0)
        record["p1_debug_sql"]         = _extract_sql(state.get("tool_trajectory", []))
        record["exec_error_p1"]        = record["exec_error_p1"] or _exec_error(state.get("_last_submit_raw"))
        record["timing"]["phase1_debug_total_secs"] = time.time() - t0

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if _current_phase is not None:
        _current_phase[0] = "phase2"
    if p1_pass and has_follow_up:
        follow_up_query = follow_up.get("query", "")
        post(f"{USER_SIM_URL}/phase_transition", {"task_id": task_id}, timeout=30.0)

        fu_msg = (
            f"Phase 1 is complete. Here is a follow-up question:\n\n{follow_up_query}\n\n"
            f"Generate the PostgreSQL query and call submit_sql."
        )
        t0 = time.time()
        resp  = post(f"{agent_url}/run_session",
                     {"task_id": task_id, "mode": "c-interact", "message": fu_msg},
                     timeout=phase_timeout)
        state = resp.get("state", {})
        p2_pass = state.get("phase2_completed", False)
        record["phase2_passed"]   = p2_pass
        record["total_reward"]    = state.get("total_reward", 0.0)
        record["p2_sql"]          = _extract_sql(state.get("tool_trajectory", []))
        record["exec_error_p2"]   = _exec_error(state.get("_last_submit_raw"))
        record["timing"]["phase2_sql_gen_secs"] = state.get("timing_sql_gen_secs")

        # ── Phase 2 Debug ─────────────────────────────────────────────────────
        if _current_phase is not None:
            _current_phase[0] = "phase2_debug"
        if not p2_pass and state.get("_submitted_this_phase"):
            debug_msg = _build_debug_message(state.get("_last_submit_raw", ""))
            t0 = time.time()
            resp  = post(f"{agent_url}/run_session",
                         {"task_id": task_id, "mode": "c-interact", "message": debug_msg},
                         timeout=phase_timeout)
            state = resp.get("state", {})
            p2_pass = state.get("phase2_completed", False)
            record["phase2_passed"]      = p2_pass
            record["phase2_debug_ran"]   = True
            record["total_reward"]       = state.get("total_reward", 0.0)
            record["p2_debug_sql"]       = _extract_sql(state.get("tool_trajectory", []))
            record["exec_error_p2"]      = record["exec_error_p2"] or _exec_error(state.get("_last_submit_raw"))
            record["timing"]["phase2_debug_total_secs"] = time.time() - t0

    elif p1_pass and not has_follow_up:
        record["phase2_passed"] = None  # N/A

    # ── Cleanup ──────────────────────────────────────────────────────────────
    try:
        post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
    except Exception:
        pass

    return record


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overnight multi-task BIRD-Interact evaluator (query category only)"
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Path to bird_interact_data_with_gt.jsonl")
    parser.add_argument("--output", default="cinteract_query_all",
                        help="Output file stem (results/<stem>.jsonl + .csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of tasks to run")
    parser.add_argument("--difficulty", default=None,
                        choices=["simple", "moderate", "challenging"], type=str.lower,
                        help="Restrict to this difficulty tier")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomize task order")
    parser.add_argument("--agent-port", type=int, default=6003,
                        help="Port to start the GSF adapter on (default: 6003)")
    parser.add_argument("--gsf-dir", type=Path, default=GSF_DIR,
                        help="Path to the GSF checkout to put on PYTHONPATH "
                             f"(default: {GSF_DIR})")
    parser.add_argument("--phase-timeout", type=float, default=900.0,
                        help="Seconds per run_session HTTP call (default: 900 = 15 min)")
    parser.add_argument("--health-timeout", type=int, default=10,
                        help="Health-check timeout per Bird service (default: 10s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files instead of failing")
    parser.add_argument("--include", nargs="+", metavar="TASK_ID", default=None,
                        help="Run only these instance_ids (space-separated)")
    parser.add_argument("--exclude", nargs="+", metavar="TASK_ID", default=None,
                        help="Skip these instance_ids (space-separated)")
    parser.add_argument("--force-include", nargs="+", metavar="TASK_ID", default=None,
                        help="Always include these instance_ids, then fill the rest of "
                             "--limit randomly from the remaining pool (space-separated)")
    args = parser.parse_args()

    agent_url = f"http://127.0.0.1:{args.agent_port}"
    run_dir = RESULTS_DIR / args.output
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / f"{args.output}.jsonl"
    csv_path   = run_dir / f"{args.output}.csv"

    if jsonl_path.exists() and not args.overwrite:
        print(f"ERROR: {jsonl_path} already exists. Use --overwrite to replace it.", file=sys.stderr)
        sys.exit(1)
    if args.overwrite:
        jsonl_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    csv_is_new = True

    # ── Load tasks ────────────────────────────────────────────────────────────
    include_ids = set(args.include) if args.include else None
    exclude_ids = set(args.exclude) if args.exclude else None
    force_include_ids = set(args.force_include) if args.force_include else None
    dataset_label = _infer_dataset_label(Path(args.data))
    tasks = _load_tasks(Path(args.data), args.difficulty, args.limit, args.shuffle,
                        include_ids=include_ids, exclude_ids=exclude_ids,
                        force_include_ids=force_include_ids)
    if not tasks:
        print("No tasks matched the given filters.", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoaded {len(tasks)} query-category tasks  (dataset: {dataset_label})")
    if args.difficulty:
        print(f"  difficulty filter: {args.difficulty}")
    if include_ids:
        print(f"  include filter:    {sorted(include_ids)}")
    if exclude_ids:
        print(f"  exclude filter:    {sorted(exclude_ids)}")
    if force_include_ids:
        print(f"  force-include:     {sorted(force_include_ids)}")
    print(f"  output dir: {run_dir}")
    print()

    # ── Check Bird services ───────────────────────────────────────────────────
    print("Checking Bird services...")
    ok = all([
        check_health(USER_SIM_URL, "user_sim :6001", args.health_timeout),
        check_health(DB_ENV_URL,   "db_env   :6002", args.health_timeout),
    ])
    if not ok:
        sys.exit(1)

    # ── Start GSF adapter (once for all tasks) ────────────────────────────────
    # Always log the first 5 tasks for diagnostics; silent after that.
    adapter_log = run_dir / f"{args.output}_adapter.log"
    print(f"\nStarting GSF adapter on :{args.agent_port} ...")
    print(f"  (adapter logs → {adapter_log})")
    adapter_proc = start_gsf_adapter(args.agent_port, args.gsf_dir, log_path=adapter_log)

    t_run_start = time.time()
    n_pass = n_fail = n_error = 0
    sum_reward = 0.0

    try:
        if not check_health(agent_url, f"GSF adapter :{args.agent_port}", timeout=60):
            sys.exit(1)
        print(f"\nRunning {len(tasks)} tasks  (phase-timeout={args.phase_timeout}s)\n")

        for i, task in enumerate(tasks, 1):
            task_id = task["instance_id"]
            db_name = task["selected_database"]
            n_critical  = len(task.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
            n_knowledge = len(task.get("knowledge_ambiguity", []))
            max_turn    = n_critical + n_knowledge + PATIENCE


            print(f"[{i}/{len(tasks)}] {task_id}  db={db_name}  max_turn={max_turn}")

            record: dict[str, Any] | None = None
            _current_phase: list = ["phase1"]
            try:
                record = run_task(task, agent_url, args.phase_timeout, dataset_label, _current_phase)
            except httpx.TimeoutException:
                phase_hint = _current_phase[0]
                err_msg = f"timeout:{phase_hint} — timed out after {args.phase_timeout:.0f}s"
                print(f"  ERROR: {err_msg}")
                record = {
                    "instance_id":  task_id,
                    "dataset":      dataset_label,
                    "database":     db_name,
                    "task_start_timestamp": time.time(),
                    "max_turn":     max_turn,
                    "turns_used":   None,
                    "phase1_passed": False,
                    "phase1_debug_ran": False,
                    "phase2_passed": None,
                    "phase2_debug_ran": False,
                    "total_reward": 0.0,
                    "p1_sql": None, "p1_debug_sql": None,
                    "p2_sql": None, "p2_debug_sql": None,
                    "exec_error_p1": None, "exec_error_p2": None,
                    "dialogue_history": [],
                    "initial_extracted_entities": [],
                    "extracted_entities": [],
                    "unresolved_entities_start": [],
                    "unresolved_entities_end": [],
                    "resolved_entities": [],
                    "timing": {
                        "phase1_clarification_secs": None,
                        "phase1_sql_gen_secs": None,
                        "phase1_debug_total_secs": None,
                        "phase2_sql_gen_secs": None,
                        "phase2_debug_total_secs": None,
                    },
                    "error": err_msg,
                }
                n_error += 1
                # Best-effort cleanup
                try:
                    post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=10.0)
                except Exception:
                    pass
            except Exception as exc:
                err_msg = repr(exc)
                print(f"  ERROR: {err_msg}")
                record = {
                    "instance_id":  task_id,
                    "dataset":      dataset_label,
                    "database":     db_name,
                    "task_start_timestamp": time.time(),
                    "max_turn":     max_turn,
                    "turns_used":   None,
                    "phase1_passed": False,
                    "phase1_debug_ran": False,
                    "phase2_passed": None,
                    "phase2_debug_ran": False,
                    "total_reward": 0.0,
                    "p1_sql": None, "p1_debug_sql": None,
                    "p2_sql": None, "p2_debug_sql": None,
                    "exec_error_p1": None, "exec_error_p2": None,
                    "dialogue_history": [],
                    "initial_extracted_entities": [],
                    "extracted_entities": [],
                    "unresolved_entities_start": [],
                    "unresolved_entities_end": [],
                    "resolved_entities": [],
                    "timing": {
                        "phase1_clarification_secs": None,
                        "phase1_sql_gen_secs": None,
                        "phase1_debug_total_secs": None,
                        "phase2_sql_gen_secs": None,
                        "phase2_debug_total_secs": None,
                    },
                    "error": err_msg,
                }
                n_error += 1
                try:
                    post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=10.0)
                except Exception:
                    pass

            # Write results immediately (safe on crash)
            _append_jsonl(record, jsonl_path)
            _append_csv(record, csv_path, write_header=csv_is_new)
            csv_is_new = False

            # Terminal summary
            if record["error"]:
                print(f"  ERROR: {record['error']}  (skipped)")
            else:
                p1 = "PASS" if record["phase1_passed"] else "FAIL"
                debug_tag = " (via debug)" if record["phase1_debug_ran"] and record["phase1_passed"] else ""
                clarify_t = record["timing"]["phase1_clarification_secs"]
                sql_t     = record["timing"]["phase1_sql_gen_secs"]
                timing_str = ""
                if clarify_t is not None and sql_t is not None:
                    timing_str = f"  clarify={clarify_t:.1f}s  sql={sql_t:.1f}s"
                elif sql_t is not None:
                    timing_str = f"  sql={sql_t:.1f}s"
                turns = record["turns_used"]
                print(f"  Phase 1: {p1}{debug_tag}  turns={turns}/{max_turn}{timing_str}")

                if record["phase2_passed"] is not None:
                    p2 = "PASS" if record["phase2_passed"] else "FAIL"
                    p2_debug = " (via debug)" if record["phase2_debug_ran"] and record["phase2_passed"] else ""
                    p2_sql_t = record["timing"]["phase2_sql_gen_secs"]
                    p2_t_str = f"  sql={p2_sql_t:.1f}s" if p2_sql_t is not None else ""
                    print(f"  Phase 2: {p2}{p2_debug}{p2_t_str}")
                elif record["phase1_passed"]:
                    print("  Phase 2: N/A (no follow-up)")
                else:
                    print("  Phase 2: skipped (phase 1 failed)")

                print(f"  reward={record['total_reward']:.2f}")
                sum_reward += record["total_reward"]

                if record["phase1_passed"] and (record["phase2_passed"] is None or record["phase2_passed"]):
                    n_pass += 1
                else:
                    n_fail += 1

            print()

    finally:
        print("\nStopping GSF adapter...")
        adapter_proc.terminate()
        try:
            adapter_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            adapter_proc.kill()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_t = time.time() - t_run_start
    total_tasks = n_pass + n_fail + n_error
    print("=" * 60)
    print("Run complete")
    print("=" * 60)
    print(f"Tasks:   {total_tasks}  |  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")
    if total_tasks > 0:
        print(f"Score: {sum_reward / total_tasks * 100:.1f}%  (sum of rewards / tasks run)")
    print(f"Wall time: {total_t / 3600:.2f}h  ({total_t:.0f}s)")
    print(f"Output:    {jsonl_path}")
    print(f"           {csv_path}")


if __name__ == "__main__":
    main()
