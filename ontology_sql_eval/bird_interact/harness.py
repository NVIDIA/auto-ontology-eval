# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side helpers shared by the BIRD-Interact runner scripts.

``scripts/bird_interact/{smoke,eval,run_all}_bird_interact.py`` each drive the
ADK services and the Auto Ontology adapter the same way; the pieces they had
each copied live here: reading ``.env``, waiting for a service's ``/health``,
posting JSON, starting the adapter, picking a task, and printing a trajectory.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx

from ontology_sql_eval.paths import REPO_ROOT


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


def start_auto_ontology_adapter(
    port: int, log_path: Path | None = None
) -> subprocess.Popen:
    """Serve ``ontology_sql_eval.bird_interact.server`` on *port* with the repo's ``.env``.

    Output goes to *log_path* when given, otherwise to this process's terminal.
    """
    dotenv_vars = load_env(REPO_ROOT / ".env")
    env = {**os.environ, **dotenv_vars}
    log_file = open(log_path, "w") if log_path else None
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ontology_sql_eval.bird_interact.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    return proc


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
        matches = [
            (i, t) for i, t in enumerate(tasks) if t.get("instance_id") == instance_id
        ]
        if not matches:
            print(f"No task found with instance_id={instance_id!r}", file=sys.stderr)
            sys.exit(1)
        idx, task = matches[0]
        if (task.get("category") or "").lower() == "management":
            print(
                f"WARNING: instance_id={instance_id!r} is a management-category task; "
                "management is not supported by this toolchain (query-category only). "
                "Skipping.",
                file=sys.stderr,
            )
            sys.exit(1)
        return idx, task
    if category_filter == "management":
        print(
            "WARNING: --category management is not supported by this toolchain "
            "(query-category only). Skipping.",
            file=sys.stderr,
        )
        sys.exit(1)
    pool = (
        [(i, t) for i, t in enumerate(tasks) if t.get("selected_database") == db_filter]
        if db_filter
        else list(enumerate(tasks))
    )
    n_before = len(pool)
    pool = [
        (i, t) for i, t in pool if (t.get("category") or "").lower() != "management"
    ]
    if n_before != len(pool):
        print(
            f"WARNING: excluded {n_before - len(pool)} management-category task(s) "
            "from the selection pool (not supported by this toolchain).",
            file=sys.stderr,
        )
    if category_filter:
        pool = [
            (i, t)
            for i, t in pool
            if (t.get("category") or "").lower() == category_filter.lower()
        ]
    if difficulty_filter:
        pool = [
            (i, t)
            for i, t in pool
            if (t.get("difficulty_tier") or "").lower() == difficulty_filter.lower()
        ]
    if output_type_filter:
        pool = [
            (i, t)
            for i, t in pool
            if (t.get("output_type") or "").lower() == output_type_filter.lower()
        ]
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
    if index < 0 or index >= len(pool):
        print(
            f"Task index {index} out of range (selection pool has {len(pool)} task(s))",
            file=sys.stderr,
        )
        sys.exit(1)
    return pool[index]


def print_trajectory(state: dict) -> None:
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
            print(
                f"    → reward={result.get('reward')}  phase_completed={result.get('phase_completed')}  msg={result.get('message', '')[:100]}"
            )
    elif not dialogue:
        print("  (no dialogue or SQL in trajectory)")


def build_debug_message(last_submit_raw: str) -> str:
    """Mirror the official orchestrator's debug message construction."""
    if "[exec_err_flg]" in last_submit_raw:
        actual_error = last_submit_raw.split("[exec_err_flg]", 1)[1].strip()
        return f"Your SQL is not executable: {actual_error}\nPlease fix and call submit_sql."
    return "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."
