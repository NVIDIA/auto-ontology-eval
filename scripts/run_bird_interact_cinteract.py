#!/usr/bin/env python3
"""
Run BIRD-Interact c-Interact evaluation using the Auto Ontology adapter.

Starts the ontology-sql-eval adapter on :6000, then invokes the official
Bird c-Interact orchestrator against the merged GT jsonl, and writes a
results CSV.

Run scripts/seed_bird_interact.py first to clone the upstream repo and
prepare datasets/bird_interact/.

WARNING: the manual uvicorn bootstrap below does not go through
scripts/start_bird_services.sh, so it does not apply our required ADK
patches (see known_issues.py) — notably the labor_certification_applications
p1snap name-collision fix. Phase 2 for that DB will silently mis-grade if
run this way. Prefer start_bird_services.sh if possible; if you must use
this path, apply patches/adk_p1snap_name_collision.patch to the ADK checkout
manually first.

Prerequisites — start separately BEFORE running this script:
  PG_WRAPPERS=/tmp/pg_wrappers   # psql/createdb/dropdb shims

  Bird :6001 (user_simulator):
    cd third_party/BIRD-Interact/BIRD-Interact-ADK
    PATH="$PG_WRAPPERS:$PATH" python -m uvicorn user_simulator.server:app --host 127.0.0.1 --port 6001

  Bird :6002 (db_environment — requires pg_wrappers in PATH for createdb/dropdb):
    cd third_party/BIRD-Interact/BIRD-Interact-ADK
    PATH="$PG_WRAPPERS:$PATH" python -m uvicorn db_environment.server:app --host 127.0.0.1 --port 6002

  pg_wrappers setup (one-time, run from ontology-sql-eval/):
    python scripts/setup_pg_wrappers.py

  Bird PostgreSQL (task DBs, only needed for :6002 scoring):
    cd third_party/BIRD-Interact/env
    docker compose up -d bird_interact_postgresql

Usage:
    python scripts/run_bird_interact_cinteract.py \\
        --data datasets/bird_interact/bird_interact_data_with_gt.jsonl \\
        --output results/cinteract_lite.csv \\
        [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ONTOLOGY_DIR = Path(__file__).resolve().parents[1]
_default_adk = ONTOLOGY_DIR / "third_party" / "BIRD-Interact" / "BIRD-Interact-ADK"
BIRD_ADK_DIR = Path(os.environ.get("BIRD_INTERACT_ADK_DIR", str(_default_adk)))
DEFAULT_DATA = ONTOLOGY_DIR / "datasets" / "bird_interact" / "bird_interact_data_with_gt.jsonl"


def _warn_if_adk_unpatched() -> None:
    """This script's manual uvicorn bootstrap doesn't go through
    scripts/start_bird_services.sh, so it never applies our required ADK
    patches (see known_issues.py). Surface it loudly, first, rather than let a
    labor_certification_applications phase 2 silently mis-grade.
    """
    db_utils = BIRD_ADK_DIR / "shared" / "db_utils.py"
    marker = "NAMEDATALEN"
    if not db_utils.is_file() or marker not in db_utils.read_text():
        print(
            "WARNING: ADK checkout at "
            f"{BIRD_ADK_DIR} does not have patches/adk_p1snap_name_collision.patch "
            "applied (see known_issues.py). labor_certification_applications "
            "phase 2 will silently mis-grade. Apply the patch or use "
            "scripts/start_bird_services.sh instead of this script's manual "
            "bootstrap.",
            file=sys.stderr,
        )


def wait_for_health(url: str, timeout: int = 30, label: str = "") -> bool:
    for _ in range(timeout):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"ERROR: {label or url} did not become healthy within {timeout}s", file=sys.stderr)
    return False


def main() -> None:
    _warn_if_adk_unpatched()
    parser = argparse.ArgumentParser(
        description="Run Bird c-Interact with the Auto Ontology adapter"
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to merged bird_interact_data_with_gt.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/bird_interact_cinteract_lite.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to run",
    )
    parser.add_argument("--db", default=None, help="Run only tasks for this database name")
    parser.add_argument("--instance-id", default=None, help="Run only this instance_id (e.g. alien_1)")
    parser.add_argument(
        "--category", default=None, choices=["query", "management"], type=str.lower,
        help="Filter by task category",
    )
    parser.add_argument(
        "--difficulty", default=None, choices=["simple", "moderate", "challenging"], type=str.lower,
        help="Filter by difficulty_tier",
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Pick one random task matching the filters",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Orchestrator writes relative to BIRD_ADK_DIR, so use absolute path
    raw_json = (BIRD_ADK_DIR / "results" / output_path.stem).with_suffix(".raw.json")
    raw_json.parent.mkdir(parents=True, exist_ok=True)

    # ── Apply task filters ────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    if args.category == "management":
        print("WARNING: --category management is not supported by this toolchain "
              "(query-category only). Skipping.", file=sys.stderr)
        sys.exit(1)

    tasks = [json.loads(line) for line in data_path.open() if line.strip()]
    if args.instance_id:
        matches = [t for t in tasks if t.get("instance_id") == args.instance_id]
        if matches and (matches[0].get("category") or "").lower() == "management":
            print(f"WARNING: instance_id={args.instance_id!r} is a management-category "
                  "task; management is not supported by this toolchain (query-category "
                  "only). Skipping.", file=sys.stderr)
            sys.exit(1)

    n_tasks_before = len(tasks)
    tasks = [t for t in tasks if (t.get("category") or "").lower() != "management"]
    if n_tasks_before != len(tasks):
        print(f"WARNING: excluded {n_tasks_before - len(tasks)} management-category "
              "task(s) (not supported by this toolchain).", file=sys.stderr)

    any_filter = args.db or args.instance_id or args.category or args.difficulty or args.random
    filtered_tmp: str | None = None
    if any_filter:
        pool = tasks
        if args.instance_id:
            pool = [t for t in pool if t.get("instance_id") == args.instance_id]
        if args.db:
            pool = [t for t in pool if t.get("selected_database") == args.db]
        if args.category:
            pool = [t for t in pool if (t.get("category") or "").lower() == args.category]
        if args.difficulty:
            pool = [t for t in pool if (t.get("difficulty_tier") or "").lower() == args.difficulty]
        if args.random:
            pool = [random.choice(pool)] if pool else []
        if not pool:
            print("ERROR: no tasks match the specified filters", file=sys.stderr)
            sys.exit(1)
        if args.limit:
            pool = pool[:args.limit]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="bird_filtered_"
        )
        for t in pool:
            tmp.write(json.dumps(t) + "\n")
        tmp.close()
        filtered_tmp = tmp.name
        data_file = filtered_tmp
        print(f"Filtered to {len(pool)} task(s) → {data_file}")
    else:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="bird_filtered_"
        )
        for t in tasks:
            tmp.write(json.dumps(t) + "\n")
        tmp.close()
        filtered_tmp = tmp.name
        data_file = filtered_tmp

    # Check Bird :6001 and :6002 are already up
    print("Checking Bird services...")
    for port, label in [(6001, "user_simulator :6001"), (6002, "db_environment :6002")]:
        if not wait_for_health(f"http://127.0.0.1:{port}/health", timeout=5, label=label):
            print(
                f"\nBird {label} is not running. Start it first:\n"
                f"  cd {BIRD_ADK_DIR}\n"
                f"  python -m uvicorn {label.split()[0]}.server:app "
                f"--host 127.0.0.1 --port {port}",
                file=sys.stderr,
            )
            sys.exit(1)
    print("Bird services OK (:6001, :6002)")

    # Start the Auto Ontology adapter on :6000.
    print("Starting Auto Ontology adapter on :6000...")
    AUTO_ONTOLOGY_DIR = ONTOLOGY_DIR.parent / "auto-ontology"

    # Load .env so Auto Ontology variables (EMBED_API_KEY, NVIDIA_API_KEY, etc.)
    # are available.
    dotenv_path = ONTOLOGY_DIR / ".env"
    dotenv_vars: dict[str, str] = {}
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    dotenv_vars[k.strip()] = v.strip()

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(ONTOLOGY_DIR), str(AUTO_ONTOLOGY_DIR)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env = {**os.environ, **dotenv_vars, "PYTHONPATH": ":".join(pythonpath_parts)}
    env.pop("VIRTUAL_ENV", None)
    adapter_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "ontology_sql_eval.bird_interact.server:app",
            "--host", "127.0.0.1",
            "--port", "6000",
            "--log-level", "info",
        ],
        cwd=str(ONTOLOGY_DIR),
        env=env,
    )

    if adapter_proc.poll() is not None:
        print(
            "ERROR: Auto Ontology adapter failed to start "
            "(port 6000 may already be in use — kill any stale process first)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not wait_for_health(
        "http://127.0.0.1:6000/health",
        timeout=60,
        label="Auto Ontology adapter :6000",
    ):
        adapter_proc.terminate()
        sys.exit(1)
    if adapter_proc.poll() is not None:
        print("ERROR: Auto Ontology adapter died during startup", file=sys.stderr)
        sys.exit(1)
    print("Auto Ontology adapter :6000 ready")

    try:
        # Invoke the official Bird c-Interact orchestrator
        print(f"Running Bird orchestrator on: {data_file}")
        cmd = [
            sys.executable, "-m", "orchestrator.cinteract",
            "--data", data_file,
            "--output", str(raw_json),
        ]
        # --limit only applies when no per-task filter was set (pool already sliced above)
        if args.limit and not any_filter:
            cmd += ["--limit", str(args.limit)]

        result = subprocess.run(cmd, cwd=str(BIRD_ADK_DIR), env=env)
        if result.returncode != 0:
            print(f"Orchestrator exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

        # Convert orchestrator JSON to CSV
        # Orchestrator writes {"metrics": {...}, "results": [...]} as a single JSON object
        print(f"Writing results CSV: {output_path}")
        fieldnames = [
            "task_id", "instance_id", "database",
            "phase1_passed", "phase2_passed", "has_follow_up",
            "total_reward", "elapsed_seconds",
        ]
        rows = []
        if raw_json.exists():
            with open(raw_json) as f:
                data = json.load(f)
            rows = data.get("results", [])

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        # Summary
        total = len(rows)
        if total == 0:
            print("No results written — orchestrator may have produced no output.")
            return

        p1 = sum(1 for r in rows if r.get("phase1_passed"))
        p2 = sum(1 for r in rows if r.get("phase2_passed"))
        reward = sum(float(r.get("total_reward", 0)) for r in rows)

        print(f"\n{'='*50}")
        print(f"Results ({total} tasks)")
        print(f"{'='*50}")
        print(f"Phase 1 pass:  {p1}/{total}  ({100*p1/total:.1f}%)")
        print(f"Phase 2 pass:  {p2}/{total}  ({100*p2/total:.1f}%)")
        print(f"Total reward:  {reward:.2f}  (avg {reward/total:.3f})")
        print(f"CSV:           {output_path}")
        print(f"Raw JSON:      {raw_json}")

    finally:
        print("Stopping Auto Ontology adapter...")
        adapter_proc.terminate()
        try:
            adapter_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            adapter_proc.kill()
        if filtered_tmp:
            try:
                os.unlink(filtered_tmp)
            except Exception:
                pass


if __name__ == "__main__":
    main()
