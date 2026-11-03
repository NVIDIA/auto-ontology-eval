# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the BIRD-Interact Full pipeline from seed through scored results.

No arguments. The API key and catalog Postgres settings are already in ``.env``.
Start catalog Postgres and apply its migrations with
``scripts/setup_auto_ontology.py`` before this script.

The script enables the approved BIRD-Interact feature flags only for its child
processes. It starts the official Full task database and ADK services, ingests
and compiles the 22 task databases, runs all c-Interact query tasks, and writes
``results/bird_interact_full_submission/``.

Usage::

    uv run python scripts/run_bird_interact.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "bird_interact_full"
MANIFEST = DATASET_DIR / "manifest.json"
DATA = DATASET_DIR / "bird_interact_data_with_gt.jsonl"
ADK_ENV_DIR = ROOT / "third_party" / "BIRD-Interact" / "env"
ADK_PYTHON = (
    ROOT
    / "third_party"
    / "BIRD-Interact"
    / "BIRD-Interact-ADK"
    / ".venv-adk"
    / "bin"
    / "python"
)
# Optional overrides (defaults reproduce the full submission run):
#   BIRD_INTERACT_OUTPUT_STEM  results/<stem>/ output name
#   BIRD_INTERACT_INCLUDE_FILE file with one instance_id per line; run only those
#   BIRD_INTERACT_DATABASES    comma-separated selected_database names. Ingest,
#                              compile, and evaluate only those databases.
OUTPUT_STEM = os.environ.get(
    "BIRD_INTERACT_OUTPUT_STEM", "bird_interact_full_submission"
)
INCLUDE_FILE = os.environ.get("BIRD_INTERACT_INCLUDE_FILE")
DATABASES = os.environ.get("BIRD_INTERACT_DATABASES")
#   BIRD_INTERACT_WORKERS      concurrent tasks (default 1)
WORKERS = os.environ.get("BIRD_INTERACT_WORKERS", "1")
OUTPUT_DIR = ROOT / "results" / OUTPUT_STEM
TASK_DB_PORT = 5433
TASK_DB_SERVICE = (
    "postgresql_full"  # compose service name (container: bird_interact_postgresql_full)
)

INTERACTIVE_FLAGS = {
    "BIRD_INTERACT": "true",
    "INTERACTIVE": "true",
    "DB_PROBE_JSONB_PATH_CHECK": "true",
    "DB_PROBE_JOIN_PATH_CHECK": "true",
    "DB_PROBE_PROACTIVE": "true",
    "DB_PROBE_VALUE_GAP_LOOKUP": "true",
    "DETECT_VACUOUS_GROUP_BY": "true",
    "RELEVANCE_FILTER_INCLUDE_COLUMNS": "true",
    "HUB_SIBLING_EXPANSION_ENABLED": "true",
    "HUB_SIBLING_CAP": "6",
    "TABLE_BRIDGE_RECONCILIATION_ENABLED": "true",
}


def _catalog_postgres_kwargs() -> dict[str, str | int]:
    env = dotenv_values(ROOT / ".env")
    return {
        "host": env.get("POSTGRES_HOST") or "localhost",
        "port": int(env.get("POSTGRES_PORT") or "5432"),
        "user": env.get("POSTGRES_USER") or "postgres",
        "password": env.get("POSTGRES_PASSWORD") or "",
        "dbname": env.get("POSTGRES_DATABASE") or "auto_ontology",
        "connect_timeout": 5,
    }


def _require_ready() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        raise SystemExit(
            f"Missing {env_path}. Copy the submission .env into the repo root."
        )
    env = dotenv_values(env_path)
    if not (env.get("DEFAULT_MODELS_API_KEY") or "").strip():
        raise SystemExit(f"{env_path} has no DEFAULT_MODELS_API_KEY.")
    if shutil.which("docker") is None:
        raise SystemExit("Docker is not installed.")

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - installed with the project
        raise SystemExit("psycopg is not installed. Run uv sync first.") from exc

    try:
        settings = _catalog_postgres_kwargs()
        with psycopg.connect(
            host=settings["host"],
            port=settings["port"],
            user=settings["user"],
            password=settings["password"],
            dbname=settings["dbname"],
            connect_timeout=settings["connect_timeout"],
        ):
            pass
    except Exception as exc:
        raise SystemExit(
            "Catalog Postgres is not reachable with POSTGRES_* from .env. "
            "Run: uv run python scripts/setup_auto_ontology.py\n"
            f"{exc}"
        ) from exc


def _selected_databases(manifest_databases: list[str]) -> list[str]:
    if not DATABASES:
        return list(manifest_databases)
    wanted = [part.strip() for part in DATABASES.split(",") if part.strip()]
    known = set(manifest_databases)
    missing = [name for name in wanted if name not in known]
    if missing:
        raise SystemExit(
            "Unknown database(s) in BIRD_INTERACT_DATABASES: " + ", ".join(missing)
        )
    print(
        f"Restricting pipeline to {len(wanted)} databases: {', '.join(wanted)}",
        flush=True,
    )
    return wanted


def _submission_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            env[key] = value
    caller_env = os.environ
    env.update(INTERACTIVE_FLAGS)
    # A/B switch: an explicit value from the caller wins for the experimental flag
    # (DB_PROBE_VALUE_GAP_LOOKUP=false uv run python scripts/run_bird_interact.py).
    if "DB_PROBE_VALUE_GAP_LOOKUP" in caller_env:
        env["DB_PROBE_VALUE_GAP_LOOKUP"] = caller_env["DB_PROBE_VALUE_GAP_LOOKUP"]
    env["PATH"] = f"/tmp/pg_wrappers:{env.get('PATH', '')}"

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    env["CONNECTION_STRINGS"] = ",".join(
        f"postgresql://root:123123@127.0.0.1:{TASK_DB_PORT}/{database}"
        for database in _selected_databases(manifest["databases"])
    )
    return env


def _require_ground_truth() -> None:
    if not MANIFEST.is_file() or not DATA.is_file():
        raise SystemExit("BIRD-Interact Full seeding did not produce its data files.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("gt_available"):
        raise SystemExit(
            "BIRD-Interact Full ground truth is missing. Follow "
            "datasets/bird_interact_full/gt/README.md and rerun this script."
        )


def run(
    name: str,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> float:
    printable = " ".join(argv[1:])
    print(f"\n=== {name} ===\n$ {printable}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(argv, cwd=ROOT, env=env, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(
            f"{name} failed with exit code {result.returncode} after {elapsed:.0f}s"
        )
    print(f"--- {name} finished in {elapsed:.0f}s", flush=True)
    return elapsed


def _database_task_ids() -> list[str]:
    wanted = {part.strip() for part in (DATABASES or "").split(",") if part.strip()}
    if not wanted:
        return []
    ids: list[str] = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if task.get("selected_database") not in wanted:
            continue
        if (task.get("category") or "").lower() != "query":
            continue
        ids.append(task["instance_id"])
    if not ids:
        raise SystemExit(
            "BIRD_INTERACT_DATABASES matched no query tasks: "
            + ", ".join(sorted(wanted))
        )
    print(
        f"Restricting evaluation to {len(ids)} query tasks "
        f"in {', '.join(sorted(wanted))}",
        flush=True,
    )
    return ids


def _include_args() -> list[str]:
    if INCLUDE_FILE:
        ids = [
            line.strip()
            for line in Path(INCLUDE_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not ids:
            raise SystemExit(f"{INCLUDE_FILE} has no instance ids.")
        print(
            f"Restricting evaluation to {len(ids)} tasks from {INCLUDE_FILE}",
            flush=True,
        )
        return ["--include", *ids]
    ids = _database_task_ids()
    if not ids:
        return []
    return ["--include", *ids]


def _service_pids() -> set[int]:
    if shutil.which("lsof") is None:
        return set()
    pids: set[int] = set()
    for port in (6000, 6001, 6002):
        result = subprocess.run(
            ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
        )
        pids.update(int(line) for line in result.stdout.splitlines() if line.strip())
    return pids


def _stop_services(pids: set[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script takes no arguments.\n"
            "Run: uv run python scripts/run_bird_interact.py"
        )

    _require_ready()
    python = sys.executable
    timings: list[tuple[str, float]] = []
    started_services: set[int] = set()

    timings.append(
        (
            "seed",
            run(
                "seed",
                [
                    python,
                    str(ROOT / "scripts" / "bird_interact" / "seed_bird_interact.py"),
                    "--dataset",
                    "full",
                ],
            ),
        )
    )
    _require_ground_truth()
    env = _submission_env()

    timings.append(
        (
            "task databases",
            run(
                "task databases",
                [
                    "docker",
                    "compose",
                    "--project-directory",
                    str(ADK_ENV_DIR),
                    "up",
                    "-d",
                    TASK_DB_SERVICE,
                ],
                env=env,
            ),
        )
    )
    timings.append(
        (
            "ingest",
            run(
                "ingest",
                [
                    python,
                    "-m",
                    "ontology_sql_eval.ingestion.ingest",
                    "--benchmark-name",
                    "bird_interact_full",
                ],
                env=env,
            ),
        )
    )
    timings.append(
        (
            "semantic compile",
            run(
                "semantic compile",
                [
                    python,
                    "-m",
                    "ontology_sql_eval.ingestion.semantic",
                    "--benchmark-name",
                    "bird_interact_full",
                ],
                env=env,
            ),
        )
    )
    timings.append(
        (
            "Postgres command shims",
            run(
                "Postgres command shims",
                [
                    # The shims import psycopg2, which only the ADK venv has.
                    str(ADK_PYTHON) if ADK_PYTHON.is_file() else python,
                    str(ROOT / "scripts" / "bird_interact" / "setup_pg_wrappers.py"),
                ],
                env=env,
            ),
        )
    )

    try:
        timings.append(
            (
                "BIRD-Interact services",
                run(
                    "BIRD-Interact services",
                    [
                        str(
                            ROOT
                            / "scripts"
                            / "bird_interact"
                            / "start_bird_services.sh"
                        )
                    ],
                    env=env,
                ),
            )
        )
        started_services = _service_pids()
        timings.append(
            (
                "evaluation",
                run(
                    "evaluation",
                    [
                        python,
                        str(
                            ROOT
                            / "scripts"
                            / "bird_interact"
                            / "run_all_bird_interact.py"
                        ),
                        "--data",
                        str(DATA),
                        "--output",
                        OUTPUT_STEM,
                        "--workers",
                        WORKERS,
                        "--overwrite",
                        *_include_args(),
                    ],
                    env=env,
                ),
            )
        )
    finally:
        _stop_services(started_services)

    result_jsonl = OUTPUT_DIR / f"{OUTPUT_STEM}.jsonl"
    timings.append(
        (
            "analysis",
            run(
                "analysis",
                [
                    python,
                    str(
                        ROOT / "scripts" / "bird_interact" / "analyze_bird_interact.py"
                    ),
                    str(result_jsonl),
                ],
                env=env,
            ),
        )
    )

    print("\n=== done ===", flush=True)
    for name, seconds in timings:
        print(f"  {name:<28} {seconds:>6.0f}s", flush=True)
    print(f"\nResults: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
