# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start Postgres and migrate the catalog after clone.

Reads ``.env`` in the repo root. Starts the pgvector container when it is not
already running, checks out the Auto Ontology revision pinned in
``pyproject.toml`` under ``/tmp/auto-ontology``, and applies that checkout's
Alembic migrations with this repo's virtualenv.

No arguments. Write ``.env`` in the repo root first.

Usage::

    uv run python scripts/setup_auto_ontology.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = Path("/tmp/auto-ontology")
REPO_URL = "https://github.com/NVIDIA/auto-ontology.git"
CONTAINER_NAME = "auto-ontology-postgres"
IMAGE = "pgvector/pgvector:pg17"
PIN_RE = re.compile(
    r"git\+https://github\.com/NVIDIA/auto-ontology\.git@([0-9a-f]{40})"
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})
_POSTGRES_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DATABASE",
)


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    user: str
    password: str
    database: str


def pinned_revision(pyproject: Path) -> str:
    """The Auto Ontology commit ``uv sync`` installs."""
    found = PIN_RE.findall(pyproject.read_text())
    if len(found) != 1:
        raise SystemExit(
            f"Expected one pinned auto-ontology revision in {pyproject}, found {found}."
        )
    return found[0]


def postgres_settings(env: Mapping[str, str | None]) -> PostgresSettings:
    missing = [
        key for key in _POSTGRES_KEYS if env.get(key) is None or env.get(key) == ""
    ]
    if missing:
        raise SystemExit(".env is missing " + ", ".join(missing) + ".")

    host = str(env["POSTGRES_HOST"]).strip()
    if host not in _LOCAL_HOSTS:
        raise SystemExit(
            f"POSTGRES_HOST is {host}. This script starts Postgres on localhost. "
            "Set POSTGRES_HOST=localhost in .env."
        )
    try:
        port = int(str(env["POSTGRES_PORT"]).strip())
    except ValueError as exc:
        raise SystemExit("POSTGRES_PORT must be an integer.") from exc

    user = str(env["POSTGRES_USER"]).strip()
    password = str(env["POSTGRES_PASSWORD"])
    database = str(env["POSTGRES_DATABASE"]).strip()
    for label, value in (("POSTGRES_USER", user), ("POSTGRES_DATABASE", database)):
        if value == "" or "\n" in value or "\r" in value:
            raise SystemExit(f"{label} must be a single non-empty line.")
    if "\n" in password or "\r" in password:
        raise SystemExit("POSTGRES_PASSWORD must be a single line.")
    return PostgresSettings(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


def docker_run_args(settings: PostgresSettings) -> list[str]:
    return [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "--restart",
        "unless-stopped",
        "-e",
        f"POSTGRES_USER={settings.user}",
        "-e",
        f"POSTGRES_PASSWORD={settings.password}",
        "-e",
        f"POSTGRES_DB={settings.database}",
        "-p",
        f"{settings.port}:5432",
        IMAGE,
    ]


def load_settings() -> PostgresSettings:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        raise SystemExit(
            f"Missing {env_path}. Copy the submission .env into the repo root, "
            "then re-run this script."
        )
    env = dotenv_values(env_path)
    if not (env.get("DEFAULT_MODELS_API_KEY") or "").strip():
        raise SystemExit(f"{env_path} has no DEFAULT_MODELS_API_KEY.")
    return postgres_settings(env)


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Docker is not installed. Install Docker and re-run this script."
        ) from exc


def _missing_container(stderr: str) -> bool:
    text = stderr.lower()
    return "no such object" in text or "no such container" in text


def ensure_postgres(settings: PostgresSettings) -> None:
    inspect = _docker("inspect", "-f", "{{.State.Status}}", CONTAINER_NAME)
    if inspect.returncode == 0:
        status = inspect.stdout.strip()
        if status == "running":
            print(
                f"Postgres container {CONTAINER_NAME} is already running.", flush=True
            )
            return
        print(f"Starting Postgres container {CONTAINER_NAME}.", flush=True)
        start = _docker("start", CONTAINER_NAME)
        if start.returncode != 0:
            raise SystemExit(start.stderr.strip() or "docker start failed")
        return

    if not _missing_container(inspect.stderr):
        detail = (inspect.stderr or inspect.stdout).strip()
        raise SystemExit(
            detail or "docker inspect failed. Is the Docker daemon running?"
        )

    print(f"Creating Postgres container {CONTAINER_NAME}.", flush=True)
    print(
        "$ docker run -d --name "
        f"{CONTAINER_NAME} --restart unless-stopped "
        f"-e POSTGRES_USER={settings.user} -e POSTGRES_PASSWORD=... "
        f"-e POSTGRES_DB={settings.database} -p {settings.port}:5432 {IMAGE}",
        flush=True,
    )
    result = subprocess.run(docker_run_args(settings), check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"docker run failed with exit code {result.returncode}. "
            f"Free host port {settings.port} if it is already in use, "
            "then re-run this script."
        )


def wait_for_postgres(settings: PostgresSettings, timeout_s: float = 90) -> None:
    import psycopg

    print("Waiting for Postgres.", flush=True)
    deadline = time.monotonic() + timeout_s
    last = "Postgres did not accept connections"
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(
                host=settings.host,
                port=settings.port,
                user=settings.user,
                password=settings.password,
                dbname=settings.database,
                connect_timeout=3,
            ):
                return
        except Exception as exc:
            last = str(exc)
            time.sleep(2)
    raise SystemExit(
        "Postgres did not accept connections with the POSTGRES_* values in .env.\n"
        f"{last}\n"
        f"If {CONTAINER_NAME} was created with different POSTGRES_* values, remove it "
        f"with `docker rm -f {CONTAINER_NAME}` and re-run this script."
    )


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(argv), flush=True)
    result = subprocess.run(argv, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{argv[0]} failed with exit code {result.returncode}")


def _git_head(dest: Path) -> str | None:
    if not (dest / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def ensure_checkout(dest: Path, sha: str) -> None:
    ready = (dest / "alembic.ini").is_file() and (dest / "alembic" / "env.py").is_file()
    if _git_head(dest) == sha and ready:
        print(f"Using {dest} at {sha}.", flush=True)
        return
    if dest.exists():
        print(f"Replacing {dest}.", flush=True)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    _run(["git", "init"], cwd=dest, env=env)
    _run(["git", "remote", "add", "origin", REPO_URL], cwd=dest, env=env)
    _run(["git", "fetch", "--depth", "1", "origin", sha], cwd=dest, env=env)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest, env=env)
    got = _git_head(dest)
    if got != sha:
        raise SystemExit(f"Checkout at {dest} is {got}, expected {sha}.")


def write_checkout_env(dest: Path, settings: PostgresSettings) -> None:
    """Postgres settings only. The API key stays in this repo's ``.env``."""
    path = dest / ".env"
    path.write_text(
        f"POSTGRES_HOST={settings.host}\n"
        f"POSTGRES_PORT={settings.port}\n"
        f"POSTGRES_USER={settings.user}\n"
        f"POSTGRES_PASSWORD={settings.password}\n"
        f"POSTGRES_DATABASE={settings.database}\n"
    )
    path.chmod(0o600)


def alembic_env(settings: PostgresSettings) -> dict[str, str]:
    env = os.environ.copy()
    env["POSTGRES_HOST"] = settings.host
    env["POSTGRES_PORT"] = str(settings.port)
    env["POSTGRES_USER"] = settings.user
    env["POSTGRES_PASSWORD"] = settings.password
    env["POSTGRES_DATABASE"] = settings.database
    return env


def run_alembic(dest: Path, settings: PostgresSettings) -> None:
    alembic = Path(sys.executable).with_name("alembic")
    if not alembic.is_file():
        raise SystemExit(
            f"Missing {alembic}. Run this script with: "
            "uv run python scripts/setup_auto_ontology.py"
        )
    _run([str(alembic), "upgrade", "head"], cwd=dest, env=alembic_env(settings))


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script takes no arguments.\n"
            "Run: uv run python scripts/setup_auto_ontology.py"
        )

    settings = load_settings()
    ensure_postgres(settings)
    wait_for_postgres(settings)
    sha = pinned_revision(ROOT / "pyproject.toml")
    ensure_checkout(CHECKOUT, sha)
    write_checkout_env(CHECKOUT, settings)
    run_alembic(CHECKOUT, settings)
    print("\nPostgres is up and the catalog schema is current.", flush=True)


if __name__ == "__main__":
    main()
