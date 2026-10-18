# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bring the Auto Ontology catalog's schema up to the installed Auto Ontology.

    uv run python scripts/migrate_catalog.py

Auto Ontology's schema changes ship as Alembic migrations in the root of its
repository (``alembic/`` and ``alembic.ini``), and its installed package does
not include them. So this fetches just those files, for exactly the commit
installed in ``.venv``, into ``cache/auto-ontology-migrations/<commit>/``, and
applies them to the database that ``POSTGRES_*`` in ``.env`` points at.

Run it once before the first ingest, and again after every Auto Ontology pin
bump. It is safe to re-run: an up-to-date catalog is left alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "auto-ontology-migrations"
DISTRIBUTION = "auto-ontology-server"


def installed_commit() -> tuple[str, str]:
    """The (repository URL, commit) that ``uv sync`` installed Auto Ontology from."""
    direct_url = distribution(DISTRIBUTION).read_text("direct_url.json")
    info = json.loads(direct_url or "{}")
    commit = info.get("vcs_info", {}).get("commit_id")
    if not commit:
        sys.exit(f"{DISTRIBUTION} was not installed from Git; run `uv sync` first.")
    return info["url"], commit


def fetch_migrations(url: str, commit: str) -> Path:
    """Check out only ``alembic/`` and ``alembic.ini`` at *commit*; reuse it if present."""
    dest = CACHE / commit
    if (dest / "alembic.ini").is_file():
        return dest

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dest), *args], check=True)

    dest.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("sparse-checkout", "set", "--no-cone", "/alembic/", "/alembic.ini")
    git("fetch", "-q", "--depth=1", "--filter=blob:none", url, commit)
    git("checkout", "-q", "FETCH_HEAD")
    return dest


def main() -> None:
    from ontology_sql_eval.env import load_env

    load_env()  # POSTGRES_* from .env, which Auto Ontology's alembic env reads

    from alembic import command
    from alembic.config import Config

    url, commit = installed_commit()
    migrations = fetch_migrations(url, commit)
    print(f"Applying Auto Ontology {commit[:12]} migrations from {migrations}", flush=True)
    command.upgrade(Config(str(migrations / "alembic.ini")), "head")
    command.current(Config(str(migrations / "alembic.ini")))


if __name__ == "__main__":
    main()
