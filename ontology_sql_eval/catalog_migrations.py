# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Auto Ontology catalog's schema in step with the installed Auto Ontology.

Auto Ontology's schema changes ship as Alembic migrations in the root of its
repository (``alembic/`` and ``alembic.ini``), and its installed package does
not include them. :func:`ensure_current` fetches just those files, for exactly
the commit installed in ``.venv``, into ``cache/auto-ontology-migrations/<commit>/``,
and upgrades the database that ``POSTGRES_*`` points at when it is behind.

``main.py`` calls it before any stage; ``scripts/migrate_catalog.py`` runs it on
its own. Auto Ontology itself never migrates from an application process, since
replicas would race each other. This eval runs one process against its own
catalog, so it does, and ``--skip-migrate`` turns that off for a shared database.
"""

from __future__ import annotations

import json
import logging
import subprocess
from importlib.metadata import distribution
from pathlib import Path

from ontology_sql_eval.paths import REPO_ROOT

logger = logging.getLogger(__name__)

CACHE = REPO_ROOT / "cache" / "auto-ontology-migrations"
DISTRIBUTION = "auto-ontology-server"


class CatalogSchemaError(RuntimeError):
    """The catalog is at a revision the installed Auto Ontology does not know."""


def installed_commit() -> tuple[str, str]:
    """The (repository URL, commit) that ``uv sync`` installed Auto Ontology from."""
    info = json.loads(distribution(DISTRIBUTION).read_text("direct_url.json") or "{}")
    commit = info.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise CatalogSchemaError(
            f"{DISTRIBUTION} was not installed from Git; run `uv sync` first."
        )
    return info["url"], commit


def fetch_migrations(url: str, commit: str) -> Path:
    """Check out only ``alembic/`` and ``alembic.ini`` at *commit*; reuse it if present."""
    dest = CACHE / commit
    if (dest / "alembic.ini").is_file():
        return dest

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dest), *args], check=True)

    logger.info("Downloading Auto Ontology %s migrations into %s", commit[:12], dest)
    dest.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("sparse-checkout", "set", "--no-cone", "/alembic/", "/alembic.ini")
    git("fetch", "-q", "--depth=1", "--filter=blob:none", url, commit)
    git("checkout", "-q", "FETCH_HEAD")
    return dest


def ensure_current() -> None:
    """Upgrade the catalog to the installed Auto Ontology's head revision if behind.

    Expects ``.env`` to be loaded already. Never downgrades: a catalog at a
    revision these migrations don't contain was migrated by something newer,
    and that is raised rather than guessed at.
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from auto_ontology.dal.session import sqlalchemy_url
    from sqlalchemy import create_engine

    url, commit = installed_commit()
    config = Config(str(fetch_migrations(url, commit) / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)
    heads = set(scripts.get_heads())

    engine = create_engine(sqlalchemy_url())
    try:
        with engine.connect() as connection:
            applied = MigrationContext.configure(connection).get_current_heads()
    finally:
        engine.dispose()

    if set(applied) == heads:
        logger.info("Catalog schema is current (%s)", ", ".join(sorted(heads)))
        return

    def known(rev: str) -> bool:
        try:
            return scripts.get_revision(rev) is not None
        except Exception:  # alembic raises for a revision it has no script for
            return False

    unknown = [rev for rev in applied if not known(rev)]
    if unknown:
        raise CatalogSchemaError(
            f"The catalog is at revision {', '.join(unknown)}, which Auto Ontology "
            f"{commit[:12]} does not have: something newer migrated it. Point "
            "POSTGRES_DATABASE at this project's own database, or bump the pin."
        )

    logger.info(
        "Migrating the catalog from %s to %s (Auto Ontology %s)",
        ", ".join(applied) or "an empty schema",
        ", ".join(sorted(heads)),
        commit[:12],
    )
    command.upgrade(config, "head")
