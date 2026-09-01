# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed a local Postgres instance with pre-generated DDL + CSV data.

Reads DDL and CSV files from ``datasets/<database_name>/`` (``ddl/`` and
``data/``) and applies them to a local Postgres instance. All artifacts are
pre-generated, so no source (e.g. MSSQL) connection is required.

Usage::

    uv run python scripts/seed_wwi.py --database-name wideworldimporters

The ``--database-name`` value both selects the source folder
``datasets/<name>/`` and names the target database (default
``wideworldimporters``). Postgres credentials come from the environment / .env:

    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD

Pass ``--drop`` to wipe and recreate the database before loading. Pass
``--refresh-collation`` if Postgres reports a "collation version mismatch" on
``template1`` (e.g. after an OS libc/ICU upgrade).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import psycopg
from psycopg.conninfo import make_conninfo

logger = logging.getLogger(__name__)

DEFAULT_DB = "wideworldimporters"

DDL_FILES_PRE_DATA = [
    "01_schemas.sql",
    "02_sequences.sql",
    "03_tables.sql",
    "04_indexes.sql",
]
DDL_FILES_POST_DATA = [
    "05_fkeys.sql",
    "06_views.sql",
]


def _paths(database_name: str) -> tuple[Path, Path]:
    """Return (ddl_dir, data_dir) for ``datasets/<database_name>/``."""
    base = Path(__file__).resolve().parents[1] / "datasets" / database_name
    return base / "ddl", base / "data"


def _conninfo(database_name: str = "postgres") -> str:
    return make_conninfo(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        dbname=database_name,
    )


def _ensure_database(database_name: str, *, drop: bool = False) -> None:
    """Create the target database (optionally dropping it first)."""
    admin_database_name = os.environ.get("POSTGRES_DATABASE", "gsf")
    with psycopg.connect(_conninfo(admin_database_name)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            if drop:
                logger.info("Dropping database %s (if exists)...", database_name)
                # WITH (FORCE) terminates lingering sessions so the drop doesn't
                # fail with "database is being accessed by other users".
                cur.execute(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)
            )
            if cur.fetchone():
                logger.info("Database %s already exists.", database_name)
                return
            logger.info("Creating database %s...", database_name)
            cur.execute(f'CREATE DATABASE "{database_name}"')


def _apply_ddl(database_name: str, ddl_dir: Path, filenames: list[str]) -> None:
    """Apply a list of DDL SQL files to the database."""
    with psycopg.connect(_conninfo(database_name)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for fname in filenames:
                path = ddl_dir / fname
                if not path.exists():
                    logger.warning("  DDL file %s not found — skipping.", fname)
                    continue
                logger.info("  Applying %s ...", fname)
                cur.execute(path.read_text())


def _load_csvs(database_name: str, data_dir: Path) -> int:
    """COPY all CSV files from data/ into the corresponding tables."""
    if not data_dir.exists():
        logger.error("Data directory %s does not exist.", data_dir)
        return 0

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        logger.warning("No CSV files found in %s", data_dir)
        return 0

    total_rows = 0
    with psycopg.connect(_conninfo(database_name)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for csv_path in csvs:
                schema_table = csv_path.stem
                parts = schema_table.split(".", 1)
                if len(parts) != 2:
                    logger.warning(
                        "  Skipping %s (unexpected name format)", csv_path.name
                    )
                    continue
                schema, table = parts[0].lower(), parts[1].lower()
                qn = f"{schema}.{table}"

                try:
                    with csv_path.open("r") as f:
                        with cur.copy(
                            f"COPY {qn} FROM STDIN WITH "
                            f"(FORMAT csv, HEADER true, NULL '\\N')"
                        ) as copy:
                            while data := f.read(65536):
                                copy.write(data.encode())

                    cur.execute(f"SELECT count(*) FROM {qn}")
                    cnt = cur.fetchone()[0]
                    total_rows += cnt
                    logger.info("  %s.%-40s %10d rows", schema, table, cnt)
                except Exception:
                    logger.error("  FAILED to load %s", csv_path.name, exc_info=True)

    return total_rows


def seed_database(
    database_name: str = DEFAULT_DB,
    *,
    drop: bool = False,
    refresh_collation: bool = False,
) -> None:
    """Full load: create DB -> DDL -> CSV data -> FKs/views."""
    ddl_dir, data_dir = _paths(database_name)

    logger.info("=" * 60)
    logger.info("Seeding Postgres database %r from %s", database_name, ddl_dir.parent)
    logger.info("=" * 60)

    _ensure_database(database_name, drop=drop)

    logger.info("\n--- Phase 1: DDL (schemas, sequences, tables, indexes) ---")
    _apply_ddl(database_name, ddl_dir, DDL_FILES_PRE_DATA)

    logger.info("\n--- Phase 2: Load CSV data ---")
    total = _load_csvs(database_name, data_dir)
    logger.info("  Total: %d rows loaded across all tables.", total)

    logger.info("\n--- Phase 3: Post-data DDL (foreign keys, views) ---")
    _apply_ddl(database_name, ddl_dir, DDL_FILES_POST_DATA)

    logger.info("\n" + "=" * 60)
    logger.info("Seed complete for database %r.", database_name)
    logger.info("=" * 60)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Seed a local Postgres database from pre-generated DDL + CSV files."
    )
    parser.add_argument(
        "--database-name",
        default=DEFAULT_DB,
        help=(
            "Dataset / target database name. Selects the source folder "
            f"datasets/<name>/ and names the target DB (default: {DEFAULT_DB})."
        ),
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop and recreate the database before loading.",
    )
    parser.add_argument(
        "--refresh-collation",
        action="store_true",
        help=(
            "Run ALTER DATABASE ... REFRESH COLLATION VERSION on template1 and the "
            "admin DB before CREATE DATABASE. Fixes 'collation version mismatch' "
            "errors after an OS libc/ICU change."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    seed_database(
        args.database_name,
        drop=args.drop,
        refresh_collation=args.refresh_collation,
    )
