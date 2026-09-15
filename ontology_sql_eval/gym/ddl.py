# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema dumping for the schema-only (no-ontology) evaluation arm.

The schema-only arm's whole premise is that the model sees the raw schema and
nothing else, so whatever this module emits *is* the independent variable of the
experiment. Keep it boring and freeze it once a full run starts.

Output mirrors the ``sql_context`` field NeMo Gym's own ``bird_sql`` environment
puts in its task records -- a SQL dump wrapped in ``BEGIN TRANSACTION;`` /
``COMMIT;`` -- so our baseline stays comparable to published BIRD numbers.

Two backends, because the corpora differ:

* SQLite (BIRD, FDABench) -- read ``CREATE TABLE`` statements back out of
  ``sqlite_master``.
* Postgres (WideWorldImporters) -- concatenate the DDL files the dataset already
  ships; there is nothing to introspect and no live database is required.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

__all__ = ["sqlite_schema_dump", "postgres_schema_dump", "column_description_block"]


def sqlite_schema_dump(db_path: Path) -> str:
    """Return the ``CREATE TABLE`` dump for a SQLite database file.

    Internal ``sqlite_*`` tables are skipped. Statements are emitted in
    ``sqlite_master`` order, which is creation order, so a reader meets parent
    tables before the tables that reference them.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"no SQLite database at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        conn.text_factory = lambda b: b.decode(errors="ignore")
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
        ).fetchall()

    statements = [r[0].strip().rstrip(";") + ";" for r in rows]
    body = "\n".join(statements)
    return f"BEGIN TRANSACTION;\n{body}\nCOMMIT;"


def postgres_schema_dump(ddl_dir: Path) -> str:
    """Concatenate the ``*.sql`` DDL files a Postgres dataset ships.

    Used for WideWorldImporters, whose schema is checked into the repo. Files are
    read in sorted order so the dump is byte-stable across runs.
    """
    if not ddl_dir.is_dir():
        raise FileNotFoundError(f"no DDL directory at {ddl_dir}")

    files = sorted(ddl_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no .sql files under {ddl_dir}")

    parts = [f"-- {f.name}\n{f.read_text().strip()}" for f in files]
    return "\n\n".join(parts)


def column_description_block(description_dir: Path) -> str:
    """Render BIRD's per-column description CSVs as a plain-text block.

    BIRD ships one CSV per table under ``database_description/`` with
    ``original_column_name`` / ``column_description`` / ``value_description``
    columns. Only genuinely informative rows are kept: a description that merely
    restates the column name adds tokens without adding information.

    Returns the empty string when the directory is absent or nothing survives
    filtering, so callers can append unconditionally.
    """
    if not description_dir.is_dir():
        return ""

    sections: list[str] = []
    for csv_path in sorted(description_dir.glob("*.csv")):
        lines: list[str] = []
        # BIRD's CSVs are not uniformly UTF-8 and several carry a BOM.
        with csv_path.open(newline="", encoding="utf-8-sig", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("original_column_name") or "").strip()
                desc = (row.get("column_description") or "").strip()
                value = (row.get("value_description") or "").strip()
                if not name:
                    continue
                if desc.lower() == name.lower():
                    desc = ""
                if not desc and not value:
                    continue
                text = "; ".join(p for p in (desc, value) if p)
                lines.append(f"  {name}: {text}")

        if lines:
            sections.append(f"Table {csv_path.stem}:\n" + "\n".join(lines))

    if not sections:
        return ""
    return "Column descriptions:\n" + "\n".join(sections)
