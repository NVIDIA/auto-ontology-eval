# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply dataset enrichment files after ingest (or as a standalone step).

Reads from ``datasets/<database_name>/``:

* ``glossary_and_prompts.json`` — inserts glossary rows into Postgres
  ``acronyms`` and the custom prompt into ``prompts`` (same tables the
  ``/semantic-input`` UI writes to).
* ``metadata.json`` — stamps table/column descriptions onto Neo4j (skipped
  when the first table entry already has a matching description).
* ``custom_analyses.json`` — creates CustomAnalysis nodes + embeddings
  (skipped when the first analysis name already exists).

Usage::

    PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.apply_enrichment \\
        --database-name kdc_ca1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from gsf.connectors.registry import create_connector
from gsf.dal.custom_analyses import find_analysis_by_name
from gsf.infra.postgres import get_postgres_connection_string
from gsf.utils import get_embed_params
from gsf.vdb import get_semantic_vdb
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

from ontology_sql_eval.ingestion.enrich_graph import (
    DEFAULT_DIR,
    add_custom_analyses,
    apply_metadata,
)

load_dotenv()

logger = logging.getLogger(__name__)

_GLOSSARY_FILENAME = "glossary_and_prompts.json"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _connection_strings() -> list[str]:
    return [
        s.strip()
        for s in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if s.strip()
    ]


def _connector_for(database_name: str):
    """Return the connector whose ``database_name`` matches *database_name*."""
    strings = _connection_strings()
    if not strings:
        raise EnvironmentError(
            "CONNECTION_STRINGS is not set. Add it to your .env so the "
            "connector dialect is available for custom-analysis SQL parsing."
        )
    for connection_string in strings:
        connector = create_connector(connection_string)
        if connector.database_name == database_name:
            return connector
    raise SystemExit(
        f"No CONNECTION_STRINGS entry resolves to database {database_name!r}. "
        f"Got: {[create_connector(s).database_name for s in strings]}"
    )


def apply_glossary_and_prompts(database_name: str) -> None:
    """Insert glossary acronyms and the custom prompt into GSF Postgres.

    Mirrors the ``/semantic-input`` UI writes:

    * glossary items → ``acronyms(name UNIQUE, description)``
    * ``custom_prompts`` string → ``prompts(content)``

    Acronyms are upserted by name. The prompt is inserted only when an
    identical ``content`` row is not already present.

    Both tables' ``id`` columns (and ``acronyms.updated_at``) are filled in by
    Prisma on the app side rather than by a database default, so these raw SQL
    inserts must supply them.
    """
    import psycopg

    path = DEFAULT_DIR / database_name / _GLOSSARY_FILENAME
    if not path.exists():
        logger.info(
            "No %s at %s — skipping glossary/prompts.", _GLOSSARY_FILENAME, path
        )
        return

    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(payload).__name__}"
        )

    glossary = payload.get("glossary") or []
    custom_prompts = (payload.get("custom_prompts") or "").strip()
    if not isinstance(glossary, list):
        raise ValueError(f"'glossary' must be a list in {path}")

    conn_str = get_postgres_connection_string()
    inserted_acronyms = 0
    updated_acronyms = 0
    inserted_prompts = 0

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            for item in glossary:
                name = (item.get("name") or "").strip()
                description = item.get("description") or ""
                if not name:
                    logger.warning("Skipping glossary entry with empty name: %r", item)
                    continue
                cur.execute(
                    """
                    INSERT INTO acronyms (id, name, description, updated_at)
                    VALUES (gen_random_uuid(), %s, %s, NOW())
                    ON CONFLICT (name) DO UPDATE
                      SET description = EXCLUDED.description,
                          updated_at = NOW()
                    RETURNING (xmax = 0) AS inserted
                    """,
                    (name, description),
                )
                row = cur.fetchone()
                if row and row[0]:
                    inserted_acronyms += 1
                else:
                    updated_acronyms += 1

            if custom_prompts:
                cur.execute(
                    "SELECT 1 FROM prompts WHERE content = %s LIMIT 1",
                    (custom_prompts,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        "INSERT INTO prompts (id, content) "
                        "VALUES (gen_random_uuid(), %s)",
                        (custom_prompts,),
                    )
                    inserted_prompts = 1
                else:
                    logger.info("Custom prompt already present — skipping insert.")

        conn.commit()

    logger.info(
        "Glossary/prompts from %s: %d acronym(s) inserted, %d updated, "
        "%d prompt(s) inserted.",
        path,
        inserted_acronyms,
        updated_acronyms,
        inserted_prompts,
    )


def _first_metadata_table(database_name: str) -> tuple[str, str] | None:
    """Return ``(table_name, description)`` for the first metadata entry."""
    path = DEFAULT_DIR / database_name / "metadata.json"
    if not path.exists():
        return None
    raw = _load_json(path)
    if not isinstance(raw, dict) or not raw:
        return None
    table_name, table_meta = next(iter(raw.items()))
    description = (table_meta or {}).get("description") or ""
    if not description:
        return None
    return table_name, description


def metadata_already_applied(database_name: str) -> bool:
    """True when the first metadata table already carries its description."""
    first = _first_metadata_table(database_name)
    if first is None:
        return False
    table_name, description = first
    rows = cast(
        list[dict[str, Any]],
        get_neo4j_conn().query_read(
            query=(
                "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
                "(:Schema)-[:CONTAINS]->(t:Table {name: $table_name}) "
                "RETURN t.description AS description "
                "LIMIT 1"
            ),
            parameters={"database_name": database_name, "table_name": table_name},
        ),
    )
    if not rows:
        return False
    existing = rows[0].get("description") or ""
    return existing == description


def _first_custom_analysis_name(database_name: str) -> str | None:
    path = DEFAULT_DIR / database_name / "custom_analyses.json"
    if not path.exists():
        return None
    analyses = _load_json(path)
    if not isinstance(analyses, list) or not analyses:
        return None
    name = (analyses[0].get("name") or "").strip()
    return name or None


def custom_analyses_already_applied(database_name: str) -> bool:
    """True when the first custom-analysis name already exists in Neo4j."""
    name = _first_custom_analysis_name(database_name)
    if not name:
        return False
    return find_analysis_by_name(name, exclude_id=None) is not None


def apply_enrichment(database_name: str) -> None:
    """Seed glossary/prompts and conditionally apply metadata + analyses."""
    logger.info("Applying enrichment for database %r", database_name)
    dataset_dir = DEFAULT_DIR / database_name
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    apply_glossary_and_prompts(database_name)

    if metadata_already_applied(database_name):
        first = _first_metadata_table(database_name)
        logger.info(
            "Metadata already applied (table %r has its description) — skipping.",
            first[0] if first else "?",
        )
    else:
        apply_metadata(database_name)

    if custom_analyses_already_applied(database_name):
        logger.info(
            "Custom analyses already applied (found %r) — skipping.",
            _first_custom_analysis_name(database_name),
        )
    else:
        connector = _connector_for(database_name)
        embed_params = get_embed_params()
        add_custom_analyses(
            database_name,
            connector.dialect,
            embed_params=embed_params,
            vdb=get_semantic_vdb(database_name=database_name),
        )

    logger.info("Enrichment complete for %r", database_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-name",
        required=True,
        help="Dataset folder under datasets/ (e.g. kdc_ca1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    apply_enrichment(args.database_name)
