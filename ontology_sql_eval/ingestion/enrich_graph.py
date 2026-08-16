# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp table/column metadata onto the Neo4j graph.

This module reads ``<database_name>/metadata.json`` and writes descriptions and
sample values onto the ``Table`` and ``Column`` nodes that the tabular ingest
pipeline created in Neo4j. It is intentionally a small, dev-tools-only helper
and is meant to be invoked at the end of an ingest run.

JSON shape (per table)::

    {
        "<table_name>": {
            "description": "...",
            "columns": [
                {
                    "name": "...",
                    "data_type": "INTEGER",
                    "description": "...",
                    "value_examples": [0, 1] | ["...", ...] | null,
                    ...
                },
                ...
            ]
        },
        ...
    }

Custom analyses (optional, ``<database_name>/custom_analyses.json``)::

    [
        {
            "name": "...",
            "description": "...",
            "sql": "SELECT ..."
        },
        ...
    ]
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "datasets"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def resolve_dataset_file(database_name: str, filename: str) -> Path | None:
    """Locate a per-database dataset file under ``datasets/``.

    Snowflake Spider2 DBs use the physical name in Neo4j / pgvector
    (e.g. ``CPTAC_PDC``), while seeded files live under
    ``datasets/spider2/<slug>/``. Resolution order:

    1. ``datasets/<database_name>/<filename>``
    2. ``datasets/spider2/<slug(database_name)>/<filename>``
    """
    candidates = [
        DEFAULT_DIR / database_name / filename,
        DEFAULT_DIR / "spider2" / _slugify(database_name) / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def get_all_schemas_ids(database_name: str | None = None) -> list[str]:
    """Return schema ids, optionally scoped to one database.

    NeMo Retriever's helper currently returns schemas from every database.
    Custom-analysis parsing must be database-local because common table names
    such as ``orders`` and ``customers`` otherwise resolve against an
    unrelated schema.
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    if database_name is None:
        query = "MATCH (s:Schema) RETURN s.id AS schema_id"
        parameters = None
    else:
        query = (
            "MATCH (d:Database {name: $database_name})-[:CONTAINS]->(s:Schema) "
            "RETURN s.id AS schema_id"
        )
        parameters = {"database_name": database_name}

    rows = cast(
        list[dict[str, Any]],
        get_neo4j_conn().query_read(query=query, parameters=parameters),
    )
    return [row["schema_id"] for row in rows if row.get("schema_id")]


def apply_metadata(database_name: str) -> None:
    """Stamp table/column metadata onto the Neo4j graph.

    Reads ``metadata.json`` for *database_name* (keyed by table name) and
    updates the following properties for every table/column belonging to
    *database_name*:

    * ``Table.description``
    * ``Column.description``
    * ``Column.data_type`` (declared upstream type, or a seed-time inference)
    * ``Column.sample_values`` (from the JSON's ``value_examples`` field, when
      present and non-empty, stored as typed JSON)

    Tables/columns that aren't present in the graph are silently skipped
    (the MATCH simply finds nothing). Properties for which the JSON has no
    value are left untouched (``coalesce`` preserves the existing value).
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    metadata_path = resolve_dataset_file(database_name, "metadata.json")

    if metadata_path is None:
        logger.info(
            "No metadata file for database %r under %s — skipping enrichment.",
            database_name,
            DEFAULT_DIR,
        )
        return

    with metadata_path.open() as f:
        raw = json.load(f)

    table_rows: list[dict[str, str]] = []
    column_rows: list[dict[str, Any]] = []
    samples_count = 0
    types_count = 0
    for table_name, table_meta in raw.items():
        table_desc = table_meta.get("description")
        if table_desc:
            table_rows.append({"table_name": table_name, "description": table_desc})

        for col in table_meta.get("columns", []) or []:
            col_desc = col.get("description")
            data_type = col.get("data_type")
            value_examples = col.get("value_examples")
            sample_values: str | None = (
                json.dumps(value_examples, ensure_ascii=False)
                if isinstance(value_examples, list) and value_examples
                else None
            )
            if not col_desc and not data_type and sample_values is None:
                continue
            if sample_values is not None:
                samples_count += 1
            if data_type:
                types_count += 1
            column_rows.append(
                {
                    "table_name": table_name,
                    "column_name": col["name"],
                    "description": col_desc or None,
                    "data_type": str(data_type) if data_type else None,
                    "sample_values": sample_values,
                }
            )

    conn = get_neo4j_conn()

    if table_rows:
        conn.query_write(
            query=(
                "UNWIND $rows AS row "
                "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
                "(:Schema)-[:CONTAINS]->(t:Table {name: row.table_name}) "
                "SET t.description = coalesce(row.description, t.description)"
            ),
            parameters={"rows": table_rows, "database_name": database_name},
        )

    if column_rows:
        conn.query_write(
            query=(
                "UNWIND $rows AS row "
                "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
                "(:Schema)-[:CONTAINS]->(t:Table {name: row.table_name})"
                "-[:CONTAINS]->(c:Column {name: row.column_name}) "
                "SET c.description = coalesce(row.description, c.description), "
                "    c.data_type = coalesce(row.data_type, c.data_type), "
                "    c.sample_values = coalesce(row.sample_values, c.sample_values)"
            ),
            parameters={"rows": column_rows, "database_name": database_name},
        )

    logger.info(
        "Applied metadata: %d table description(s), %d column description(s), "
        "%d column data type(s), %d column sample_values from %s",
        len(table_rows),
        sum(1 for r in column_rows if r.get("description")),
        types_count,
        samples_count,
        metadata_path,
    )


def add_custom_analyses(
    database_name: str,
    dialect: str,
    embed_params: "EmbedParams | None" = None,
    vdb: "VDB | None" = None,
) -> None:
    """Ingest custom analyses for *database_name* into the Neo4j graph and the VDB.

    Reads ``<this dir>/<database_name>/custom_analyses.json`` — a list of
    ``{"name", "description", "sql"}`` entries — and, for each entry:

    * parses the SQL against the schemas already in the graph (via
      :func:`parse_query_single`), which produces a :class:`Sql` node and the
      corresponding ``Sql -> Table/Column`` edges;
    * creates a :class:`CustomAnalysis` node with ``name`` and ``description``;
    * connects ``CustomAnalysis -[:HAS_SQL]-> Sql``.

    When *embed_params* and *vdb* are provided, the function then embeds
    each newly-ingested analysis (name + description + SQL) and **appends**
    the rows to the supplied vector store — so they live alongside the rows
    the main embed pipeline writes for ``Table`` and ``Column`` nodes. The
    append semantics mean the main pipeline must run *before* this function.

    Entries with no SQL, or whose SQL doesn't resolve to any known table, are
    skipped with a warning. Must be called *after* schema ingestion so the
    parser can resolve table/column references.
    """
    from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
    from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
    from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels, Props
    from nemo_retriever.tabular_data.ingestion.services.queries import (
        parse_query_single,
    )
    from nemo_retriever.tabular_data.retrieval.data_access.graph_schemas import (
        get_schemas_by_ids,
    )
    from gsf.dal.custom_analyses import embed_custom_analyses

    analyses_path = resolve_dataset_file(database_name, "custom_analyses.json")

    if analyses_path is None:
        logger.info(
            "custom analyses file not found for database %r under %s; skipping",
            database_name,
            DEFAULT_DIR,
        )
        return

    with analyses_path.open() as f:
        analyses = json.load(f)

    if not isinstance(analyses, list) or not analyses:
        logger.info("No custom analyses to ingest from %s.", analyses_path)
        return

    schemas_ids = get_all_schemas_ids(database_name=database_name)
    if not schemas_ids:
        logger.warning(
            "No schemas found in Neo4j for database %r; skipping custom analyses.",
            database_name,
        )
        return
    schemas = get_schemas_by_ids(schemas_ids)

    before = time.time()
    logger.info(
        "Starting to ingest %d custom analyses from %s.", len(analyses), analyses_path
    )

    ingested = 0
    ingested_analysis_ids: list[str] = []
    for entry in analyses:
        name = entry.get("name", "")
        sql = (entry.get("sql") or "").strip()
        if not sql:
            logger.warning("Skipping custom analysis %r — no SQL provided.", name)
            continue

        query_obj = parse_query_single(sql=sql, dialects=[dialect], schemas=schemas)
        if query_obj is None:
            logger.warning(
                "Could not resolve any tables for custom analysis %r — skipping.",
                name,
            )
            continue

        # Match the Sql node by its full text so re-runs reuse the existing
        # node instead of creating a fresh one (which would cause duplicate
        # HAS_SQL edges from the merged CustomAnalysis node).
        query_obj.sql_node.match_props = {"sql_full_query": sql}

        # Match the CustomAnalysis node by name so re-running the script is
        # idempotent (Tables/Columns merge by id derived from their fully
        # qualified path; CustomAnalysis has no such id, so name is the
        # natural key from the JSON spec).
        analysis_node = Neo4jNode(
            name=name,
            label=Labels.CUSTOM_ANALYSIS,
            props={
                "name": name,
                "description": entry.get("description", ""),
            },
            match_props={"name": name},
        )

        edge_props = {Props.ANALYSIS_ID: analysis_node.get_id()}
        query_obj.edges.append((analysis_node, query_obj.sql_node, edge_props))

        add_query(query_obj.get_edges())
        ingested += 1
        # MERGE matches CustomAnalysis by name, so an existing node's id is kept.
        # Resolve the persisted id before embedding — analysis_node.get_id() is a
        # freshly generated UUID and will miss the Neo4j row on re-ingest.
        from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

        id_rows = cast(
            list[dict[str, Any]],
            get_neo4j_conn().query_read(
                "MATCH (ca:CustomAnalysis {name: $name}) RETURN ca.id AS id",
                parameters={"name": name},
            ),
        )
        persisted_id = id_rows[0]["id"] if id_rows and id_rows[0].get("id") else None
        if persisted_id:
            ingested_analysis_ids.append(str(persisted_id))
        else:
            logger.warning(
                "Could not resolve persisted id for custom analysis %r; "
                "embedding may be skipped.",
                name,
            )
            ingested_analysis_ids.append(analysis_node.get_id())

    logger.info(
        "Ingested %d/%d custom analyses in %.2fs.",
        ingested,
        len(analyses),
        time.time() - before,
    )

    if ingested == 0:
        return

    if embed_params is None or vdb is None:
        logger.info(
            "Skipping custom-analysis embedding: embed_params/vdb not provided."
        )
        return

    for analysis_id in ingested_analysis_ids:
        # Re-ingestion should replace, not duplicate, the semantic embedding.
        delete_by_id = getattr(vdb, "delete_by_id", None)
        if callable(delete_by_id):
            delete_by_id(analysis_id)
        embed_custom_analyses(
            embed_params,
            vdb,
            analysis_id=analysis_id,
            database_name=database_name,
        )
