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
                    "description": "...",
                    "value_examples": ["...", ...] | null,
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
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "datasets"


def apply_metadata(database_name: str) -> None:
    """Stamp table/column metadata onto the Neo4j graph.

    Reads ``<this dir>/<database_name>/metadata.json`` (keyed by table name) and
    updates the following properties for every table/column belonging to
    *database_name*:

    * ``Table.description``
    * ``Column.description``
    * ``Column.sample_values`` (from the JSON's ``value_examples`` field, when
      present and non-empty)

    Tables/columns that aren't present in the graph are silently skipped
    (the MATCH simply finds nothing). Properties for which the JSON has no
    value are left untouched (``coalesce`` preserves the existing value).
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    metadata_path = DEFAULT_DIR / database_name / "metadata.json"

    if not metadata_path.exists():
        logger.info("No metadata file at %s — skipping enrichment.", metadata_path)
        return

    with metadata_path.open() as f:
        raw = json.load(f)

    table_rows: list[dict[str, str]] = []
    column_rows: list[dict[str, str | list[str] | None]] = []
    samples_count = 0
    for table_name, table_meta in raw.items():
        table_desc = table_meta.get("description")
        if table_desc:
            table_rows.append({"table_name": table_name, "description": table_desc})

        for col in table_meta.get("columns", []) or []:
            col_desc = col.get("description")
            value_examples = col.get("value_examples")
            sample_values: list[str] | None = (
                [str(v) for v in value_examples]
                if isinstance(value_examples, list) and value_examples
                else None
            )
            if not col_desc and sample_values is None:
                continue
            if sample_values is not None:
                samples_count += 1
            column_rows.append(
                {
                    "table_name": table_name,
                    "column_name": col["name"],
                    "description": col_desc or None,
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
                "    c.sample_values = coalesce(row.sample_values, c.sample_values)"
            ),
            parameters={"rows": column_rows, "database_name": database_name},
        )

    logger.info(
        "Applied metadata: %d table description(s), %d column description(s), "
        "%d column sample_values from %s",
        len(table_rows),
        sum(1 for r in column_rows if r.get("description")),
        samples_count,
        metadata_path,
    )


def _format_sample_value(value: object, *, max_len: int = 60) -> str | None:
    """Normalize a raw cell value to a short display string, or ``None`` to skip."""
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if len(text) > max_len:
        text = text[: max_len - 1] + "\u2026"
    return text


def _collect_column_samples(
    frame: "pd.DataFrame", *, max_values: int
) -> dict[str, list[str]]:
    """Pull up to *max_values* distinct, non-null display values per column."""
    samples: dict[str, list[str]] = {}
    for column in frame.columns:
        seen: list[str] = []
        for raw in frame[column].tolist():
            formatted = _format_sample_value(raw)
            if formatted is None or formatted in seen:
                continue
            seen.append(formatted)
            if len(seen) >= max_values:
                break
        if seen:
            samples[str(column)] = seen
    return samples


def backfill_sample_values(
    database_name: str,
    connector: object,
    *,
    sample_row_limit: int = 200,
    max_values_per_column: int = 5,
) -> None:
    """Sample real values from the source DB onto ``Column.sample_values`` nodes.

    Tabular ingest records column names/types but no example values, so the
    text-to-SQL prompt can't tell that e.g. a ``coordinates`` TEXT column holds
    ``(lon,lat)`` tuples rather than JSON. This scans up to *sample_row_limit*
    rows per table and stores a few distinct non-null values per column, giving
    the model the actual value shape to parse against.

    Existing sample values (e.g. curated ``value_examples`` from
    ``metadata.json`` applied by :func:`apply_metadata`) are preserved — this
    only fills columns that don't already have them.
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    dialect = str(getattr(connector, "dialect", "") or "").lower()
    execute = getattr(connector, "execute", None)
    get_columns = getattr(connector, "get_columns", None)
    if not callable(execute) or not callable(get_columns):
        logger.info(
            "Connector for %s exposes no execute/get_columns; "
            "skipping sample-value backfill.",
            database_name,
        )
        return

    try:
        columns_df = get_columns()
    except Exception:
        logger.warning(
            "Could not list columns for %s; skipping sample-value backfill.",
            database_name,
            exc_info=True,
        )
        return
    if columns_df is None or columns_df.empty:
        return

    column_rows: list[dict[str, object]] = []
    for (schema_name, table_name), _group in columns_df.groupby(
        ["table_schema", "table_name"], sort=False
    ):
        if dialect == "sqlite":
            ref = f'"{table_name}"'
        elif schema_name:
            ref = f'"{schema_name}"."{table_name}"'
        else:
            ref = f'"{table_name}"'

        try:
            frame = execute(f"SELECT * FROM {ref} LIMIT {int(sample_row_limit)}")
        except Exception:
            logger.debug(
                "Sampling failed for %s; skipping table.", ref, exc_info=True
            )
            continue
        if frame is None or frame.empty:
            continue

        for col_name, values in _collect_column_samples(
            frame, max_values=max_values_per_column
        ).items():
            column_rows.append(
                {
                    "table_name": str(table_name),
                    "column_name": col_name,
                    "sample_values": values,
                }
            )

    if not column_rows:
        logger.info("No sample values collected for %s.", database_name)
        return

    conn = get_neo4j_conn()
    conn.query_write(
        query=(
            "UNWIND $rows AS row "
            "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
            "(:Schema)-[:CONTAINS]->(t:Table {name: row.table_name})"
            "-[:CONTAINS]->(c:Column {name: row.column_name}) "
            "SET c.sample_values = coalesce(c.sample_values, row.sample_values)"
        ),
        parameters={"rows": column_rows, "database_name": database_name},
    )
    logger.info(
        "Backfilled sample values for %d column(s) in %s.",
        len(column_rows),
        database_name,
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
        get_all_schemas_ids,
        get_schemas_by_ids,
    )
    from gsf.dal.custom_analyses import embed_custom_analyses

    analyses_path = DEFAULT_DIR / database_name / "custom_analyses.json"

    if not analyses_path.exists():
        logger.info("custom analyses file not found at %s; skipping", analyses_path)
        return

    with analyses_path.open() as f:
        analyses = json.load(f)

    if not isinstance(analyses, list) or not analyses:
        logger.info("No custom analyses to ingest from %s.", analyses_path)
        return

    schemas_ids = get_all_schemas_ids()
    schemas = get_schemas_by_ids(schemas_ids)

    before = time.time()
    logger.info(
        "Starting to ingest %d custom analyses from %s.", len(analyses), analyses_path
    )

    ingested = 0
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

    embed_custom_analyses(embed_params, vdb, database_name=database_name)
