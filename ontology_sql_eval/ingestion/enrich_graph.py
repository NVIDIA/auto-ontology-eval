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
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "datasets"


def _dataset_file(database_name: str, filename: str) -> Path | None:
    """Resolve a per-database enrichment file under ``datasets/``.

    Looks first at the single-DB layout ``datasets/<database_name>/<filename>``
    (WideWorldImporters), then at multi-DB layouts
    ``datasets/<benchmark>/<database_name>/<filename>`` (BIRD, FDABench).
    """
    direct = DEFAULT_DIR / database_name / filename
    if direct.is_file():
        return direct
    matches = sorted(DEFAULT_DIR.glob(f"*/{database_name}/{filename}"))
    return matches[0] if matches else None


def apply_metadata(database_name: str) -> None:
    """Stamp table/column metadata onto the Neo4j graph.

    Reads ``datasets/<database_name>/metadata.json`` (or, for multi-DB
    benchmarks, ``datasets/<benchmark>/<database_name>/metadata.json``), keyed
    by table name, and updates:

    * ``Table.description``
    * ``Column.description``
    * ``Column.sample_values`` (from the JSON's ``value_examples`` field, when
      present and non-empty)

    Tables/columns that aren't present in the graph are silently skipped
    (the MATCH simply finds nothing). Properties for which the JSON has no
    value are left untouched (``coalesce`` preserves the existing value).
    """
    from gsf.dal.datasources import apply_metadata_batch

    metadata_path = _dataset_file(database_name, "metadata.json")

    if metadata_path is None:
        logger.info(
            "No metadata.json for %r under %s — skipping enrichment.",
            database_name,
            DEFAULT_DIR,
        )
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

    # `apply_metadata_batch` writes both shapes in one call and coalesces, so a
    # curated description already in the catalog survives a batch that has
    # nothing to say about it -- the same semantics the two Cypher statements
    # had with `coalesce(row.description, t.description)`.
    apply_metadata_batch(database_name, table_rows, column_rows)

    logger.info(
        "Applied metadata: %d table description(s), %d column description(s), "
        "%d column sample_values from %s",
        len(table_rows),
        sum(1 for r in column_rows if r.get("description")),
        samples_count,
        metadata_path,
    )


def add_custom_analyses(
    database_name: str,
    dialect: str,  # noqa: ARG001 — kept for call-site compatibility; GSF resolves
    # the dialect from the connector itself now.
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
    from gsf.dal.custom_analyses import embed_custom_analyses
    from gsf.server.custom_analyses.service import (
        CustomAnalysisNameConflict,
        CustomAnalysisSqlConflict,
        CustomAnalysisSqlError,
        create_custom_analysis,
    )

    analyses_path = _dataset_file(database_name, "custom_analyses.json")

    if analyses_path is None:
        logger.info(
            "No custom_analyses.json for %r under %s; skipping",
            database_name,
            DEFAULT_DIR,
        )
        return

    with analyses_path.open() as f:
        analyses = json.load(f)

    if not isinstance(analyses, list) or not analyses:
        logger.info("No custom analyses to ingest from %s.", analyses_path)
        return

    before = time.time()
    logger.info(
        "Starting to ingest %d custom analyses from %s.", len(analyses), analyses_path
    )

    # GSF's service owns the whole sequence: it parses the SQL against the
    # ingested catalog, creates the Sql row and its Table/Column links, creates
    # the CustomAnalysis, and joins them. Previously this file assembled a
    # Neo4jNode and edge tuples by hand and called the library's `add_query`,
    # neither of which exists now that the catalog is relational.
    #
    # Idempotency is preserved and is now the service's job: it raises on a
    # duplicate name or a statement already attached to another analysis, so a
    # re-run reports "already present" instead of duplicating rows.
    # Embed once at the end, not once per analysis -- unless there is no batch
    # pass to defer to. `create_custom_analysis` embeds inline by default, which
    # is right for a user creating one analysis and wrong here: each embed is a
    # network round trip (~2s), so nineteen analyses cost ~40s of pure latency.
    # Worse, embedding inline *and* running the batch below writes every
    # analysis into the semantic index twice, and nothing there dedupes -- a
    # duplicated analysis just occupies two of retrieval's top-k slots.
    will_batch = embed_params is not None and vdb is not None

    ingested = skipped = 0
    for entry in analyses:
        name = entry.get("name", "")
        sql = (entry.get("sql") or "").strip()
        if not sql:
            logger.warning("Skipping custom analysis %r — no SQL provided.", name)
            continue
        try:
            create_custom_analysis(
                name=name,
                description=entry.get("description", ""),
                sql=sql,
                embed=not will_batch,
            )
            ingested += 1
        except (CustomAnalysisNameConflict, CustomAnalysisSqlConflict):
            skipped += 1
        except CustomAnalysisSqlError as exc:
            logger.warning(
                "Could not resolve any tables for custom analysis %r — skipping (%s).",
                name,
                exc,
            )

    if skipped:
        logger.info(
            "%d custom analysis/analyses already present — left alone.", skipped
        )

    logger.info(
        "Ingested %d/%d custom analyses in %.2fs.",
        ingested,
        len(analyses),
        time.time() - before,
    )

    if ingested == 0:
        return

    if not will_batch:
        # Already embedded inline above, one at a time -- the slow path, taken
        # only when there is no vdb to batch into.
        logger.info("Custom analyses embedded inline: embed_params/vdb not provided.")
        return

    embed_custom_analyses(embed_params, vdb, database_name=database_name)
