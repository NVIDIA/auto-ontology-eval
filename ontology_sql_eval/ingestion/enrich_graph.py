# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp table/column metadata onto the Neo4j graph.

This module reads ``metadata.json`` and writes descriptions and sample values
onto the ``Table`` and ``Column`` nodes that the tabular ingest pipeline created
in Neo4j. It is intentionally a small, dev-tools-only helper and is meant to be
invoked at the end of an ingest run.

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

Custom analyses (optional, ``custom_analyses.json``)::

    [
        {
            "name": "...",
            "description": "...",
            "sql": "SELECT ..."
        },
        ...
    ]

Both files above describe the *downloaded / seeded* dataset. Our own analyses
and table/column description corrections are kept outside
``datasets/spider2/``, which is gitignored and replaced on re-seed. Name them
in ``.env``::

    SAVED_CUSTOM_ANALYSES_DIR=annotations/spider2/custom_analyses
    SAVED_METADATA_DIR=annotations/spider2/metadata
    SAVED_DESCRIPTIONS_CSV=annotations/spider2/semantic_descriptions.csv

* ``SAVED_CUSTOM_ANALYSES_DIR`` — directory of ``<database_name>.json`` CA
  specs; wins over beside-DB ``custom_analyses.json`` at ingest.
* ``SAVED_METADATA_DIR`` — directory of ``<database_name>.json`` description
  overlays merged into seeded ``metadata.json`` at ingest (table/column
  ``description`` fields only; types/samples still come from the seed file).
* ``SAVED_DESCRIPTIONS_CSV`` — optional column/ColumnAttribute overrides
  applied after semantic compile with ``--override-descriptions``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "datasets"

SAVED_DESCRIPTIONS_ENV = "SAVED_DESCRIPTIONS_CSV"
SAVED_ANALYSES_DIR_ENV = "SAVED_CUSTOM_ANALYSES_DIR"
SAVED_METADATA_DIR_ENV = "SAVED_METADATA_DIR"

SAVED_DESCRIPTION_FIELDS = (
    "database",
    "table",
    "column",
    "column_description",
    "column_attribute",
    "column_attribute_description",
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _repo_relative(value: str) -> Path:
    """Resolve *value* against the repository root when it is not absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


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


def custom_analyses_json_path(database_name: str) -> Path | None:
    """Resolve *database_name*'s custom analyses, or ``None`` when it has none.

    ``SAVED_CUSTOM_ANALYSES_DIR`` wins when set, as a directory of
    ``<database_name>.json`` specs — durable annotations that outlive
    ``datasets/spider2/``. Unset, the database's own ``custom_analyses.json``
    is used (see :func:`resolve_dataset_file`).
    """
    named = os.environ.get(SAVED_ANALYSES_DIR_ENV, "").strip()
    if named:
        path = _repo_relative(named) / f"{database_name}.json"
        if not path.is_file():
            logger.warning("No analyses spec for %s at %s", database_name, path)
        return path if path.is_file() else None

    return resolve_dataset_file(database_name, "custom_analyses.json")


def saved_metadata_json_path(database_name: str) -> Path | None:
    """Resolve a durable metadata description overlay for *database_name*."""
    named = os.environ.get(SAVED_METADATA_DIR_ENV, "").strip()
    if not named:
        return None
    path = _repo_relative(named) / f"{database_name}.json"
    return path if path.is_file() else None


def _merge_metadata_descriptions(
    base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    """Overlay table/column ``description`` fields from *overlay* onto *base*."""
    merged = dict(base)
    for table_name, table_meta in overlay.items():
        if not isinstance(table_meta, dict):
            continue
        target = dict(merged.get(table_name) or {})
        if table_meta.get("description"):
            target["description"] = table_meta["description"]
        overlay_cols = {
            str(col.get("name")): col
            for col in (table_meta.get("columns") or [])
            if isinstance(col, dict) and col.get("name")
        }
        if overlay_cols:
            columns = [dict(col) for col in (target.get("columns") or [])]
            by_name = {str(col.get("name")): col for col in columns if col.get("name")}
            for name, overlay_col in overlay_cols.items():
                if name in by_name:
                    if overlay_col.get("description"):
                        by_name[name]["description"] = overlay_col["description"]
                else:
                    columns.append(
                        {
                            "name": name,
                            "description": overlay_col.get("description"),
                            "data_type": overlay_col.get("data_type"),
                            "value_examples": overlay_col.get("value_examples"),
                        }
                    )
                    by_name[name] = columns[-1]
            target["columns"] = columns
        merged[table_name] = target
    return merged


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
    overlay_path = saved_metadata_json_path(database_name)

    if metadata_path is None and overlay_path is None:
        logger.info(
            "No metadata file for database %r under %s — skipping enrichment.",
            database_name,
            DEFAULT_DIR,
        )
        return

    raw: dict[str, Any] = {}
    if metadata_path is not None:
        with metadata_path.open() as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            raw = loaded
        source_label = str(metadata_path)
    else:
        source_label = "(overlay only)"

    if overlay_path is not None:
        with overlay_path.open() as f:
            overlay = json.load(f)
        if isinstance(overlay, dict):
            raw = _merge_metadata_descriptions(raw, overlay)
            source_label = f"{source_label} + {overlay_path}"
            logger.info(
                "Merged saved metadata descriptions for %s from %s",
                database_name,
                overlay_path,
            )

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
        source_label,
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

    analyses_path = custom_analyses_json_path(database_name)

    if analyses_path is None:
        logger.info(
            "custom analyses file not found for database %r; skipping",
            database_name,
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


# ---------------------------------------------------------------------------
# Writing a saved description set over a freshly compiled graph
# ---------------------------------------------------------------------------
#
# ColumnAttribute nodes exist only after semantic compile, so column/attribute
# overrides from SAVED_DESCRIPTIONS_CSV are applied there (see semantic.py
# --override-descriptions), not during ingest.

_LIVE_COLUMNS = """
UNWIND $rows AS row
MATCH (d:Database {name: $database_name})-[:CONTAINS]->(:Schema)
      -[:CONTAINS]->(t:Table {name: row.table_name})
      -[:CONTAINS]->(c:Column {name: row.column_name})
OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:ColumnAttribute)
RETURN row.table_name AS table_name, row.column_name AS column_name,
       c.id AS column_id, c.description AS column_description,
       collect(CASE WHEN a IS NULL THEN NULL ELSE {
           id: a.id, name: a.name, term_name: a.term_name,
           source_column: a.source_column, description: a.description
       } END) AS attributes
"""

_SET_ATTRIBUTE_DESCRIPTIONS = """
UNWIND $rows AS row
MATCH (a:ColumnAttribute {id: row.id})
SET a.description = row.description
"""


def saved_descriptions_csv_path(
    database_name: str,
    explicit: str | Path | None = None,
) -> Path | None:
    """Resolve the saved description CSV for *database_name*, or ``None``.

    Order: *explicit* / ``SAVED_DESCRIPTIONS_CSV``, then
    ``semantic_descriptions.csv`` beside the database's ``metadata.json``.
    """
    named = str(explicit or os.environ.get(SAVED_DESCRIPTIONS_ENV, "")).strip()
    if named:
        path = _repo_relative(named)
        if not path.is_file():
            logger.warning("Saved description set %s does not exist", path)
        return path if path.is_file() else None

    beside_database = resolve_dataset_file(database_name, "semantic_descriptions.csv")
    return beside_database


def _squash(text: str | None) -> str:
    """Collapse whitespace, so a reflowed line does not read as a real change."""
    return " ".join((text or "").split())


def _read_saved_descriptions(
    csv_path: Path, database_name: str
) -> dict[tuple[str, str], dict[str, str]]:
    """The saved rows for one database, keyed by ``(table, column)``."""
    saved: dict[tuple[str, str], dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = [f for f in SAVED_DESCRIPTION_FIELDS if f not in fields]
        if missing:
            raise ValueError(f"{csv_path} is missing column(s): {', '.join(missing)}")
        for row in reader:
            if (row.get("database") or "").strip() != database_name:
                continue
            table = (row.get("table") or "").strip()
            column = (row.get("column") or "").strip()
            if not table or not column:
                continue
            saved[(table, column)] = {
                "column_description": row.get("column_description") or "",
                "attribute_name": (row.get("column_attribute") or "").strip(),
                "attribute_description": row.get("column_attribute_description") or "",
            }
    return saved


def apply_saved_descriptions(
    database_name: str,
    csv_path: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> int:
    """Overwrite column/attribute descriptions with a saved CSV set.

    Run after the semantic layer is compiled. Returns the number of nodes written.
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    from gsf.semantic.embed import build_semantic_embedder
    from gsf.server.datasources.service import update_node_properties

    resolved = saved_descriptions_csv_path(database_name, csv_path)
    if resolved is None:
        logger.info(
            "No saved description set for %s (%s unset, none beside its "
            "metadata.json) — leaving descriptions as ingested.",
            database_name,
            SAVED_DESCRIPTIONS_ENV,
        )
        return 0
    csv_path = resolved

    saved = _read_saved_descriptions(csv_path, database_name)
    if not saved:
        logger.info("%s holds no rows for %s — skipping.", csv_path, database_name)
        return 0

    conn = get_neo4j_conn()
    rows = cast(
        list[dict[str, Any]],
        conn.query_read(
            _LIVE_COLUMNS,
            parameters={
                "database_name": database_name,
                "rows": [
                    {"table_name": table, "column_name": column}
                    for table, column in sorted(saved)
                ],
            },
        ),
    )
    live = {(r["table_name"], r["column_name"]): r for r in rows}

    column_writes: list[tuple[tuple[str, str], str, str]] = []
    attribute_writes: list[tuple[tuple[str, str], dict[str, Any]]] = []
    renamed: list[tuple[tuple[str, str], str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    without_attribute: list[tuple[str, str]] = []

    for key in sorted(saved):
        record, row = saved[key], live.get(key)
        if row is None:
            continue
        wanted = record["column_description"]
        if wanted and _squash(wanted) != _squash(row["column_description"]):
            column_writes.append((key, row["column_id"], wanted))

        wanted_attribute = record["attribute_description"]
        attributes = row["attributes"] or []
        if not wanted_attribute:
            continue
        if not attributes:
            without_attribute.append(key)
            continue
        if len(attributes) > 1:
            ambiguous.append(key)
            continue
        attribute = attributes[0]
        if record["attribute_name"] and record["attribute_name"] != attribute["name"]:
            renamed.append((key, record["attribute_name"], attribute["name"]))
        if _squash(wanted_attribute) != _squash(attribute["description"]):
            attribute_writes.append(
                (key, {**attribute, "description": wanted_attribute})
            )

    logger.info(
        "Saved descriptions for %s from %s: %d saved column(s), %d matched in the "
        "graph, %d column description(s) and %d attribute description(s) to write",
        database_name,
        csv_path,
        len(saved),
        len(live),
        len(column_writes),
        len(attribute_writes),
    )
    for key, _column_id, wanted in column_writes:
        logger.info("  column %s.%s", *key)
        logger.info("      was: %s", _squash(live[key]["column_description"])[:150])
        logger.info("      now: %s", _squash(wanted)[:150])
    for key, attribute in attribute_writes:
        logger.info("  attribute %s.%s (%s)", key[0], key[1], attribute["name"])
        logger.info("      now: %s", _squash(attribute["description"])[:150])
    for key, was, now in renamed:
        logger.info(
            "  %s.%s: attribute renamed by the rebuild, %r -> %r; matched by column",
            key[0],
            key[1],
            was,
            now,
        )
    for label, keys in (
        ("absent from the graph", sorted(set(saved) - set(live))),
        ("no ColumnAttribute", without_attribute),
        ("several ColumnAttributes, left alone", ambiguous),
    ):
        if keys:
            logger.info(
                "  %d column(s) %s: %s",
                len(keys),
                label,
                ", ".join(f"{t}.{c}" for t, c in keys[:10]),
            )

    if dry_run:
        logger.info("dry run — nothing written")
        return 0

    for key, column_id, wanted in column_writes:
        if not update_node_properties(column_id, {"description": wanted}):
            logger.warning("  %s.%s: column patch reported no change", *key)

    if attribute_writes:
        conn.query_write(
            _SET_ATTRIBUTE_DESCRIPTIONS,
            parameters={
                "rows": [
                    {"id": attribute["id"], "description": attribute["description"]}
                    for _key, attribute in attribute_writes
                ]
            },
        )
        embedder = build_semantic_embedder(database_name, reset=False)
        if embedder is None:
            logger.warning(
                "Semantic embedding disabled — %d attribute description(s) written to "
                "the graph but the vector index still serves the old text",
                len(attribute_writes),
            )
        else:
            for _key, attribute in attribute_writes:
                embedder.vdb.delete_by_id(attribute["id"])
            written = embedder.embed_column_attributes(
                [attribute for _key, attribute in attribute_writes]
            )
            logger.info("re-embedded %d attribute row(s)", written)

    total = len(column_writes) + len(attribute_writes)
    logger.info("Applied saved descriptions: %d node(s) written", total)
    return total
