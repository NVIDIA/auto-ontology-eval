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

Both files live beside their database: ``datasets/<dataset>/dev/<database_name>/``
for a multi-database dataset such as BIRD, or ``datasets/<database_name>/`` for a
standalone one. See :func:`metadata_json_path` and
:func:`custom_analyses_json_path`.

Custom analyses (optional, ``custom_analyses.json``)::

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
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "datasets"


def train_json_for_dataset(dataset_name: str) -> Path:
    """Return ``datasets/<dataset_name>/train/train.json``."""
    return DEFAULT_DIR / dataset_name / "train" / "train.json"


def _dataset_file_path(
    filename: str, database_name: str, dataset: str | None = None
) -> Path:
    """Resolve a per-database data file, preferring paths that exist.

    A standalone dataset is one database and keeps its files at the dataset root
    (``datasets/dor_prod/``); a multi-database dataset gives each database its
    own folder (``datasets/bird/dev/card_games/``). Both layouts are searched,
    the explicitly named *dataset* first.

    When the caller does not name a dataset, the per-database folder of every
    dataset is searched as a last resort, so an ingest run that omits
    ``--dataset-name`` still finds the file instead of silently skipping it.
    Returns the first existing path, else the most likely one for an error
    message.
    """
    candidates: list[Path] = []
    if dataset:
        dev_root = DEFAULT_DIR / dataset / "dev"
        if dev_root.is_dir():
            candidates.append(dev_root / database_name / filename)
        candidates.append(DEFAULT_DIR / dataset / database_name / filename)
    candidates.append(DEFAULT_DIR / database_name / filename)
    if not dataset:
        candidates.extend(sorted(DEFAULT_DIR.glob(f"*/dev/{database_name}/{filename}")))

    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def metadata_json_path(database_name: str, dataset: str | None = None) -> Path:
    """Resolve ``metadata.json`` for *database_name*. See :func:`_dataset_file_path`."""
    return _dataset_file_path("metadata.json", database_name, dataset)


def custom_analyses_json_path(database_name: str, dataset: str | None = None) -> Path:
    """Resolve ``custom_analyses.json`` for *database_name*. See :func:`_dataset_file_path`."""
    return _dataset_file_path("custom_analyses.json", database_name, dataset)


def _existing_few_shot_questions(vdb, label: str, database_name: str) -> set[str]:
    """Return normalized questions already stored in the semantic VDB."""
    import psycopg
    from psycopg import sql

    if not vdb._table_exists():
        return set()

    query = sql.SQL(
        """
        SELECT langchain_metadata ->> 'question'
        FROM {table}
        WHERE {db_col} = %s AND {label_col} = %s
        """
    ).format(
        table=sql.Identifier(vdb.schema_name, vdb.collection_name),
        db_col=sql.Identifier("database_name"),
        label_col=sql.Identifier("label"),
    )
    with psycopg.connect(vdb.connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (database_name, label))
            return {
                str(row[0]).strip().casefold()
                for row in cur.fetchall()
                if row[0] and str(row[0]).strip()
            }


def add_few_shot_examples(
    *,
    train_json: Path,
    embed_params: "EmbedParams",
    vdb: "VDB",
    batch_size: int = 64,
) -> int:
    """Mask and embed new Train Q→SQL examples into ``train_qa``.

    *train_json* must be an explicit corpus file (typically
    ``datasets/<dataset>/train/train.json``). Existing questions are read from
    Postgres and skipped, so repeated ingest runs are incremental.
    """
    from gsf.retrieval.text_to_sql.question_masking import mask_question
    from gsf.semantic.constants import FEW_SHOT_DATABASE_NAME, LABEL_FEW_SHOT_QA
    from gsf.utils.embedding import embed_docs_into_vdb

    if not train_json.is_file():
        logger.info("Few-shot corpus not found at %s; skipping.", train_json)
        return 0

    with train_json.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{train_json} must contain a JSON list")

    existing = _existing_few_shot_questions(
        vdb,
        LABEL_FEW_SHOT_QA,
        FEW_SHOT_DATABASE_NAME,
    )
    docs: list[dict] = []
    skipped_existing = 0
    skipped_empty = 0
    dataset_name = (
        train_json.parents[1].name
        if train_json.parent.name == "train"
        else train_json.stem
    )

    for row in rows:
        question = str(row.get("question") or "").strip()
        sql = str(row.get("SQL") or row.get("sql") or "").strip()
        if not question or not sql:
            skipped_empty += 1
            continue

        normalized = question.casefold()
        if normalized in existing:
            skipped_existing += 1
            continue
        existing.add(normalized)

        masked = mask_question(question) or question
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:24]
        docs.append(
            {
                "id": f"{dataset_name}:train:{digest}",
                "name": question,
                "label": LABEL_FEW_SHOT_QA,
                "text": masked,
                "masked_question": masked,
                "question": question,
                "sql": sql,
                "evidence": str(row.get("evidence") or ""),
                "db_id": str(row.get("db_id") or ""),
            }
        )

    if not docs:
        logger.info(
            "Few-shot enrichment: nothing new from %s (%d existing, %d empty).",
            train_json,
            skipped_existing,
            skipped_empty,
        )
        return 0

    logger.info(
        "Few-shot enrichment: embedding %d new question(s) from %s "
        "(skipped %d existing, %d empty).",
        len(docs),
        train_json,
        skipped_existing,
        skipped_empty,
    )
    written = 0
    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        written += embed_docs_into_vdb(
            chunk,
            embed_params,
            vdb,
            database_name=FEW_SHOT_DATABASE_NAME,
        )
        logger.info(
            "Few-shot enrichment progress: %d/%d",
            min(start + len(chunk), len(docs)),
            len(docs),
        )
    return written


def apply_metadata(database_name: str, dataset: str | None = None) -> None:
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

    metadata_path = metadata_json_path(database_name, dataset=dataset)

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


def profile_and_describe_columns(connector, database_name: str) -> int:
    """Profile every column and describe the ones worth describing.

    Which those are is ``SEMANTIC_DESCRIBE_MODE``'s decision, made in
    :func:`gsf.semantic.deterministic.columns_to_describe`; unset, it means the
    ones the source metadata left blank.

    Must run after :func:`apply_metadata` (so a supplied annotation is in view
    when deciding, and re-stamped ahead of any description this replaces) and
    before the embed graph (so the descriptions it writes are part of what the
    Column nodes are embedded from). The semantic layer then only copies them
    onto ColumnAttributes.

    Profiling has to happen here rather than in the semantic layer because a
    description is only worth generating with the column's values in view, and the
    embeddings are computed at ingest. Doing it here also reaches the foreign-key
    columns, which the semantic layer excludes from description generation by
    design — they are linked to the attribute they reference instead of owning one.

    Returns the number of descriptions written.
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    from gsf.dal.datasources import store_column_descriptions
    from gsf.semantic.deterministic import (
        blank_column_descriptions,
        columns_to_describe,
        describe_mode,
    )
    from gsf.semantic.visit_enter import calculate_columns_profiling

    def _profile(
        table: dict[str, object], columns: list[dict[str, object]]
    ) -> dict[str, dict]:
        """Profile one table, degrading to no values rather than aborting ingest."""
        try:
            return calculate_columns_profiling(table, columns, connector)
        except Exception:
            logger.warning(
                "profiling failed for %s — describing without values",
                table.get("name"),
                exc_info=True,
            )
            return {}

    # Reads ``c.description`` rather than going through the DAL's table fetch,
    # which resolves a column's description through its ColumnAttribute and would
    # therefore report every column as documented on a re-run over a graph the
    # semantic layer had already populated.
    rows = get_neo4j_conn().query_read(
        query=(
            "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
            "(s:Schema)-[:CONTAINS]->(t:Table)-[:CONTAINS]->(c:Column) "
            "WITH t, s, c ORDER BY c.ordinal_position "
            "RETURN t.id AS table_id, t.name AS table_name, "
            "       s.name AS schema_name, "
            "       collect({name: c.name, data_type: c.data_type, "
            "                description: c.description}) AS columns "
            "ORDER BY table_name"
        ),
        parameters={"database_name": database_name},
    )

    mode = describe_mode()
    written = 0
    target_total = 0
    for row in rows:
        columns = [c for c in row.get("columns") or [] if c.get("name")]
        if not columns:
            continue
        # Asking the same question the describe step will ask, rather than
        # counting blanks: under the wider modes a table with no blank column
        # still has work, and a mismatch here silently skips it.
        wanted = columns_to_describe(columns, mode)
        target_total += len(wanted)
        table = {
            "id": row["table_id"],
            "name": row.get("table_name") or "",
            "schema_name": row.get("schema_name"),
        }
        if not wanted:
            # Still profiled: sample values and uniqueness are persisted by the
            # profiler, and the embed graph below reads them off the Column nodes.
            _profile(table, columns)
            continue
        profiling = _profile(table, columns)
        descriptions = blank_column_descriptions(
            columns, profiling, table_name=table["name"]
        )
        store_column_descriptions(table["id"], descriptions)
        written += len(descriptions)

    logger.info(
        "Column descriptions (mode=%s): %d of %d targeted column(s) described "
        "at ingest",
        mode,
        written,
        target_total,
    )
    return written


def sync_graph_metadata_into_schema_data(
    schema_data: tuple, database_name: str
) -> tuple:
    """Copy graph descriptions and sample values back into the embed input.

    ``TabularSchemaExtractOp`` returns ``(tables_df, columns_df)`` and
    ``TabularFetchEmbeddingsOp`` builds its text from that pair directly, "without
    a Neo4j round-trip" — so anything written to the graph *after* extraction is
    invisible to the embeddings. For a SQLite source that is everything worth
    embedding: the introspected DataFrames carry no descriptions at all, and both
    :func:`apply_metadata` and :func:`profile_and_describe_columns` write only to
    Neo4j. Without this step a Column is embedded as name, type and nothing else.

    Joins on the Neo4j UUID that extraction already placed in each frame's ``id``
    column, so it is immune to name-casing and duplicate table names across
    schemas. Returns the patched pair; frames missing ``id`` are passed through.
    """
    import pandas as pd
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    tables_df, columns_df = schema_data
    conn = get_neo4j_conn()

    def _apply(df, rows: list[dict], fields: tuple[str, ...]):
        if df is None or getattr(df, "empty", True) or "id" not in df.columns:
            return df, 0
        by_id = {r["id"]: r for r in rows if r.get("id")}
        patched = 0
        for field in fields:
            # object dtype so a list assignment is legal, and so an all-empty
            # column is not inferred as float64 (whose NaN is truthy — the embed
            # operator does `(sample_values or [])[:5]` and would raise on it).
            df[field] = (
                df[field].astype(object) if field in df.columns else pd.Series(
                    [None] * len(df), index=df.index, dtype=object
                )
            )
        for idx, node_id in df["id"].items():
            row = by_id.get(node_id)
            if not row:
                continue
            for field in fields:
                value = row.get(field)
                if field == "sample_values" and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = None
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                df.at[idx, field] = value
                if field == "description":
                    patched += 1
        if "sample_values" in fields:
            # NaN reaches the embed operator as a truthy float and breaks it; None
            # is what its `or []` fallback expects.
            df["sample_values"] = df["sample_values"].apply(
                lambda v: v if isinstance(v, list) else None
            )
        return df, patched

    table_rows = conn.query_read(
        query=(
            "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
            "(:Schema)-[:CONTAINS]->(t:Table) "
            "RETURN t.id AS id, t.description AS description"
        ),
        parameters={"database_name": database_name},
    )
    column_rows = conn.query_read(
        query=(
            "MATCH (d:Database {name: $database_name})-[:CONTAINS]->"
            "(:Schema)-[:CONTAINS]->(:Table)-[:CONTAINS]->(c:Column) "
            "RETURN c.id AS id, c.description AS description, "
            "       c.sample_values AS sample_values"
        ),
        parameters={"database_name": database_name},
    )

    tables_df, n_tables = _apply(tables_df, table_rows, ("description",))
    columns_df, n_columns = _apply(
        columns_df, column_rows, ("description", "sample_values")
    )

    logger.info(
        "Synced graph metadata into embed input: %d table and %d column "
        "description(s)",
        n_tables,
        n_columns,
    )
    return tables_df, columns_df


def add_custom_analyses(
    database_name: str,
    dialect: str,
    embed_params: "EmbedParams | None" = None,
    vdb: "VDB | None" = None,
    dataset: str | None = None,
) -> None:
    """Ingest custom analyses for *database_name* into the Neo4j graph and the VDB.

    Reads the database's ``custom_analyses.json`` (see
    :func:`custom_analyses_json_path`) — a list of
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

    analyses_path = custom_analyses_json_path(database_name, dataset=dataset)

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
