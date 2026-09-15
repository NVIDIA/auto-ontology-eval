# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp table/column metadata onto the Postgres catalog.

This module reads ``<database_name>/metadata.json`` and writes descriptions and
sample values onto the ``Table`` and ``Column`` rows that the tabular ingest
pipeline created in Postgres. It is intentionally a small, dev-tools-only helper
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

Both files above describe the *downloaded* dataset. Our own analyses and our
corrections to its annotations live in ``annotations/<dataset>/`` instead, since
``datasets/bird/`` is downloaded and replaced wholesale and cannot hold anything
that has to survive::

    annotations/<dataset>/custom_analyses/<database_name>.json
    annotations/<dataset>/semantic_descriptions.csv   # one export, every database

Nothing has to be configured to use them: both are found by that layout and take
precedence over the dataset's own copies. Each is picked up by the stage that can
use it. The analyses go in during ingest, which is where analyses always come from
(see :func:`custom_analyses_json_path`). The descriptions have to wait for the
semantic compile, since half of them belong to ColumnAttribute nodes that do not
exist until it runs, so the compile applies them when asked::

    python -m ontology_sql_eval.ingestion.ingest --dataset-name bird
    python -m ontology_sql_eval.ingestion.semantic --override-descriptions

Without the flag the compile leaves its own text in place. Re-run the same command
after editing the saved set to push the edit into the graph and the vector store;
the compile itself only revisits tables that have no ``Term``, so a second pass is
cheap. The export's header is::

    database,schema,table,column,column_description,column_attribute,
    column_attribute_description,term,term_description,term_synonyms

Only the column and attribute description fields are read; the rest identify the
row. See :func:`apply_saved_descriptions`.
"""
from __future__ import annotations

import csv
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "datasets"
ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "annotations"
TRAIN_QA_COLLECTION_NAME = "train_qa"
TRAIN_QA_DATABASE_NAME = "train_qa"
TRAIN_QA_LABEL = "FewShotQA"


def _removed_graph_connection() -> Any:
    """Fail closed if a retired graph-era helper is called."""
    raise RuntimeError(
        "This graph-era helper was retired by the Postgres catalog migration."
    )


def train_json_for_dataset(dataset_name: str) -> Path:
    """Return ``datasets/<dataset_name>/train/train.json``."""
    return DEFAULT_DIR / dataset_name / "train" / "train.json"


def _dataset_file_path(
    filename: str, database_name: str, dataset: str | None = None
) -> Path | None:
    """Resolve a per-database data file, or ``None`` when no layout holds it.

    A standalone dataset is one database and keeps its files at the dataset root
    (``datasets/wideworldimporters/``); a multi-database dataset gives each
    database its own folder, either under a split (``datasets/bird/dev/card_games/``)
    or directly beneath the dataset (``datasets/fdabench/<database_name>/``). All
    three layouts are searched, the explicitly named *dataset* first.

    When the caller does not name a dataset, the per-database folder of every
    dataset is searched as a last resort, so an ingest run that omits
    ``--dataset-name`` still finds the file instead of silently skipping it.
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
        candidates.extend(sorted(DEFAULT_DIR.glob(f"*/{database_name}/{filename}")))

    for path in candidates:
        if path.is_file():
            return path
    return None


def metadata_json_path(database_name: str, dataset: str | None = None) -> Path | None:
    """Resolve ``metadata.json`` for *database_name*. See :func:`_dataset_file_path`."""
    return _dataset_file_path("metadata.json", database_name, dataset)


def _repo_relative(value: str | Path) -> Path:
    """Resolve *value* against the repository root when it is not absolute.

    So one path works no matter which directory a run starts from.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _annotations_file_path(relative: str, dataset: str | None = None) -> Path | None:
    """Resolve one of *our own* annotation files, or ``None`` when we have none.

    Ours live in ``annotations/<dataset>/`` rather than beside the database, because
    ``datasets/bird/`` is downloaded, gitignored and replaced wholesale, so nothing
    put inside it survives. When the caller does not name a dataset — an ingest run
    without ``--dataset-name`` — every dataset's folder is searched, so the file is
    still found instead of being silently skipped.
    """
    if dataset:
        candidates = [ANNOTATIONS_DIR / dataset / relative]
    else:
        candidates = sorted(ANNOTATIONS_DIR.glob(f"*/{relative}"))

    for path in candidates:
        if path.is_file():
            return path
    return None


def custom_analyses_json_path(
    database_name: str, dataset: str | None = None
) -> Path | None:
    """Resolve *database_name*'s custom analyses, or ``None`` when it has none.

    Ours win: ``annotations/<dataset>/custom_analyses/<database_name>.json``, one
    spec per database (see :func:`_annotations_file_path` for why they sit there).
    Without one, the database's own ``custom_analyses.json`` is used (see
    :func:`_dataset_file_path`), which is what a hand-maintained dataset has.
    """
    return _annotations_file_path(
        f"custom_analyses/{database_name}.json", dataset
    ) or _dataset_file_path("custom_analyses.json", database_name, dataset)


def _existing_few_shot_questions(vdb, label: str, database_name: str) -> set[str]:
    """Return normalized questions already stored in the semantic VDB."""
    import psycopg
    from psycopg import sql

    if not vdb._table_exists():
        return set()

    query = sql.SQL(
        """
        SELECT COALESCE(
            langchain_metadata ->> 'question',
            langchain_metadata ->> 'name'
        )
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


def _embed_train_qa_docs(
    docs: list[dict],
    embed_params: "EmbedParams",
    vdb: "VDB",
) -> int:
    """Embed Train QA docs without dropping retrieval metadata."""
    from gsf.utils.embedding import batch_embed
    from nemo_retriever.operators.vdb import IngestVdbOperator

    rows: list[dict] = []
    for item in docs:
        node_id = item["id"]
        path = f"bird-train:{node_id}"
        metadata = {
            "id": node_id,
            "label": item["label"],
            "name": item["name"],
            "question": item["question"],
            "sql": item["sql"],
            "evidence": item["evidence"],
            "db_id": item["db_id"],
            "source_path": path,
            "database_name": TRAIN_QA_DATABASE_NAME,
        }
        rows.append(
            {
                "text": item["text"],
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {
                    **metadata,
                    "content_metadata": dict(metadata),
                },
            }
        )

    embedded = batch_embed(rows, embed_params)
    records = [
        row
        for row in embedded.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not records:
        raise RuntimeError(
            f"Embedding step produced no vectors for {len(rows)} Train QA row(s)."
        )
    IngestVdbOperator(vdb=vdb)(records)
    return len(records)


def add_few_shot_examples(
    *,
    train_json: Path,
    embed_params: "EmbedParams",
    vdb: "VDB",
    batch_size: int = 64,
) -> int:
    """Embed new Train Q→SQL examples into ``train_qa``.

    *train_json* must be an explicit corpus file (typically
    ``datasets/<dataset>/train/train.json``). Existing questions are read from
    Postgres and skipped, so repeated ingest runs are incremental. Questions
    are embedded verbatim because GSF main does not expose a masking API.
    """
    if not train_json.is_file():
        logger.info("Few-shot corpus not found at %s; skipping.", train_json)
        return 0

    with train_json.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{train_json} must contain a JSON list")

    existing = _existing_few_shot_questions(
        vdb,
        TRAIN_QA_LABEL,
        TRAIN_QA_DATABASE_NAME,
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

        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:24]
        docs.append(
            {
                "id": f"{dataset_name}:train:{digest}",
                "name": question,
                "label": TRAIN_QA_LABEL,
                "text": question,
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
        written += _embed_train_qa_docs(
            chunk,
            embed_params,
            vdb,
        )
        logger.info(
            "Few-shot enrichment progress: %d/%d",
            min(start + len(chunk), len(docs)),
            len(docs),
        )
    return written


def apply_metadata(database_name: str, dataset: str | None = None) -> None:
    """Stamp table/column metadata onto the Postgres catalog.

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

    metadata_path = metadata_json_path(database_name, dataset=dataset)

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

    apply_metadata_batch(database_name, table_rows, column_rows)

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
    from gsf.dal.datasources import store_column_descriptions  # type: ignore
    from gsf.semantic.deterministic import (
        blank_column_descriptions,  # type: ignore
        columns_to_describe,  # type: ignore
        describe_mode,  # type: ignore
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
    rows = _removed_graph_connection().query_read(
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
    a catalog round-trip" — so anything written to the old graph *after* extraction is
    invisible to the embeddings. For a SQLite source that is everything worth
    embedding: the introspected DataFrames carry no descriptions at all, and both
    :func:`apply_metadata` and :func:`profile_and_describe_columns` wrote only to
    that graph. Without this step a Column was embedded as name and type alone.

    Joins on the catalog UUID that extraction already placed in each frame's ``id``
    column, so it is immune to name-casing and duplicate table names across
    schemas. Returns the patched pair; frames missing ``id`` are passed through.
    """
    import pandas as pd
    tables_df, columns_df = schema_data
    conn = _removed_graph_connection()

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
    dialect: str,  # noqa: ARG001 — kept for call-site compatibility; GSF resolves
    # the dialect from the connector itself now.
    embed_params: "EmbedParams | None" = None,
    vdb: "VDB | None" = None,
    dataset: str | None = None,
) -> None:
    """Ingest custom analyses for *database_name* into Postgres and the VDB.

    Reads the database's analyses — ours from ``annotations/<dataset>/`` if we have
    any, else its own ``custom_analyses.json``; see
    :func:`custom_analyses_json_path` — a list of ``{"name", "description", "sql"}``
    entries, and for each entry:

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

    analyses_path = custom_analyses_json_path(database_name, dataset=dataset)
    if analyses_path is None:
        logger.info("No custom analyses for %s; skipping", database_name)
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

    assert embed_params is not None and vdb is not None
    embed_custom_analyses(embed_params, vdb, database_name=database_name)


# ---------------------------------------------------------------------------
# Writing a saved description set over a freshly compiled graph
# ---------------------------------------------------------------------------
#
# BIRD ships its own column annotations, so a graph built from a fresh download
# carries the original text. Replacing it with ours happens after the semantic
# layer is compiled — the ColumnAttribute nodes that hold the string retrieval
# embeds do not exist before then — which is why the semantic stage, not ingest,
# calls in here. See :func:`apply_saved_descriptions`.

# One row per (column, attribute), as exported by
# ``experiments/scripts/make_baseline_image_v2.py``.
SAVED_DESCRIPTION_FIELDS = (
    "database",
    "table",
    "column",
    "column_description",
    "column_attribute",
    "column_attribute_description",
)


def saved_descriptions_csv_path(
    database_name: str,
    dataset: str | None = None,
    explicit: str | Path | None = None,
) -> Path | None:
    """Resolve the saved description set for *database_name*, or ``None``.

    Three sources, in order:

    1. *explicit*, for a caller that names the file itself;
    2. ``annotations/<dataset>/semantic_descriptions.csv``. This is the one that
       matters in practice: the corrections have to outlive ``datasets/bird/``,
       which is downloaded, ignored by git and replaced wholesale, so they are kept
       outside it — one export covering every database;
    3. ``semantic_descriptions.csv`` beside the database's own ``metadata.json``,
       for a dataset whose folder is hand-maintained rather than downloaded.

    ``None`` means no set was found, and every caller reads that as "leave the
    descriptions as they were ingested" rather than as an error. A relative
    *explicit* path resolves against the repository root, not the caller's cwd.
    """
    if explicit:
        path = _repo_relative(explicit)
        if not path.is_file():
            logger.warning("Saved description set %s does not exist", path)
        return path if path.is_file() else None

    return _annotations_file_path(
        "semantic_descriptions.csv", dataset
    ) or _dataset_file_path("semantic_descriptions.csv", database_name, dataset)


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
    dataset: str | None = None,
    csv_path: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> int:
    """Overwrite descriptions with a saved set, columns and attributes alike.

    Run this after the semantic layer is compiled: the ColumnAttribute nodes it
    corrects do not exist before then. Both stores are updated, since a description
    the graph holds and the vector index does not is invisible to retrieval:

    * ``Column.description`` goes through the same call the server makes when a
      description is edited in the UI, which also deletes and re-appends the
      column's and its parent table's rows in the data-objects collection.
    * ``ColumnAttribute.description`` — the string the semantic search embeds — is
      written for the attribute hanging off each column, and its row in the
      semantic collection is deleted and re-embedded. A second ``semantic`` run
      would not revisit it: the compile only visits tables that have no ``Term``.

    The saved attribute text is written verbatim rather than recomputed from the
    column description: the file is the record of what a measured baseline read,
    and recomputing would fold in whatever the fresh profile sampled, which is the
    drift this step exists to remove.

    Columns absent from the graph, columns whose saved attribute text is empty, and
    columns carrying more than one attribute are counted and named in the log
    rather than guessed at. Returns the number of nodes written.
    """
    from gsf.dal.attributes import (
        fetch_attr_column_contexts,
        find_column_attribute_by_column_id,
        update_column_attribute,
    )
    from gsf.dal.datasources import (
        fetch_schema_ids_for_database,
        fetch_tables_and_columns_by_node_ids,
        fetch_tables_for_schema,
    )
    from gsf.semantic.embed import build_semantic_embedder
    from gsf.server.datasources.service import update_node_properties

    resolved = saved_descriptions_csv_path(database_name, dataset, csv_path)
    if resolved is None:
        logger.info(
            "No saved description set for %s (none under annotations/, none "
            "beside its metadata.json) — leaving descriptions as ingested.",
            database_name,
        )
        return 0
    csv_path = resolved

    saved = _read_saved_descriptions(csv_path, database_name)
    if not saved:
        logger.info("%s holds no rows for %s — skipping.", csv_path, database_name)
        return 0

    table_ids: list[str] = []
    for schema_id in fetch_schema_ids_for_database(database_name):
        table_ids.extend(
            str(table["id"])
            for table in fetch_tables_for_schema(
                schema_id, database_name=database_name
            )
            if table.get("id")
        )
    _tables_df, columns_df, _database_name = fetch_tables_and_columns_by_node_ids(
        table_ids
    )
    column_rows = columns_df.to_dict(orient="records")
    attr_id_by_column = {
        str(row["id"]): find_column_attribute_by_column_id(str(row["id"]))
        for row in column_rows
        if row.get("id")
    }
    attr_contexts = fetch_attr_column_contexts(
        [attr_id for attr_id in attr_id_by_column.values() if attr_id],
        database_name=database_name,
    )
    rows: list[dict] = []
    for column in column_rows:
        key = (str(column.get("table_name") or ""), str(column.get("column_name") or ""))
        if key not in saved:
            continue
        attr_id = attr_id_by_column.get(str(column.get("id") or ""))
        context = attr_contexts.get(attr_id or "")
        attributes = []
        if attr_id and context:
            attributes.append(
                {
                    "id": attr_id,
                    "name": context.get("attr_name", ""),
                    "term_id": context.get("term_id"),
                    "term_name": context.get("term_name", ""),
                    "source_column": context.get("col_name", ""),
                    "description": context.get("attr_description", ""),
                }
            )
        rows.append(
            {
                "table_name": key[0],
                "column_name": key[1],
                "column_id": str(column.get("id") or ""),
                "column_description": column.get("description") or "",
                "attributes": attributes,
            }
        )
    live = {(r["table_name"], r["column_name"]): r for r in rows}

    column_writes: list[tuple[tuple[str, str], str, str]] = []
    attribute_writes: list[tuple[tuple[str, str], dict]] = []
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
        updated_attributes: list[dict] = []
        for key, attribute in attribute_writes:
            term_id = attribute.get("term_id")
            if not term_id:
                logger.warning("  %s.%s: attribute has no owning term", *key)
                continue
            updated = update_column_attribute(
                attribute["id"],
                term_id,
                description=attribute["description"],
            )
            if updated:
                updated_attributes.append(updated)
            else:
                logger.warning("  %s.%s: attribute patch reported no change", *key)
        embedder = build_semantic_embedder(database_name, reset=False)
        if embedder is None:
            logger.warning(
                "Semantic embedding disabled — %d attribute description(s) written to "
                "Postgres but the vector index still serves the old text",
                len(updated_attributes),
            )
        else:
            for attribute in updated_attributes:
                # The embed path appends, so the superseded row has to go first.
                embedder.vdb.delete_by_id(attribute["id"])
            written = embedder.embed_column_attributes(updated_attributes)
            logger.info("re-embedded %d attribute row(s)", written)

    total = len(column_writes) + len(attribute_writes)
    logger.info("Applied saved descriptions: %d node(s) written", total)
    return total
