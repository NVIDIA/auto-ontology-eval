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
after editing the saved set to push the edit into the catalog and vector store;
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
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import EmbedParams
    from nemo_retriever.common.vdb.adt_vdb import VDB

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "datasets"
ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "annotations"


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


def apply_metadata(database_name: str, dataset: str | None = None) -> None:
    """Stamp table/column metadata onto the Postgres catalog.

    Reads ``datasets/<database_name>/metadata.json`` (or, for multi-DB
    benchmarks, ``datasets/<benchmark>/<database_name>/metadata.json``), keyed
    by table name, and updates:

    * ``Table.description``
    * ``Column.description``
    * ``Column.sample_values`` (from the JSON's ``value_examples`` field, when
      present and non-empty)

    Tables/columns that aren't present in the catalog are silently skipped.
    Properties for which the JSON has no value are left untouched.
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
    column_rows: list[dict[str, str | None]] = []
    samples_count = 0
    for table_name, table_meta in raw.items():
        table_desc = table_meta.get("description")
        if table_desc:
            table_rows.append({"table_name": table_name, "description": table_desc})

        for col in table_meta.get("columns", []) or []:
            col_desc = col.get("description")
            value_examples = col.get("value_examples")
            # Column.sample_values is stored as a JSON string, matching
            # gsf.dal.datasources.store_column_sample_values (profiling's own
            # writer) and gsf.utils.sample_values.parse_sample_values (the
            # shared reader, which accepts either a JSON string or a list).
            sample_values: str | None = (
                json.dumps([str(v) for v in value_examples])
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


def add_custom_analyses(
    database_name: str,
    dialect: str,  # noqa: ARG001 — kept for call-site compatibility; GSF resolves
    # the dialect from the connector itself now.
    embed_params: "EmbedParams | None" = None,
    vdb: "VDB | None" = None,
    dataset: str | None = None,
) -> None:
    """Ingest custom analyses for *database_name* into Postgres and the VDB.

    Reads the database's analyses — ours from ``annotations/<dataset>/`` when
    present, otherwise its own ``custom_analyses.json``; see
    :func:`custom_analyses_json_path`. For each
    ``{"name", "description", "sql"}`` entry it creates a ``CustomAnalysis``
    catalog row linked to its parsed ``Sql`` row via
    :func:`~gsf.server.custom_analyses.service.create_custom_analysis` (the
    same write path the server's create-analysis endpoint uses).

    When *embed_params* and *vdb* are provided, the function then embeds
    each newly-ingested analysis (name + description + SQL) and **appends**
    the rows to the supplied vector store — so they live alongside the rows
    the main embed pipeline writes for ``Table`` and ``Column`` nodes. The
    append semantics mean the main pipeline must run *before* this function.

    Entries with no SQL, or whose SQL doesn't resolve to any known table, are
    skipped with a warning. An entry whose name or SQL already exists — e.g.
    re-running ingest without resetting the store first — is treated as
    already-ingested and skipped without a warning, so the script stays
    re-runnable. Must be called *after* schema ingestion so the parser can
    resolve table/column references.
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
        description = entry.get("description", "")
        sql = (entry.get("sql") or "").strip()
        if not sql:
            logger.warning("Skipping custom analysis %r — no SQL provided.", name)
            continue
        try:
            create_custom_analysis(
                name=name,
                description=description,
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
# Writing a saved description set over a freshly compiled catalog
# ---------------------------------------------------------------------------
#
# BIRD ships its own column annotations, so a catalog built from a fresh download
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

    Run this after the semantic layer is compiled: the ColumnAttribute rows it
    corrects do not exist before then. Both stores are updated, since a description
    the catalog holds and the vector index does not is invisible to retrieval:

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

    Columns absent from the catalog, columns whose saved attribute text is empty, and
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
            for table in fetch_tables_for_schema(schema_id, database_name=database_name)
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
        key = (
            str(column.get("table_name") or ""),
            str(column.get("column_name") or ""),
        )
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
        "catalog, %d column description(s) and %d attribute description(s) to write",
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
        ("absent from the catalog", sorted(set(saved) - set(live))),
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
    logger.info("Applied saved descriptions: %d catalog row(s) written", total)
    return total
