# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download Spider2-lite's BigQuery (``bq*``, ``ga*``) questions and metadata.

The local (SQLite) questions are seeded by ``seed_spider2_lite.py``; this
covers the 205 that run on Google's public BigQuery datasets::

    datasets/spider2_lite_bq/evaluation.json              # the bq*/ga* questions
    datasets/spider2_lite_bq/<db_id>/metadata.json        # descriptions, samples
    datasets/spider2_lite_bq/<db_id>/schema.sql           # schema-only arm's dump
    datasets/spider2_lite_bq/gold/<instance_id>[_x].csv   # gold execution results
    datasets/spider2_lite_bq/manifest.json                # upstream commit + counts

Nothing is downloaded from BigQuery: the schema dump and metadata come from
the DDL and per-table JSON Spider2 ships under
``resource/databases/bigquery/<db>/<project>.<dataset>/``, read from the same
sparse checkout under ``third_party/Spider2/`` the local seeder uses. Querying
needs Google credentials and a billing project, which only the ingest and the
benchmark use; this script writes them into ``CONNECTION_STRINGS`` as one
``bigquery://<billing_project>/<db_id>?dataset=...`` entry per database (see
``ontology_sql_eval/bigquery.py``).

Date-sharded tables are collapsed into one wildcard table (``events_*``), as
the connector collapses them for the treatment arm. Some databases are still
far too large for a prompt -- gnomAD's DDL alone is ~5 MB -- so the
schema-only arm's dump is cut at ``--schema-char-budget`` characters: the tables
past it are listed by name only, and ``manifest.json`` records which dumps
were cut.

Usage::

    uv run python scripts/seed/seed_spider2_lite_bq.py
    uv run python scripts/seed/seed_spider2_lite_bq.py --db ga4 --db stackoverflow
    uv run python scripts/seed/seed_spider2_lite_bq.py --billing-project my-project

The billing project defaults to ``GOOGLE_CLOUD_PROJECT``, else the
``quota_project_id`` of the Application Default Credentials file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import sqlglot
from sqlglot import exp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from auto_ontology.connectors.bigquery import (  # noqa: E402
    TABLE_SUFFIX_COLUMN,
    ShardFamily,
    TableShape,
    shard_families,
)

from ontology_sql_eval.bigquery import (  # noqa: E402
    DEFAULT_MAXIMUM_BYTES_BILLED,
    MAXIMUM_BYTES_BILLED_ENV,
    bigquery_url,
    parse_byte_size,
)
from scripts.seed import seed_spider2_lite as lite  # noqa: E402

logger = logging.getLogger(__name__)

BIGQUERY_INSTANCE_PREFIXES = ("bq", "ga")

_BIGQUERY_METADATA = "spider2-lite/resource/databases/bigquery"
_SPARSE_PATHS = (*lite._SPARSE_PATHS, _BIGQUERY_METADATA)

# About 50k tokens. The largest Spider2 schemas run to millions of characters
# even with shards collapsed, so some cut is unavoidable; the treatment arm
# retrieves from the full catalog instead.
DEFAULT_SCHEMA_CHAR_BUDGET = 200_000

# Some DDL cells (gnomAD's nested VEP records) exceed csv's 128 KiB default.
csv.field_size_limit(sys.maxsize)


def _default_dest() -> Path:
    return lite._repo_root() / "datasets" / "spider2_lite_bq"


@dataclass(frozen=True)
class _Table:
    """One table as the catalog and the schema dump show it."""

    dataset: str  # project.dataset
    name: str  # table name, or a family's wildcard
    ddl: str
    source: Optional[dict[str, Any]]  # upstream per-table JSON, if any
    family: Optional[ShardFamily] = None

    @property
    def full_name(self) -> str:
        return f"{self.dataset}.{self.name}"


@dataclass
class _Database:
    db_id: str
    spider2_db: str
    datasets: list[str] = field(default_factory=list)
    tables: list[_Table] = field(default_factory=list)
    question_count: int = 0
    collapsed_shards: int = 0
    schema_chars: int = 0
    tables_listed_by_name: int = 0


def _ddl_columns(ddl: str) -> dict[str, str]:
    """Top-level column name to type in a ``CREATE TABLE``; empty if unparsable."""
    try:
        create = sqlglot.parse_one(ddl, read="bigquery")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
        return {}
    schema = create.this
    return {
        column.name: column.kind.sql("bigquery") if column.kind else ""
        for column in create.find_all(exp.ColumnDef)
        if column.parent is schema
    }


def _read_dataset(dataset_dir: Path) -> list[_Table]:
    """The tables of one ``<project>.<dataset>`` folder, shards collapsed."""
    ddl_path = dataset_dir / "DDL.csv"
    ddl: dict[str, str] = {}
    if ddl_path.is_file():
        with ddl_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("table_name") or "").strip()
                if name:
                    ddl[name] = (row.get("ddl") or "").strip()

    sources: dict[str, dict[str, Any]] = {}
    for path in sorted(dataset_dir.glob("*.json")):
        try:
            source = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("  could not parse %s: %s", path, exc)
            continue
        name = str(source.get("table_fullname") or "").split(".")[-1] or path.stem
        sources[name] = source

    dataset = dataset_dir.name
    columns = {name: _ddl_columns(text) for name, text in ddl.items()}
    shapes = {name: TableShape(columns=cols) for name, cols in columns.items()}
    families = shard_families(list(ddl), shapes)
    in_family = {shard for f in families for shard in f.shards}
    tables: list[_Table] = []
    for family in families:
        old = f"`{dataset}.{family.schema_table}`"
        new = f"`{dataset}.{family.wildcard}`"
        source = sources.get(family.schema_table) or next(
            (sources[s] for s in reversed(family.shards) if s in sources), None
        )
        tables.append(
            _Table(
                dataset,
                family.wildcard,
                ddl[family.schema_table].replace(old, new, 1),
                source,
                family,
            )
        )
    for name, text in ddl.items():
        if name not in in_family:
            tables.append(_Table(dataset, name, text, sources.get(name)))
    return tables


# --- metadata.json -----------------------------------------------------------

# A top-level column's description, with its nested fields' appended, is cut
# here: GA360's ``hits`` record alone has some 200 described fields.
MAX_COLUMN_DESCRIPTION_CHARS = 2000


def _nested_descriptions(source: dict[str, Any]) -> list[tuple[str, str]]:
    """``(dotted field path, description)`` for every described field.

    Upstream's ``description`` list is aligned with ``nested_column_names``
    (``event_params``, ``event_params.key``, ...), not with ``column_names``.
    """
    names = source.get("nested_column_names") or source.get("column_names") or []
    descriptions = source.get("description") or []
    return [
        (str(name), text)
        for name, raw in zip(names, descriptions)
        if (text := " ".join(str(raw or "").split()))
    ]


def _top_level_source(source: dict[str, Any]) -> dict[str, Any]:
    """*source* with ``description`` re-aligned to ``column_names``.

    Each top-level column keeps its own description followed by its nested
    fields', so the catalog column for a RECORD says what is inside it.
    """
    described = _nested_descriptions(source)
    by_column: dict[str, list[str]] = {}
    for path, text in described:
        top, _, rest = path.partition(".")
        by_column.setdefault(top, []).append(f"{rest}: {text}" if rest else text)
    aligned = []
    for name in source.get("column_names") or []:
        parts = by_column.get(str(name), [])
        if len(parts) > 1:
            text = f"{parts[0]} Fields: " + "; ".join(parts[1:])
        else:
            text = parts[0] if parts else ""
        if len(text) > MAX_COLUMN_DESCRIPTION_CHARS:
            text = text[: MAX_COLUMN_DESCRIPTION_CHARS - 1] + "…"
        aligned.append(text or None)
    return {**source, "description": aligned}


def _metadata(tables: list[_Table]) -> dict[str, Any]:
    """``apply_metadata`` entries keyed by catalog table name.

    The catalog keys tables by name within the database, so when two datasets
    share a table name the first dataset's entry is kept.
    """
    metadata: dict[str, Any] = {}
    for table in tables:
        if table.name in metadata:
            logger.debug("  %s: duplicate table name, keeping the first", table.name)
            continue
        entry: dict[str, Any] = {"description": "", "columns": []}
        if table.source is not None:
            try:
                _, entry = lite._table_metadata(_top_level_source(table.source))
            except ValueError as exc:
                logger.warning("  %s: %s", table.full_name, exc)
        if table.family is not None:
            prefix = table.family.prefix
            entry = {
                **entry,
                "description": table.family.note(),
                "columns": [
                    *entry.get("columns", []),
                    {
                        "name": TABLE_SUFFIX_COLUMN,
                        "description": (
                            "Pseudo-column of the wildcard table: the part of each "
                            f"shard's name after '{prefix}'."
                        ),
                        "value_examples": [
                            shard[len(prefix) :]
                            for shard in (
                                table.family.shards[0],
                                table.family.schema_table,
                            )
                        ],
                    },
                ],
            }
        if entry.get("description") or entry.get("columns"):
            metadata[table.name] = entry
    return metadata


# --- schema dump -------------------------------------------------------------


def _column_body(ddl: str) -> str:
    """The DDL after its table name: what two same-schema tables share."""
    start = ddl.find("(")
    return ddl[start:] if start >= 0 else ddl


def _description_lines(table: _Table) -> list[str]:
    if table.source is None:
        return []
    return [
        f"  `{table.full_name}`.{path}: {text}"
        for path, text in _nested_descriptions(table.source)
    ]


def render_schema(db_id: str, tables: list[_Table], budget: int) -> tuple[str, int]:
    """The schema-only arm's dump; returns ``(text, tables_listed_by_name)``.

    Tables whose columns repeat an earlier table's are listed under it rather
    than printed again. Blocks go in upstream order until the next would pass
    *budget*; the rest are named without columns. Column descriptions follow,
    for the tables shown, while budget remains.
    """
    header = (
        f"-- Spider2-lite database {db_id} on BigQuery. Name every table in "
        "full, as `project.dataset.table`."
    )
    groups: dict[str, list[_Table]] = {}
    for table in tables:
        groups.setdefault(_column_body(table.ddl), []).append(table)

    blocks: list[tuple[str, list[_Table]]] = []
    for members in groups.values():
        first, rest = members[0], members[1:]
        lines = [first.ddl.rstrip().rstrip(";") + ";"]
        if first.family is not None:
            lines.append(f"-- {first.name}: {first.family.note()}")
        if rest:
            same = ", ".join(f"`{t.full_name}`" for t in rest)
            lines.append(f"-- Same columns as `{first.full_name}`: {same}")
        blocks.append(("\n".join(lines), members))

    parts = [header]
    used = len(header)
    shown: list[_Table] = []
    omitted: list[_Table] = []
    for text, members in blocks:
        if omitted or used + len(text) + 2 > budget:
            omitted.extend(members)
            continue
        parts.append(text)
        used += len(text) + 2
        shown.append(members[0])

    descriptions = [line for table in shown for line in _description_lines(table)]
    if descriptions:
        kept = ["Column descriptions:"]
        size = len(kept[0])
        for line in descriptions:
            if used + size + len(line) + 3 > budget:
                break
            kept.append(line)
            size += len(line) + 1
        if len(kept) > 1:
            parts.append("\n".join(kept))
            used += size + 2

    if omitted:
        names = ", ".join(f"`{t.full_name}`" for t in omitted)
        parts.append(
            f"-- {len(omitted)} more table(s), columns omitted for length: {names}"
        )
    return "\n\n".join(parts) + "\n", len(omitted)


# --- dataset -----------------------------------------------------------------


def _build_dataset(
    upstream: lite._Upstream,
    dest: Path,
    *,
    only: set[str] | None,
    schema_char_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, _Database]]:
    rows = [
        row
        for row in lite._load_jsonl(upstream.questions_jsonl)
        if str(row.get("instance_id", "")).startswith(BIGQUERY_INSTANCE_PREFIXES)
    ]
    if only:
        rows = [
            row
            for row in rows
            if str(row["db"]) in only or lite.slugify(str(row["db"])) in only
        ]
    if not rows:
        raise SystemExit(
            f"No bq*/ga* questions selected from {upstream.questions_jsonl}"
        )

    standards = {
        row["instance_id"]: row for row in lite._load_jsonl(upstream.eval_jsonl)
    }
    metadata_root = upstream.root / _BIGQUERY_METADATA

    databases: dict[str, _Database] = {}
    for spider2_db in sorted({str(row["db"]) for row in rows}):
        db_id = lite.slugify(spider2_db)
        db_dir = metadata_root / spider2_db
        if not db_dir.is_dir():
            logger.warning("  %s: no BigQuery metadata upstream; skipped", spider2_db)
            continue
        database = _Database(db_id, spider2_db)
        for dataset_dir in sorted(p for p in db_dir.iterdir() if p.is_dir()):
            if "." not in dataset_dir.name:
                continue
            database.datasets.append(dataset_dir.name)
            tables = _read_dataset(dataset_dir)
            database.tables.extend(tables)
            database.collapsed_shards += sum(
                len(t.family.shards) - 1 for t in tables if t.family is not None
            )
        if not database.datasets:
            logger.warning("  %s: no datasets upstream; skipped", spider2_db)
            continue
        databases[db_id] = database

    gold_dest = dest / "gold"
    if gold_dest.exists():
        shutil.rmtree(gold_dest)
    gold_dest.mkdir(parents=True)

    evaluation: list[dict[str, Any]] = []
    missing_results: list[str] = []
    for row in sorted(rows, key=lambda r: r["instance_id"]):
        instance_id = row["instance_id"]
        db_id = lite.slugify(str(row["db"]))
        database = databases.get(db_id)
        if database is None:
            continue
        database.question_count += 1

        sql_path = upstream.gold_sql_dir / f"{instance_id}.sql"
        gold_sql = (
            sql_path.read_text(encoding="utf-8").strip() if sql_path.is_file() else ""
        )
        result_files = lite._gold_result_files(upstream.gold_result_dir, instance_id)
        if not result_files:
            missing_results.append(instance_id)
        for path in result_files:
            shutil.copy2(path, gold_dest / path.name)
        standard = standards.get(instance_id) or {}

        evaluation.append(
            {
                "question_id": instance_id,
                "db_id": db_id,
                "question": row.get("question", ""),
                "evidence": lite._load_evidence(
                    upstream.documents_dir, row.get("external_knowledge")
                ),
                "SQL": gold_sql,
                "difficulty": "",
                "gold_results": [f"gold/{path.name}" for path in result_files],
                "condition_cols": standard.get("condition_cols"),
                "ignore_order": bool(standard.get("ignore_order", False)),
                "spider2_db": database.spider2_db,
                "external_knowledge": row.get("external_knowledge"),
                "bq_datasets": database.datasets,
            }
        )
    if missing_results:
        logger.warning(
            "%d question(s) have no gold result CSV: %s",
            len(missing_results),
            ", ".join(missing_results),
        )

    for database in databases.values():
        db_dest = dest / database.db_id
        if db_dest.exists():
            shutil.rmtree(db_dest)
        db_dest.mkdir(parents=True)
        metadata = _metadata(database.tables)
        (db_dest / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        schema, listed = render_schema(
            database.db_id, database.tables, schema_char_budget
        )
        (db_dest / "schema.sql").write_text(schema, encoding="utf-8")
        database.schema_chars = len(schema)
        database.tables_listed_by_name = listed
        logger.info(
            "  %s: %d dataset(s), %d table(s) (%d shard(s) collapsed), "
            "schema %d chars%s",
            database.db_id,
            len(database.datasets),
            len(database.tables),
            database.collapsed_shards,
            database.schema_chars,
            f", {listed} table(s) by name only" if listed else "",
        )

    (dest / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "  evaluation.json (%d question(s), %d with gold SQL, %d database(s))",
        len(evaluation),
        sum(1 for q in evaluation if q["SQL"]),
        len(databases),
    )
    return evaluation, databases


# --- connection strings ------------------------------------------------------


def _adc_path() -> Path:
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return Path(explicit).expanduser()
    config = os.environ.get("CLOUDSDK_CONFIG")
    base = Path(config).expanduser() if config else Path.home() / ".config" / "gcloud"
    return base / "application_default_credentials.json"


def resolve_billing_project(explicit: str | None) -> str | None:
    """``--billing-project``, else ``GOOGLE_CLOUD_PROJECT``, else the ADC file's project."""
    if explicit:
        return explicit
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ["GOOGLE_CLOUD_PROJECT"]
    try:
        credentials = json.loads(_adc_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return credentials.get("quota_project_id") or credentials.get("project_id")


def _write_manifest(
    dest: Path,
    *,
    upstream_commit: str | None,
    evaluation: list[dict[str, Any]],
    databases: dict[str, _Database],
    schema_char_budget: int,
) -> None:
    manifest = {
        "source": {
            "repository": lite.UPSTREAM_REPO,
            "commit": upstream_commit,
            "questions": "spider2-lite/spider2-lite.jsonl",
            "metadata": _BIGQUERY_METADATA,
        },
        "question_count": len(evaluation),
        "questions_with_gold_sql": sum(1 for q in evaluation if q["SQL"]),
        "schema_char_budget": schema_char_budget,
        "databases": [
            {
                "db_id": db.db_id,
                "spider2_db": db.spider2_db,
                "datasets": db.datasets,
                "question_count": db.question_count,
                "table_count": len(db.tables),
                "collapsed_shards": db.collapsed_shards,
                "schema_chars": db.schema_chars,
                "tables_listed_by_name": db.tables_listed_by_name,
            }
            for db in sorted(databases.values(), key=lambda d: d.db_id)
        ],
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def seed_spider2_lite_bq(
    *,
    dest: Path | None = None,
    upstream_root: Path | None = None,
    ref: str = lite.DEFAULT_UPSTREAM_REF,
    only: set[str] | None = None,
    billing_project: str | None = None,
    maximum_bytes_billed: str | None = None,
    schema_char_budget: int = DEFAULT_SCHEMA_CHAR_BUDGET,
    write_env: bool = True,
) -> list[str]:
    """Seed the BigQuery questions under *dest*; return the seeded db_ids."""
    target = dest or _default_dest()
    target.mkdir(parents=True, exist_ok=True)
    # Auto Ontology reads the cap only from the URL, while the seeder also
    # accepts an environment override; writing the resolved value keeps both arms equal.
    if maximum_bytes_billed is None:
        maximum_bytes_billed = os.environ.get(
            MAXIMUM_BYTES_BILLED_ENV, ""
        ).strip() or str(DEFAULT_MAXIMUM_BYTES_BILLED)
    parse_byte_size(maximum_bytes_billed)

    logger.info("=" * 60)
    logger.info("Seeding Spider2-lite (BigQuery) to %s", target)
    logger.info("=" * 60)

    if upstream_root is None:
        upstream_root = lite._default_upstream_root()
        lite._bootstrap_upstream(upstream_root, ref, _SPARSE_PATHS)
    upstream = lite._Upstream(upstream_root)
    required = (
        upstream.questions_jsonl,
        upstream.eval_jsonl,
        upstream.gold_sql_dir,
        upstream.gold_result_dir,
        upstream.documents_dir,
        upstream_root / _BIGQUERY_METADATA,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Spider2 checkout is missing required paths:\n{listing}")
    commit = lite._git_commit(upstream_root)
    logger.info(
        "Spider2 upstream: %s (%s)", upstream_root, commit or "not a git checkout"
    )

    evaluation, databases = _build_dataset(
        upstream, target, only=only, schema_char_budget=schema_char_budget
    )
    _write_manifest(
        target,
        upstream_commit=commit,
        evaluation=evaluation,
        databases=databases,
        schema_char_budget=schema_char_budget,
    )

    cut = sorted(db.db_id for db in databases.values() if db.tables_listed_by_name)
    logger.info("=" * 60)
    logger.info("Spider2-lite (BigQuery) seed complete.")
    logger.info("  Destination : %s", target)
    logger.info("  Questions   : %d", len(evaluation))
    logger.info("  Databases   : %d", len(databases))
    if cut:
        logger.info(
            "  Schema dumps cut at %d chars: %s", schema_char_budget, ", ".join(cut)
        )
    logger.info("=" * 60)

    project = resolve_billing_project(billing_project)
    if project is None:
        logger.warning(
            "No billing project: pass --billing-project, set GOOGLE_CLOUD_PROJECT, "
            "or run `gcloud auth application-default set-quota-project <PROJECT_ID>`. "
            "CONNECTION_STRINGS was not written."
        )
        return sorted(databases)

    value = ",".join(
        bigquery_url(
            project,
            db.db_id,
            db.datasets,
            maximum_bytes_billed=maximum_bytes_billed,
        )
        for db in sorted(databases.values(), key=lambda d: d.db_id)
    )
    # Quoted: the URLs join their datasets with "&", which a shell sourcing .env
    # (scripts/run_gym_arm.sh does) reads as "run in the background", leaving
    # CONNECTION_STRINGS unset.
    value = f"'{value}'"
    env_path = lite._repo_root() / ".env"
    if write_env and lite._update_env_file(env_path, value):
        logger.info(
            "Wrote CONNECTION_STRINGS (billing project %s) to %s", project, env_path
        )
    else:
        logger.info("Add this to your .env:")
        logger.info("CONNECTION_STRINGS=%s", value)
    return sorted(databases)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed Spider2-lite's BigQuery (bq*, ga*) questions, gold results, "
            "metadata and schema dumps into datasets/spider2_lite_bq/."
        )
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: datasets/spider2_lite_bq/).",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help=(
            "Use an existing Spider2 checkout as-is, without git operations "
            "(default: clone or update third_party/Spider2/)."
        ),
    )
    parser.add_argument(
        "--ref",
        default=lite.DEFAULT_UPSTREAM_REF,
        help="Upstream branch, tag or commit (ignored with --upstream-root).",
    )
    parser.add_argument(
        "--db",
        action="append",
        default=None,
        help=(
            "Seed only this Spider2 database (upstream name or db_id, e.g. ga4). "
            "Repeat for several. Default: all of them."
        ),
    )
    parser.add_argument(
        "--billing-project",
        default=None,
        help=(
            "GCP project billed for queries (default: GOOGLE_CLOUD_PROJECT, else "
            "the quota_project_id of the Application Default Credentials)."
        ),
    )
    parser.add_argument(
        "--maximum-bytes-billed",
        default=None,
        help=(
            "Byte cap written into each connection string, e.g. 100GiB or 0 for "
            "none. Default: BIGQUERY_MAXIMUM_BYTES_BILLED, else 40 GiB."
        ),
    )
    parser.add_argument(
        "--schema-char-budget",
        type=int,
        default=DEFAULT_SCHEMA_CHAR_BUDGET,
        help=(
            "Longest schema dump the schema-only arm gets, in characters "
            f"(default: {DEFAULT_SCHEMA_CHAR_BUDGET})."
        ),
    )
    parser.add_argument(
        "--no-write-env",
        action="store_true",
        help="Don't write CONNECTION_STRINGS into .env; just print it.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed_spider2_lite_bq(
        dest=args.dest,
        upstream_root=args.upstream_root,
        ref=args.ref,
        only=set(args.db) if args.db else None,
        billing_project=args.billing_project,
        maximum_bytes_billed=args.maximum_bytes_billed,
        schema_char_budget=args.schema_char_budget,
        write_env=not args.no_write_env,
    )


if __name__ == "__main__":
    main()
