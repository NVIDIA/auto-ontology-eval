# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation for the Spider2 SQLite and Snow seeders.

Use ``seed_spider2_sqlite.py`` or ``seed_spider2_snow.py`` rather than
executing this module directly.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sqlglot import transpile

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "https://github.com/xlang-ai/Spider2.git"
DEFAULT_UPSTREAM_REF = "main"

# Only these spider2-lite subpaths are read by this script. Scoping the sparse
# checkout to them (cone mode also keeps files along the parent path, e.g.
# spider2-lite/spider2-lite.jsonl) trims the working tree from ~900M to ~2M by
# excluding resource/databases (the SQLite bundle comes from Google Drive) and
# the large gold/ execution-result files.
_SQLITE_SPARSE_PATHS = (
    "spider2-lite/evaluation_suite/gold/sql",
    "spider2-lite/evaluation_suite/gold/exec_result",
    "spider2-lite/resource/documents",
)
_SNOW_SPARSE_PATHS = (
    "spider2-snow/evaluation_suite/gold/sql",
    "spider2-snow/evaluation_suite/gold/exec_result",
    "spider2-snow/resource/databases",
    "spider2-lite/resource/documents",
)

# Spider2-lite local SQLite bundle (see spider2-lite/README.md).
LOCAL_SQLITE_DRIVE_ID = "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG"
LOCAL_SQLITE_DRIVE_URL = (
    f"https://drive.google.com/uc?export=download&id={LOCAL_SQLITE_DRIVE_ID}"
)
ARCHIVE_NAME = "local_sqlite.zip"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _datasets_dir() -> Path:
    return _repo_root() / "datasets"


def _spider2_datasets_dir() -> Path:
    return _datasets_dir() / "spider2"


def _upstream_root() -> Path:
    return _repo_root() / "third_party" / "Spider2"


def _spider2_lite_root() -> Path:
    return _upstream_root() / "spider2-lite"


def _spider2_snow_root() -> Path:
    return _upstream_root() / "spider2-snow"


def _snow_databases_dir() -> Path:
    return _spider2_snow_root() / "resource" / "databases"


def _spider2_jsonl() -> Path:
    return _spider2_lite_root() / "spider2-lite.jsonl"


def _spider2_snow_jsonl() -> Path:
    return _spider2_snow_root() / "spider2-snow.jsonl"


def _gold_sql_dir() -> Path:
    return _spider2_lite_root() / "evaluation_suite" / "gold" / "sql"


def _snow_gold_sql_dir() -> Path:
    return _spider2_snow_root() / "evaluation_suite" / "gold" / "sql"


def _documents_dir() -> Path:
    return _spider2_lite_root() / "resource" / "documents"


def _archive_cache_path(dest: Path) -> Path:
    return dest / f".{ARCHIVE_NAME}"


def slugify(name: str) -> str:
    """Normalize a Spider2 db name into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def dataset_dir(slug: str) -> Path:
    return _spider2_datasets_dir() / slug


def sqlite_file(slug: str) -> Path:
    return dataset_dir(slug) / f"{slug}.sqlite"


def sqlite_connection_string(slug: str) -> str:
    """Return a GSF-ready SQLite URI with ``metadata_database=spider2/<slug>``."""
    sqlite_path = sqlite_file(slug).resolve()
    dataset_name = f"spider2/{slug}"
    return (
        f"sqlite:///{sqlite_path}"
        f"?metadata_database={quote(dataset_name, safe='/')}"
    )


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _current_ref(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _bootstrap_upstream(
    ref: str = DEFAULT_UPSTREAM_REF,
    *,
    mode: str,
) -> str:
    """Clone/update only the upstream paths required by the selected seeder."""
    if mode == "sqlite":
        sparse_paths = _SQLITE_SPARSE_PATHS
    elif mode == "snow":
        sparse_paths = _SNOW_SPARSE_PATHS
    else:
        raise ValueError(f"Unsupported Spider2 seed mode: {mode!r}")

    upstream = _upstream_root()
    upstream.parent.mkdir(parents=True, exist_ok=True)

    if upstream.exists():
        logger.info("Updating existing checkout at %s", upstream)
        # Preserve paths materialized by the other seeder.
        _run(["git", "sparse-checkout", "add", *sparse_paths], cwd=upstream)
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=upstream)
        _run(["git", "checkout", ref], cwd=upstream)
        _run(["git", "pull", "--ff-only", "origin", ref], cwd=upstream)
    else:
        logger.info("Cloning %s (sparse %s) into %s", UPSTREAM_REPO, mode, upstream)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                ref,
                UPSTREAM_REPO,
                str(upstream),
            ]
        )
        _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=upstream)

    commit = _current_ref(upstream)
    logger.info("Upstream Spider2 at %s", commit)

    required = (
        (
            _spider2_jsonl(),
            _gold_sql_dir(),
            _gold_exec_result_dir(),
            _documents_dir(),
        )
        if mode == "sqlite"
        else (
            _spider2_snow_jsonl(),
            _snow_gold_sql_dir(),
            _snow_gold_exec_result_dir(),
            _snow_databases_dir(),
            _documents_dir(),
        )
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            f"Upstream checkout is missing required Spider2 {mode} paths:\n{missing_list}"
        )

    return commit


def _is_local_instance(instance_id: str) -> bool:
    return instance_id.startswith("local")


def _is_snowflake_instance(instance_id: str) -> bool:
    return str(instance_id or "").startswith("sf")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return rows


def _gold_exec_result_dir() -> Path:
    return _spider2_lite_root() / "evaluation_suite" / "gold" / "exec_result"


def _snow_gold_exec_result_dir() -> Path:
    return _spider2_snow_root() / "evaluation_suite" / "gold" / "exec_result"


def _load_gold_sql(
    instance_id: str,
    gold_sql_dir: Path | None = None,
) -> tuple[str, str | None]:
    sql_path = (gold_sql_dir or _gold_sql_dir()) / f"{instance_id}.sql"
    if not sql_path.exists():
        return "", f"missing gold SQL file: {sql_path.name}"
    return sql_path.read_text(encoding="utf-8").strip(), None


def _load_evidence(filename: str | None) -> str:
    if not filename:
        return ""
    doc_path = _documents_dir() / filename
    if not doc_path.exists():
        logger.warning("External knowledge file not found: %s", doc_path)
        return ""
    return doc_path.read_text(encoding="utf-8").strip()


def _sample_value(value: Any) -> str:
    """Return a stable string representation suitable for Neo4j metadata."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _metadata_from_snow_table(source: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert one Spider2-Snow table JSON to enrichment ``metadata.json`` format.

    Output matches ``ontology_sql_eval.ingestion.enrich_graph.apply_metadata``::

        {
            "<table_name>": {
                "description": "...",
                "columns": [
                    {"name": "...", "description": "...", "value_examples": [...]}
                ]
            }
        }

    Spider2 stores column descriptions as a list aligned with ``column_names``
    and examples as ``sample_rows`` dictionaries. Table keys use the bare
    Snowflake table name (as stored on Neo4j ``Table.name`` after ingest).
    """
    fullname = str(source.get("table_fullname") or "")
    source_table_name = str(source.get("table_name") or "")
    table_name = (fullname.split(".")[-1] if fullname else "") or (
        source_table_name.split(".")[-1] if source_table_name else ""
    )
    if not table_name:
        raise ValueError("Spider2 table metadata is missing table_name")

    column_names = source.get("column_names") or []
    descriptions = source.get("description") or []
    sample_rows = source.get("sample_rows") or []

    columns: list[dict[str, Any]] = []
    for index, raw_name in enumerate(column_names):
        source_name = str(raw_name)
        # Snowflake INFORMATION_SCHEMA returns unquoted identifiers in uppercase,
        # and enrich_graph matches Column.name exactly.
        name = source_name.upper()
        description = descriptions[index] if index < len(descriptions) else None

        examples: list[str] = []
        seen: set[str] = set()
        for row in sample_rows:
            if not isinstance(row, dict):
                continue
            value = row.get(source_name)
            if value is None:
                continue
            rendered = _sample_value(value)
            if rendered in seen:
                continue
            seen.add(rendered)
            examples.append(rendered)
            if len(examples) == 5:
                break

        columns.append(
            {
                "name": name,
                "description": str(description) if description else None,
                "value_examples": examples or None,
            }
        )

    return table_name, {
        "description": None,
        "columns": columns,
    }


def _build_snow_metadata() -> dict[str, dict[str, int]]:
    """Generate enrichment metadata for every Spider2-Snow database.

    Writes ``datasets/spider2/<slug>/metadata.json``. Snowflake connectors
    keep the physical database name (e.g. ``CPTAC_PDC``) for Neo4j / pgvector
    and resolve this path by slug. Introspection is then restricted to the
    tables/columns listed in the metadata file.
    """
    source_root = _snow_databases_dir()
    if not source_root.is_dir():
        raise SystemExit(
            f"Missing Spider2-Snow database metadata at {source_root}. "
            "Ensure spider2-snow/resource/databases is in the sparse checkout."
        )

    summary: dict[str, dict[str, int]] = {}
    for database_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        metadata: dict[str, dict[str, Any]] = {}
        column_count = 0
        sample_count = 0
        duplicate_tables = 0

        for source_path in sorted(database_dir.rglob("*.json")):
            source = json.loads(source_path.read_text(encoding="utf-8"))
            table_name, table = _metadata_from_snow_table(source)
            if table_name in metadata:
                # enrich_graph matches tables by bare name only; keep the last
                # definition when the same table name appears in multiple schemas.
                duplicate_tables += 1
                logger.warning(
                    "Duplicate table name %r in %s; overwriting previous metadata entry",
                    table_name,
                    database_dir.name,
                )
            metadata[table_name] = table
            column_count += len(table["columns"])
            sample_count += sum(
                bool(column.get("value_examples")) for column in table["columns"]
            )

        out_dir = dataset_dir(slugify(database_dir.name))
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary[database_dir.name] = {
            "table_count": len(metadata),
            "column_count": column_count,
            "columns_with_samples": sample_count,
            "duplicate_tables": duplicate_tables,
        }
        logger.info(
            "Wrote metadata %s (%d tables, %d columns, %d columns with samples)",
            metadata_path,
            len(metadata),
            column_count,
            sample_count,
        )

    logger.info("Wrote metadata for %d Spider2 databases", len(summary))
    return summary


def _transpile_sqlite_to_postgres(sql: str) -> tuple[str, str | None]:
    if not sql.strip():
        return "", "empty SQL"
    try:
        results = transpile(sql, read="sqlite", write="postgres", pretty=True)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if not results:
        return "", "empty transpile result"
    return results[0], None


def _write_manifest(filename: str, manifest: dict[str, Any]) -> Path:
    dest = _spider2_datasets_dir()
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / filename
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Manifest: %s (%d databases, %d questions)",
        manifest_path,
        manifest["database_count"],
        manifest["question_count"],
    )
    return manifest_path


def _build_sqlite_datasets(*, upstream_commit: str | None = None) -> Path:
    """Build evaluations and a manifest for Spider2-lite local SQLite DBs."""
    jsonl_path = _spider2_jsonl()
    if not jsonl_path.exists():
        raise SystemExit(
            f"Missing {jsonl_path}. Upstream bootstrap may have failed."
        )

    rows = _load_jsonl(jsonl_path)
    local_rows = [row for row in rows if _is_local_instance(row.get("instance_id", ""))]
    if not local_rows:
        raise SystemExit("No local Spider2-lite questions found in spider2-lite.jsonl")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in local_rows:
        grouped[row["db"]].append(row)

    manifest_databases: list[dict[str, Any]] = []
    transpile_failures: list[dict[str, str]] = []
    missing_gold_sql: list[dict[str, str]] = []
    all_questions: list[dict[str, Any]] = []

    for spider2_db_name in sorted(grouped):
        slug = slugify(spider2_db_name)
        out_dir = dataset_dir(slug)
        out_dir.mkdir(parents=True, exist_ok=True)

        evaluation: list[dict[str, Any]] = []
        question_ids: list[str] = []

        for row in sorted(grouped[spider2_db_name], key=lambda r: r["instance_id"]):
            instance_id = row["instance_id"]
            question_ids.append(instance_id)
            sqlite_sql, missing_sql = _load_gold_sql(instance_id)
            if missing_sql:
                missing_gold_sql.append(
                    {
                        "instance_id": instance_id,
                        "database": spider2_db_name,
                        "reason": missing_sql,
                    }
                )
                postgres_sql = ""
                transpile_error = None
            else:
                postgres_sql, transpile_error = _transpile_sqlite_to_postgres(sqlite_sql)
                if transpile_error:
                    transpile_failures.append(
                        {
                            "instance_id": instance_id,
                            "database": spider2_db_name,
                            "error": transpile_error,
                        }
                    )

            evaluation.append(
                {
                    "question_id": instance_id,
                    "db_id": spider2_db_name,
                    "question": row.get("question", ""),
                    "evidence": _load_evidence(row.get("external_knowledge")),
                    "SQL": sqlite_sql,
                    "SQL_postgres": postgres_sql,
                    "difficulty": "",
                    "answer_raw": "",
                    "answer": "",
                }
            )

        all_questions.extend(evaluation)

        manifest_databases.append(
            {
                "slug": slug,
                "spider2_db_name": spider2_db_name,
                "dataset_name": f"spider2/{slug}",
                "postgres_database_name": f"spider2_{slug}",
                "dataset_dir": str(out_dir.relative_to(_datasets_dir())),
                "question_count": len(evaluation),
                "question_ids": question_ids,
                "sqlite_filename": f"{slug}.sqlite",
                "sqlite_path": f"spider2/{slug}/{slug}.sqlite",
            }
        )
        logger.debug("Collected %s (%d questions)", f"spider2/{slug}", len(evaluation))

    # Write a single merged evaluation.json at datasets/spider2/evaluation.json so
    # eval_chatbot.py --database-name spider2 can load all 135 questions in one shot.
    merged_eval_path = _spider2_datasets_dir() / "evaluation.json"
    all_questions_sorted = sorted(all_questions, key=lambda q: q["question_id"])
    with merged_eval_path.open("w", encoding="utf-8") as f:
        json.dump(all_questions_sorted, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info(
        "Wrote merged evaluation.json (%d questions) -> %s",
        len(all_questions_sorted),
        merged_eval_path,
    )

    manifest = {
        "source": {
            "repository": UPSTREAM_REPO,
            "path": "spider2-lite",
            "upstream_root": str(_upstream_root()),
            "upstream_commit": upstream_commit,
            "jsonl": str(_spider2_jsonl().relative_to(_upstream_root())),
        },
        "scope": "local_sqlite",
        "question_count": len(local_rows),
        "database_count": len(manifest_databases),
        "databases": manifest_databases,
        "transpile_failures": transpile_failures,
        "missing_gold_sql": missing_gold_sql,
    }
    if missing_gold_sql:
        logger.warning(
            "%d question(s) have no published gold SQL in upstream (see manifest.missing_gold_sql)",
            len(missing_gold_sql),
        )
    if transpile_failures:
        logger.warning(
            "%d question(s) failed SQLite->Postgres transpile",
            len(transpile_failures),
        )
    return _write_manifest("manifest_sqlite.json", manifest)


def _build_snow_datasets(*, upstream_commit: str | None = None) -> Path:
    """Build all 547 Spider2-Snow evaluations plus enrichment metadata."""
    jsonl_path = _spider2_snow_jsonl()
    if not jsonl_path.exists():
        raise SystemExit(f"Missing {jsonl_path}. Upstream bootstrap may have failed.")

    rows = _load_jsonl(jsonl_path)
    if not rows:
        raise SystemExit("No Spider2-Snow questions found in spider2-snow.jsonl")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["db_id"])].append(row)

    manifest_databases: list[dict[str, Any]] = []
    missing_gold_sql: list[dict[str, str]] = []
    metadata_summary = _build_snow_metadata()

    for database_name in sorted(grouped):
        slug = slugify(database_name)
        out_dir = dataset_dir(slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        evaluation: list[dict[str, Any]] = []
        question_ids: list[str] = []

        for row in sorted(grouped[database_name], key=lambda item: item["instance_id"]):
            instance_id = str(row["instance_id"])
            question_ids.append(instance_id)
            gold_sql, missing_sql = _load_gold_sql(
                instance_id,
                _snow_gold_sql_dir(),
            )
            if missing_sql:
                missing_gold_sql.append(
                    {
                        "instance_id": instance_id,
                        "database": database_name,
                        "reason": missing_sql,
                    }
                )

            evaluation.append(
                {
                    "question_id": instance_id,
                    "db_id": database_name,
                    "question": row.get("instruction", ""),
                    "evidence": _load_evidence(row.get("external_knowledge")),
                    "SQL": gold_sql,
                    "SQL_postgres": "",
                    "difficulty": "",
                    "answer_raw": "",
                    "answer": "",
                }
            )

        eval_path = out_dir / "evaluation.json"
        eval_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest_databases.append(
            {
                "slug": slug,
                "dialect": "snowflake",
                "snowflake_database": database_name,
                "spider2_db_name": database_name,
                "dataset_name": f"spider2/{slug}",
                "dataset_dir": str(out_dir.relative_to(_datasets_dir())),
                "evaluation_json": str(eval_path.relative_to(_datasets_dir())),
                "question_count": len(evaluation),
                "question_ids": question_ids,
            }
        )
        logger.info(
            "Wrote Snowflake %s (%d questions) -> %s",
            database_name,
            len(evaluation),
            eval_path,
        )

    manifest = {
        "source": {
            "repository": UPSTREAM_REPO,
            "path": "spider2-snow",
            "upstream_root": str(_upstream_root()),
            "upstream_commit": upstream_commit,
            "jsonl": str(jsonl_path.relative_to(_upstream_root())),
        },
        "scope": "snowflake",
        "question_count": len(rows),
        "database_count": len(manifest_databases),
        "databases": manifest_databases,
        "missing_gold_sql": missing_gold_sql,
        "metadata": metadata_summary,
    }
    if missing_gold_sql:
        logger.warning("%d Snow questions have no published gold SQL", len(missing_gold_sql))
    return _write_manifest("manifest_snow.json", manifest)


def _load_manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and "databases" in manifest:
        return manifest["databases"]
    if isinstance(manifest, list):
        return manifest
    return []


def _build_slug_map(entries: list[dict[str, Any]]) -> dict[str, str]:
    slug_map: dict[str, str] = {}
    for entry in entries:
        slug = entry["slug"]
        spider2_db_name = entry.get("spider2_db_name", slug)
        for key in (_norm_key(spider2_db_name), _norm_key(slug)):
            slug_map.setdefault(key, slug)
    return slug_map


def _resolve_slug(sqlite_stem: str, slug_map: dict[str, str]) -> str:
    for candidate in (sqlite_stem, slugify(sqlite_stem)):
        slug = slug_map.get(_norm_key(candidate))
        if slug:
            return slug
    return slugify(sqlite_stem)


def _download_sqlite_archive(archive_path: Path, *, force: bool = False) -> None:
    if archive_path.exists() and not force:
        logger.info("Using cached archive at %s", archive_path)
        return

    if archive_path.exists() and force:
        archive_path.unlink()

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Spider2-lite local SQLite archive to %s", archive_path)
    logger.info("Drive file ID: %s", LOCAL_SQLITE_DRIVE_ID)

    try:
        import gdown  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "gdown is required to download from Google Drive. "
            "Install it with:  uv pip install gdown\n"
            "Manual fallback: download local_sqlite.zip from spider2-lite/README.md "
            f"and place it at {archive_path}"
        ) from exc

    # gdown handles the virus-scan confirmation redirect that plain curl misses.
    gdown.download(
        id=LOCAL_SQLITE_DRIVE_ID,
        output=str(archive_path),
        quiet=False,
    )

    if not archive_path.exists() or not zipfile.is_zipfile(archive_path):
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid zip archive: {archive_path}\n"
            "Try downloading manually from spider2-lite/README.md and "
            f"placing the zip at {archive_path}"
        )

    logger.info("Download complete (%d bytes).", archive_path.stat().st_size)


def _install_sqlite_databases(
    archive_path: Path,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing {archive_path}. Run seed_spider2_sqlite.py first."
        )

    manifest_file = manifest_path or (
        _spider2_datasets_dir() / "manifest_sqlite.json"
    )
    slug_map = _build_slug_map(_load_manifest_entries(manifest_file))

    installed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="spider2-sqlite-") as tmp_dir:
        with zipfile.ZipFile(archive_path) as zf:
            members = [
                name
                for name in zf.namelist()
                if name.endswith(".sqlite")
                and not name.startswith("__MACOSX/")
                and "/._" not in name
            ]
            if not members:
                raise RuntimeError(f"No .sqlite files found in {archive_path}")

            zf.extractall(tmp_dir, members=members)

        tmp_root = Path(tmp_dir)
        sqlite_files = sorted(tmp_root.rglob("*.sqlite"))
        logger.info("Found %d SQLite files in archive.", len(sqlite_files))

        for src in sqlite_files:
            slug = _resolve_slug(src.stem, slug_map)
            dest = sqlite_file(slug)
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and not force:
                logger.debug("Skipping %s (already installed).", dest)
                skipped.append(slug)
                continue

            try:
                if dest.exists():
                    dest.unlink()
                shutil.move(str(src), str(dest))
                installed.append(slug)
                logger.info(
                    "Installed %s -> %s",
                    src.name,
                    dest.relative_to(_datasets_dir()),
                )
            except OSError as exc:
                failed.append({"slug": slug, "source": src.name, "error": str(exc)})
                logger.error("Failed installing %s: %s", src.name, exc)

    return {
        "archive": str(archive_path),
        "installed": installed,
        "skipped": skipped,
        "failed": failed,
        "installed_count": len(installed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
    }


def _print_summary(dest: Path, slugs: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("Spider2-lite download complete.")
    logger.info("  Destination : %s", dest)
    logger.info("  Databases   : %d", len(slugs))
    for slug in slugs:
        logger.info("    - %s", slug)
    logger.info("=" * 60)

    if not slugs:
        return

    conn_strings = [sqlite_connection_string(slug) for slug in slugs]
    print()
    print("Add this to your .env (all databases, comma-separated):")
    print()
    print(f"CONNECTION_STRINGS={','.join(conn_strings)}")
    print()
    print("Run the full pipeline for a single database, e.g.:")
    slug = slugs[0]
    print(
        f"  PYTHONPATH=../GSF uv run python main.py "
        f"--database-name spider2/{slug}"
    )


def seed_spider2_sqlite(
    *,
    ref: str = DEFAULT_UPSTREAM_REF,
    dest: Path | None = None,
    force: bool = False,
    keep_archive: bool = False,
    skip_build: bool = False,
) -> list[str]:
    """Download and organize the Spider2-lite local dataset.

    Returns the list of dataset slugs installed under *dest*.
    """
    target = dest or _spider2_datasets_dir()
    archive_path = _archive_cache_path(target)

    logger.info("=" * 60)
    logger.info("Downloading Spider2-lite to %s", target)
    logger.info("=" * 60)

    commit = _bootstrap_upstream(ref=ref, mode="sqlite")
    logger.info("Upstream Spider2 commit: %s", commit)

    if skip_build:
        logger.info("Skipping evaluation.json / manifest build (--skip-build).")
    else:
        manifest_path = _build_sqlite_datasets(upstream_commit=commit)
        logger.info("Wrote manifest: %s", manifest_path)

    _download_sqlite_archive(archive_path, force=force)
    summary = _install_sqlite_databases(archive_path, force=force)

    logger.info(
        "SQLite install: %d installed, %d skipped, %d failed",
        summary["installed_count"],
        summary["skipped_count"],
        summary["failed_count"],
    )
    if summary["failed"]:
        for failure in summary["failed"]:
            logger.error(
                "  %s (%s): %s",
                failure["slug"],
                failure["source"],
                failure["error"],
            )
        raise SystemExit("One or more SQLite databases failed to install.")

    if not keep_archive and archive_path.exists():
        archive_path.unlink()
        logger.info("Removed cached archive %s", archive_path)
    elif keep_archive:
        logger.info("Kept cached archive at %s", archive_path)

    manifest_entries = _load_manifest_entries(target / "manifest_sqlite.json")
    slugs = sorted(
        {
            entry["slug"]
            for entry in manifest_entries
            if sqlite_file(entry["slug"]).exists()
        }
        or set(summary["installed"] + summary["skipped"])
    )
    _print_summary(target, slugs)
    return slugs


def seed_spider2_snow(
    *,
    ref: str = DEFAULT_UPSTREAM_REF,
    skip_build: bool = False,
) -> Path | None:
    """Seed Spider2-Snow evaluations and metadata without downloading SQLite."""
    logger.info("Seeding Spider2-Snow under %s", _spider2_datasets_dir())
    commit = _bootstrap_upstream(ref=ref, mode="snow")
    logger.info("Upstream Spider2 commit: %s", commit)
    if skip_build:
        logger.info("Skipping evaluation.json / metadata build (--skip-build).")
        return None
    return _build_snow_datasets(upstream_commit=commit)
