# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the Spider2-lite local (SQLite) questions and databases.

Spider2-lite mixes BigQuery (``bq*``), Snowflake (``sf*``) and local SQLite
(``local*``) questions. Only the local ones run without cloud credentials, so
only those are seeded::

    datasets/spider2_lite/evaluation.json             # the 135 local* questions
    datasets/spider2_lite/<db_id>/<db_id>.sqlite      # one database per db_id
    datasets/spider2_lite/<db_id>/metadata.json       # sample values per column
    datasets/spider2_lite/gold/<instance_id>[_x].csv  # gold execution results
    datasets/spider2_lite/manifest.json               # upstream commit + counts

Questions, gold artifacts and per-table metadata come from a sparse checkout of
the upstream repository under ``third_party/Spider2/``. The SQLite files are
not in that repository: they ship as a separate zip, fetched from the
HuggingFace mirror first and then the authors' Google Drive (see
``SQLITE_SOURCES``).

Spider2 scores a prediction against gold *result* CSVs, not gold SQL. Upstream
publishes gold SQL for only a minority of the local questions, but every one
has one or more acceptable result CSVs plus the scoring options its evaluator
uses (``condition_cols``, ``ignore_order``). Each ``evaluation.json`` row
carries those alongside the paths of its gold CSVs, so the comparison can be
reproduced without the upstream checkout.

``db_id`` is the upstream ``db`` name slugified (``Db-IMDB`` -> ``db_imdb``,
``IPL`` -> ``ipl``), which is also the installed file stem and therefore the
database name the SQLite connector reports at ingest.

Usage::

    uv run python scripts/seed/seed_spider2_lite.py
    uv run python scripts/seed/seed_spider2_lite.py --sqlite-root /path/to/spider2-localdb
    uv run python scripts/seed/seed_spider2_lite.py --upstream-root third_party/Spider2

After seeding, ingest the databases via::

    uv run python -m ontology_sql_eval.ingestion.ingest --benchmark-name spider2_lite
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sqlite3
import ssl
import subprocess
import urllib.error
import urllib.request
import zipfile
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import certifi
except ImportError:  # pragma: no cover - stdlib fallback
    certifi = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context, preferring certifi's CA bundle when available."""
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


UPSTREAM_REPO = "https://github.com/xlang-ai/Spider2.git"
DEFAULT_UPSTREAM_REF = "main"

# Cone-mode sparse checkout also materializes the files sitting directly in each
# parent directory, which is how spider2-lite/spider2-lite.jsonl and
# evaluation_suite/gold/spider2lite_eval.jsonl arrive without being listed.
_SPARSE_PATHS = (
    "spider2-lite/evaluation_suite/gold/sql",
    "spider2-lite/evaluation_suite/gold/exec_result",
    "spider2-lite/resource/documents",
    "spider2-lite/resource/databases/sqlite",
)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
ZIP_MAGIC = b"PK\x03\x04"

LOCAL_INSTANCE_PREFIX = "local"

# Some columns hold whole documents (wwe's scraped HTML pages, eu_soccer's match
# XML). As samples they say nothing a type and name don't, they get embedded at
# ingest, and Auto Ontology's default nemotron-3-embed-1b endpoint rejects text
# containing <img> tags as image input. Longer values are left out; truncating
# them instead could keep a partial tag.
MAX_SAMPLE_CHARS = 500


@dataclass(frozen=True)
class ArchiveSource:
    """A downloadable zip that may contain some of the wanted databases."""

    name: str
    url: str
    archive_name: str


# The complete local pack is only on Google Drive, which regularly answers large
# downloads with a "Quota exceeded" HTML page. The HuggingFace mirror is
# reliable but an older, partial snapshot, so it goes first and Google Drive
# fills in whatever it lacks.
SQLITE_SOURCES = (
    ArchiveSource(
        name="Spider2-lite localdb (HuggingFace mirror, partial)",
        url=(
            "https://huggingface.co/datasets/xlangai/spider2-localdb"
            "/resolve/main/sqlite.zip"
        ),
        archive_name="spider2_localdb_hf.zip",
    ),
    ArchiveSource(
        name="Spider2-lite local pack (Google Drive, complete)",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG&export=download&confirm=t"
        ),
        archive_name="spider2_local_sqlite.zip",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_dest() -> Path:
    return _repo_root() / "datasets" / "spider2_lite"


def _default_upstream_root() -> Path:
    return _repo_root() / "third_party" / "Spider2"


@dataclass(frozen=True)
class _Upstream:
    """Paths inside a Spider2 checkout that this seeder reads."""

    root: Path

    @property
    def lite(self) -> Path:
        return self.root / "spider2-lite"

    @property
    def questions_jsonl(self) -> Path:
        return self.lite / "spider2-lite.jsonl"

    @property
    def eval_jsonl(self) -> Path:
        return self.lite / "evaluation_suite" / "gold" / "spider2lite_eval.jsonl"

    @property
    def gold_sql_dir(self) -> Path:
        return self.lite / "evaluation_suite" / "gold" / "sql"

    @property
    def gold_result_dir(self) -> Path:
        return self.lite / "evaluation_suite" / "gold" / "exec_result"

    @property
    def documents_dir(self) -> Path:
        return self.lite / "resource" / "documents"

    @property
    def table_metadata_dir(self) -> Path:
        return self.lite / "resource" / "databases" / "sqlite"

    def required(self) -> tuple[Path, ...]:
        return (
            self.questions_jsonl,
            self.eval_jsonl,
            self.gold_sql_dir,
            self.gold_result_dir,
            self.documents_dir,
            self.table_metadata_dir,
        )


def slugify(name: str) -> str:
    """Turn an upstream Spider2 ``db`` name into this repo's ``db_id``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _norm_key(name: str) -> str:
    """Match key for the same database spelled differently across upstream.

    The question file, the metadata folders and the SQLite pack disagree on
    case and punctuation (``Db-IMDB`` / ``DB_IMDB``, ``sqlite-sakila`` /
    ``SQLITE_SAKILA``); dropping both makes them compare equal.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _is_macos_junk(name: str) -> bool:
    """True for AppleDouble / ``__MACOSX`` sidecar paths."""
    parts = Path(name).parts
    return "__MACOSX" in parts or any(part.startswith("._") for part in parts)


# --- upstream checkout -------------------------------------------------------


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _git_commit(repo_dir: Path) -> str | None:
    """Return *repo_dir*'s HEAD commit, or ``None`` when it isn't a git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _is_sparse_checkout(repo_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "config", "--get", "core.sparseCheckout"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() == "true"


def _bootstrap_upstream(
    upstream_root: Path, ref: str, sparse_paths: tuple[str, ...] = _SPARSE_PATHS
) -> None:
    """Clone or update the sparse Spider2 checkout at *upstream_root*."""
    if (upstream_root / ".git").exists():
        logger.info("Updating Spider2 checkout at %s to %s", upstream_root, ref)
        # ``add`` rather than ``set``, so paths another seeder (e.g. Spider2-Snow)
        # materialized in the same checkout are kept.
        if _is_sparse_checkout(upstream_root):
            _run(["git", "sparse-checkout", "add", *sparse_paths], cwd=upstream_root)
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=upstream_root)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=upstream_root)
        return

    logger.info("Cloning %s (sparse) into %s", UPSTREAM_REPO, upstream_root)
    upstream_root.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--no-checkout",
            UPSTREAM_REPO,
            str(upstream_root),
        ]
    )
    _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=upstream_root)
    _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=upstream_root)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=upstream_root)


def _check_upstream(upstream: _Upstream) -> None:
    missing = [path for path in upstream.required() if not path.exists()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Spider2 checkout is missing required paths:\n{listing}")


# --- questions, gold and metadata --------------------------------------------


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


def _gold_result_files(gold_result_dir: Path, instance_id: str) -> list[Path]:
    """Gold CSVs for *instance_id*, resolved as upstream ``evaluate.py`` does.

    A single ``<id>.csv`` wins; otherwise every ``<id>_<letter>.csv`` is an
    acceptable answer and a prediction matching any one of them scores.
    """
    single = gold_result_dir / f"{instance_id}.csv"
    if single.is_file():
        return [single]
    pattern = re.compile(rf"^{re.escape(instance_id)}_[a-z]\.csv$")
    return sorted(
        path for path in gold_result_dir.iterdir() if pattern.match(path.name)
    )


def _load_evidence(documents_dir: Path, filename: str | None) -> str:
    if not filename:
        return ""
    path = documents_dir / filename
    if not path.is_file():
        logger.warning("External knowledge file not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8").strip()


def _sample_value(value: Any) -> Any:
    """Keep JSON scalars as they are; render nested values as canonical JSON."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _table_metadata(source: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert one upstream per-table JSON into an ``apply_metadata`` entry.

    Upstream keeps column descriptions in a list aligned with ``column_names``
    and samples as ``sample_rows`` dicts; the entry keeps up to five distinct
    non-null samples per column as ``value_examples``, leaving out any longer
    than ``MAX_SAMPLE_CHARS`` once rendered. The table key is the bare
    table name, which is what ingest stores for a SQLite table.
    """
    raw_name = str(source.get("table_fullname") or source.get("table_name") or "")
    table_name = raw_name.split(".")[-1]
    if not table_name:
        raise ValueError("Spider2 table metadata is missing table_name")

    descriptions = source.get("description") or []
    sample_rows = [
        row for row in source.get("sample_rows") or [] if isinstance(row, dict)
    ]

    columns: list[dict[str, Any]] = []
    for index, raw_column in enumerate(source.get("column_names") or []):
        name = str(raw_column)
        raw_description = descriptions[index] if index < len(descriptions) else None
        description = str(raw_description or "").strip() or None

        examples: list[Any] = []
        seen: set[str] = set()
        for row in sample_rows:
            value = row.get(name)
            if value is None:
                continue
            rendered = _sample_value(value)
            if isinstance(rendered, str) and len(rendered) > MAX_SAMPLE_CHARS:
                continue
            identity = json.dumps(rendered, ensure_ascii=False, sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            examples.append(rendered)
            if len(examples) == 5:
                break

        columns.append(
            {
                "name": name,
                "description": description,
                "value_examples": examples or None,
            }
        )

    return table_name, {"description": "", "columns": columns}


def _write_metadata_json(source_dir: Path, db_dest: Path) -> int:
    """Write ``<db_dest>/metadata.json`` from *source_dir*'s per-table JSONs.

    Returns the number of tables written. A malformed table file is logged and
    skipped rather than failing the whole seed.
    """
    metadata: dict[str, Any] = {}
    for source_path in sorted(source_dir.glob("*.json")):
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
            table_name, table = _table_metadata(source)
        except (OSError, ValueError) as exc:
            logger.warning("  could not parse %s: %s", source_path.name, exc)
            continue
        metadata[table_name] = table

    if not metadata:
        return 0
    db_dest.mkdir(parents=True, exist_ok=True)
    (db_dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(metadata)


@dataclass
class _Database:
    """One seeded database and where its upstream pieces live."""

    db_id: str
    spider2_db: str
    question_count: int = 0
    table_count: int = 0
    dropped_views: list[str] = field(default_factory=list)


def _build_dataset(
    upstream: _Upstream, dest: Path
) -> tuple[list[dict[str, Any]], dict[str, _Database]]:
    """Write questions, gold results and metadata; return the rows and databases."""
    local_rows = [
        row
        for row in _load_jsonl(upstream.questions_jsonl)
        if str(row.get("instance_id", "")).startswith(LOCAL_INSTANCE_PREFIX)
    ]
    if not local_rows:
        raise SystemExit(f"No local* questions in {upstream.questions_jsonl}")

    standards = {row["instance_id"]: row for row in _load_jsonl(upstream.eval_jsonl)}

    gold_dest = dest / "gold"
    if gold_dest.exists():
        shutil.rmtree(gold_dest)
    gold_dest.mkdir(parents=True)

    databases: dict[str, _Database] = {}
    evaluation: list[dict[str, Any]] = []
    missing_results: list[str] = []
    missing_standards: list[str] = []

    for row in sorted(local_rows, key=lambda r: r["instance_id"]):
        instance_id = row["instance_id"]
        spider2_db = str(row["db"])
        db_id = slugify(spider2_db)
        database = databases.setdefault(db_id, _Database(db_id, spider2_db))
        database.question_count += 1

        sql_path = upstream.gold_sql_dir / f"{instance_id}.sql"
        gold_sql = (
            sql_path.read_text(encoding="utf-8").strip() if sql_path.is_file() else ""
        )

        result_files = _gold_result_files(upstream.gold_result_dir, instance_id)
        if not result_files:
            missing_results.append(instance_id)
        for path in result_files:
            shutil.copy2(path, gold_dest / path.name)

        standard = standards.get(instance_id)
        if standard is None:
            missing_standards.append(instance_id)
            standard = {}

        evaluation.append(
            {
                "question_id": instance_id,
                "db_id": db_id,
                "question": row.get("question", ""),
                "evidence": _load_evidence(
                    upstream.documents_dir, row.get("external_knowledge")
                ),
                "SQL": gold_sql,
                "difficulty": "",
                "gold_results": [f"gold/{path.name}" for path in result_files],
                "condition_cols": standard.get("condition_cols"),
                "ignore_order": bool(standard.get("ignore_order", False)),
                "spider2_db": spider2_db,
                "external_knowledge": row.get("external_knowledge"),
            }
        )

    if missing_results:
        logger.warning(
            "%d question(s) have no gold result CSV: %s",
            len(missing_results),
            ", ".join(missing_results),
        )
    if missing_standards:
        logger.warning(
            "%d question(s) have no entry in %s: %s",
            len(missing_standards),
            upstream.eval_jsonl.name,
            ", ".join(missing_standards),
        )

    metadata_dirs = {
        _norm_key(path.name): path
        for path in upstream.table_metadata_dir.iterdir()
        if path.is_dir()
    }
    for database in databases.values():
        source_dir = metadata_dirs.get(_norm_key(database.spider2_db))
        if source_dir is None:
            logger.warning("  %s: no table metadata upstream", database.db_id)
            continue
        database.table_count = _write_metadata_json(source_dir, dest / database.db_id)
        logger.info(
            "  %s/metadata.json (%d table(s) from %s)",
            database.db_id,
            database.table_count,
            source_dir.name,
        )

    eval_path = dest / "evaluation.json"
    eval_path.write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "  evaluation.json (%d question(s), %d with gold SQL, %d database(s))",
        len(evaluation),
        sum(1 for q in evaluation if q["SQL"]),
        len(databases),
    )
    return evaluation, databases


# --- SQLite databases --------------------------------------------------------


class DownloadError(RuntimeError):
    """A download failed, or returned something other than the expected file."""


def _check_zip_payload(path: Path, url: str) -> None:
    """Raise if *path* isn't a zip (hosts serve HTML error pages with HTTP 200)."""
    with path.open("rb") as f:
        head = f.read(4)
    if head.startswith(ZIP_MAGIC):
        return

    size = path.stat().st_size
    path.unlink(missing_ok=True)
    if head.lstrip().lower().startswith((b"<!do", b"<htm")):
        raise DownloadError(
            f"{url} returned an HTML page ({size} bytes), not a zip. "
            "Google Drive serves a 'Quota exceeded' page for heavily "
            "downloaded files; retry later or pass --sqlite-root."
        )
    raise DownloadError(f"{url} returned {size} bytes that are not a zip archive.")


def _download(url: str, dest_path: Path, *, force: bool = False) -> None:
    """Stream-download the zip at *url* to *dest_path*, logging progress."""
    if dest_path.exists() and not force:
        logger.info("Using cached archive at %s", dest_path)
        _check_zip_payload(dest_path, url)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, context=_ssl_context()) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_pct = -1

            with dest_path.open("wb") as out:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct >= last_pct + 5:
                            logger.info(
                                "  %d%% (%d / %d bytes)", pct, downloaded, total
                            )
                            last_pct = pct
                    elif downloaded % (10 * CHUNK_SIZE) == 0:
                        logger.info("  %d bytes downloaded ...", downloaded)

    except urllib.error.URLError as exc:
        dest_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    logger.info("Download complete: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    _check_zip_payload(dest_path, url)


def _sqlite_path(dest: Path, db_id: str) -> Path:
    return dest / db_id / f"{db_id}.sqlite"


def _install_from_root(root: Path, wanted: dict[str, str], dest: Path) -> set[str]:
    """Copy wanted ``*.sqlite`` files found under *root*; *wanted* maps match key -> db_id."""
    installed: set[str] = set()
    for src in sorted(root.rglob("*.sqlite")):
        if _is_macos_junk(str(src)):
            continue
        db_id = wanted.get(_norm_key(src.stem))
        if db_id is None or db_id in installed:
            continue
        target = _sqlite_path(dest, db_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        logger.info("  %s -> %s", src, target)
        installed.add(db_id)
    return installed


def _install_from_archive(
    archive_path: Path, wanted: dict[str, str], dest: Path
) -> set[str]:
    """Stream wanted ``*.sqlite`` members of *archive_path* into *dest*.

    Members are written to a ``.part`` file and renamed, so an interrupted run
    never leaves a truncated database that a re-run would then skip.
    """
    installed: set[str] = set()
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.endswith(".sqlite") or _is_macos_junk(name):
                continue
            db_id = wanted.get(_norm_key(Path(name).stem))
            if db_id is None or db_id in installed:
                continue
            target = _sqlite_path(dest, db_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".sqlite.part")
            with zf.open(info) as src, partial.open("wb") as out:
                shutil.copyfileobj(src, out, CHUNK_SIZE)
            partial.replace(target)
            logger.info("  %s -> %s (%d bytes)", name, target, target.stat().st_size)
            installed.add(db_id)
    return installed


def _install_databases(
    databases: dict[str, _Database],
    dest: Path,
    *,
    sqlite_root: Path | None,
    force: bool,
    keep_archive: bool,
) -> set[str]:
    """Install every database's SQLite file; return the db_ids still missing."""
    remaining = {
        db_id for db_id in databases if force or not _sqlite_path(dest, db_id).is_file()
    }
    if not remaining:
        logger.info("All %d SQLite database(s) already installed.", len(databases))
        return set()

    def wanted() -> dict[str, str]:
        return {_norm_key(databases[db_id].spider2_db): db_id for db_id in remaining}

    if sqlite_root is not None:
        logger.info("Copying SQLite databases from %s ...", sqlite_root)
        remaining -= _install_from_root(sqlite_root, wanted(), dest)
        return remaining

    for source in SQLITE_SOURCES:
        if not remaining:
            break
        logger.info("Source: %s (need %d database(s))", source.name, len(remaining))
        archive_path = dest / f".{source.archive_name}"
        try:
            _download(source.url, archive_path, force=force)
        except DownloadError as exc:
            logger.warning("  %s unavailable: %s", source.name, exc)
            continue

        remaining -= _install_from_archive(archive_path, wanted(), dest)

        if not keep_archive:
            archive_path.unlink(missing_ok=True)
            logger.info("  removed cached archive %s", archive_path)
        else:
            logger.info("  kept cached archive at %s", archive_path)

    return remaining


def _drop_broken_views(sqlite_path: Path) -> list[str]:
    """Drop the views in *sqlite_path* that SQLite cannot compile; return their names.

    Upstream ships some views over columns that do not exist (``oracle_sql``'s
    ``emp_hire_periods_with_name`` selects ``start_date`` from a table whose
    column is ``start_``). Such a view cannot be queried, so no question uses
    it, but reading its columns raises, which aborts ingest for the database.
    A view built on a broken view fails to compile too and goes with it.
    """

    def quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    with closing(sqlite3.connect(sqlite_path)) as conn, conn:
        views = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'view'"
            )
        ]
        broken: list[str] = []
        for view in views:
            try:
                conn.execute(f"SELECT * FROM {quote(view)} LIMIT 0")
            except sqlite3.Error as exc:
                logger.warning(
                    "  %s: dropping broken view %s (%s)", sqlite_path.name, view, exc
                )
                broken.append(view)
        for view in broken:
            conn.execute(f"DROP VIEW {quote(view)}")
    return broken


# --- summary -----------------------------------------------------------------


def _write_manifest(
    dest: Path,
    *,
    upstream_commit: str | None,
    evaluation: list[dict[str, Any]],
    databases: dict[str, _Database],
) -> None:
    manifest = {
        "source": {
            "repository": UPSTREAM_REPO,
            "commit": upstream_commit,
            "questions": "spider2-lite/spider2-lite.jsonl",
        },
        "question_count": len(evaluation),
        "questions_with_gold_sql": sum(1 for q in evaluation if q["SQL"]),
        "databases": [
            {
                "db_id": db.db_id,
                "spider2_db": db.spider2_db,
                "question_count": db.question_count,
                "table_count": db.table_count,
                "sqlite_installed": _sqlite_path(dest, db.db_id).is_file(),
                "dropped_views": db.dropped_views,
            }
            for db in sorted(databases.values(), key=lambda d: d.db_id)
        ],
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _update_env_file(env_path: Path, connection_strings_value: str) -> bool:
    """Set ``CONNECTION_STRINGS`` in *env_path*, preserving everything else.

    Replaces the first uncommented ``CONNECTION_STRINGS=...`` line if one
    exists, otherwise appends a new line. Returns ``True`` if *env_path* was
    written, ``False`` if it doesn't exist (nothing is created).
    """
    if not env_path.exists():
        return False

    new_line = f"CONNECTION_STRINGS={connection_strings_value}"
    lines = env_path.read_text().splitlines()

    for i, line in enumerate(lines):
        if line.startswith("CONNECTION_STRINGS="):
            lines[i] = new_line
            break
    else:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(new_line)

    env_path.write_text("\n".join(lines) + "\n")
    return True


def _print_summary(
    dest: Path,
    evaluation: list[dict[str, Any]],
    databases: dict[str, _Database],
    missing: set[str],
    *,
    write_env: bool,
) -> None:
    installed = sorted(db_id for db_id in databases if db_id not in missing)
    logger.info("=" * 60)
    logger.info("Spider2-lite (local) seed complete.")
    logger.info("  Destination : %s", dest)
    logger.info("  Questions   : %d", len(evaluation))
    logger.info("  Databases   : %d of %d installed", len(installed), len(databases))
    for db_id in sorted(databases):
        status = "" if db_id in installed else "  (MISSING sqlite)"
        logger.info("    - %s%s", db_id, status)
    logger.info("=" * 60)

    if missing:
        stranded = sum(1 for q in evaluation if q["db_id"] in missing)
        logger.warning(
            "%d database(s) have no SQLite file, stranding %d question(s). "
            "Re-run later (Google Drive quota resets) or pass --sqlite-root.",
            len(missing),
            stranded,
        )

    if not installed:
        return

    connection_strings_value = ",".join(
        f"sqlite:///{_sqlite_path(dest, db_id).resolve()}" for db_id in installed
    )
    env_path = _repo_root() / ".env"
    if write_env and _update_env_file(env_path, connection_strings_value):
        logger.info("Wrote CONNECTION_STRINGS to %s", env_path)
    else:
        logger.info("Add this to your .env:")
        logger.info("CONNECTION_STRINGS=%s", connection_strings_value)


def seed_spider2_lite(
    *,
    dest: Path | None = None,
    upstream_root: Path | None = None,
    ref: str = DEFAULT_UPSTREAM_REF,
    sqlite_root: Path | None = None,
    skip_databases: bool = False,
    force: bool = False,
    keep_archive: bool = False,
    write_env: bool = True,
) -> list[str]:
    """Seed the Spider2-lite local questions and databases under *dest*.

    With *upstream_root* the checkout there is used as-is; without it,
    ``third_party/Spider2/`` is cloned or updated to *ref*. Returns the sorted
    db_ids whose SQLite file is installed.
    """
    target = dest or _default_dest()
    target.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Seeding Spider2-lite (local) to %s", target)
    logger.info("=" * 60)

    if upstream_root is None:
        upstream_root = _default_upstream_root()
        _bootstrap_upstream(upstream_root, ref)
    upstream = _Upstream(upstream_root)
    _check_upstream(upstream)
    commit = _git_commit(upstream_root)
    logger.info(
        "Spider2 upstream: %s (%s)", upstream_root, commit or "not a git checkout"
    )

    evaluation, databases = _build_dataset(upstream, target)

    if skip_databases:
        logger.info("Skipping SQLite install (--skip-databases).")
        missing = {
            db_id for db_id in databases if not _sqlite_path(target, db_id).is_file()
        }
    else:
        missing = _install_databases(
            databases,
            target,
            sqlite_root=sqlite_root,
            force=force,
            keep_archive=keep_archive,
        )

    for db_id, database in databases.items():
        if db_id not in missing:
            database.dropped_views = _drop_broken_views(_sqlite_path(target, db_id))

    _write_manifest(
        target, upstream_commit=commit, evaluation=evaluation, databases=databases
    )
    _print_summary(target, evaluation, databases, missing, write_env=write_env)
    return sorted(db_id for db_id in databases if db_id not in missing)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed the Spider2-lite local (SQLite) questions, gold results, "
            "metadata and databases into datasets/spider2_lite/."
        ),
        epilog="SQLite sources, tried in order:\n"
        + "\n".join(f"  {s.name}: {s.url}" for s in SQLITE_SOURCES),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: datasets/spider2_lite/ in the repo root).",
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
        default=DEFAULT_UPSTREAM_REF,
        help="Upstream branch, tag or commit to check out (ignored with --upstream-root).",
    )
    parser.add_argument(
        "--sqlite-root",
        type=Path,
        default=None,
        help=(
            "Directory already holding the Spider2-lite *.sqlite files "
            "(e.g. an unpacked spider2-localdb); skips the download."
        ),
    )
    parser.add_argument(
        "--skip-databases",
        action="store_true",
        help="Write questions, gold results and metadata only; install no SQLite.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archives and overwrite installed databases.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zips cached under the destination directory.",
    )
    parser.add_argument(
        "--no-write-env",
        action="store_true",
        help=(
            "Don't write CONNECTION_STRINGS into .env; just print it for "
            "manual copy-paste instead."
        ),
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
    seed_spider2_lite(
        dest=args.dest,
        upstream_root=args.upstream_root,
        ref=args.ref,
        sqlite_root=args.sqlite_root,
        skip_databases=args.skip_databases,
        force=args.force,
        keep_archive=args.keep_archive,
        write_env=not args.no_write_env,
    )


if __name__ == "__main__":
    main()
