# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download BIRD evaluation split(s) and the Dev training corpus.

Fetches the official zip(s) for one or more BIRD splits and writes::

    datasets/bird/evaluation.json                 # Mini-Dev / Dev questions only
    datasets/bird/dev/<db_id>/<db_id>.sqlite      # evaluation databases
    datasets/bird/dev/<db_id>/database_description/*.csv
    datasets/bird/dev/<db_id>/metadata.json       # derived column descriptions
    datasets/bird/train/train.json                # Train Q/evidence/SQL rows

Mini-Dev / Dev SQLite DBs land under ``dev/`` and drive evaluation.
Selecting Dev also stores the Train questions under ``train/`` for future
few-shot use; Train databases are not installed.

The per-database ``metadata.json`` is derived from BIRD's own
``database_description/*.csv`` files and written in the shape consumed by the
ingest enrichment step (``enrich_graph.apply_metadata``), so column meanings /
value descriptions reach the text-to-SQL prompt at eval time.

Two evaluation splits are available via ``--splits`` (default: ``dev``):

- ``mini-dev`` — 500 questions, 11 SQLite DBs under ``dev/``.
- ``dev`` — the full 1,534-question Dev split plus the 9,428-row Train corpus.

Usage::

    uv run python scripts/seed_bird.py
    uv run python scripts/seed_bird.py --splits dev
    uv run python scripts/seed_bird.py --force

The default ``mini-dev`` download URL is the Google Drive "Complete Package"
from the BIRD Mini-Dev README's 2025-07-04 update — the corrected
500-question set (no duplicate rows, fixed gold SQL). The older Aliyun OSS
``minidev.zip`` is kept as ``LEGACY_OSS_URL`` for reference but is stale; pass
``--url`` to override (only valid with exactly one ``--splits`` value).

After download, ingest the Dev databases into the Postgres catalog via::

    CONNECTION_STRINGS=sqlite:///<abs-path>/datasets/bird/dev/<db_id>/<db_id>.sqlite \\
      PYTHONPATH=../GSF uv run python main.py --database-name bird --skip-eval --skip-judge
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
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


GOOGLE_DRIVE_FILE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"
GOOGLE_DRIVE_URL = (
    f"https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view?usp=sharing"
)
# Direct-download form that streams the raw bytes (the ``confirm=t`` token
# bypasses Google Drive's large-file virus-scan interstitial).
DEFAULT_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={GOOGLE_DRIVE_FILE_ID}&export=download&confirm=t"
)
# Legacy Aliyun OSS mirror linked from the top README badge. Kept for reference
# only: it serves the stale 2024 snapshot (duplicate question_ids 137/138,
# missing 119/120, wrong gold SQL for 1322).
LEGACY_OSS_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class _SplitConfig:
    url: str
    archive_name: str
    # Candidate filenames for the split's question JSON, tried in order.
    json_names: tuple[str, ...]


DEFAULT_SPLIT = "dev"
# Offset applied to question_id per split index when combining >1 split, so
# evaluation splits do not collide once merged.
_SPLIT_ID_OFFSET = 100_000

_SPLITS: dict[str, _SplitConfig] = {
    "mini-dev": _SplitConfig(
        url=DEFAULT_URL,
        archive_name="minidev.zip",
        json_names=("mini_dev_sqlite.json",),
    ),
    "dev": _SplitConfig(
        url="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        archive_name="dev.zip",
        json_names=("dev.json",),
    ),
}

_TRAIN_CONFIG = _SplitConfig(
    url="https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    archive_name="train.zip",
    json_names=("train.json",),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dest() -> Path:
    return _repo_root() / "datasets" / "bird"


def _archive_cache_path(dest: Path, archive_name: str) -> Path:
    return dest / f".{archive_name}"


def _eval_db_root(dest: Path) -> Path:
    """SQLite evaluation databases live under ``datasets/bird/dev/``."""
    return dest / "dev"


def _train_root(dest: Path) -> Path:
    """Train question rows live under ``datasets/bird/train/``."""
    return dest / "train"


def _is_macos_junk(path: Path) -> bool:
    """True for AppleDouble / ``__MACOSX`` sidecar paths."""
    return "__MACOSX" in path.parts or any(part.startswith("._") for part in path.parts)


def _download(url: str, dest_path: Path, *, force: bool = False) -> None:
    """Stream-download *url* to *dest_path*, logging progress."""
    if dest_path.exists() and not force:
        logger.info("Using cached archive at %s", dest_path)
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
                                "  %d%% (%d / %d bytes)",
                                pct,
                                downloaded,
                                total,
                            )
                            last_pct = pct
                    elif downloaded % (10 * CHUNK_SIZE) == 0:
                        logger.info("  %d bytes downloaded ...", downloaded)

    except urllib.error.URLError as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    logger.info("Download complete: %s (%d bytes)", dest_path, dest_path.stat().st_size)


def _extract(archive_path: Path, extract_dir: Path) -> None:
    """Extract *archive_path* into *extract_dir*, including nested zips."""
    logger.info("Extracting %s ...", archive_path.name)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    _extract_nested_zips(extract_dir)
    logger.info("Extraction complete.")


def _extract_train_questions(archive_path: Path, extract_dir: Path) -> None:
    """Extract only ``train.json`` and skip the Train database archives."""
    logger.info("Extracting Train questions from %s ...", archive_path.name)
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [
            name
            for name in zf.namelist()
            if Path(name).name == "train.json" and not _is_macos_junk(Path(name))
        ]
        if not members:
            raise RuntimeError(
                f"No train.json found in {archive_path}. "
                "The archive layout may have changed."
            )
        for name in members:
            zf.extract(name, extract_dir)
            logger.info("  extracted %s", name)
    logger.info("Train question extraction complete.")


def _extract_nested_zips(root: Path) -> None:
    """Recursively extract any nested zip archives found under *root*.

    BIRD's ``dev.zip`` ships one level of nesting (for example,
    ``dev_databases.zip`` inside the outer archive); Mini-Dev's ``minidev.zip``
    is already flat.
    """
    seen: set[Path] = set()
    while True:
        # Skip AppleDouble / __MACOSX sidecar stubs (e.g. ._train_databases.zip),
        # which match *.zip but are not valid archives.
        nested = [
            p for p in root.rglob("*.zip") if p not in seen and not _is_macos_junk(p)
        ]
        if not nested:
            return
        for zip_path in nested:
            seen.add(zip_path)
            logger.info(
                "  extracting nested archive %s ...", zip_path.relative_to(root)
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_path.parent)


def _db_root_for_sqlite(sqlite_path: Path) -> Path:
    """Return the BIRD database folder that owns *sqlite_path*."""
    if sqlite_path.parent.name == "sqlite":
        return sqlite_path.parent.parent
    return sqlite_path.parent


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree, replacing *dst* if it already exists."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _clean_cell(text: str | None) -> str:
    """Collapse a CSV cell's internal whitespace/newlines into one line."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _read_description_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse one BIRD ``database_description`` CSV into column metadata.

    BIRD ships one CSV per table with the columns ``original_column_name``
    (the real DB column), ``column_name`` (a friendlier/expanded label),
    ``column_description``, ``data_format`` and ``value_description``. A handful
    of these CSVs are Windows-1252 rather than UTF-8 encoded, so decoding falls
    back to cp1252.

    Each returned entry matches the ``columns`` shape ``apply_metadata`` reads:
    ``name`` is the real DB column (so it matches the catalog's ``Column`` rows),
    ``description`` is ``column_description`` and ``value_description`` joined by
    a comma, and ``value_examples`` is left ``None`` (BIRD gives prose, not
    discrete values). A ``value_description`` that is exactly ``not useful``
    (case-insensitive) is dropped from the description — it's an annotator note
    with no signal — but the column entry is still kept.
    """
    try:
        raw = csv_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = csv_path.read_text(encoding="cp1252")

    columns: list[dict[str, Any]] = []
    for row in csv.DictReader(raw.splitlines()):
        # Normalise header keys (strip stray BOM/whitespace) for stable lookups.
        norm = {(k or "").strip().lstrip("\ufeff"): v for k, v in row.items()}
        original = _clean_cell(norm.get("original_column_name"))
        friendly = _clean_cell(norm.get("column_name"))
        name = original or friendly
        if not name:
            continue

        col_desc = _clean_cell(norm.get("column_description"))
        val_desc = _clean_cell(norm.get("value_description"))
        # Drop annotator "not useful" notes: they carry no signal (they flag
        # opaque ID/redundant columns), but the column itself must stay — some
        # (e.g. card_games.cards.uuid) are join keys used by many gold queries.
        if val_desc.lower() == "not useful":
            val_desc = ""
        description = ", ".join(p for p in (col_desc, val_desc) if p) or None

        columns.append(
            {"name": name, "description": description, "value_examples": None}
        )
    return columns


def _write_metadata_json(db_dest: Path) -> int:
    """Generate ``<db_dest>/metadata.json`` from that DB's description CSVs.

    Writes an object keyed by table name (one CSV = one table, the file stem is
    the table name), each with a ``description`` and a list of ``columns`` —
    exactly the shape ``enrich_graph.apply_metadata`` consumes. Returns the
    number of tables written (0 when there is no ``database_description`` folder
    or nothing parseable in it). A malformed CSV is logged and skipped rather
    than failing the whole download.
    """
    desc_dir = db_dest / "database_description"
    if not desc_dir.is_dir():
        return 0

    metadata: dict[str, Any] = {}
    for csv_path in sorted(desc_dir.glob("*.csv")):
        try:
            columns = _read_description_csv(csv_path)
        except Exception as exc:
            logger.warning("  could not parse %s: %s", csv_path.name, exc)
            continue
        if columns:
            # BIRD has no table-level description; leave it blank so enrichment
            # preserves the existing catalog value instead of overwriting it.
            metadata[csv_path.stem] = {"description": "", "columns": columns}

    if not metadata:
        return 0

    (db_dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(metadata)


def _organize_extracted(extract_dir: Path, dest: Path) -> list[str]:
    """Copy SQLite DBs and metadata from *extract_dir* into *dest*.

    Returns the sorted list of ``db_id`` values that were installed.
    """
    dest.mkdir(parents=True, exist_ok=True)

    sqlite_files = sorted(
        p for p in extract_dir.rglob("*.sqlite") if not _is_macos_junk(p)
    )
    if not sqlite_files:
        raise RuntimeError(
            f"No .sqlite files found under {extract_dir}. "
            "The archive layout may have changed."
        )

    db_ids: list[str] = []
    for sqlite_path in sqlite_files:
        db_id = sqlite_path.stem
        db_dest = dest / db_id
        db_dest.mkdir(parents=True, exist_ok=True)

        sqlite_dest = db_dest / f"{db_id}.sqlite"
        shutil.copy2(sqlite_path, sqlite_dest)
        logger.info("  %s -> %s", sqlite_path.relative_to(extract_dir), sqlite_dest)

        db_root = _db_root_for_sqlite(sqlite_path)
        desc_src = db_root / "database_description"
        if desc_src.is_dir():
            desc_dest = db_dest / "database_description"
            _copy_tree(desc_src, desc_dest)
            csv_count = len(list(desc_dest.glob("*.csv")))
            logger.info(
                "  %s/database_description/ (%d CSV file(s))",
                db_id,
                csv_count,
            )
            # Derive metadata.json from the description CSVs so the ingest
            # enrichment step can stamp column meanings onto the catalog.
            table_count = _write_metadata_json(db_dest)
            if table_count:
                logger.info(
                    "  %s/metadata.json (%d table(s) from descriptions)",
                    db_id,
                    table_count,
                )

        db_ids.append(db_id)

    return sorted(set(db_ids))


def _load_eval_rows(
    extract_dir: Path, json_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Load a split's question JSON, trying *json_names* in order.

    BIRD ships one question JSON per split (``mini_dev_sqlite.json``,
    ``dev.json``, ``train.json``); each row carries its own ``db_id``. The
    MySQL / PostgreSQL JSONs and the ``*_gold.sql`` / ``*_tables.json`` files
    are intentionally ignored — this SQLite-only pipeline doesn't use them.
    """
    for name in json_names:
        matches = sorted(extract_dir.rglob(name))
        if matches:
            with matches[0].open() as f:
                return json.load(f)

    logger.warning(
        "No question JSON (%s) found under %s; skipping.",
        " / ".join(json_names),
        extract_dir,
    )
    return []


def _offset_question_ids(
    rows: list[dict[str, Any]], offset: int
) -> list[dict[str, Any]]:
    """Shift each row's ``question_id`` by *offset* (no-op when *offset* is 0).

    Used to keep ids unique when combining multiple splits into one
    ``evaluation.json`` — each split independently numbers its own questions
    starting from 0, so a fixed per-split offset avoids collisions.
    """
    if not offset:
        return rows

    offset_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        if new_row.get("question_id") is not None:
            new_row["question_id"] = int(new_row["question_id"]) + offset
        offset_rows.append(new_row)
    return offset_rows


def _write_evaluation_file(dest: Path, rows: list[dict[str, Any]]) -> None:
    """Write Mini-Dev / Dev questions to ``datasets/bird/evaluation.json``."""
    eval_path = dest / "evaluation.json"
    with eval_path.open("w") as f:
        json.dump(rows, f, indent=2)
    logger.info("  evaluation.json (%d question(s))", len(rows))


def _write_train_file(dest: Path, rows: list[dict[str, Any]]) -> None:
    """Preserve complete BIRD Train rows for future few-shot retrieval."""
    dest.mkdir(parents=True, exist_ok=True)
    train_path = dest / "train.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info("  train/train.json (%d question(s))", len(rows))


def _cleanup_legacy_flat_layout(dest: Path) -> None:
    """Remove pre-split flat ``datasets/bird/<db_id>/`` entries if present."""
    keep_names = {"dev", "train", "evaluation.json", "README.md", ".gitkeep"}
    removed = 0
    for child in dest.iterdir():
        if child.name.startswith(".") or child.name in keep_names:
            continue
        is_legacy_db = child.is_dir() and (
            (child / f"{child.name}.sqlite").exists() or child.name.startswith("._")
        )
        if is_legacy_db or (child.is_dir() and child.name.startswith("._")):
            shutil.rmtree(child)
            removed += 1
        elif child.is_file() and child.name.startswith("._"):
            child.unlink()
            removed += 1
    if removed:
        logger.info("Removed %d legacy flat entry(ies) under %s", removed, dest)


def _migrate_legacy_flat_layout(dest: Path) -> None:
    """Move a pre-split flat ``datasets/bird/<db_id>/`` tree into ``dev/``.

    Evaluation DB ids are taken from ``evaluation.json`` when present; their
    SQLite folders move under ``dev/``. Remaining flat DB folders (Train-only
    SQLite or leftover history dirs) are deleted.
    """
    if not dest.is_dir():
        return

    eval_path = dest / "evaluation.json"
    eval_db_ids: set[str] = set()
    if eval_path.exists():
        with eval_path.open() as f:
            rows = json.load(f)
        eval_db_ids = {
            str(row.get("db_id") or "").strip()
            for row in rows
            if str(row.get("db_id") or "").strip()
        }

    eval_root = _eval_db_root(dest)
    keep_names = {"dev", "train", "evaluation.json", "README.md", ".gitkeep"}
    moved_dev = 0

    for child in sorted(dest.iterdir()):
        if child.name.startswith(".") or child.name in keep_names or not child.is_dir():
            continue
        if child.name.startswith("._"):
            shutil.rmtree(child)
            continue

        sqlite_src = child / f"{child.name}.sqlite"
        if child.name in eval_db_ids and sqlite_src.exists():
            db_dest = eval_root / child.name
            if not (db_dest / f"{child.name}.sqlite").exists():
                _copy_tree(child, db_dest)
                leftover_history = db_dest / "query_history.csv"
                if leftover_history.exists():
                    leftover_history.unlink()
                moved_dev += 1

        shutil.rmtree(child)

    if moved_dev:
        logger.info(
            "Migrated legacy flat layout: %d eval DB(s) -> %s",
            moved_dev,
            eval_root,
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
    db_ids: list[str],
    *,
    splits: list[str],
    write_env: bool = True,
) -> None:
    """Log a summary, write ``CONNECTION_STRINGS`` to ``.env``, and print it.

    Connection strings point at evaluation SQLite files under
    ``datasets/bird/dev/``.
    """
    eval_root = _eval_db_root(dest)
    logger.info("=" * 60)
    logger.info("BIRD download complete (splits: %s).", ", ".join(splits))
    logger.info("  Destination : %s", dest)
    logger.info("  Eval DBs    : %s (%d)", eval_root, len(db_ids))
    for db_id in db_ids:
        logger.info("    - %s", db_id)
    if "dev" in splits:
        logger.info("  Train Qs    : %s/train.json", _train_root(dest))
    logger.info("=" * 60)

    if not db_ids:
        return

    conn_strings = [
        f"sqlite:///{(eval_root / db_id / f'{db_id}.sqlite').resolve()}"
        for db_id in db_ids
    ]
    connection_strings_value = ",".join(conn_strings)

    env_path = _repo_root() / ".env"
    if write_env and _update_env_file(env_path, connection_strings_value):
        logger.info("Wrote CONNECTION_STRINGS to %s", env_path)
    else:
        logger.info("Add this to your .env:")
        logger.info("CONNECTION_STRINGS=%s", connection_strings_value)


def download_bird(
    *,
    splits: list[str] | None = None,
    url: str | None = None,
    dest: Path | None = None,
    force: bool = False,
    keep_archive: bool = False,
    write_env: bool = True,
) -> list[str]:
    """Download and organize one or more BIRD splits under *dest*.

    Returns the sorted list of evaluation ``db_id`` values installed under
    ``dest/dev/``.
    """
    split_names = splits or [DEFAULT_SPLIT]
    if url and len(split_names) != 1:
        raise ValueError("--url can only be used with a single --splits value.")

    target = dest or _default_dest()
    multi = len(split_names) > 1
    eval_root = _eval_db_root(target)

    logger.info("=" * 60)
    logger.info("Downloading BIRD split(s) [%s] to %s", ", ".join(split_names), target)
    logger.info("=" * 60)

    _migrate_legacy_flat_layout(target)

    all_db_ids: set[str] = set()
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_split_index = 0

    for split_name in split_names:
        config = _SPLITS[split_name]
        effective_url = url or config.url
        archive_path = _archive_cache_path(target, config.archive_name)

        logger.info("--- Split: %s ---", split_name)
        _download(effective_url, archive_path, force=force)

        with tempfile.TemporaryDirectory(prefix=f"bird_{split_name}_") as tmp:
            extract_dir = Path(tmp)
            _extract(archive_path, extract_dir)
            logger.info("Organizing evaluation DBs into %s ...", eval_root)
            db_ids = _organize_extracted(extract_dir, eval_root)
            all_db_ids.update(db_ids)

            rows = _load_eval_rows(extract_dir, config.json_names)
            offset = (
                evaluation_split_index * _SPLIT_ID_OFFSET
                if multi and evaluation_split_index
                else 0
            )
            evaluation_rows.extend(_offset_question_ids(rows, offset))
            evaluation_split_index += 1

        if not keep_archive and archive_path.exists():
            archive_path.unlink()
            logger.info("Removed cached archive %s", archive_path)
        elif keep_archive:
            logger.info("Kept cached archive at %s", archive_path)

    if "dev" in split_names:
        train_archive = _archive_cache_path(target, _TRAIN_CONFIG.archive_name)
        logger.info("--- Training corpus (included with Dev) ---")
        _download(_TRAIN_CONFIG.url, train_archive, force=force)
        with tempfile.TemporaryDirectory(prefix="bird_train_") as tmp:
            extract_dir = Path(tmp)
            _extract_train_questions(train_archive, extract_dir)
            train_rows = _load_eval_rows(extract_dir, _TRAIN_CONFIG.json_names)
            if train_rows:
                _write_train_file(_train_root(target), train_rows)
            else:
                logger.warning("No Train rows collected; train/train.json not written.")

        if not keep_archive and train_archive.exists():
            train_archive.unlink()
            logger.info("Removed cached archive %s", train_archive)
        elif keep_archive:
            logger.info("Kept cached archive at %s", train_archive)

    if evaluation_rows:
        _write_evaluation_file(target, evaluation_rows)
    else:
        logger.warning(
            "No Mini-Dev / Dev rows collected across split(s) [%s]; "
            "evaluation.json not written.",
            ", ".join(split_names),
        )
    _cleanup_legacy_flat_layout(target)

    db_ids = sorted(all_db_ids)
    _print_summary(target, db_ids, splits=split_names, write_env=write_env)
    return db_ids


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download BIRD evaluation split(s) (mini-dev / dev). Dev also "
            "installs the Train question corpus."
        ),
        epilog=(
            f"mini-dev source (corrected 2025-07-04 set): {GOOGLE_DRIVE_URL}\n"
            f"mini-dev legacy Aliyun OSS mirror (stale 2024 data): {LEGACY_OSS_URL}\n"
            f"dev source: {_SPLITS['dev'].url}\n"
            f"train source (included with dev): {_TRAIN_CONFIG.url}\n"
            "Pass --url to override the URL for a single --splits value "
            "(e.g. to host a local copy of the zip)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(_SPLITS),
        default=[DEFAULT_SPLIT],
        help=(
            "Which BIRD split(s) to install (default: "
            f"{DEFAULT_SPLIT}). mini-dev/dev write SQLite under "
            "datasets/bird/dev/ and evaluation.json; dev also writes "
            "datasets/bird/train/train.json."
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "Override the download URL for the (single) requested split "
            "(default: that split's official URL). Only valid with exactly "
            "one --splits value."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: datasets/bird/ in the repo root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive and overwrite existing files.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip cached under the destination directory.",
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
    download_bird(
        splits=args.splits,
        url=args.url,
        dest=args.dest,
        force=args.force,
        keep_archive=args.keep_archive,
        write_env=not args.no_write_env,
    )


if __name__ == "__main__":
    main()
