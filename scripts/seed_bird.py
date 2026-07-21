# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download BIRD split(s) (Mini-Dev / Dev / Train) and populate ``datasets/bird/``.

Fetches the official zip(s) for one or more BIRD splits and writes::

    datasets/bird/<db_id>/<db_id>.sqlite
    datasets/bird/<db_id>/database_description/*.csv
    datasets/bird/evaluation.json

Only the SQLite dialect is kept. Each split's question JSON is merged into a
single combined ``evaluation.json`` (each row carries its own ``db_id``). The
MySQL / PostgreSQL JSONs and the ``*_gold.sql`` files are ignored.

Three splits are available via ``--splits`` (default: ``mini-dev``):

- ``mini-dev`` — 500 questions, 11 SQLite DBs. Lite/cheap default for
  day-to-day development.
- ``dev`` — the full 1,534-question Dev split, same 11 DBs as mini-dev.
- ``train`` — the 9,428-question Train split, ~69 additional DBs. Rows have
  no ``difficulty`` label (fine-tuning split, not part of the official
  EX/VES leaderboard).

Pass more than one split to combine them into one shared ``datasets/bird/``
(e.g. ``--splits dev train``); when combining splits, each split's
``question_id`` values are offset so they don't collide across splits.

Usage::

    uv run python scripts/seed_bird.py
    uv run python scripts/seed_bird.py --splits dev
    uv run python scripts/seed_bird.py --splits dev train
    uv run python scripts/seed_bird.py --force

The default ``mini-dev`` download URL is the Google Drive "Complete Package"
from the BIRD Mini-Dev README's 2025-07-04 update — the corrected
500-question set (no duplicate rows, fixed gold SQL). The older Aliyun OSS
``minidev.zip`` is kept as ``LEGACY_OSS_URL`` for reference but is stale; pass
``--url`` to override (only valid with exactly one ``--splits`` value).

After download, ingest a single database into Neo4j via::

    CONNECTION_STRINGS=sqlite:///<abs-path>/datasets/bird/<db_id>/<db_id>.sqlite \\
      PYTHONPATH=../GSF uv run python main.py --database-name bird --skip-eval --skip-judge
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
# e.g. dev's ids 0-1533 and train's ids 0-9427 don't collide once merged.
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
    "train": _SplitConfig(
        url="https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
        archive_name="train.zip",
        json_names=("train.json",),
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dest() -> Path:
    return _repo_root() / "datasets" / "bird"


def _archive_cache_path(dest: Path, archive_name: str) -> Path:
    return dest / f".{archive_name}"


def _download(url: str, dest_path: Path, *, force: bool = False) -> None:
    """Stream-download *url* to *dest_path*, logging progress."""
    if dest_path.exists() and not force:
        logger.info("Using cached archive at %s", dest_path)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
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


def _extract_nested_zips(root: Path) -> None:
    """Recursively extract any nested zip archives found under *root*.

    BIRD's ``dev.zip`` / ``train.zip`` ship one level of nesting (e.g. a
    ``dev_databases.zip`` inside the outer archive); Mini-Dev's ``minidev.zip``
    is already flat, so this is a no-op for it.
    """
    seen: set[Path] = set()
    while True:
        nested = [p for p in root.rglob("*.zip") if p not in seen]
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


def _organize_extracted(extract_dir: Path, dest: Path) -> list[str]:
    """Copy SQLite DBs and metadata from *extract_dir* into *dest*.

    Returns the sorted list of ``db_id`` values that were installed.
    """
    dest.mkdir(parents=True, exist_ok=True)

    sqlite_files = sorted(extract_dir.rglob("*.sqlite"))
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

        db_ids.append(db_id)

    return sorted(set(db_ids))


def _load_eval_rows(extract_dir: Path, json_names: tuple[str, ...]) -> list[dict[str, Any]]:
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
    """Write the combined question set to ``datasets/bird/evaluation.json``."""
    eval_path = dest / "evaluation.json"
    with eval_path.open("w") as f:
        json.dump(rows, f, indent=2)
    logger.info("  evaluation.json (%d question(s))", len(rows))


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
    """Log a summary, write ``CONNECTION_STRINGS`` to ``.env``, and print it."""
    logger.info("=" * 60)
    logger.info("BIRD download complete (splits: %s).", ", ".join(splits))
    logger.info("  Destination : %s", dest)
    logger.info("  Databases   : %d", len(db_ids))
    for db_id in db_ids:
        logger.info("    - %s", db_id)
    logger.info("=" * 60)

    if not db_ids:
        return

    conn_strings = [
        f"sqlite:///{(dest / db_id / f'{db_id}.sqlite').resolve()}" for db_id in db_ids
    ]
    connection_strings_value = ",".join(conn_strings)

    env_path = _repo_root() / ".env"
    if write_env and _update_env_file(env_path, connection_strings_value):
        logger.info("Wrote CONNECTION_STRINGS to %s", env_path)
    else:
        logger.info("Add this to your .env (all databases, comma-separated):")
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
    """Download and organize one or more BIRD splits into a shared *dest*.

    Returns the sorted list of ``db_id`` values installed under *dest*,
    merged across every requested split.
    """
    split_names = splits or [DEFAULT_SPLIT]
    if url and len(split_names) != 1:
        raise ValueError("--url can only be used with a single --splits value.")

    target = dest or _default_dest()
    multi = len(split_names) > 1

    logger.info("=" * 60)
    logger.info("Downloading BIRD split(s) [%s] to %s", ", ".join(split_names), target)
    logger.info("=" * 60)

    all_db_ids: set[str] = set()
    all_rows: list[dict[str, Any]] = []

    for i, split_name in enumerate(split_names):
        config = _SPLITS[split_name]
        effective_url = url or config.url
        archive_path = _archive_cache_path(target, config.archive_name)

        logger.info("--- Split: %s ---", split_name)
        _download(effective_url, archive_path, force=force)

        with tempfile.TemporaryDirectory(prefix=f"bird_{split_name}_") as tmp:
            extract_dir = Path(tmp)
            _extract(archive_path, extract_dir)

            logger.info("Organizing files into %s ...", target)
            db_ids = _organize_extracted(extract_dir, target)
            all_db_ids.update(db_ids)

            rows = _load_eval_rows(extract_dir, config.json_names)
            offset = i * _SPLIT_ID_OFFSET if multi else 0
            all_rows.extend(_offset_question_ids(rows, offset))

        if not keep_archive and archive_path.exists():
            archive_path.unlink()
            logger.info("Removed cached archive %s", archive_path)
        elif keep_archive:
            logger.info("Kept cached archive at %s", archive_path)

    if all_rows:
        _write_evaluation_file(target, all_rows)
    else:
        logger.warning(
            "No eval rows collected across split(s) [%s]; evaluation.json not written.",
            ", ".join(split_names),
        )

    db_ids = sorted(all_db_ids)
    _print_summary(target, db_ids, splits=split_names, write_env=write_env)
    return db_ids


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download BIRD split(s) (mini-dev / dev / train) and populate "
            "datasets/bird/ with SQLite databases, schema descriptions, and "
            "a merged evaluation.json."
        ),
        epilog=(
            f"mini-dev source (corrected 2025-07-04 set): {GOOGLE_DRIVE_URL}\n"
            f"mini-dev legacy Aliyun OSS mirror (stale 2024 data): {LEGACY_OSS_URL}\n"
            f"dev source: {_SPLITS['dev'].url}\n"
            f"train source: {_SPLITS['train'].url}\n"
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
            "Which BIRD split(s) to download, merged into one datasets/bird/ "
            f"(default: {DEFAULT_SPLIT}, 500 Qs / 11 DBs). 'dev' is the full "
            "1,534-question Dev split (same 11 DBs as mini-dev); 'train' is "
            "the 9,428-question Train split (~69 additional DBs, no "
            "difficulty labels). Pass more than one to combine them, e.g. "
            "--splits dev train."
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
