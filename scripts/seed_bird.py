# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the BIRD Mini-Dev dataset and populate ``datasets/bird/``.

Fetches the official Mini-Dev zip and writes::

    datasets/bird/<db_id>/<db_id>.sqlite
    datasets/bird/<db_id>/database_description/*.csv
    datasets/bird/evaluation.json

Only the SQLite dialect is kept. The archive's ``mini_dev_sqlite.json`` is kept
as a single combined ``evaluation.json`` (each row carries its own ``db_id``).
The MySQL / PostgreSQL JSONs and the ``*_gold.sql`` files are ignored.

Usage::

    uv run python scripts/seed_bird.py
    uv run python scripts/seed_bird.py --force

The default download URL is the Google Drive "Complete Package" from the BIRD
Mini-Dev README's 2025-07-04 update — the corrected 500-question set (no
duplicate rows, fixed gold SQL). The older Aliyun OSS ``minidev.zip`` is kept
as ``LEGACY_OSS_URL`` for reference but is stale; pass ``--url`` to override.

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
from pathlib import Path

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
ARCHIVE_NAME = "minidev.zip"
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dest() -> Path:
    return _repo_root() / "datasets" / "bird"


def _archive_cache_path(dest: Path) -> Path:
    return dest / f".{ARCHIVE_NAME}"


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
    """Extract *archive_path* into *extract_dir*."""
    logger.info("Extracting %s ...", archive_path.name)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extraction complete.")


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

    _write_evaluation_file(extract_dir, dest)

    return sorted(set(db_ids))


def _write_evaluation_file(extract_dir: Path, dest: Path) -> None:
    """Write the combined SQLite question set to ``datasets/bird/evaluation.json``.

    BIRD ships a single ``mini_dev_sqlite.json`` spanning every database; each
    row carries its own ``db_id``, so we keep it as one file (renamed to
    ``evaluation.json``). The MySQL / PostgreSQL JSONs and the ``*_gold.sql``
    files are intentionally ignored — this SQLite-only pipeline doesn't use them.
    """
    matches = sorted(extract_dir.rglob("mini_dev_sqlite.json"))
    if not matches:
        logger.warning(
            "No mini_dev_sqlite.json found in the archive; "
            "skipping evaluation.json generation."
        )
        return

    with matches[0].open() as f:
        rows = json.load(f)

    eval_path = dest / "evaluation.json"
    with eval_path.open("w") as f:
        json.dump(rows, f, indent=2)
    logger.info("  evaluation.json (%d question(s))", len(rows))


def _print_summary(dest: Path, db_ids: list[str]) -> None:
    """Log a summary and print ingest command hints."""
    logger.info("=" * 60)
    logger.info("BIRD Mini-Dev download complete.")
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
    print()
    print("Add this to your .env (all databases, comma-separated):")
    print()
    print(f"CONNECTION_STRINGS={','.join(conn_strings)}")


def download_bird(
    *,
    url: str = DEFAULT_URL,
    dest: Path | None = None,
    force: bool = False,
    keep_archive: bool = False,
) -> list[str]:
    """Download and organize the BIRD Mini-Dev dataset.

    Returns the list of ``db_id`` values installed under *dest*.
    """
    target = dest or _default_dest()
    archive_path = _archive_cache_path(target)

    logger.info("=" * 60)
    logger.info("Downloading BIRD Mini-Dev to %s", target)
    logger.info("=" * 60)

    _download(url, archive_path, force=force)

    with tempfile.TemporaryDirectory(prefix="bird_minidev_") as tmp:
        extract_dir = Path(tmp)
        _extract(archive_path, extract_dir)

        logger.info("Organizing files into %s ...", target)
        db_ids = _organize_extracted(extract_dir, target)

    if not keep_archive and archive_path.exists():
        archive_path.unlink()
        logger.info("Removed cached archive %s", archive_path)
    elif keep_archive:
        logger.info("Kept cached archive at %s", archive_path)

    _print_summary(target, db_ids)
    return db_ids


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the BIRD Mini-Dev dataset and populate datasets/bird/ "
            "with SQLite databases, schema descriptions, question JSONs, and gold SQL."
        ),
        epilog=(
            f"Default source (corrected 2025-07-04 set): {GOOGLE_DRIVE_URL}\n"
            f"Legacy Aliyun OSS mirror (stale 2024 data): {LEGACY_OSS_URL}\n"
            "Pass --url if you host a local copy of the zip."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Direct zip download URL (default: {DEFAULT_URL}).",
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
        url=args.url,
        dest=args.dest,
        force=args.force,
        keep_archive=args.keep_archive,
    )


if __name__ == "__main__":
    main()
