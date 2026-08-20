# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download FDABench-Lite and populate ``datasets/fdabench/``.

Fetches the HuggingFace Lite task JSONLs, converts gold SQL subtasks into a
combined ``evaluation.json``, and installs only the SQLite databases those
tasks need::

    datasets/fdabench/evaluation.json
    datasets/fdabench/<db_id>/<db_id>.sqlite
    datasets/fdabench/<db_id>/database_description/*.csv  # BIRD only
    datasets/fdabench/<db_id>/metadata.json               # from those CSVs

Dabstep tasks are skipped (no redistributable ``merchant_data.db``). Tasks
without ``expected_SQL`` in ``gold_subtasks`` are also skipped. All remaining
Lite tasks use SQLite (BIRD train, Spider1, Spider2-lite ``local*``).

Database archives (downloaded only when the matching ``--*-root`` is omitted)::

    BIRD train          ~9 GB   Aliyun OSS train.zip
    Spider2-lite local  ~1.6 GB Google Drive local_sqlite.zip
    Spider1             ~1 GB   Google Drive Spider dataset zip

Usage::

    uv run python scripts/seed_fdabench.py
    uv run python scripts/seed_fdabench.py --tasks-only
    uv run python scripts/seed_fdabench.py --bird-root /path/to/train_databases

After seeding::

    PYTHONPATH=../GSF uv run python main.py --database-name fdabench
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


CHUNK_SIZE = 1024 * 1024  # 1 MiB
ZIP_MAGIC = b"PK\x03\x04"

HF_LITE_BASE = (
    "https://huggingface.co/datasets/FDAbench2026/Fdabench-Lite/resolve/main"
)
HF_SPLITS = ("report", "single", "multiple")


@dataclass(frozen=True)
class ArchiveSource:
    """A downloadable zip that may contain some of the wanted databases."""

    name: str
    url: str
    archive_name: str


# BIRD train ships every database inside a nested ``train/train_databases.zip``.
BIRD_SOURCES = (
    ArchiveSource(
        name="BIRD train (Aliyun OSS)",
        url="https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
        archive_name="bird_train.zip",
    ),
)

# The complete Spider2-lite local pack is only on Google Drive, which regularly
# refuses large downloads with a "Quota exceeded" HTML page. The HuggingFace
# mirror is reliable but is an older, partial snapshot, so try it first and let
# Google Drive fill in whatever it is missing.
SPIDER2_SOURCES = (
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

SPIDER1_SOURCES = (
    ArchiveSource(
        name="Spider 1.0 databases (HuggingFace mirror)",
        url=(
            "https://huggingface.co/datasets/HAL-9001/spider-databases"
            "/resolve/main/spider_data.zip"
        ),
        archive_name="spider1_data_hf.zip",
    ),
    ArchiveSource(
        name="Spider 1.0 dataset (Google Drive)",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
        ),
        archive_name="spider1.zip",
    ),
)

# Known FDABench-Lite database_type groupings (used when --*-root is given and
# when deciding which archive to pull). Updated dynamically from the tasks.
BIRD_DB_TYPES = frozenset({"bird"})
SPIDER2_DB_TYPES = frozenset({"spider2-lite"})
SPIDER1_DB_TYPES = frozenset({"spider1"})
SKIP_DB_TYPES = frozenset({"dabstep"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dest() -> Path:
    return _repo_root() / "datasets" / "fdabench"


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
            "downloaded files; retry later or pass the matching --*-root flag."
        )
    raise DownloadError(f"{url} returned {size} bytes that are not a zip archive.")


def _download(
    url: str, dest_path: Path, *, force: bool = False, expect_zip: bool = False
) -> None:
    """Stream-download *url* to *dest_path*, logging progress."""
    if dest_path.exists() and not force:
        logger.info("Using cached archive at %s", dest_path)
        if expect_zip:
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
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    logger.info("Download complete: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    if expect_zip:
        _check_zip_payload(dest_path, url)


_SQL_START_RE = re.compile(r"\s*(with|select)\b", re.IGNORECASE)


def _parse_gold_subtask(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, dict) else None


def _extract_sql_from_task(task: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(natural_language_query, expected_SQL)`` from gold subtasks."""
    for raw in task.get("gold_subtasks") or []:
        subtask = _parse_gold_subtask(raw)
        if not subtask:
            continue
        sql = subtask.get("expected_SQL")
        if not sql or not isinstance(sql, str):
            continue
        # A chunk of Lite tasks carry "N/A" as their gold SQL; they have no
        # answerable query, so they can't be scored as text-to-SQL.
        if not _SQL_START_RE.match(sql):
            continue
        inp = subtask.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        question = inp.get("natural_language_query") or task.get("query") or ""
        if not isinstance(question, str) or not question.strip():
            continue
        return question.strip(), sql.strip()
    return None


def _normalize_db_type(database_type: str | None) -> str:
    return (database_type or "").strip().lower()


def _should_keep_task(task: dict[str, Any]) -> tuple[bool, str]:
    db_type = _normalize_db_type(task.get("database_type"))
    if db_type in SKIP_DB_TYPES:
        return False, "dabstep"
    if db_type in SPIDER2_DB_TYPES:
        instance_id = str(task.get("instance_id") or "")
        if not instance_id.startswith("local"):
            return False, "non_local_spider2"
    if _extract_sql_from_task(task) is None:
        return False, "placeholder_sql" if _has_placeholder_sql(task) else "no_sql"
    return True, ""


def _has_placeholder_sql(task: dict[str, Any]) -> bool:
    """True when the task declares gold SQL but it is a stub such as ``N/A``."""
    for raw in task.get("gold_subtasks") or []:
        subtask = _parse_gold_subtask(raw)
        if subtask and isinstance(subtask.get("expected_SQL"), str):
            return True
    return False


def _task_to_eval_row(task: dict[str, Any]) -> dict[str, Any]:
    extracted = _extract_sql_from_task(task)
    assert extracted is not None
    question, sql = extracted
    answer_raw = task.get("sql_result")
    if answer_raw is None:
        answer_raw = ""
    elif not isinstance(answer_raw, str):
        answer_raw = json.dumps(answer_raw)

    return {
        "question_id": task.get("task_id", ""),
        "db_id": task.get("db", ""),
        "question": question,
        "evidence": "",
        "SQL": sql,
        "difficulty": task.get("level") or "",
        "answer_raw": answer_raw,
        "answer": "",
    }


def _download_lite_tasks(cache_dir: Path, *, force: bool = False) -> list[dict[str, Any]]:
    """Download FDABench-Lite JSONLs and return parsed task dicts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for split in HF_SPLITS:
        url = f"{HF_LITE_BASE}/{split}/data.jsonl"
        dest = cache_dir / f"{split}.jsonl"
        _download(url, dest, force=force)

        split_count = 0
        with dest.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSONL in {dest} line {line_no}: {exc}"
                    ) from exc
                if not isinstance(task, dict):
                    raise RuntimeError(f"Expected object in {dest} line {line_no}")
                task_id = str(task.get("task_id") or "")
                if task_id and task_id in seen_ids:
                    continue
                if task_id:
                    seen_ids.add(task_id)
                task["_split"] = split
                tasks.append(task)
                split_count += 1
        logger.info("  %s: %d task(s)", split, split_count)

    logger.info("Loaded %d unique FDABench-Lite task(s)", len(tasks))
    return tasks


def _convert_tasks(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], Counter[str]]:
    """Convert Lite tasks to evaluation rows.

    Returns ``(rows, db_id -> database_type, skip_counts)``.
    """
    rows: list[dict[str, Any]] = []
    db_types: dict[str, str] = {}
    skip_counts: Counter[str] = Counter()

    for task in tasks:
        keep, reason = _should_keep_task(task)
        if not keep:
            skip_counts[reason] += 1
            continue
        row = _task_to_eval_row(task)
        db_id = str(row["db_id"])
        if not db_id:
            skip_counts["missing_db"] += 1
            continue
        rows.append(row)
        db_types[db_id] = _normalize_db_type(task.get("database_type"))

    return rows, db_types, skip_counts


def _write_evaluation_json(dest: Path, rows: list[dict[str, Any]]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    eval_path = dest / "evaluation.json"
    with eval_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    logger.info("Wrote %s (%d question(s))", eval_path, len(rows))
    return eval_path


def _target_sqlite_path(dest: Path, db_id: str) -> Path:
    return dest / db_id / f"{db_id}.sqlite"


def _target_metadata_path(dest: Path, db_id: str) -> Path:
    return dest / db_id / "metadata.json"


def _target_description_dir(dest: Path, db_id: str) -> Path:
    return dest / db_id / "database_description"


def _install_sqlite(src: Path, dest: Path, db_id: str) -> None:
    """Copy *src* sqlite file to ``dest/<db_id>/<db_id>.sqlite``."""
    target = _target_sqlite_path(dest, db_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    logger.info("  installed %s (%d bytes)", target, target.stat().st_size)


def _is_macos_metadata(member_name: str) -> bool:
    """True for ``__MACOSX`` / AppleDouble entries that shadow real members."""
    parts = Path(member_name).parts
    return "__MACOSX" in parts or Path(member_name).name.startswith("._")


def _clean_bird_text(value: str | None) -> str:
    """Normalize BIRD description cells (strip, collapse whitespace)."""
    if not value:
        return ""
    return " ".join(value.replace("\r", "\n").split())


def _metadata_from_bird_descriptions(desc_dir: Path) -> dict[str, Any]:
    """Convert a BIRD ``database_description/`` folder into our metadata.json shape.

    Each ``*.csv`` is one table. Columns come from BIRD's
    ``original_column_name`` / ``column_name`` / ``column_description`` /
    ``value_description`` fields.
    """
    metadata: dict[str, Any] = {}
    for csv_path in sorted(desc_dir.glob("*.csv")):
        if csv_path.name.startswith("._"):
            continue
        table_name = csv_path.stem
        columns: list[dict[str, Any]] = []
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (
                    (row.get("original_column_name") or row.get("column_name") or "")
                    .strip()
                )
                if not name:
                    continue
                col_desc = _clean_bird_text(row.get("column_description"))
                value_desc = _clean_bird_text(row.get("value_description"))
                description = " | ".join(p for p in (col_desc, value_desc) if p)
                entry: dict[str, Any] = {"name": name}
                if description:
                    entry["description"] = description
                columns.append(entry)
        metadata[table_name] = {"description": "", "columns": columns}
    return metadata


def _write_metadata_from_descriptions(dest: Path, db_id: str) -> Path | None:
    """Write ``metadata.json`` for *db_id* when ``database_description/`` exists."""
    desc_dir = _target_description_dir(dest, db_id)
    if not desc_dir.is_dir():
        return None
    metadata = _metadata_from_bird_descriptions(desc_dir)
    if not metadata:
        logger.warning("  %s: database_description/ has no usable CSV files", db_id)
        return None
    out = _target_metadata_path(dest, db_id)
    with out.open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    n_cols = sum(len(t.get("columns") or []) for t in metadata.values())
    logger.info(
        "  wrote %s (%d table(s), %d column(s))",
        out,
        len(metadata),
        n_cols,
    )
    return out


def _copy_description_dir(src_dir: Path, dest: Path, db_id: str) -> None:
    """Replace ``dest/<db_id>/database_description`` with *src_dir*."""
    target = _target_description_dir(dest, db_id)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src_dir, target)
    csv_count = len(list(target.glob("*.csv")))
    logger.info(
        "  installed %s/database_description/ (%d CSV file(s))", db_id, csv_count
    )


def _find_description_dir(sqlite_path: Path, root: Path, db_id: str) -> Path | None:
    """Locate BIRD ``database_description/`` next to a sqlite file or under *root*."""
    candidates = [
        sqlite_path.parent / "database_description",
        sqlite_path.parent.parent / "database_description",
        root / db_id / "database_description",
        root / "database_description",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
    matches = [
        p
        for p in root.rglob("database_description")
        if p.is_dir() and db_id.lower() in {x.lower() for x in p.parts}
    ]
    for match in matches:
        if any(match.glob("*.csv")):
            return match
    return None


def _find_sqlite_in_tree(root: Path, db_id: str) -> Path | None:
    """Locate ``<db_id>.sqlite`` under *root*, preferring canonical layouts."""
    candidates = [
        root / db_id / f"{db_id}.sqlite",
        root / f"{db_id}.sqlite",
        root / db_id / "sqlite" / f"{db_id}.sqlite",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = [
        match
        for match in sorted(root.rglob(f"{db_id}.sqlite"))
        if not _is_macos_metadata(str(match))
    ]
    return matches[0] if matches else None


def _copy_dbs_from_root(
    root: Path,
    dest: Path,
    db_ids: Iterable[str],
    *,
    label: str,
    with_descriptions: bool = False,
) -> set[str]:
    installed: set[str] = set()
    missing: list[str] = []
    for db_id in sorted(set(db_ids)):
        src = _find_sqlite_in_tree(root, db_id)
        if src is None:
            missing.append(db_id)
            continue
        _install_sqlite(src, dest, db_id)
        if with_descriptions:
            desc = _find_description_dir(src, root, db_id)
            if desc is not None:
                _copy_description_dir(desc, dest, db_id)
                _write_metadata_from_descriptions(dest, db_id)
            else:
                logger.warning(
                    "  %s: no database_description/ found under %s", db_id, root
                )
        installed.add(db_id)

    if missing:
        logger.warning(
            "%s: could not find sqlite for %s under %s",
            label,
            ", ".join(missing),
            root,
        )
    return installed


def _db_id_from_member(member_name: str, wanted: set[str]) -> str | None:
    """Return the wanted db_id if *member_name* belongs to that database tree."""
    if _is_macos_metadata(member_name):
        return None
    lower_map = {db_id.lower(): db_id for db_id in wanted}
    parts = Path(member_name).parts
    for part in parts:
        if part.lower() in lower_map:
            return lower_map[part.lower()]
    if member_name.lower().endswith(".sqlite"):
        return lower_map.get(Path(member_name).stem.lower())
    return None


def _stream_member(zf: zipfile.ZipFile, member: str, target: Path) -> None:
    """Copy a single zip *member* out to *target* without loading it in memory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, target.open("wb") as out:
        shutil.copyfileobj(src, out, CHUNK_SIZE)


def _relocate_extracted_member(
    member_name: str, db_id: str, dest: Path
) -> Path | None:
    """Map an archive member path onto ``dest/<db_id>/...``."""
    path = Path(member_name)
    lower_name = path.name.lower()
    if lower_name == f"{db_id.lower()}.sqlite":
        return _target_sqlite_path(dest, db_id)

    parts_lower = [p.lower() for p in path.parts]
    if "database_description" in parts_lower:
        # Keep only the filename under database_description/
        return _target_description_dir(dest, db_id) / path.name

    return None


def _extract_db_assets_from_archive(
    archive_path: Path,
    wanted: set[str],
    dest: Path,
    work_dir: Path,
    *,
    with_descriptions: bool = False,
) -> set[str]:
    """Extract sqlite files (and optionally BIRD description CSVs) for *wanted*.

    Recurses into nested zips (BIRD's ``train.zip`` wraps
    ``train/train_databases.zip``).
    """
    installed: set[str] = set()
    remaining = set(wanted)

    with zipfile.ZipFile(archive_path, "r") as zf:
        nested_zips: list[str] = []
        members_by_db: dict[str, list[str]] = {db_id: [] for db_id in wanted}

        for info in zf.infolist():
            if info.is_dir() or _is_macos_metadata(info.filename):
                continue
            if info.filename.lower().endswith(".zip"):
                nested_zips.append(info.filename)
                continue
            db_id = _db_id_from_member(info.filename, remaining)
            if db_id is None:
                continue
            lower = info.filename.lower()
            is_sqlite = lower.endswith(".sqlite")
            is_desc = with_descriptions and "database_description" in lower
            if is_sqlite or is_desc:
                members_by_db[db_id].append(info.filename)

        for db_id, members in sorted(members_by_db.items()):
            if not members:
                continue
            got_sqlite = False
            got_desc = False
            for member in members:
                target = _relocate_extracted_member(member, db_id, dest)
                if target is None:
                    continue
                logger.info("  extracting %s -> %s", member, target)
                _stream_member(zf, member, target)
                if target.suffix.lower() == ".sqlite":
                    got_sqlite = True
                    logger.info(
                        "  installed %s (%d bytes)", target, target.stat().st_size
                    )
                elif "database_description" in target.parts:
                    got_desc = True
            if with_descriptions and got_desc:
                _write_metadata_from_descriptions(dest, db_id)
            # Count as done when we installed a sqlite, or when we only needed
            # descriptions and the sqlite was already on disk.
            if got_sqlite or (
                with_descriptions
                and got_desc
                and _target_sqlite_path(dest, db_id).is_file()
            ):
                installed.add(db_id)
                remaining.discard(db_id)

        for nested in nested_zips:
            if not remaining:
                break
            nested_path = work_dir / Path(nested).name
            logger.info(
                "  unpacking nested archive %s (still need: %s)",
                nested,
                ", ".join(sorted(remaining)),
            )
            _stream_member(zf, nested, nested_path)
            try:
                found = _extract_db_assets_from_archive(
                    nested_path,
                    remaining,
                    dest,
                    work_dir,
                    with_descriptions=with_descriptions,
                )
            finally:
                nested_path.unlink(missing_ok=True)
            installed |= found
            remaining -= found

    return installed


def _install_from_sources(
    *,
    sources: tuple[ArchiveSource, ...],
    dest: Path,
    cache_dir: Path,
    db_ids: set[str],
    force: bool,
    keep_archive: bool,
    label: str,
    with_descriptions: bool = False,
) -> set[str]:
    """Install *db_ids* from *sources*, trying each until nothing is missing.

    Sources are complementary rather than interchangeable: mirrors can be
    partial snapshots, so each one contributes whatever it has.
    """
    installed: set[str] = set()
    remaining = set(db_ids)

    for source in sources:
        if not remaining:
            break

        archive_path = cache_dir / source.archive_name
        logger.info(
            "  source: %s (need: %s)", source.name, ", ".join(sorted(remaining))
        )
        try:
            _download(source.url, archive_path, force=force, expect_zip=True)
        except DownloadError as exc:
            logger.warning("  %s unavailable: %s", source.name, exc)
            continue

        with tempfile.TemporaryDirectory(
            prefix=f"fdabench_{label}_", dir=cache_dir
        ) as tmp:
            found = _extract_db_assets_from_archive(
                archive_path,
                remaining,
                dest,
                Path(tmp),
                with_descriptions=with_descriptions,
            )

        installed |= found
        remaining -= found

        if not keep_archive and archive_path.exists():
            archive_path.unlink()
            logger.info("  removed cached archive %s", archive_path)
        elif keep_archive:
            logger.info("  kept cached archive at %s", archive_path)

    if remaining:
        logger.warning(
            "%s: no source provided sqlite for: %s",
            label,
            ", ".join(sorted(remaining)),
        )

    return installed


def _dbs_for_types(
    db_types: dict[str, str], allowed: frozenset[str]
) -> set[str]:
    return {db_id for db_id, dtype in db_types.items() if dtype in allowed}


def _update_env_file(env_path: Path, connection_strings_value: str) -> bool:
    """Set ``CONNECTION_STRINGS`` in *env_path*, preserving everything else."""
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
    n_questions: int,
    write_env: bool = True,
) -> None:
    logger.info("=" * 60)
    logger.info("FDABench-Lite seed complete.")
    logger.info("  Destination : %s", dest)
    logger.info("  Questions   : %d", n_questions)
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


def seed_fdabench(
    *,
    dest: Path | None = None,
    force: bool = False,
    keep_archive: bool = False,
    write_env: bool = True,
    tasks_only: bool = False,
    bird_root: Path | None = None,
    spider2_root: Path | None = None,
    spider1_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Download FDABench-Lite tasks and (unless *tasks_only*) required SQLite DBs.

    Returns ``(evaluation_rows, installed_db_ids)``.
    """
    target = dest or _default_dest()
    cache_dir = target / ".cache"
    target.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Seeding FDABench-Lite into %s", target)
    logger.info("=" * 60)

    tasks = _download_lite_tasks(cache_dir / "lite_tasks", force=force)
    rows, db_types, skip_counts = _convert_tasks(tasks)
    _write_evaluation_json(target, rows)

    if skip_counts:
        logger.info(
            "Skipped tasks: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(skip_counts.items())),
        )

    if tasks_only:
        logger.info("--tasks-only: skipping SQLite database install")
        return rows, []

    bird_dbs = _dbs_for_types(db_types, BIRD_DB_TYPES)
    spider2_dbs = _dbs_for_types(db_types, SPIDER2_DB_TYPES)
    spider1_dbs = _dbs_for_types(db_types, SPIDER1_DB_TYPES)
    other = {
        db_id: dtype
        for db_id, dtype in db_types.items()
        if dtype not in (BIRD_DB_TYPES | SPIDER2_DB_TYPES | SPIDER1_DB_TYPES)
    }
    if other:
        raise RuntimeError(
            "Unsupported database_type(s) in kept tasks: "
            + ", ".join(f"{k}={v}" for k, v in sorted(other.items()))
        )

    # Resume support: a partial seed shouldn't re-download multi-GB archives
    # just to reinstall databases that are already in place. BIRD DBs still
    # re-enter the install path when metadata.json is missing so we can pull
    # database_description CSVs without --force.
    already_present = {
        db_id
        for db_id in db_types
        if _target_sqlite_path(target, db_id).is_file()
    }
    bird_need_metadata = {
        db_id
        for db_id in bird_dbs
        if force or not _target_metadata_path(target, db_id).is_file()
    }
    if already_present and not force:
        logger.info(
            "Already installed (pass --force to reinstall): %s",
            ", ".join(sorted(already_present)),
        )
        spider2_dbs -= already_present
        spider1_dbs -= already_present
        # Keep BIRD DBs that still need metadata/descriptions.
        bird_dbs = (bird_dbs - already_present) | (
            bird_need_metadata & already_present
        )
        if bird_need_metadata & already_present:
            logger.info(
                "Will refresh BIRD descriptions/metadata for: %s",
                ", ".join(sorted(bird_need_metadata & already_present)),
            )

    installed: set[str] = set(already_present) if not force else set()

    groups = (
        ("BIRD", bird_dbs, bird_root, BIRD_SOURCES, "bird", True),
        ("Spider2-lite", spider2_dbs, spider2_root, SPIDER2_SOURCES, "spider2", False),
        ("Spider 1.0", spider1_dbs, spider1_root, SPIDER1_SOURCES, "spider1", False),
    )
    for group_label, group_dbs, group_root, sources, slug, with_desc in groups:
        if not group_dbs:
            continue
        logger.info(
            "Installing %d %s database(s)%s ...",
            len(group_dbs),
            group_label,
            " (+ descriptions)" if with_desc else "",
        )
        if group_root is not None:
            installed.update(
                _copy_dbs_from_root(
                    group_root,
                    target,
                    group_dbs,
                    label=group_label,
                    with_descriptions=with_desc,
                )
            )
        else:
            installed.update(
                _install_from_sources(
                    sources=sources,
                    dest=target,
                    cache_dir=cache_dir,
                    db_ids=group_dbs,
                    force=force,
                    keep_archive=keep_archive,
                    label=slug,
                    with_descriptions=with_desc,
                )
            )

    _warn_about_missing_dbs(db_types, installed, rows)
    _warn_about_missing_metadata(db_types, target)

    installed_ids = sorted(installed)
    _print_summary(
        target, installed_ids, n_questions=len(rows), write_env=write_env
    )
    return rows, installed_ids


def _warn_about_missing_metadata(
    db_types: dict[str, str], dest: Path
) -> None:
    """Log bird DBs that still lack metadata.json after seeding."""
    bird_missing = sorted(
        db_id
        for db_id, dtype in db_types.items()
        if dtype in BIRD_DB_TYPES
        and _target_sqlite_path(dest, db_id).is_file()
        and not _target_metadata_path(dest, db_id).is_file()
    )
    if not bird_missing:
        return
    logger.warning(
        "BIRD databases without metadata.json (no database_description CSVs): %s",
        ", ".join(bird_missing),
    )


def _warn_about_missing_dbs(
    db_types: dict[str, str],
    installed: set[str],
    rows: list[dict[str, Any]],
) -> None:
    """Log which databases are absent and how many questions that strands."""
    missing = sorted(set(db_types) - installed)
    if not missing:
        return

    per_db = Counter(str(row["db_id"]) for row in rows)
    stranded = sum(per_db[db_id] for db_id in missing)
    logger.warning("=" * 60)
    logger.warning(
        "%d database(s) could not be installed, leaving %d of %d question(s) "
        "unrunnable:",
        len(missing),
        stranded,
        len(rows),
    )
    for db_id in missing:
        logger.warning(
            "    - %s (%s, %d question(s))",
            db_id,
            db_types[db_id],
            per_db[db_id],
        )
    logger.warning(
        "Re-run this script later, or point it at a local copy with "
        "--bird-root / --spider2-root / --spider1-root."
    )
    logger.warning("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FDABench-Lite tasks from HuggingFace and populate "
            "datasets/fdabench/ with evaluation.json plus the required SQLite "
            "databases (BIRD train / Spider2-lite local / Spider1)."
        ),
        epilog=(
            "Archives are large (~9 GB BIRD train + ~1.6 GB Spider2-lite + "
            "Spider1). Prefer --bird-root / --spider2-root / --spider1-root "
            "when you already have those trees locally. Pass --tasks-only to "
            "write evaluation.json without downloading databases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: datasets/fdabench/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download cached archives/JSONLs and overwrite installed DBs.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep downloaded zip archives under datasets/fdabench/.cache/.",
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
        "--tasks-only",
        action="store_true",
        help="Only download/convert Lite tasks into evaluation.json.",
    )
    parser.add_argument(
        "--bird-root",
        type=Path,
        default=None,
        help=(
            "Existing BIRD train_databases/ (or parent) tree; skip BIRD train.zip "
            "download and copy needed DBs from here."
        ),
    )
    parser.add_argument(
        "--spider2-root",
        type=Path,
        default=None,
        help=(
            "Existing Spider2-lite local sqlite directory "
            "(e.g. spider2-localdb/); skip Google Drive download."
        ),
    )
    parser.add_argument(
        "--spider1-root",
        type=Path,
        default=None,
        help=(
            "Existing Spider1 database/ or test_database/ tree; skip Spider "
            "dataset download."
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
    seed_fdabench(
        dest=args.dest,
        force=args.force,
        keep_archive=args.keep_archive,
        write_env=not args.no_write_env,
        tasks_only=args.tasks_only,
        bird_root=args.bird_root,
        spider2_root=args.spider2_root,
        spider1_root=args.spider1_root,
    )


if __name__ == "__main__":
    main()
