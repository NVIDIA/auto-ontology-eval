# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ``eval_chatbot`` over a dataset in parallel index ranges.

Each API key owns one worker slot. A slot evaluates its assigned chunks
sequentially, so the same key is never used by multiple subprocesses at once.
Different keys run concurrently. Every subprocess writes a separate shard CSV;
successful shards are merged in original dataset order at the end.

Examples::

    uv run python scripts/run_parallel_eval.py \
        --database-name bird --chunk-size 100

    uv run python scripts/run_parallel_eval.py \
        --database-name bird --ranges 0:250 250:500 500:

Keys come from comma-separated ``API_KEYS_LIST``. When it is unset, the runner
falls back to ``NVIDIA_API_KEY`` and therefore uses one worker slot.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger("run_parallel_eval")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
INPUT_DIR = REPO_ROOT / "input"
EVAL_MODULE = "ontology_sql_eval.retrieval.eval_chatbot"
_ROW_INDEX_FIELD = "row_index"

load_dotenv(REPO_ROOT / ".env")

Range = tuple[int, int]
Job = tuple[int, int, int]


def _model_slug() -> str:
    return os.environ.get("MODEL_NAME", "nemotron").rsplit("/", 1)[-1]


def _api_keys() -> list[str]:
    """Return configured keys, falling back to the normal single key."""
    raw = os.environ.get("API_KEYS_LIST", "")
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if keys:
        return keys
    single = os.environ.get("NVIDIA_API_KEY", "").strip()
    return [single] if single else []


def _mask_key(key: str | None) -> str:
    if not key:
        return "<inherited>"
    return f"…{key[-4:]}" if len(key) > 4 else "…"


def _dataset_input_path(database_name: str) -> Path:
    return DATASETS_DIR / database_name / "evaluation.json"


def _load_questions(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise SystemExit(
            f"Evaluation file not found: {input_path}\n"
            "Pass a valid --database-name."
        )
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(
            f"Expected a JSON array in {input_path}, got {type(data).__name__}"
        )
    return data


def _parse_ranges(raw_ranges: list[str], total: int) -> list[Range]:
    """Parse and validate ``start:end`` tokens (end optional)."""
    ranges: list[Range] = []
    for token in raw_ranges:
        if ":" not in token:
            raise SystemExit(f"--ranges expects start:end tokens, got {token!r}")
        start_s, end_s = token.split(":", 1)
        start = int(start_s) if start_s.strip() else 0
        end = int(end_s) if end_s.strip() else total
        if start < 0 or end < 0:
            raise SystemExit(f"Range {token!r} cannot contain negative indexes.")
        start = min(start, total)
        end = min(end, total)
        if end <= start:
            raise SystemExit(
                f"Range {token!r} is empty after clamping to dataset size {total}."
            )
        ranges.append((start, end))
    return ranges


def _chunk_ranges(total: int, chunk_size: int) -> list[Range]:
    """Split ``[0, total)`` into contiguous chunks."""
    if chunk_size <= 0:
        raise SystemExit(f"--chunk-size must be positive, got {chunk_size}.")
    return [
        (start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    ]


def _shard_output(
    database_name: str, slug: str, idx: int, start: int, end: int
) -> Path:
    return INPUT_DIR / f"{database_name}_{slug}.part{idx}_{start}_{end}.csv"


def _write_slice(
    questions: list[dict[str, Any]], start: int, end: int, dest: Path
) -> None:
    with dest.open("w", encoding="utf-8") as f:
        json.dump(questions[start:end], f)


def _stream_output(proc: subprocess.Popen[str], prefix: str) -> None:
    """Stream one worker's combined stdout/stderr with a prefix."""
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()


def _run_range(
    *,
    job: Job,
    slot_idx: int,
    api_key: str | None,
    questions: list[dict[str, Any]],
    database_name: str,
    slug: str,
    tmp_dir: Path,
) -> tuple[int, Path, bool]:
    idx, start, end = job
    slice_input = tmp_dir / f"part{idx}_{start}_{end}.json"
    output_path = _shard_output(database_name, slug, idx, start, end)
    _write_slice(questions, start, end, slice_input)

    # Never merge a stale shard left by an earlier failed run.
    output_path.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-m",
        EVAL_MODULE,
        "--input",
        str(slice_input),
        "--output",
        str(output_path),
        "--start-index",
        "0",
    ]
    env = os.environ.copy()
    if api_key:
        env["NVIDIA_API_KEY"] = api_key

    logger.info(
        "slot %d chunk %d: questions [%d, %d) key=%s -> %s",
        slot_idx,
        idx,
        start,
        end,
        _mask_key(api_key),
        output_path,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tee = threading.Thread(
        target=_stream_output,
        args=(proc, f"[s{slot_idx}:c{idx}]"),
        daemon=True,
    )
    tee.start()
    rc = proc.wait()
    tee.join()

    if rc != 0:
        logger.error("chunk %d exited with code %d (%s)", idx, rc, output_path)
        return idx, output_path, False

    row_count = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as f:
            row_count = sum(1 for _ in csv.DictReader(f))
    expected = end - start
    if row_count != expected:
        logger.error(
            "chunk %d wrote %d/%d rows (%s)",
            idx,
            row_count,
            expected,
            output_path,
        )
        return idx, output_path, False

    logger.info("chunk %d finished: %d rows (%s)", idx, row_count, output_path)
    return idx, output_path, True


def _run_slot(
    *,
    slot_idx: int,
    api_key: str | None,
    jobs: list[Job],
    questions: list[dict[str, Any]],
    database_name: str,
    slug: str,
    tmp_dir: Path,
) -> list[tuple[int, Path, bool]]:
    """Run one key's chunks sequentially."""
    results: list[tuple[int, Path, bool]] = []
    for job in jobs:
        results.append(
            _run_range(
                job=job,
                slot_idx=slot_idx,
                api_key=api_key,
                questions=questions,
                database_name=database_name,
                slug=slug,
                tmp_dir=tmp_dir,
            )
        )
    return results


def _merge_shards(shard_paths: list[Path], merged_path: Path) -> None:
    """Merge shards in original range order and renumber ``row_index``."""
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in shard_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames)
                elif list(reader.fieldnames) != fieldnames:
                    raise RuntimeError(f"CSV fields differ in shard {path}")
            rows.extend(reader)

    if fieldnames is None:
        raise RuntimeError("No shard rows available to merge.")

    if _ROW_INDEX_FIELD in fieldnames:
        for i, row in enumerate(rows):
            row[_ROW_INDEX_FIELD] = str(i)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "Merged %d rows from %d shards -> %s",
        len(rows),
        len(shard_paths),
        merged_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--database-name",
        default="bird",
        help="Dataset / database name under datasets/ (default: bird).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--chunk-size",
        type=int,
        help="Split the dataset into chunks of this many questions.",
    )
    group.add_argument(
        "--ranges",
        nargs="+",
        help="Explicit start:end ranges, e.g. 0:250 250:500 500:.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Maximum simultaneous key slots. Defaults to the number of API keys; "
            "one when only NVIDIA_API_KEY is configured."
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    questions = _load_questions(_dataset_input_path(args.database_name))
    total = len(questions)
    ranges = (
        _parse_ranges(args.ranges, total)
        if args.ranges
        else _chunk_ranges(total, args.chunk_size)
    )
    jobs: list[Job] = [
        (idx, start, end) for idx, (start, end) in enumerate(ranges)
    ]

    keys: list[str | None] = _api_keys() or [None]
    if args.max_workers is not None:
        if args.max_workers <= 0:
            raise SystemExit("--max-workers must be positive.")
        keys = keys[: args.max_workers]
    slot_count = min(len(keys), len(jobs))
    keys = keys[:slot_count]

    # Round-robin distribution keeps each key busy while guaranteeing that no
    # key is used by more than one active subprocess.
    slot_jobs = [jobs[slot_idx::slot_count] for slot_idx in range(slot_count)]
    logger.info(
        "%d questions, %d chunks, %d simultaneous key slot(s)",
        total,
        len(jobs),
        slot_count,
    )

    slug = _model_slug()
    tmp_dir = Path(tempfile.mkdtemp(prefix="parallel_eval_"))
    results: list[tuple[int, Path, bool]] = []
    try:
        with ThreadPoolExecutor(max_workers=slot_count) as pool:
            futures = [
                pool.submit(
                    _run_slot,
                    slot_idx=slot_idx,
                    api_key=keys[slot_idx],
                    jobs=slot_jobs[slot_idx],
                    questions=questions,
                    database_name=args.database_name,
                    slug=slug,
                    tmp_dir=tmp_dir,
                )
                for slot_idx in range(slot_count)
            ]
            for future in as_completed(futures):
                results.extend(future.result())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    results.sort(key=lambda result: result[0])
    failures = [result for result in results if not result[2]]
    if failures:
        logger.error(
            "%d/%d chunks failed; merged CSV was not overwritten.",
            len(failures),
            len(jobs),
        )
        return 1

    shard_paths = [path for _, path, _ in results]
    merged_path = INPUT_DIR / f"{args.database_name}_{slug}.csv"
    _merge_shards(shard_paths, merged_path)
    logger.info("All %d chunks completed successfully.", len(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
