# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the eval pipeline across BIRD Mini-Dev databases, then judge with BIRD EX/VES.

For each ``db_id`` under ``datasets/bird/<db_id>/<db_id>.sqlite`` this script:

1. Resets shared state so each database starts from a clean slate:
   * Neo4j   -- ``MATCH (n) DETACH DELETE n``
   * pgvector -- ``DROP TABLE`` on ``vdb.data_objects_layer`` and
     ``vdb.semantic_layer`` (recreated by the next ingest).
2. Materialises a per-database ``datasets/bird/<db_id>/evaluation.json`` slice
   from the combined ``datasets/bird/evaluation.json``.
3. Runs ``main.py --database-name bird/<db_id> --skip-judge`` (ingest ->
   semantic -> eval) with ``CONNECTION_STRINGS`` pointed at that SQLite file.

Once every database has produced its eval CSV, the files are concatenated and
scored with the official BIRD judge (``ontology_sql_eval.judge.bird``).

Usage::

    PYTHONPATH=../GSF uv run python scripts/run_bird_pipeline.py
    PYTHONPATH=../GSF uv run python scripts/run_bird_pipeline.py --first-only
    PYTHONPATH=../GSF uv run python scripts/run_bird_pipeline.py --db-ids california_schools
    PYTHONPATH=../GSF uv run python scripts/run_bird_pipeline.py --eval-only --db-ids california_schools
    PYTHONPATH=../GSF uv run python scripts/run_bird_pipeline.py --start-from card_games --skip-completed
    PYTHONPATH=../GSF uv run python scripts/run_bird_pipeline.py --skip-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger("bird-pipeline")

REPO_ROOT = Path(__file__).resolve().parents[1]
BIRD_DIR = REPO_ROOT / "datasets" / "bird"
MASTER_EVAL_PATH = BIRD_DIR / "evaluation.json"
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"

VDB_SCHEMA = "vdb"
VDB_TABLES = ("data_objects_layer", "semantic_layer")


# ---------------------------------------------------------------------------
# Discovery / paths
# ---------------------------------------------------------------------------


def _model_slug() -> str:
    return os.environ.get("MODEL_NAME", "nemotron").rsplit("/", 1)[-1]


def _sqlite_file(db_id: str) -> Path:
    return BIRD_DIR / db_id / f"{db_id}.sqlite"


def _dataset_name(db_id: str) -> str:
    return f"bird/{db_id}"


def _per_db_eval_path(db_id: str) -> Path:
    return BIRD_DIR / db_id / "evaluation.json"


def _connection_string(db_id: str) -> str:
    sqlite_path = _sqlite_file(db_id).resolve()
    return (
        f"sqlite:///{sqlite_path}"
        f"?metadata_database={quote(db_id, safe='')}"
    )


def _eval_csv_path(db_id: str) -> Path:
    return INPUT_DIR / f"{_dataset_name(db_id)}_{_model_slug()}.csv"


def _load_master_evaluation() -> list[dict]:
    if not MASTER_EVAL_PATH.is_file():
        raise SystemExit(
            f"BIRD evaluation file not found: {MASTER_EVAL_PATH}\n"
            "Run scripts/download_bird.py first."
        )
    with MASTER_EVAL_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {MASTER_EVAL_PATH}")
    return data


def list_db_ids() -> list[str]:
    """Return sorted ``db_id`` values that have a local SQLite file."""
    if not BIRD_DIR.is_dir():
        return []
    db_ids: list[str] = []
    for child in sorted(BIRD_DIR.iterdir()):
        if not child.is_dir():
            continue
        db_id = child.name
        if _sqlite_file(db_id).is_file():
            db_ids.append(db_id)
    return db_ids


def ensure_per_db_evaluation(db_id: str, master_rows: list[dict] | None = None) -> Path:
    """Write ``datasets/bird/<db_id>/evaluation.json`` filtered from the master file."""
    rows = master_rows if master_rows is not None else _load_master_evaluation()
    slice_rows = [row for row in rows if str(row.get("db_id", "")) == db_id]
    if not slice_rows:
        raise ValueError(
            f"No questions for db_id={db_id!r} in {MASTER_EVAL_PATH}. "
            "Re-run scripts/download_bird.py."
        )

    out_path = _per_db_eval_path(db_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(slice_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d question(s) for %s -> %s",
        len(slice_rows),
        db_id,
        out_path,
    )
    return out_path


def ensure_all_per_db_evaluations(db_ids: list[str]) -> None:
    master_rows = _load_master_evaluation()
    for db_id in db_ids:
        ensure_per_db_evaluation(db_id, master_rows)


# ---------------------------------------------------------------------------
# Store resets
# ---------------------------------------------------------------------------


def _postgres_url() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ.get("POSTGRES_DATABASE", "gsf")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def reset_neo4j() -> None:
    from neo4j import GraphDatabase

    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    with GraphDatabase.driver(uri, auth=auth) as driver:
        summary = driver.execute_query(
            "MATCH (n) DETACH DELETE n", database_="neo4j"
        ).summary
        deleted = summary.counters.nodes_deleted
    logger.info("Neo4j reset: detached-deleted %d node(s).", deleted)


def reset_pgvector() -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(_postgres_url()) as conn:
        with conn.cursor() as cur:
            for table in VDB_TABLES:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(VDB_SCHEMA, table)
                    )
                )
        conn.commit()
    logger.info(
        "pgvector reset: dropped %s.",
        ", ".join(f"{VDB_SCHEMA}.{t}" for t in VDB_TABLES),
    )


def reset_between_datasets() -> None:
    reset_neo4j()
    reset_pgvector()


# ---------------------------------------------------------------------------
# Per-database run
# ---------------------------------------------------------------------------


def _activate_db(db_id: str) -> None:
    """Point env + GSF caches at *db_id* for an in-process stage run.

    Sets ``CONNECTION_STRINGS`` for this database and drops GSF's cached
    connectors and retrievers so the next stage rebuilds them against this
    database's freshly ingested Neo4j + pgvector data. This is what makes it
    safe to run several databases sequentially in one process (the subprocess
    isolation the runner used to rely on).
    """
    os.environ["CONNECTION_STRINGS"] = _connection_string(db_id)

    from gsf.connectors import invalidate_connectors_cache
    from gsf.utils import invalidate_retrievers_cache

    invalidate_connectors_cache()
    invalidate_retrievers_cache()


def _load_pipeline():
    """Import the repo-root ``main.py`` pipeline module (adds repo root to path)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import main as pipeline

    return pipeline


def run_eval_only(db_id: str) -> Path | None:
    """Run retrieval eval only (skip ingest, semantic, store resets), in-process."""
    logger.info("=" * 70)
    logger.info("EVAL ONLY: bird/%s", db_id)
    logger.info("=" * 70)

    ensure_per_db_evaluation(db_id)
    _activate_db(db_id)

    logger.info("Running in-process: CONNECTION_STRINGS=<%s> eval-only", db_id)
    pipeline = _load_pipeline()
    try:
        pipeline.main(
            [
                "--database-name",
                _dataset_name(db_id),
                "--skip-ingest",
                "--skip-semantic",
                "--skip-judge",
            ]
        )
    except SystemExit as exc:
        logger.error("bird/%s eval aborted: %s", db_id, exc)
        return None

    csv_path = _eval_csv_path(db_id)
    if not csv_path.exists():
        logger.error("bird/%s produced no eval CSV at %s.", db_id, csv_path)
        return None
    logger.info("bird/%s eval CSV: %s", db_id, csv_path)
    return csv_path


def run_database(db_id: str) -> Path | None:
    """Reset stores and run ingest->semantic->eval for one BIRD database, in-process."""
    logger.info("=" * 70)
    logger.info("DATABASE: bird/%s", db_id)
    logger.info("=" * 70)

    ensure_per_db_evaluation(db_id)
    _activate_db(db_id)
    reset_between_datasets()

    logger.info("Running in-process: CONNECTION_STRINGS=<%s> ingest->semantic->eval", db_id)
    pipeline = _load_pipeline()
    try:
        pipeline.main(
            [
                "--database-name",
                _dataset_name(db_id),
                "--skip-judge",
            ]
        )
    except SystemExit as exc:
        logger.error("bird/%s aborted: %s", db_id, exc)
        return None

    csv_path = _eval_csv_path(db_id)
    if not csv_path.exists():
        logger.error("bird/%s produced no eval CSV at %s.", db_id, csv_path)
        return None
    logger.info("bird/%s eval CSV: %s", db_id, csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Concatenate + judge
# ---------------------------------------------------------------------------


def concatenate_csvs(csv_items: list[tuple[str, Path]], combined_path: Path) -> int:
    fieldnames: list[str] = ["source_database"]
    for _db_id, path in csv_items:
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        for col in header:
            if col not in fieldnames:
                fieldnames.append(col)

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with combined_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for db_id, path in csv_items:
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row["source_database"] = _dataset_name(db_id)
                    writer.writerow(row)
                    total += 1
    logger.info(
        "Concatenated %d database CSV(s) -> %s (%d rows).",
        len(csv_items),
        combined_path,
        total,
    )
    return total


def judge_with_bird(
    csv_items: list[tuple[str, Path]],
    *,
    skip_ves: bool = True,
    num_cpus: int = 1,
    meta_time_out: float = 30.0,
    iterate_num: int = 100,
) -> dict:
    """Run the official BIRD judge on each database's eval CSV."""
    from ontology_sql_eval.judge.bird import run as bird_run

    summaries: list[dict] = []
    for db_id, csv_path in csv_items:
        work_dir = OUTPUT_DIR / "bird_eval" / db_id
        work_dir.mkdir(parents=True, exist_ok=True)
        scored_path = work_dir / f"{csv_path.stem}_bird_scores.csv"
        logger.info("BIRD judging bird/%s from %s", db_id, csv_path)
        bird_run(
            csv_path,
            scored_path,
            evaluation_json=MASTER_EVAL_PATH,
            db_root=BIRD_DIR,
            num_cpus=num_cpus,
            meta_time_out=meta_time_out,
            iterate_num=iterate_num,
            skip_ves=skip_ves,
        )
        summaries.append(
            {
                "db_id": db_id,
                "eval_csv": str(csv_path),
                "scored_csv": str(scored_path),
            }
        )

    combined = {
        "databases": len(summaries),
        "skip_ves": skip_ves,
        "per_database": summaries,
    }
    combined_path = OUTPUT_DIR / "bird_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    logger.info("BIRD judge complete. Summary: %s", combined_path)
    return combined


def judge_combined_csv(
    combined_path: Path,
    *,
    skip_ves: bool = True,
    num_cpus: int = 1,
    meta_time_out: float = 30.0,
    iterate_num: int = 100,
) -> Path:
    """Score one concatenated eval CSV with the BIRD judge."""
    from ontology_sql_eval.judge.bird import run as bird_run

    scored_path = OUTPUT_DIR / f"{combined_path.stem}_bird_scores.csv"
    logger.info("BIRD judging combined CSV %s", combined_path)
    bird_run(
        combined_path,
        scored_path,
        evaluation_json=MASTER_EVAL_PATH,
        db_root=BIRD_DIR,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
        iterate_num=iterate_num,
        skip_ves=skip_ves,
    )
    return scored_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _expected_question_count(db_id: str, master_rows: list[dict] | None = None) -> int:
    rows = master_rows if master_rows is not None else _load_master_evaluation()
    return sum(1 for row in rows if str(row.get("db_id", "")) == db_id)


def _eval_csv_complete(db_id: str, master_rows: list[dict] | None = None) -> bool:
    """True when the per-database eval CSV exists and has all questions."""
    csv_path = _eval_csv_path(db_id)
    if not csv_path.is_file():
        return False
    expected = _expected_question_count(db_id, master_rows)
    if expected <= 0:
        return False
    with csv_path.open(encoding="utf-8", newline="") as f:
        actual = sum(1 for _ in csv.DictReader(f))
    return actual >= expected


def _select_db_ids(
    requested: list[str] | None,
    *,
    first_only: bool,
    start_from: str | None = None,
    skip_completed: bool = False,
) -> list[str]:
    available = list_db_ids()
    if not available:
        raise SystemExit(
            f"No BIRD databases found under {BIRD_DIR}.\n"
            "Run scripts/download_bird.py first."
        )
    if requested:
        missing = [db_id for db_id in requested if db_id not in available]
        if missing:
            raise SystemExit(
                f"Requested db_id(s) not available: {', '.join(missing)}\n"
                f"Available: {', '.join(available)}"
            )
        db_ids = list(requested)
    elif first_only:
        db_ids = available[:1]
    else:
        db_ids = available

    if start_from:
        if start_from not in available:
            raise SystemExit(
                f"--start-from db_id not available: {start_from!r}\n"
                f"Available: {', '.join(available)}"
            )
        db_ids = available[available.index(start_from) :]

    if skip_completed:
        master_rows = _load_master_evaluation()
        skipped = [db_id for db_id in db_ids if _eval_csv_complete(db_id, master_rows)]
        db_ids = [db_id for db_id in db_ids if db_id not in skipped]
        if skipped:
            logger.info(
                "Skipping %d completed database(s): %s",
                len(skipped),
                ", ".join(skipped),
            )

    if not db_ids:
        raise SystemExit("No databases selected to run.")

    return db_ids


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ingest->semantic->eval for BIRD Mini-Dev databases "
            "(resetting Neo4j + pgvector between each), then score with the "
            "official BIRD EX/VES judge."
        )
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Run only the first available database.",
    )
    parser.add_argument(
        "--db-ids",
        nargs="+",
        default=None,
        help="Subset of BIRD db_id values to run (default: all available).",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        help=(
            "Resume from this db_id onward in sorted order "
            "(e.g. card_games skips california_schools)."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip databases that already have a complete eval CSV "
            "(row count matches evaluation.json)."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the whole run if any database fails (default: keep going).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip per-database runs; only concatenate existing CSVs and judge.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Rerun retrieval eval only (skip ingest, semantic, store resets). "
            "Neo4j + pgvector must already be populated for each db_id."
        ),
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the BIRD judge stage.",
    )
    parser.add_argument(
        "--with-ves",
        action="store_true",
        help="Also run BIRD VES timing (default: EX only).",
    )
    parser.add_argument(
        "--judge-cpus",
        type=int,
        default=1,
        help="Parallel workers for BIRD judge (default: 1).",
    )
    parser.add_argument(
        "--combined-name",
        default=None,
        help="Basename for the combined CSV (default: bird_all_<model>.csv).",
    )
    parser.add_argument(
        "--skip-concat",
        action="store_true",
        help="Skip concatenating per-database CSVs (judge uses per-DB files only).",
    )
    parser.add_argument(
        "--judge-combined",
        action="store_true",
        help="Also score the concatenated CSV when multiple databases ran.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if args.skip_run and args.eval_only:
        raise SystemExit("--skip-run and --eval-only are mutually exclusive.")
    if args.first_only and args.db_ids:
        raise SystemExit("--first-only and --db-ids are mutually exclusive.")
    if args.first_only and args.start_from:
        raise SystemExit("--first-only and --start-from are mutually exclusive.")
    if args.db_ids and args.start_from:
        raise SystemExit("--db-ids and --start-from are mutually exclusive.")

    load_dotenv(REPO_ROOT / ".env")

    db_ids = _select_db_ids(
        args.db_ids,
        first_only=args.first_only,
        start_from=args.start_from,
        skip_completed=args.skip_completed,
    )
    logger.info("Selected %d database(s): %s", len(db_ids), ", ".join(db_ids))

    combined_name = args.combined_name or f"bird_all_{_model_slug()}.csv"
    combined_path = INPUT_DIR / combined_name
    skip_ves = not args.with_ves

    produced: list[tuple[str, Path]] = []
    failed: list[str] = []

    if args.skip_run:
        produced = [
            (db_id, _eval_csv_path(db_id))
            for db_id in db_ids
            if _eval_csv_path(db_id).exists()
        ]
        missing = [db_id for db_id in db_ids if not _eval_csv_path(db_id).exists()]
        if missing:
            logger.warning(
                "--skip-run: no existing CSV for %d database(s): %s",
                len(missing),
                ", ".join(missing),
            )
    else:
        ensure_all_per_db_evaluations(db_ids)
        run_fn = run_eval_only if args.eval_only else run_database
        for i, db_id in enumerate(db_ids, start=1):
            label = "eval" if args.eval_only else "run"
            logger.info("[%d/%d] Starting bird/%s (%s)", i, len(db_ids), db_id, label)
            try:
                csv_path = run_fn(db_id)
            except Exception:
                logger.exception("bird/%s raised an exception.", db_id)
                csv_path = None

            if csv_path is None:
                failed.append(db_id)
                if args.stop_on_error:
                    raise SystemExit(f"Aborting: bird/{db_id} failed.")
            else:
                produced.append((db_id, csv_path))

    logger.info("=" * 70)
    logger.info(
        "Per-database runs complete: %d succeeded, %d failed.",
        len(produced),
        len(failed),
    )
    if failed:
        logger.warning("Failed database(s): %s", ", ".join(failed))

    if not produced:
        raise SystemExit("No eval CSVs were produced; nothing to judge.")

    if not args.skip_concat and len(produced) > 1:
        concatenate_csvs(produced, combined_path)

    if args.skip_judge:
        logger.info("Skipping BIRD judge.")
        return

    judge_with_bird(
        produced,
        skip_ves=skip_ves,
        num_cpus=args.judge_cpus,
    )

    if args.judge_combined and not args.skip_concat and len(produced) > 1:
        scored = judge_combined_csv(
            combined_path,
            skip_ves=skip_ves,
            num_cpus=args.judge_cpus,
        )
        logger.info("Combined BIRD scores: %s", scored)

    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    if not args.skip_concat and len(produced) > 1:
        logger.info("  Combined eval CSV : %s", combined_path)
    logger.info("  Summary JSON        : %s", OUTPUT_DIR / "bird_summary.json")


if __name__ == "__main__":
    main()
