# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the eval pipeline across Spider2 databases (SQLite and Snowflake), then judge.

For each database listed in ``datasets/spider2/manifest.json`` this script:

1. Resets shared state so each database starts from a clean slate:
   * Neo4j   -- ``MATCH (n) DETACH DELETE n``
   * pgvector -- ``DROP TABLE`` on ``vdb.data_objects_layer`` and
     ``vdb.semantic_layer`` (recreated by the next ingest).
   * Snowflake judge cache -- ``output/spider2_eval_suite/<slug>/exec_results/``
     (only for ``sf_*`` slugs; skipped on ``--eval-only``).
2. Runs ``main.py --database-name spider2/<slug> --skip-judge`` (ingest ->
   semantic -> eval) with ``CONNECTION_STRINGS`` pointed at that database.

SQLite slugs require a local ``.sqlite`` file. Snowflake slugs connect live via
``snowflake_credential.json`` (PAT in the ``password`` field).

Once every database has produced its per-database eval CSV, the CSVs are
concatenated and scored with the upstream Spider2-lite evaluation suite
(execution-result comparison against gold ``exec_result`` CSVs).

Usage::

    PYTHONPATH=../GSF uv run python scripts/run_spider2_pipeline.py --first-only
    PYTHONPATH=../GSF uv run python scripts/run_spider2_pipeline.py --slugs adventureworks
    PYTHONPATH=../GSF uv run python scripts/run_spider2_pipeline.py --skip-run --first-only
    PYTHONPATH=../GSF uv run python scripts/run_spider2_pipeline.py --eval-only --slugs bank_sales_trading
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
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger("spider2-pipeline")

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"
DATASETS_DIR = REPO_ROOT / "datasets"
SPIDER2_DIR = DATASETS_DIR / "spider2"
MANIFEST_PATH = SPIDER2_DIR / "manifest.json"
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"

# pgvector embedding tables to drop between runs (schema is ``vdb`` in the
# local ``gsf`` Postgres instance -- see gsf.vdb.__init__).
VDB_SCHEMA = "vdb"
VDB_TABLES = ("data_objects_layer", "semantic_layer")


# ---------------------------------------------------------------------------
# Manifest / path helpers
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run scripts/download_spider2.py first."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_entry(slug: str) -> dict:
    manifest = _load_manifest()
    for entry in manifest.get("databases", []):
        if entry["slug"] == slug:
            return entry
    raise KeyError(f"Unknown Spider2 slug: {slug!r}")


def _dialect(slug: str) -> str:
    return _manifest_entry(slug).get("dialect", "sqlite")


def _sqlite_file(slug: str) -> Path:
    return SPIDER2_DIR / slug / f"{slug}.sqlite"


def _connection_string(slug: str) -> str:
    """GSF-ready connection URI for one manifest slug (SQLite or Snowflake)."""
    entry = _manifest_entry(slug)
    dialect = entry.get("dialect", "sqlite")
    if dialect == "snowflake":
        from ontology_sql_eval.spider2_snowflake import snowflake_connection_string

        snowflake_db = entry.get("snowflake_database") or entry["spider2_db_name"]
        return snowflake_connection_string(
            snowflake_db,
            metadata_database=slug,
        )

    sqlite_path = _sqlite_file(slug).resolve()
    return f"sqlite:///{sqlite_path}?metadata_database={quote(slug, safe='')}"


def _model_slug() -> str:
    return os.environ.get("MODEL_NAME", "nemotron").rsplit("/", 1)[-1]


def _eval_csv_path(slug: str) -> Path:
    """Where ``main.py`` writes the eval CSV for ``spider2/<slug>``.

    Mirrors ``ontology_sql_eval.retrieval.eval_chatbot._resolve_paths``:
    ``input/<database_name>_<model_slug>.csv`` with ``database_name`` =
    ``spider2/<slug>``.
    """
    return INPUT_DIR / f"spider2/{slug}_{_model_slug()}.csv"


# ---------------------------------------------------------------------------
# Store resets
# ---------------------------------------------------------------------------


def _postgres_url() -> str:
    """Build the local pgvector Postgres URL from ``POSTGRES_*`` env vars."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ.get("POSTGRES_DATABASE", "gsf")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def reset_neo4j() -> None:
    """Detach-delete every node in the Neo4j graph."""
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
    """Drop the two pgvector embedding tables in the ``vdb`` schema."""
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


def reset_stores() -> None:
    reset_neo4j()
    reset_pgvector()


def _judge_exec_results_dir(slug: str) -> Path:
    return OUTPUT_DIR / "spider2_eval_suite" / slug / "exec_results"


def reset_between_datasets(slug: str) -> None:
    """Reset Neo4j + pgvector (and Snowflake judge cache) before a full pipeline run."""
    reset_stores()
    exec_results = _judge_exec_results_dir(slug)
    if exec_results.exists():
        shutil.rmtree(exec_results)
        logger.info("Cleared Snowflake judge exec_results cache: %s", exec_results)


# ---------------------------------------------------------------------------
# Per-database run
# ---------------------------------------------------------------------------


def _subprocess_env(slug: str) -> dict[str, str]:
    env = os.environ.copy()
    # Our value must win over the .env default; main.py calls load_dotenv()
    # which does NOT override already-set env vars.
    env["CONNECTION_STRINGS"] = _connection_string(slug)
    gsf_path = str((REPO_ROOT.parent / "GSF").resolve())
    existing = env.get("PYTHONPATH", "")
    if gsf_path not in existing.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(p for p in (gsf_path, existing) if p)
    return env


def run_eval_only(slug: str) -> Path | None:
    """Run retrieval eval only (skip ingest, semantic, LLM judge, store resets).

    Assumes Neo4j + pgvector already contain the graph/embeddings for *slug*.
    Returns the eval CSV path on success, or ``None`` if the run failed.
    """
    logger.info("=" * 70)
    logger.info("EVAL ONLY: spider2/%s", slug)
    logger.info("=" * 70)

    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--database-name",
        f"spider2/{slug}",
        "--skip-ingest",
        "--skip-semantic",
        "--skip-judge",
    ]
    logger.info("Running: CONNECTION_STRINGS=<%s> %s", slug, " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_subprocess_env(slug))
    if result.returncode != 0:
        logger.error("spider2/%s eval FAILED (exit code %d).", slug, result.returncode)
        return None

    csv_path = _eval_csv_path(slug)
    if not csv_path.exists():
        logger.error("spider2/%s produced no eval CSV at %s.", slug, csv_path)
        return None
    logger.info("spider2/%s eval CSV: %s", slug, csv_path)
    return csv_path


def run_database(slug: str) -> Path | None:
    """Reset stores and run ingest->semantic->eval for one database.

    Returns the eval CSV path on success, or ``None`` if the run failed or
    produced no CSV.
    """
    logger.info("=" * 70)
    logger.info("DATABASE: spider2/%s (%s)", slug, _dialect(slug))
    logger.info("=" * 70)

    reset_between_datasets(slug)

    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--database-name",
        f"spider2/{slug}",
        "--skip-judge",
    ]
    logger.info("Running: CONNECTION_STRINGS=<%s> %s", slug, " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_subprocess_env(slug))
    if result.returncode != 0:
        logger.error("spider2/%s FAILED (exit code %d).", slug, result.returncode)
        return None

    csv_path = _eval_csv_path(slug)
    if not csv_path.exists():
        logger.error("spider2/%s produced no eval CSV at %s.", slug, csv_path)
        return None
    logger.info("spider2/%s eval CSV: %s", slug, csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Concatenate + judge
# ---------------------------------------------------------------------------


def concatenate_csvs(csv_items: list[tuple[str, Path]], combined_path: Path) -> int:
    """Concatenate eval CSVs into ``combined_path``. Returns total data rows.

    A ``source_database`` column (``spider2/<slug>``) is prepended to each row
    so the combined file records which database each question came from.
    """
    fieldnames: list[str] = ["source_database"]
    for _slug, path in csv_items:
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
        for slug, path in csv_items:
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row["source_database"] = f"spider2/{slug}"
                    writer.writerow(row)
                    total += 1
    logger.info(
        "Concatenated %d database CSV(s) -> %s (%d rows).",
        len(csv_items),
        combined_path,
        total,
    )
    return total


def judge_with_spider2_suite(
    csv_items: list[tuple[str, Path]],
    *,
    max_workers: int = 4,
) -> dict:
    """Run the upstream Spider2 evaluation suite on each database's submission."""
    from ontology_sql_eval.judge.spider2_suite import judge_eval_csv

    summaries: list[dict] = []
    for slug, csv_path in csv_items:
        if _dialect(slug) == "snowflake":
            os.environ["CONNECTION_STRINGS"] = _connection_string(slug)
            from gsf.connectors.registry import invalidate_connectors_cache

            invalidate_connectors_cache()
        logger.info("Spider2 suite judging spider2/%s from %s", slug, csv_path)
        summary = judge_eval_csv(
            csv_path,
            slug,
            output_dir=OUTPUT_DIR / "spider2_eval_suite" / slug,
            max_workers=max_workers,
        )
        summaries.append(summary)

    correct = sum(s["correct"] for s in summaries)
    total = sum(s["total"] for s in summaries)
    combined = {
        "databases": len(summaries),
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "per_database": summaries,
    }
    combined_path = OUTPUT_DIR / "spider2_suite_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Spider2 suite overall: %d/%d correct (%.1f%%). Summary: %s",
        correct,
        total,
        combined["accuracy"] * 100,
        combined_path,
    )
    return combined


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _snowflake_slugs(manifest: dict) -> list[str]:
    """Return every Snowflake slug that has a local ``evaluation.json``."""
    slugs: list[str] = []
    for entry in manifest.get("databases", []):
        if entry.get("dialect") != "snowflake":
            continue
        slug = entry["slug"]
        if (SPIDER2_DIR / slug / "evaluation.json").is_file():
            slugs.append(slug)
    return sorted(slugs)


def _select_slugs(manifest: dict, requested: list[str] | None) -> list[str]:
    available: list[str] = []
    for entry in manifest.get("databases", []):
        slug = entry["slug"]
        dialect = entry.get("dialect", "sqlite")
        if dialect == "snowflake":
            eval_json = SPIDER2_DIR / slug / "evaluation.json"
            if eval_json.exists():
                available.append(slug)
            continue
        if _sqlite_file(slug).exists():
            available.append(slug)
    if requested:
        missing = [s for s in requested if s not in available]
        if missing:
            raise SystemExit(
                f"Requested slug(s) not available: {', '.join(missing)}\n"
                f"Available: {', '.join(available)}"
            )
        return list(requested)
    return available


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ingest->semantic->eval for Spider2 databases "
            "(resetting Neo4j + pgvector between each), then score with the "
            "upstream Spider2-lite evaluation suite."
        )
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Run only the first available database (default when --slugs omitted).",
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help="Subset of database slugs to run (default: first database only).",
    )
    parser.add_argument(
        "--snowflake-all",
        action="store_true",
        help=(
            "Run every Snowflake slug from the manifest (mutually exclusive "
            "with --slugs). Requires evaluation.json per slug from download_spider2.py."
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
            "Rerun retrieval eval only (skip ingest, semantic, store resets, and "
            "LLM judge). Neo4j + pgvector must already be populated for each slug."
        ),
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the Spider2 evaluation-suite judge stage.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=4,
        help="Worker threads for Spider2 evaluate.py (default: 4).",
    )
    parser.add_argument(
        "--combined-name",
        default=None,
        help="Basename for the combined CSV (default: spider2_all_<model>.csv).",
    )
    parser.add_argument(
        "--skip-concat",
        action="store_true",
        help="Skip concatenating per-database CSVs (judge uses per-DB files).",
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
    if args.snowflake_all and args.slugs:
        raise SystemExit("--snowflake-all and --slugs are mutually exclusive.")
    load_dotenv(REPO_ROOT / ".env")

    manifest = _load_manifest()
    available = _select_slugs(manifest, None)
    if args.snowflake_all:
        slugs = _snowflake_slugs(manifest)
        if not slugs:
            raise SystemExit(
                "No Snowflake databases available.\n"
                "Run scripts/download_spider2.py to build Snowflake evaluation.json files."
            )
    elif args.slugs:
        slugs = _select_slugs(manifest, args.slugs)
    else:
        slugs = available[:1]
    logger.info("Selected %d database(s): %s", len(slugs), ", ".join(slugs))

    combined_name = args.combined_name or f"spider2_all_{_model_slug()}.csv"
    combined_path = INPUT_DIR / combined_name

    produced: list[tuple[str, Path]] = []
    failed: list[str] = []

    if args.skip_run:
        produced = [(s, _eval_csv_path(s)) for s in slugs if _eval_csv_path(s).exists()]
        missing = [s for s in slugs if not _eval_csv_path(s).exists()]
        if missing:
            logger.warning(
                "--skip-run: no existing CSV for %d database(s): %s",
                len(missing),
                ", ".join(missing),
            )
    else:
        run_fn = run_eval_only if args.eval_only else run_database
        for i, slug in enumerate(slugs, start=1):
            label = "eval" if args.eval_only else "run"
            logger.info("[%d/%d] Starting spider2/%s (%s)", i, len(slugs), slug, label)
            try:
                csv_path = run_fn(slug)
            except Exception:
                logger.exception("spider2/%s raised an exception.", slug)
                csv_path = None

            if csv_path is None:
                failed.append(slug)
                if args.stop_on_error:
                    raise SystemExit(f"Aborting: spider2/{slug} failed.")
            else:
                produced.append((slug, csv_path))

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
        logger.info("Skipping Spider2 suite judge.")
        return

    summary = judge_with_spider2_suite(produced, max_workers=args.judge_workers)
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    if not args.skip_concat and len(produced) > 1:
        logger.info("  Combined eval CSV : %s", combined_path)
    logger.info(
        "  Spider2 suite score : %d/%d (%.1f%%)",
        summary["correct"],
        summary["total"],
        summary["accuracy"] * 100,
    )
    logger.info("  Summary JSON        : %s", OUTPUT_DIR / "spider2_suite_summary.json")


if __name__ == "__main__":
    main()
