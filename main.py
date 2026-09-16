# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline entry point.

Runs the full evaluation pipeline for a dataset, in order:

1. Ingest      source DB schema -> Postgres + pgvector (``CONNECTION_STRINGS``)
2. Semantic    compile the semantic layer (``ingestion.semantic``)
3. Eval        run the text-to-SQL agent -> ``datasets/<db>/<model>.csv``
4. Judge       LLM re-score the eval CSV -> ``datasets/<db>/<model>_scores.csv``

Stages 1-3 import GSF, so run with the sibling checkout on the path::

    PYTHONPATH=../GSF uv run python main.py --database-name wideworldimporters

Any stage can be skipped with ``--skip-ingest`` / ``--skip-semantic`` /
``--skip-eval`` / ``--skip-judge`` (skipped stages don't import their deps).

The eval stage runs ``--eval-workers`` questions concurrently and can be
restricted with ``--start-index`` / ``--end-index`` / ``--limit``. It writes a
full instrumentation bundle (per-question logs, node timings, a phase timeline
and an aggregated summary) to ``logs/<run-id>/``::

    PYTHONPATH=../GSF uv run python main.py --database-name bird \
        --skip-ingest --skip-semantic --limit 10 --eval-workers 2
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger("pipeline")

_REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _REPO_ROOT / "output"
LOG_DIR = _REPO_ROOT / "logs"


def _graph_database_name(fallback: str) -> str:
    """Graph / pgvector database name used by ingest + semantic compile.

    Ingest keys the Neo4j DB node and pgvector rows off the connector's
    ``database_name`` (the ``metadata_database`` query param of
    ``CONNECTION_STRINGS``). Semantic compile must filter by the *same* name, so
    derive it from that param rather than from ``--database-name`` (which may
    carry a logical/path-style prefix used only for file layout). Falls back
    to *fallback* when no ``metadata_database`` is present
    (e.g. Postgres, where ``database_name`` equals the URL's database).
    """
    from urllib.parse import parse_qs, unquote, urlparse

    conns = [
        s for s in os.environ.get("CONNECTION_STRINGS", "").split(",") if s.strip()
    ]
    if conns:
        values = parse_qs(urlparse(conns[0]).query).get("metadata_database")
        if values:
            name = unquote(values[0]).strip()
            if name:
                return name
    return fallback


def _banner(step: int, title: str) -> None:
    sep = "=" * 70
    logger.info("%s", sep)
    logger.info("STAGE %d: %s", step, title)
    logger.info("%s", sep)


def _connection_strings() -> list[str]:
    """Return non-empty entries from the comma-separated ``CONNECTION_STRINGS`` env var."""
    return [
        s.strip()
        for s in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if s.strip()
    ]


def stage_ingest(benchmark_name: str) -> None:
    """Ingest source schemas and apply this benchmark's enrichment artifacts."""
    _banner(1, "Ingest (source DB -> Postgres + pgvector)")
    connection_strings = _connection_strings()
    if not connection_strings:
        raise EnvironmentError(
            "CONNECTION_STRINGS is not set. Add it to your .env, e.g.:\n\n"
            "    CONNECTION_STRINGS=postgresql://user:password@host:5432/wideworldimporters"
        )

    from ontology_sql_eval.ingestion.ingest import run_ingest

    for i, connection_string in enumerate(connection_strings, start=1):
        logger.info(
            "Ingesting database %d/%d: %s",
            i,
            len(connection_strings),
            connection_string,
        )
        run_ingest(connection_string, benchmark_name)


def stage_semantic(benchmark_name: str) -> None:
    """Compile semantic layers and apply this benchmark's saved descriptions."""
    _banner(2, "Semantic compile (ingestion.semantic)")
    from ontology_sql_eval.ingestion.semantic import run_semantic

    connection_strings = _connection_strings()
    if connection_strings:
        from ontology_sql_eval.ingestion.ingest import database_name_for

        for i, connection_string in enumerate(connection_strings, start=1):
            database_name = database_name_for(connection_string)
            logger.info(
                "Compiling semantic layer %d/%d: %s",
                i,
                len(connection_strings),
                database_name,
            )
            run_semantic(database_name, benchmark_name)
    else:
        run_semantic(benchmark_name)


def stage_eval(
    *,
    database_name: str,
    workers: int = 1,
    start_index: int = 0,
    end_index: int | None = None,
    log_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Run the retrieval eval; return the path of the model CSV it wrote.

    The CSV lands in the repo-root ``input/`` folder so the judge stage (and the
    standalone judge) can pick it up directly. ``workers`` questions run
    concurrently; the run's instrumentation bundle is written to
    ``logs/<run-id>/``.
    """
    _banner(3, "Retrieval eval (text-to-SQL agent)")
    from ontology_sql_eval.retrieval.eval_chatbot import _resolve_paths, run_evaluation
    from ontology_sql_eval.retrieval.run_logging import setup_run_logging

    input_path, output_path = _resolve_paths(database_name, None, None)
    run_log = setup_run_logging(log_dir or LOG_DIR, run_id)
    run_log.log_environment(
        dataset=database_name,
        input_path=str(input_path),
        output_path=str(output_path),
        workers=workers,
        start_index=start_index,
        end_index=end_index,
        stage="pipeline",
    )
    logger.info("Eval run logs: %s", run_log.dir)
    try:
        run_evaluation(
            input_path=input_path,
            output_path=output_path,
            start_index=start_index,
            end_index=end_index,
            workers=workers,
            run_log=run_log,
        )
        summary = run_log.write_summary(workers=workers, dataset=database_name)
        logger.info(
            "Eval finished: %d questions in %.1fs (%s q/min, effective parallelism %s)",
            summary["questions"],
            summary["wall_clock_seconds"],
            summary["questions_per_minute"],
            summary["effective_parallelism"],
        )
    finally:
        run_log.close()
    return output_path


def stage_judge(eval_csv: Path, workers: int = 1) -> Path:
    """LLM re-score ``eval_csv`` from ``input/``; write scores to ``output/``."""
    _banner(4, "LLM judge (re-score eval CSV)")
    from ontology_sql_eval.judge.runner import run

    scored_path = OUTPUT_DIR / f"{eval_csv.stem}_scores.csv"
    run(eval_csv, scored_path, workers=workers)
    return scored_path


def _eval_output_path(*, database_name: str) -> Path:
    """Resolve the eval CSV path without running eval (for --skip-eval)."""
    from ontology_sql_eval.retrieval.eval_chatbot import _resolve_paths

    _, output_path = _resolve_paths(database_name, None, None)
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ontology-sql-eval-pipeline",
        description=(
            "Run the full evaluation pipeline: ingest -> semantic compile -> "
            "retrieval eval -> LLM judge."
        ),
    )
    parser.add_argument(
        "--database-name",
        required=True,
        help="Dataset / database name. "
        "Selects datasets/<name>/evaluation.json; per-question db_id routes the connector.",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Skip the ingest stage."
    )
    parser.add_argument(
        "--skip-semantic", action="store_true", help="Skip the semantic-compile stage."
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip the retrieval-eval stage."
    )
    parser.add_argument(
        "--skip-judge", action="store_true", help="Skip the LLM-judge stage."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent judge scoring workers (default: 1).",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of questions the retrieval eval runs concurrently (default: 1).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First question index the eval runs (0-based, default: 0).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Stop the eval before this question index (default: run to the end).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many questions from --start-index "
        "(ignored when --end-index is given).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help=f"Root directory for eval run logs (default: {LOG_DIR}).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Name for the eval run's log directory (default: a timestamp).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    database_name = args.database_name

    from ontology_sql_eval.env import load_env

    load_env()

    if not args.skip_ingest:
        stage_ingest(database_name)

    if not args.skip_semantic:
        stage_semantic(database_name)

    end_index = args.end_index
    if end_index is None and args.limit is not None:
        end_index = args.start_index + args.limit

    if not args.skip_eval:
        eval_csv = stage_eval(
            database_name=database_name,
            workers=args.eval_workers,
            start_index=args.start_index,
            end_index=end_index,
            log_dir=args.log_dir,
            run_id=args.run_id,
        )
    else:
        eval_csv = _eval_output_path(database_name=database_name)
        logger.info("Skipping eval; using existing %s", eval_csv)

    if not args.skip_judge:
        if not eval_csv.exists():
            raise SystemExit(
                f"Eval CSV not found: {eval_csv}\n"
                "Run without --skip-eval, or ensure the file exists before judging."
            )
        scored = stage_judge(eval_csv, workers=args.workers)
        logger.info("Pipeline complete. Scored output: %s", scored)


if __name__ == "__main__":
    main()
