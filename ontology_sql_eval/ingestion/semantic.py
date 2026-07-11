# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic-layer compilation: build the business taxonomy over the graph.

Thin in-process wrapper around GSF's ``run_semantic_compilation`` (the same
routine as ``python -m gsf.semantic``) so it can be imported, called, and
stepped through in a debugger without spawning a subprocess.

Run after :mod:`ontology_sql_eval.ingestion.ingest` has populated the schema
graph and embeddings::

    uv run python -m ontology_sql_eval.ingestion.semantic --database-name <database_name>
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from gsf.semantic.compile import run_semantic_compilation

logger = logging.getLogger(__name__)


def run_semantic(database_name: str) -> int:
    """Compile the semantic layer for ``database_name`` in-process.

    Runs the same routine as ``python -m gsf.semantic`` (Term/ColumnAttribute
    taxonomy, embeddings, semantic FK edges, SqlAttribute suggestions) but
    without spawning a subprocess, so it can be stepped through in a debugger.
    Requires the schema graph and embeddings to already exist (i.e. run
    :func:`ontology_sql_eval.ingestion.ingest.run_ingest` first).
    """
    logger.info("Compiling semantic layer for %s", database_name)
    return run_semantic_compilation(database_name)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ontology-sql-eval-semantic",
        description="Compile the semantic layer for a database (Term/ColumnAttribute "
        "taxonomy, embeddings, semantic FK edges, SqlAttribute suggestions).",
    )
    parser.add_argument(
        "--database-name",
        required=True,
        help="Database name — must match the SQL connector and tabular ingest.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    try:
        run_semantic(args.database_name)
    except KeyboardInterrupt:
        logger.info("semantic: shutting down")
        raise SystemExit(0)
