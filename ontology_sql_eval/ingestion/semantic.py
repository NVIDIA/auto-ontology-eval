# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic-layer compilation: build the business taxonomy in Postgres.

Thin in-process wrapper around GSF's ``run_semantic_compilation`` (the same
routine as ``python -m gsf.semantic``) so it can be imported, called, and
stepped through in a debugger without spawning a subprocess.

Run after :mod:`ontology_sql_eval.ingestion.ingest` has populated the schema
catalog and embeddings. Compile a single database, or omit ``--database-name``
to compile every database in ``CONNECTION_STRINGS`` (same source as ingest)::

    uv run python -m ontology_sql_eval.ingestion.semantic                            # all CONNECTION_STRINGS
    uv run python -m ontology_sql_eval.ingestion.semantic --database-name <name>     # a single database

When ``annotations/<dataset>/semantic_descriptions.csv`` exists, each database
is automatically finished with its saved column and attribute descriptions.
See :mod:`ontology_sql_eval.ingestion.enrich_graph`.
"""

from __future__ import annotations

import argparse
import logging
import os

from ontology_sql_eval.env import load_env

# ruff: noqa: E402 - GSF imports must follow environment bootstrap.
# Load this repository's environment before importing GSF modules that read it,
# while retaining GSF's environment as a fallback.
load_env()

from gsf.semantic.compile import run_semantic_compilation  # noqa: E402

from ontology_sql_eval.ingestion.enrich_graph import (  # noqa: E402
    apply_saved_descriptions,
)

logger = logging.getLogger(__name__)


def run_semantic(
    database_name: str,
    benchmark_name: str | None = None,
) -> int:
    """Compile the semantic layer for ``database_name`` in-process.

    Runs the same routine as ``python -m gsf.semantic`` (Term/ColumnAttribute
    taxonomy, embeddings, semantic FK edges, SqlAttribute suggestions) but
    without spawning a subprocess, so it can be stepped through in a debugger.
    Requires the schema catalog and embeddings to already exist (i.e. run
    :func:`ontology_sql_eval.ingestion.ingest.run_ingest` first).

    Saved column and attribute descriptions are written over freshly compiled
    ones when an annotation file exists — see
    :func:`ontology_sql_eval.ingestion.enrich_graph.apply_saved_descriptions`. Here
    is the earliest they can be applied: the ColumnAttribute nodes half of them
    belong to are what the compile creates. (Custom analyses need no such wait and
    are ingested during ingest.)

    When *benchmark_name* is omitted, annotation resolution searches every
    benchmark folder for a matching saved description set.
    """
    logger.info("Compiling semantic layer for %s", database_name)
    result = run_semantic_compilation(database_name)
    apply_saved_descriptions(database_name, benchmark_name)
    return result


def database_names_from_env() -> list[str]:
    """Resolve database names from ``CONNECTION_STRINGS`` (same source as ingest)."""
    from ontology_sql_eval.ingestion.ingest import database_name_for

    connection_strings = [
        s.strip()
        for s in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if s.strip()
    ]
    return [database_name_for(cs) for cs in connection_strings]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auto-ontology-eval-semantic",
        description="Compile the semantic layer for a database (Term/ColumnAttribute "
        "taxonomy, embeddings, semantic FK edges, SqlAttribute suggestions).",
    )
    parser.add_argument(
        "--database-name",
        default=None,
        help="Database name — must match the SQL connector and tabular ingest. "
        "When omitted, every database in CONNECTION_STRINGS is compiled in turn.",
    )
    parser.add_argument(
        "--benchmark-name",
        "--dataset-name",
        dest="benchmark_name",
        default=None,
        help="e.g. bird — same value as ingest. Used to locate saved artifacts "
        "under annotations/<benchmark>/ and beside the database. "
        "--dataset-name is a compatibility alias.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    try:
        if args.database_name:
            run_semantic(args.database_name, args.benchmark_name)
        else:
            database_names = database_names_from_env()
            if not database_names:
                raise EnvironmentError(
                    "No --database-name given and CONNECTION_STRINGS is not set. "
                    "Pass --database-name, or add CONNECTION_STRINGS to your .env."
                )
            for i, database_name in enumerate(database_names, start=1):
                logger.info(
                    "Compiling semantic layer %d/%d: %s",
                    i,
                    len(database_names),
                    database_name,
                )
                run_semantic(database_name, args.benchmark_name)
    except KeyboardInterrupt:
        logger.info("semantic: shutting down")
        raise SystemExit(0)
