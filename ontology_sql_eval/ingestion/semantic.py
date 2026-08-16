# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic-layer compilation: build the business taxonomy over the graph.

Thin in-process wrapper around GSF's ``run_semantic_compilation`` (the same
routine as ``python -m gsf.semantic``) so it can be imported, called, and
stepped through in a debugger without spawning a subprocess.

Run after :mod:`ontology_sql_eval.ingestion.ingest` has populated the schema
graph and embeddings. Compile a single database, or omit ``--database-name``
to compile every database in ``CONNECTION_STRINGS`` (same source as ingest)::

    uv run python -m ontology_sql_eval.ingestion.semantic                            # all CONNECTION_STRINGS
    uv run python -m ontology_sql_eval.ingestion.semantic --database-name <name>     # a single database

Pass ``--override-descriptions`` to finish each database with our own saved
column/attribute descriptions from ``SAVED_DESCRIPTIONS_CSV`` (see
:mod:`ontology_sql_eval.ingestion.enrich_graph`).
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv
from gsf.semantic.compile import run_semantic_compilation

from ontology_sql_eval.ingestion.enrich_graph import apply_saved_descriptions

load_dotenv()

logger = logging.getLogger(__name__)


def run_semantic(
    database_name: str,
    *,
    override_descriptions: bool = False,
) -> int:
    """Compile the semantic layer for ``database_name`` in-process.

    Runs the same routine as ``python -m gsf.semantic`` (Term/ColumnAttribute
    taxonomy, embeddings, semantic FK edges, SqlAttribute suggestions) but
    without spawning a subprocess, so it can be stepped through in a debugger.
    Requires the schema graph and embeddings to already exist (i.e. run
    :func:`ontology_sql_eval.ingestion.ingest.run_ingest` first).

    With *override_descriptions*, apply :func:`apply_saved_descriptions` after
    compile so ColumnAttribute nodes receive the durable CSV text.
    """
    logger.info("Compiling semantic layer for %s", database_name)
    result = run_semantic_compilation(database_name)
    if override_descriptions:
        apply_saved_descriptions(database_name)
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
        prog="ontology-sql-eval-semantic",
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
        "--override-descriptions",
        action="store_true",
        help="After compiling, overwrite column and attribute descriptions with the "
        "saved set named by SAVED_DESCRIPTIONS_CSV in .env. Without that variable "
        "(or a semantic_descriptions.csv beside the database) nothing is written.",
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
            run_semantic(
                args.database_name,
                override_descriptions=args.override_descriptions,
            )
        else:
            database_names = database_names_from_env()
            if not database_names:
                raise EnvironmentError(
                    "No --database-name given and CONNECTION_STRINGS is not set. "
                    "Pass --database-name, or add CONNECTION_STRINGS to your .env."
                )
            for i, db_name in enumerate(database_names, start=1):
                logger.info(
                    "Compiling semantic layer %d/%d: %s",
                    i,
                    len(database_names),
                    db_name,
                )
                run_semantic(
                    db_name,
                    override_descriptions=args.override_descriptions,
                )
    except KeyboardInterrupt:
        logger.info("semantic: shutting down")
        raise SystemExit(0)
