# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ingest: source DB schema -> GSF Postgres catalog + pgvector embeddings.

Vendored from ``../GSF/dev_tools/local_ingest.py`` and retargeted so the
enrichment step reads this repo's ``datasets/`` and ``annotations/`` files
instead of GSF's copy.

Run after the Postgres catalog + pgvector services are up (see GSF's
``docker-compose.yml``) and ``CONNECTION_STRINGS`` points at the source DB::

    PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest
    PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest \\
      --dataset-name bird
"""

from __future__ import annotations

import argparse
import logging
import os

# ruff: noqa: E402 - GSF imports must follow environment bootstrap.
from ontology_sql_eval.env import load_env

# Must run before any `gsf`/`ontology_sql_eval.ingestion.enrich_graph` import:
# gsf.retrieval.generate_sql calls its own load_dotenv() at import time (no
# explicit path), which finds ../GSF*/.env first and — since load_dotenv()
# never overrides already-set vars — silently wins over this repo's .env for
# any var it defines (e.g. a stale CONNECTION_STRINGS left in a sibling GSF
# checkout's .env). Loading ours first ensures it wins the race instead.
load_env()

from gsf.connectors.registry import create_connector
from gsf.ingestion_service.ingest import run_ingest as gsf_run_ingest  # noqa: E402
from gsf.utils import get_embed_params  # noqa: E402
from gsf.vdb import get_semantic_vdb, get_vdb  # noqa: E402
from ontology_sql_eval.ingestion.enrich_graph import (  # noqa: E402
    TRAIN_QA_COLLECTION_NAME,
    TRAIN_QA_DATABASE_NAME,
    add_custom_analyses,
    add_few_shot_examples,
    apply_metadata,
    train_json_for_dataset,
)

logger = logging.getLogger(__name__)


def database_name_for(connection_string: str) -> str:
    """Return the database name a connection string resolves to."""
    connector = create_connector(connection_string)
    try:
        return connector.database_name
    finally:
        connector.close()


def run_ingest(connection_string: str, dataset_name: str | None = None) -> None:
    """Extract the source schema into GSF's store and write embeddings."""
    connector = create_connector(connection_string)
    try:
        database_name = connector.database_name
        logger.info("Starting ingest for database %r", database_name)

        gsf_run_ingest(connector)

        # Dataset metadata is persisted after GSF's catalog/embedding pipeline.
        apply_metadata(database_name, dataset=dataset_name)

        embed_params = get_embed_params()

        # Custom analyses live in the semantic-layer collection, so they go
        # through a dedicated semantic VDB rather than the tabular one the
        # ingest wrote to.
        add_custom_analyses(
            database_name,
            connector.dialect,
            embed_params=embed_params,
            vdb=get_semantic_vdb(database_name=database_name),
            dataset=dataset_name,
        )
    finally:
        connector.close()


def run_train_qa_ingest(dataset_name: str) -> int:
    """Embed a dataset's Train Q→SQL corpus into the dedicated VDB."""
    return add_few_shot_examples(
        train_json=train_json_for_dataset(dataset_name),
        embed_params=get_embed_params(),
        vdb=get_vdb(
            database_name=TRAIN_QA_DATABASE_NAME,
            collection_name=TRAIN_QA_COLLECTION_NAME,
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest source DB schema(s) into the GSF Postgres catalog + pgvector, "
            "then optionally embed a dataset's Train few-shot corpus."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Dataset folder under datasets/ whose Train corpus to embed "
            "(e.g. bird → datasets/bird/train/train.json). "
            "Omit to skip few-shot enrichment."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    # Remote source DB to extract tabular schema/embeddings from. Kept separate
    # from the local POSTGRES_* vars (which point at the pgvector store).
    connection_strings = [
        s.strip()
        for s in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if s.strip()
    ]
    if not connection_strings:
        raise EnvironmentError(
            "CONNECTION_STRINGS is not set. Add it to your .env, e.g.:\n\n"
            "    CONNECTION_STRINGS=postgresql://user:password@host:5432/dbname"
        )
    try:
        for i, connection_string in enumerate(connection_strings, start=1):
            logger.info(
                "Ingesting database %d/%d: %s",
                i,
                len(connection_strings),
                connection_string,
            )
            run_ingest(connection_string, dataset_name=args.dataset_name)

        # Train few-shots are dataset-scoped (e.g. BIRD), not per SQLite DB.
        # Only run when the caller names the dataset; missing train.json is a no-op.
        if args.dataset_name:
            run_train_qa_ingest(args.dataset_name)
    except KeyboardInterrupt:
        logger.info("ingestion: shutting down")
        raise SystemExit(0)
