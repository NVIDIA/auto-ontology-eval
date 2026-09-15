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

# ruff: noqa: E402 - file-scoped: the imports after load_dotenv() below are
# deliberately late, for the reason described next.
# Load .env BEFORE importing gsf: gsf.utils.embedding (and semantic_fk/embed)
# capture EMBED_API_KEY / EMBED_ENDPOINT / EMBED_MODEL into module-level
# constants at import time. Importing gsf first freezes those to the shell's
# NVIDIA_API_KEY fallback (an sk- proxy key), causing 401s against the public
# integrate.api.nvidia.com embeddings endpoint.
from dotenv import load_dotenv

load_dotenv()

from gsf.connectors.registry import create_connector
from gsf.catalog import ingest_catalog
from gsf.dal.datasources import fetch_tables_and_columns_by_node_ids
from gsf.utils import get_embed_params
from gsf.utils.embedding import batch_embed
from gsf.utils.embedding_rows import CatalogEmbeddingRowsOp
from nemo_retriever.operators.vdb import IngestVdbOperator
from gsf.vdb import get_semantic_vdb, get_vdb
from gsf.vdb import get_data_vdb
from ontology_sql_eval.ingestion.enrich_graph import (
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
    return create_connector(connection_string).database_name


def run_ingest(connection_string: str, dataset_name: str | None = None) -> None:
    """Extract the source schema into GSF's store and write embeddings."""
    connector = create_connector(connection_string)
    database_name = connector.database_name
    logger.info("Starting ingest for database %r", database_name)

    # Populate the catalog first, then stamp dataset metadata before building
    # embedding rows. Embedding the frames returned directly by ingest_catalog
    # would miss the descriptions and samples applied below.
    tables_df, _columns_df = ingest_catalog(connector)
    apply_metadata(database_name, dataset=dataset_name)

    embed_params = get_embed_params()
    table_ids = (
        tables_df["id"].dropna().astype(str).tolist()
        if tables_df is not None and not tables_df.empty
        else []
    )
    refreshed = fetch_tables_and_columns_by_node_ids(table_ids)
    embed_rows = CatalogEmbeddingRowsOp(database_name=database_name)(refreshed[:2])
    result_df = batch_embed(embed_rows, embed_params)
    if result_df is not None and not result_df.empty:
        vdb = get_data_vdb(database_name=database_name, reset=True)
        IngestVdbOperator(vdb=vdb)(result_df.to_dict(orient="records"))
        logger.info("Tabular ingest: %d rows written to pgvector", len(result_df))
    else:
        logger.info("Tabular ingest: no embedding rows produced for %s", database_name)

    add_custom_analyses(
        database_name,
        connector.dialect,
        embed_params=embed_params,
        vdb=get_semantic_vdb(database_name=database_name),
        dataset=dataset_name,
    )


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
