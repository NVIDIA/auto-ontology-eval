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
from contextlib import nullcontext

# ruff: noqa: E402 - GSF imports must follow environment bootstrap.
from ontology_sql_eval.env import load_env

# Must run before any `gsf`/`ontology_sql_eval.ingestion.enrich_graph` import:
# gsf.retrieval.generate_sql calls its own load_dotenv() at import time (no
# explicit path), which finds ../GSF*/.env first and — since load_dotenv()
# never overrides already-set vars — silently wins over this repo's .env for
# any var it defines (e.g. a stale CONNECTION_STRINGS left in a sibling GSF
# checkout's .env). Loading ours first ensures it wins the race instead.
load_env()

from gsf.catalog import ingest_catalog
from gsf.connectors.registry import create_connector
from gsf.dal.datasources import fetch_tables_and_columns_by_node_ids
from gsf.utils import get_embed_params  # noqa: E402
from gsf.utils.embedding import batch_embed_chunks
from gsf.utils.embedding_rows import CatalogEmbeddingRowsOp
from gsf.vdb import get_data_vdb, get_semantic_vdb  # noqa: E402
from nemo_retriever.common.vdb.records import to_client_vdb_records
from ontology_sql_eval.ingestion.enrich_graph import (  # noqa: E402
    add_custom_analyses,
    apply_metadata,
)

logger = logging.getLogger(__name__)


def database_name_for(connection_string: str) -> str:
    """Return the database name a connection string resolves to."""
    connector = create_connector(connection_string)
    try:
        return connector.database_name
    finally:
        connector.close()


def _ingest_catalog_with_metadata(
    connector, dataset_name: str | None, embed_params
) -> None:
    """Write the catalog, enrich it, then embed the enriched Postgres rows."""
    database_name = connector.database_name
    reuse_connection = getattr(connector, "reuse_connection", None)
    connection_context = reuse_connection() if reuse_connection else nullcontext()
    with connection_context:
        tables_df, columns_df = ingest_catalog(connector)

    # Enrichment must precede embedding. CatalogEmbeddingRowsOp builds text from
    # DataFrames, so refresh them from Postgres after apply_metadata updates the
    # catalog instead of embedding the pre-enrichment extraction frames.
    apply_metadata(database_name, dataset=dataset_name)
    table_ids = (
        [
            str(value)
            for value in tables_df["id"].dropna().tolist()
            if str(value).strip()
        ]
        if "id" in tables_df.columns
        else []
    )
    if table_ids:
        tables_df, columns_df, _ = fetch_tables_and_columns_by_node_ids(table_ids)

    embed_rows = CatalogEmbeddingRowsOp(database_name=database_name)(
        (tables_df, columns_df)
    )
    data_vdb = None
    rows_written = 0
    for chunk in batch_embed_chunks(embed_rows, embed_params, label=database_name):
        records = to_client_vdb_records(chunk)
        if not records:
            continue
        if data_vdb is None:
            data_vdb = get_data_vdb(database_name=database_name, reset=True)
        rows_written += data_vdb.run(records)

    logger.info(
        "Tabular ingest: %d enriched catalog row(s) written to pgvector.",
        rows_written,
    )


def run_ingest(connection_string: str, dataset_name: str | None = None) -> None:
    """Extract the source schema into GSF's store and write embeddings."""
    connector = create_connector(connection_string)
    try:
        database_name = connector.database_name
        logger.info("Starting ingest for database %r", database_name)

        embed_params = get_embed_params()
        _ingest_catalog_with_metadata(connector, dataset_name, embed_params)

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest source DB schema(s) into the GSF Postgres catalog + pgvector."
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Dataset folder under datasets/ used to locate metadata and annotations "
            "(for example, bird)."
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

    except KeyboardInterrupt:
        logger.info("ingestion: shutting down")
        raise SystemExit(0)
