# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ingest: source DB schema -> pgvector embeddings store.

Vendored from ``../GSF/dev_tools/local_ingest.py`` and retargeted so the graph
enrichment step reads this repo's ``datasets/<database_name>/`` data files
(``metadata.json`` / ``custom_analyses.json``) instead of GSF's copy.

Run after the Neo4j + Postgres (pgvector) services are up (see GSF's
``docker-compose.yml``) and ``CONNECTION_STRINGS`` points at the source DB::

    uv run python -m ontology_sql_eval.ingestion.ingest
"""

from __future__ import annotations

import logging
import os

from ontology_sql_eval.env import load_env

# Must run before any `gsf`/`ontology_sql_eval.ingestion.enrich_graph` import:
# gsf.retrieval.generate_sql calls its own load_dotenv() at import time (no
# explicit path), which finds ../GSF*/.env first and — since load_dotenv()
# never overrides already-set vars — silently wins over this repo's .env for
# any var it defines (e.g. a stale CONNECTION_STRINGS left in a sibling GSF
# checkout's .env). Loading ours first ensures it wins the race instead.
load_env()

from gsf.ingestion_service.ingest import run_ingest as gsf_run_ingest  # noqa: E402
from gsf.utils import get_embed_params  # noqa: E402
from gsf.vdb import get_semantic_vdb  # noqa: E402
from gsf.connectors.registry import create_connector  # noqa: E402
from ontology_sql_eval.ingestion.enrich_graph import (  # noqa: E402
    add_custom_analyses,
    apply_metadata,
)

logger = logging.getLogger(__name__)


def database_name_for(connection_string: str) -> str:
    """Return the database name a connection string resolves to."""
    return create_connector(connection_string).database_name


def run_ingest(connection_string: str) -> None:
    """Extract the source schema into GSF's store and write embeddings."""
    connector = create_connector(connection_string)
    database_name = connector.database_name
    logger.info("Starting ingest for database %r", database_name)

    gsf_run_ingest(connector)

    # After the catalog write, as before. Metadata never fed the embeddings --
    # those are built from the frames the extract step returned, not re-read
    # from the store -- so its position relative to embedding does not matter.
    apply_metadata(database_name)

    embed_params = get_embed_params()

    # Custom analyses live in the semantic-layer collection, so they go through
    # a dedicated semantic VDB rather than the tabular one the ingest wrote to.
    add_custom_analyses(
        database_name,
        connector.dialect,
        embed_params=embed_params,
        vdb=get_semantic_vdb(database_name=database_name),
    )
    connector.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
            run_ingest(connection_string)
    except KeyboardInterrupt:
        logger.info("ingestion: shutting down")
        raise SystemExit(0)
