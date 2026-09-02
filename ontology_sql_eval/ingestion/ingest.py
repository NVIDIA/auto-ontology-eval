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

from dotenv import load_dotenv

# Must run before any `gsf`/`ontology_sql_eval.ingestion.enrich_graph` import:
# gsf.retrieval.generate_sql calls its own load_dotenv() at import time (no
# explicit path), which finds ../GSF*/.env first and — since load_dotenv()
# never overrides already-set vars — silently wins over this repo's .env for
# any var it defines (e.g. a stale CONNECTION_STRINGS left in a sibling GSF
# checkout's .env). Loading ours first ensures it wins the race instead.
load_dotenv()

from nemo_retriever.graph import Graph  # noqa: E402
from nemo_retriever.tabular_data.operators.tabular_schema_extract_operator import (  # noqa: E402
    TabularSchemaExtractOp,
)
from nemo_retriever.tabular_data.operators.tabular_fetch_embeddings_operator import (  # noqa: E402
    TabularFetchEmbeddingsOp,
)
from nemo_retriever.operators.embed.operators import _BatchEmbedActor  # noqa: E402
from nemo_retriever.operators.vdb import IngestVdbOperator  # noqa: E402
from nemo_retriever.common.params.models import TabularExtractParams  # noqa: E402
from gsf.utils import get_embed_params  # noqa: E402
from gsf.vdb import get_data_vdb, get_semantic_vdb  # noqa: E402
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
    """Build the tabular ingest graph, run it, and write embeddings to pgvector."""
    connector = create_connector(connection_string)
    database_name = connector.database_name
    logger.info("Starting ingest for database %r", database_name)

    TABULAR_PARAMS = TabularExtractParams(
        connector=connector,
    )

    extract_graph = Graph() >> TabularSchemaExtractOp(tabular_params=TABULAR_PARAMS)
    extract_results = extract_graph.execute(None)
    schema_data = extract_results[0] if extract_results else None
    if not (isinstance(schema_data, tuple) and len(schema_data) == 2):
        raise RuntimeError(
            "TabularSchemaExtractOp did not return (tables_df, columns_df); "
            f"got {type(schema_data).__name__}."
        )

    # Curated value_examples (if any) win over what TabularSchemaExtractOp's
    # own profiling wrote; sample-value backfill from the live DB is otherwise
    # handled by GSF's own ingestion/semantic-compilation profiling step, so
    # there is no separate backfill call here anymore.
    apply_metadata(database_name)
    embed_params = get_embed_params()

    embed_graph = (
        Graph()
        >> TabularFetchEmbeddingsOp(database_name=database_name)
        >> _BatchEmbedActor(params=embed_params)
    )
    results = embed_graph.execute(schema_data)
    result_df = results[0] if results else None

    # Build the pgvector VDB once. PostgresVDB.__init__ wipes existing rows
    # for `database_name`, so reuse the same instance for the custom-analysis
    # append below — calling get_data_vdb(database_name=...) again would re-delete
    # everything we just wrote.
    vdb = get_data_vdb(database_name=database_name)

    if result_df is not None and not result_df.empty:
        ingest_op = IngestVdbOperator(vdb=vdb)
        ingest_op(result_df.to_dict(orient="records"))
        print(
            "Tabular ingest result:",
            len(result_df),
            "rows written to pgvector)",
        )
    else:
        print("Tabular ingest result: no rows produced")

    # Custom analyses live in the semantic-layer collection, so embed them
    # through a dedicated semantic VDB rather than the tabular `vdb` above.
    add_custom_analyses(
        connector.database_name,
        connector.dialect,
        embed_params=embed_params,
        vdb=get_semantic_vdb(database_name=connector.database_name),
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
