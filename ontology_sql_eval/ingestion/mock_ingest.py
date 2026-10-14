# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny in-memory mock ingest for fast semantic-layer iteration.

Vendored from Auto Ontology's ``dev_tools/mock_ingest.py``. Builds a 4-table shop schema
(customer / order / orderline / products) entirely in-memory and pushes it
through the same tabular ingest pipeline that ``ontology_sql_eval.ingestion.ingest`` uses.
No remote DB, no docker dependency, no metadata JSON files. Embeddings still go
through the real NVIDIA endpoint because the semantic-compile path needs the
data-layer VDB populated.

Usage::

    uv run python -m ontology_sql_eval.ingestion.mock_ingest
"""

from __future__ import annotations

import logging
from typing import Optional

from ontology_sql_eval.env import load_env

# ruff: noqa: E402 - Auto Ontology imports must follow environment bootstrap.
# Auto Ontology resolves its model and store settings when it is imported, so
# .env has to be loaded first or those settings silently miss it.
load_env()

import pandas as pd
from nemo_retriever.operators.vdb import IngestVdbOperator
from auto_ontology.catalog import ingest_catalog
from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.utils import get_embed_params
from auto_ontology.utils.embedding import batch_embed
from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp

from auto_ontology.vdb import get_data_vdb

logger = logging.getLogger(__name__)

MOCK_DATABASE_NAME = "mock_shop"
MOCK_SCHEMA = "public"

# The same embedding settings the real ingest uses: EMBED_*, then
# DEFAULT_MODELS_*, then Auto Ontology's built-in default for the key's prefix.
# Vectors from a different model would live in a different space from the
# queries' and retrieve nothing useful.
EMBED_PARAMS = get_embed_params()

if not EMBED_PARAMS.api_key:
    raise EnvironmentError(
        "No embedding API key. Set DEFAULT_MODELS_API_KEY (or EMBED_API_KEY) in "
        ".env; see .env.example."
    )


# Intentionally non-user-friendly names (underscore prefixes, numeric suffixes)
# so the semantic extractor has to actually clean them up.
# (table_name, pk_column, [(column_name, data_type), ...])
_TABLES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "_customer",
        "_id",
        [
            ("_id", "uuid"),
            ("name1", "text"),
            ("_phone", "text"),
            ("email2", "text"),
        ],
    ),
    (
        "orders1",
        "id1",
        [
            ("id1", "uuid"),
            ("_customer_id", "uuid"),
            ("date2", "date"),
        ],
    ),
    (
        "_orderline",
        "_id",
        [
            ("_id", "uuid"),
            ("order_id1", "uuid"),
            ("_price_paid", "numeric"),
            ("product_id2", "uuid"),
        ],
    ),
    (
        "products2",
        "_id",
        [
            ("_id", "uuid"),
            ("name1", "text"),
            ("_price", "numeric"),
            ("sinceDate2", "date"),
        ],
    ),
]

# (source_table, source_column, target_table, target_column)
_FKS: list[tuple[str, str, str, str]] = [
    ("orders1", "_customer_id", "_customer", "_id"),
    ("_orderline", "order_id1", "orders1", "id1"),
    ("_orderline", "product_id2", "products2", "_id"),
]


class MockDatabase(SQLDatabase):
    """In-memory schema source: 4 hand-built tables, no real backend."""

    def __init__(self, connection_string: str = "") -> None:
        self._database_name = MOCK_DATABASE_NAME

    @property
    def dialect(self) -> str:
        return "postgres"

    @property
    def database_name(self) -> str:
        return self._database_name

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_tables(self) -> pd.DataFrame:
        rows = [
            {
                "table_schema": MOCK_SCHEMA,
                "table_name": name,
                "table_type": "base table",
            }
            for name, _, _ in _TABLES
        ]
        return pd.DataFrame(rows)

    def get_columns(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for table_name, pk_column, columns in _TABLES:
            for ordinal, (col_name, data_type) in enumerate(columns, start=1):
                rows.append(
                    {
                        "table_schema": MOCK_SCHEMA,
                        "table_name": table_name,
                        "column_name": col_name,
                        "data_type": data_type,
                        "is_nullable": "NO" if col_name == pk_column else "YES",
                        "ordinal_position": ordinal,
                    }
                )
        return pd.DataFrame(rows)

    def get_pks(self) -> pd.DataFrame:
        rows = [
            {
                "table_schema": MOCK_SCHEMA,
                "table_name": table_name,
                "column_name": pk_column,
                "ordinal_position": 1,
            }
            for table_name, pk_column, _ in _TABLES
        ]
        return pd.DataFrame(rows)

    def get_fks(self) -> pd.DataFrame:
        rows = [
            {
                "table_schema": MOCK_SCHEMA,
                "table_name": src_table,
                "column_name": src_col,
                "referenced_schema": MOCK_SCHEMA,
                "referenced_table": tgt_table,
                "referenced_column": tgt_col,
            }
            for src_table, src_col, tgt_table, tgt_col in _FKS
        ]
        return pd.DataFrame(rows)

    def get_views(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=pd.Index(["table_schema", "table_name", "view_definition"])
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        return pd.DataFrame(columns=pd.Index(["end_time", "query_text"]))

    def close(self) -> None:
        pass


def run_mock_ingest() -> None:
    """Push the mock 4-table schema through the real ingest + embed pipeline."""
    connector = MockDatabase()
    database_name = connector.database_name

    logger.info("Extracting schema for %r (mock)", database_name)
    schema_data = ingest_catalog(connector)

    logger.info("Embedding %r schema rows", database_name)
    embed_rows = CatalogEmbeddingRowsOp(database_name=database_name)(schema_data)
    result_df = batch_embed(embed_rows, EMBED_PARAMS)

    vdb = get_data_vdb(database_name=database_name, reset=True)

    if result_df is not None and not result_df.empty:
        ingest_op = IngestVdbOperator(vdb=vdb)
        ingest_op(result_df.to_dict(orient="records"))
        logger.info("Mock ingest wrote %d rows to pgvector", len(result_df))
    else:
        logger.warning("Mock ingest produced 0 rows")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_mock_ingest()
