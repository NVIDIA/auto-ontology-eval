# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Spider2 Snowflake credentials (PAT in snowflake_credential.json).

Usage::

    PYTHONPATH=../GSF uv run python scripts/test_snowflake_connection.py
    PYTHONPATH=../GSF uv run python scripts/test_snowflake_connection.py --database PATENTS
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ontology_sql_eval.spider2_snowflake import snowflake_connection_string  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Spider2 Snowflake connectivity.")
    parser.add_argument(
        "--database",
        default="PATENTS",
        help="Snowflake database name (default: PATENTS).",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    os.environ["CONNECTION_STRINGS"] = snowflake_connection_string(args.database)

    from gsf.connectors import get_connectors  # noqa: WPS433 — after env is set

    connector = get_connectors()[0]
    print(f"Connected: dialect={connector.dialect} database={connector.database_name}")
    session = connector.execute(
        "SELECT CURRENT_DATABASE() AS db, CURRENT_ROLE() AS role, "
        "CURRENT_WAREHOUSE() AS warehouse"
    )
    print(session.to_dict("records")[0])
    schemas = connector.get_schemas()
    print(f"Visible schemas: {len(schemas)} (first 5: {schemas[:5]})")


if __name__ == "__main__":
    main()
