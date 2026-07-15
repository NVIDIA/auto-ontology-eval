# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spider2-lite Snowflake connection helpers.

Snowflake databases are hosted — nothing is downloaded. Point
``CONNECTION_STRINGS`` at the live instance using credentials from
``third_party/Spider2/spider2-lite/evaluation_suite/snowflake_credential.json``.

Per the upstream Spider2 guideline, put your **Programmatic Access Token (PAT)**
in the ``password`` field (not your interactive login password).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIAL_PATH = (
    _REPO_ROOT
    / "third_party"
    / "Spider2"
    / "spider2-lite"
    / "evaluation_suite"
    / "snowflake_credential.json"
)


def load_snowflake_credentials(
    credential_path: Path | None = None,
) -> dict[str, Any]:
    """Load Spider2 ``snowflake_credential.json`` (``user``/``username``, PAT in ``password``)."""
    path = credential_path or DEFAULT_CREDENTIAL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Snowflake credentials not found: {path}\n"
            "Copy snowflake_credential.example.json and set your PAT as password."
        )
    cred = json.loads(path.read_text(encoding="utf-8"))
    user = cred.get("user") or cred.get("username")
    if not user:
        raise ValueError(f"{path}: missing user/username")
    password = cred.get("password")
    if not password:
        raise ValueError(f"{path}: missing password (use your PAT here)")
    account = cred.get("account")
    warehouse = cred.get("warehouse")
    if not account or not warehouse:
        raise ValueError(f"{path}: missing account and/or warehouse")
    return cred


def snowflake_connection_string(
    database: str,
    *,
    credential_path: Path | None = None,
    metadata_database: str | None = None,
) -> str:
    """Build a GSF ``snowflake://`` URI for a Spider2 hosted database.

    Parameters
    ----------
    database:
        Snowflake database name from ``spider2-lite.jsonl`` (e.g. ``PATENTS``).
    metadata_database:
        Optional graph/pgvector name override (e.g. ``spider2/patents``).
    """
    cred = load_snowflake_credentials(credential_path)
    user = cred.get("user") or cred["username"]
    password = cred["password"]
    account = cred["account"]
    warehouse = cred["warehouse"]

    query = [
        f"warehouse={quote(str(warehouse), safe='')}",
        f"database={quote(database, safe='')}",
    ]
    role = cred.get("role")
    if role:
        query.append(f"role={quote(str(role), safe='')}")
    if metadata_database:
        query.append(f"metadata_database={quote(metadata_database, safe='')}")

    return (
        f"snowflake://{quote(str(user), safe='')}:{quote(str(password), safe='')}"
        f"@{account}?{'&'.join(query)}"
    )
