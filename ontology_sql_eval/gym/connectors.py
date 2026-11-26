# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick the Auto Ontology connector a task's ``db_id`` names."""

from __future__ import annotations

from typing import Mapping, TypeVar

T = TypeVar("T")


def connector_for(connectors: Mapping[str, T], db_id: str) -> T | None:
    """The connector for *db_id*: an exact name first, else the one name that
    matches ignoring case.

    A connector is named after its database as the connection string spells it,
    and Postgres folds unquoted names to lower case, so Wide World Importers'
    ``WIDEWORLDIMPORTERS`` tasks run against a ``wideworldimporters`` database.
    Two connectors whose names differ only in case are ambiguous, so neither is
    returned.
    """
    if db_id in connectors:
        return connectors[db_id]
    folded = [
        c for name, c in connectors.items() if name.casefold() == db_id.casefold()
    ]
    return folded[0] if len(folded) == 1 else None
