# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task's db_id finds its connector whatever case the dataset spells it in."""

from __future__ import annotations

from ontology_sql_eval.gym.connectors import connector_for


def test_an_exact_name_wins() -> None:
    assert connector_for({"bird": "exact", "BIRD": "other"}, "bird") == "exact"


def test_wide_world_importers_upper_case_db_id_finds_its_postgres_database() -> None:
    """The dataset's db_id is upper case; Postgres names the database lower case."""
    assert connector_for({"wideworldimporters": "wwi"}, "WIDEWORLDIMPORTERS") == "wwi"


def test_names_differing_only_in_case_are_ambiguous() -> None:
    assert connector_for({"Shop": "a", "shop": "b"}, "SHOP") is None


def test_an_unknown_db_id_finds_nothing() -> None:
    assert connector_for({"wideworldimporters": "wwi"}, "bird") is None
