# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The BEAM graph scripts stop or narrow themselves where a mistake would stick."""

from __future__ import annotations

import types
from typing import Any

import pytest
import sqlalchemy as sa

import build_graph
import verify_graph


def test_a_failed_vector_wipe_stops_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Carrying on would leave the old vectors pointing at deleted rows."""

    class Store:
        def delete_by_database(self, database_name: str) -> None:
            raise RuntimeError("vector store down")

    vdb = types.ModuleType("auto_ontology.vdb")
    vdb.get_conversation_semantic_vdb = lambda **_: Store()  # type: ignore[attr-defined]
    vdb.get_one_hop_path_vdb = lambda **_: Store()  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "auto_ontology.vdb", vdb)

    with pytest.raises(RuntimeError, match="vector store down"):
        build_graph._wipe(database_name="beam_100K_2", conversation_id="2")


def test_deleting_an_edge_matches_its_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """A semantic edge with the same endpoints and role is a different row."""
    edge = sa.Table(
        "conversation_edge",
        sa.MetaData(),
        sa.Column("source_id", sa.Text, primary_key=True),
        sa.Column("target_id", sa.Text, primary_key=True),
        sa.Column("type", sa.Text, primary_key=True),
        sa.Column("role", sa.Text, primary_key=True),
    )
    statements: list[Any] = []

    class Store:
        def query_write(self, statement: Any) -> list[Any]:
            statements.append(statement)
            return [("supersedes",)]

    sys_modules = __import__("sys").modules
    schema = types.ModuleType("auto_ontology.dal.schema")
    schema.conversation_edge = edge  # type: ignore[attr-defined]
    session = types.ModuleType("auto_ontology.dal.session")
    session.store = Store  # type: ignore[attr-defined]
    dal = types.ModuleType("auto_ontology.dal")
    dal.schema = schema  # type: ignore[attr-defined]
    dal.session = session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys_modules, "auto_ontology.dal", dal)
    monkeypatch.setitem(sys_modules, "auto_ontology.dal.schema", schema)
    monkeypatch.setitem(sys_modules, "auto_ontology.dal.session", session)

    draft = object()
    row = {
        "source_id": "a",
        "target_id": "b",
        "type": "knowledge",
        "role": "supersedes",
    }
    verify_graph._delete({id(draft): row}, [draft])

    [statement] = statements
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "conversation_edge.type = 'knowledge'" in sql
