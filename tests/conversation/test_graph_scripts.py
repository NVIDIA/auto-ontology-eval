# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph commands stop or narrow themselves where a mistake would stick.

Only what this repository owns. Judging a stored graph and deleting the edges
it got wrong is the library's, and is tested there -- see
``auto_ontology/conversations/tests/test_repair.py``.
"""

from __future__ import annotations

import types

import pytest

from auto_ontology_eval.ingestion import conversation_semantic


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
        conversation_semantic._wipe(database_name="beam_100K_2")
