# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The BEAM download is pinned, recorded, and keeps its files where they belong."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from auto_ontology_eval.beam import run
from scripts.seed import seed_beam


def test_every_repo_is_read_at_a_pinned_commit() -> None:
    for repo, _ in seed_beam.SCALES.values():
        assert len(seed_beam.REVISIONS[repo]) == 40


@pytest.mark.parametrize("bad", ["../../etc", "a/b", "2x", ""])
def test_a_conversation_id_that_is_not_a_number_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        seed_beam._conversation_id(bad)


def test_the_download_reads_and_records_the_pinned_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened: list[str] = []

    class FileSystem:
        def open(self, remote: str, mode: str) -> Any:
            opened.append(remote)
            return open(__file__, "rb")

    table = types.SimpleNamespace(
        to_pylist=lambda: [
            {"conversation_id": 7, "chat": [], "probing_questions": "{}"}
        ]
    )
    hub = types.ModuleType("huggingface_hub")
    hub.HfFileSystem = FileSystem  # type: ignore[attr-defined]
    parquet = types.ModuleType("pyarrow.parquet")
    parquet.read_table = lambda handle, columns: table  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setattr(seed_beam, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run, "DATA_DIR", tmp_path)

    seed_beam._write_scale("100K", force=True)

    revision = seed_beam.REVISIONS["Mohammadta/BEAM"]
    assert opened == [
        f"datasets/Mohammadta/BEAM@{revision}/data/100K-00000-of-00001.parquet"
    ]
    source = json.loads((tmp_path / "100K" / "source.json").read_text())
    assert source["revision"] == revision
    assert run.dataset_source("100K") == source
