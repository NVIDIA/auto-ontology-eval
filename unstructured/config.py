# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared paths and settings for the unstructured eval."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Ray GPU override: must be set before nemo_retriever is imported -----------
os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data" / "vidore_v3_hr"
PDF_DIR = DATA_DIR / "pdfs"
QUERIES_CSV = DATA_DIR / "queries_en.csv"
QRELS_DIR = DATA_DIR / "qrels"

# --- auto-ontology path -------------------------------------------------------
# Add the auto-ontology checkout to sys.path so its `unstructured_ao` package is
# importable without installing it.  That checkout must be on the
# `feature/unstructured` branch — the package does not exist on main.
_REPO_ROOT = MODULE_DIR.parent
_SIBLING_CANDIDATES = ("ao-unstructured", "auto-ontology")


def _find_auto_ontology_root() -> Path:
    override = os.environ.get("AUTO_ONTOLOGY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for name in _SIBLING_CANDIDATES:
        candidate = (_REPO_ROOT.parent / name).resolve()
        if (candidate / "unstructured_ao").is_dir():
            return candidate
    return (_REPO_ROOT.parent / _SIBLING_CANDIDATES[0]).resolve()


_AUTO_ONTOLOGY_ROOT = _find_auto_ontology_root()
if str(_AUTO_ONTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ONTOLOGY_ROOT))

# Collection name that auto-ontology's ingest uses — must match.
CHUNK_COLLECTION = "unstructured_chunks"

# Retrieval eval settings
RETRIEVE_TOP_K = 10
