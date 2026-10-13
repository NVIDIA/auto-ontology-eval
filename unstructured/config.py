# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared paths and settings for the unstructured eval."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

MODULE_DIR = Path(__file__).parent

# auto-ontology reads POSTGRES_* and EMBED_* straight from the environment, and
# its own loader looks in the auto-ontology checkout, not here.  Every script in
# this package imports this module first, so loading the repo-root .env here is
# what lets them run without the variables having been exported by hand.
#
# override=True because a shell that has sourced a sibling repo's .env carries a
# stale EMBED_API_KEY, and model_config.resolve() prefers that over this repo's
# DEFAULT_MODELS_API_KEY — pairing another gateway's key with our endpoint for a
# 401 that reads like a bad key rather than the wrong one.
load_dotenv(MODULE_DIR.parent / ".env", override=True)

# --- Ray GPU override: must be set before nemo_retriever is imported -----------
os.environ.setdefault(
    "RAY_OVERRIDE_RESOURCES",
    json.dumps({"GPU": 0, "CPU": max(16, os.cpu_count() or 8)}),
)

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

# --- .env -------------------------------------------------------------------
# Must run after the sys.path insertion above: load_env() reaches through to
# auto_ontology.env for the fallback .env.  Without this the POSTGRES_* and
# model triplets are never read and every script here falls back to built-in
# defaults chosen from an API key that isn't set either.
from ontology_sql_eval.env import load_env as _load_env  # noqa: E402

_load_env()

# The vector table is named by auto-ontology's `unstructured_ao.config`; it is
# deliberately not restated here, so the two cannot drift apart.

# Retrieval eval settings
RETRIEVE_TOP_K = 10
