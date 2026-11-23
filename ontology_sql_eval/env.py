# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment bootstrap shared by this repo's entry points.

Loads this repo's own ``.env`` first, then Auto Ontology's ``.env`` (via
``auto_ontology.env.load_env()``) as a fallback for anything not already set --
``dotenv.load_dotenv()`` called with no explicit path resolves relative to
the *calling file's* location (frame-based search, not CWD). Loading ours first
preserves the existing priority (this repo's .env wins for any variable already
set; see ``ingestion/ingest.py`` and ``ingestion/semantic.py`` for why that
matters).
"""

from __future__ import annotations

from dotenv import load_dotenv

from auto_ontology.env import load_env as _load_auto_ontology_env


def load_env() -> None:
    """Load ``.env`` into the process environment.

    Auto Ontology reads its feature flags (e.g. ``INTERACTIVE``) straight off
    ``os.environ``, so this must run before any ``auto_ontology.*`` import that
    depends on them.
    """
    load_dotenv()
    _load_auto_ontology_env()
