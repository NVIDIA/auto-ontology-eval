# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment bootstrap shared by this repo's entry points.

Loads this repo's own ``.env`` first, then GSF's ``.env`` (via
``gsf.env.load_env()``) as a fallback for anything not already set --
``dotenv.load_dotenv()`` called with no explicit path resolves relative to
the *calling file's* location (frame-based search, not CWD), so a bare call
from this module only ever finds ``auto-ontology-eval/.env``; it never finds
GSF's sibling checkout on its own. Loading ours first preserves the existing
priority (this repo's .env wins over GSF's for any var both define -- see
``ingestion/ingest.py`` and ``ingestion/semantic.py`` for why that matters).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from gsf.env import load_env as _load_gsf_env

# BIRD_INTERACT is a master flag: set BIRD_INTERACT=true (with no per-flag
# overrides in .env) to turn on every flag this deployment has approved for
# BIRD-Interact runs, in one place, instead of setting each individually.
#
# ASK_OUTPUT_TYPE, INJECT_CONDITIONAL_OUTPUT_HINT, and CLARIFY_MAX_DISTANCE
# are deliberately NOT listed here — they already default to INTERACTIVE's
# value when unset (see GSF's entity_resolution.py, output_type.py,
# conditional_output.py), so setting INTERACTIVE=true covers them.
#
# Each entry is applied with os.environ.setdefault(), so an explicit value
# already set in .env (or the process environment) always wins over this
# master flag's default.
_BIRD_INTERACT_DEFAULTS: dict[str, str] = {
    "INTERACTIVE": "true",
    "DB_PROBE_JSONB_PATH_CHECK": "true",
    "DB_PROBE_JOIN_PATH_CHECK": "true",
    "DB_PROBE_PROACTIVE": "true",
    "DETECT_VACUOUS_GROUP_BY": "true",
    "RELEVANCE_FILTER_INCLUDE_COLUMNS": "true",
    "HUB_SIBLING_EXPANSION_ENABLED": "true",
    "HUB_SIBLING_CAP": "6",
    "TABLE_BRIDGE_RECONCILIATION_ENABLED": "true",
}


def load_env() -> None:
    """Load ``.env`` into the process environment.

    If BIRD_INTERACT is truthy after loading, also applies the approved
    BIRD-Interact flag defaults (see _BIRD_INTERACT_DEFAULTS) to any of them
    left unset by .env. GSF's own retrieval/ingestion code reads these flags
    straight off ``os.environ``, so this must run before any ``gsf.*`` import
    that depends on them.
    """
    load_dotenv()
    _load_gsf_env()
    if os.environ.get("BIRD_INTERACT", "").strip().lower() in ("true", "1"):
        for key, value in _BIRD_INTERACT_DEFAULTS.items():
            os.environ.setdefault(key, value)
