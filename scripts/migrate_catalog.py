# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migrate the Auto Ontology catalog to the installed Auto Ontology's schema.

    uv run python scripts/migrate_catalog.py

``main.py`` already does this before every run (unless ``--skip-migrate``);
this runs it on its own. See ``ontology_sql_eval/catalog_migrations.py``.
"""

from __future__ import annotations

import logging

from ontology_sql_eval.catalog_migrations import ensure_current
from ontology_sql_eval.env import load_env

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    load_env()
    ensure_current()
