#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the auto-ontology unstructured ingest pipeline on the vidore_v3_hr PDFs.

This script calls auto-ontology's ingest → extract → retrieve pipeline on
every PDF in ``unstructured/data/vidore_v3_hr/pdfs/`` and writes the results
into the auto-ontology Postgres instance (configured via ``POSTGRES_*`` env
vars) and the pgvector collection ``unstructured_chunks``.

Pre-requisites
--------------
- auto-ontology Postgres running and migrated:
      cd ../auto-ontology && uv run alembic upgrade head
- NVIDIA API key (embedding + LLM) set in the environment.
- auto-ontology checked out on ``feature/unstructured``.

Usage
-----
    uv run python unstructured/run_ingest.py
    uv run python unstructured/run_ingest.py --skip-retrieve   # skip link step
    uv run python unstructured/run_ingest.py --file path.pdf   # single file
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# config.py adds auto-ontology to sys.path and sets the Ray GPU override.
import unstructured.config as config  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", help="Ingest a single file instead of the whole PDF directory."
    )
    parser.add_argument(
        "--skip-retrieve",
        action="store_true",
        help="Skip the attribute → chunk linking step.",
    )
    args = parser.parse_args()

    # Import here so auto-ontology's sys.path addition (in config.py) has run.
    try:
        from auto_ontology.dal.session import get_engine
        from unstructured_ao.ingest import ingest_file  # auto-ontology module
        from unstructured_ao.retrieve import link_document_attributes
    except ImportError as exc:
        logger.error(
            "Cannot import auto-ontology modules: %s\n"
            "Make sure auto-ontology is checked out at %s on branch feature/unstructured.",
            exc,
            config._AUTO_ONTOLOGY_ROOT,
        )
        sys.exit(1)

    pdf_paths: list[Path]
    if args.file:
        pdf_paths = [Path(args.file)]
    else:
        if not config.PDF_DIR.exists():
            logger.error("PDF directory not found: %s", config.PDF_DIR)
            sys.exit(1)
        pdf_paths = sorted(config.PDF_DIR.glob("*.pdf"))

    if not pdf_paths:
        logger.error("No PDF files found in %s.", config.PDF_DIR)
        sys.exit(1)

    logger.info("Ingesting %d PDF(s) …", len(pdf_paths))
    engine = get_engine()

    doc_ids: list[str] = []
    with engine.begin() as conn:
        for pdf in pdf_paths:
            logger.info("  → %s", pdf.name)
            doc_id = ingest_file(str(pdf), conn)
            if doc_id:
                doc_ids.append(doc_id)
                if not args.skip_retrieve:
                    link_document_attributes(doc_id, conn)

    logger.info(
        "Ingest complete: %d / %d document(s) processed.",
        len(doc_ids),
        len(pdf_paths),
    )
    for doc_id in doc_ids:
        logger.info("  doc_id=%s", doc_id)


if __name__ == "__main__":
    main()
