#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest the vidore_v3_hr PDFs into Postgres: chunks and embeddings only.

This is the mechanical half of the pipeline — NeMo Retriever extracts and
embeds each PDF, the chunks land in ``unstructured_chunk`` and their vectors in
the pgvector collection ``unstructured_chunks``.  No LLM runs here, so a corpus
can be re-ingested for the cost of the embeddings alone.

Terms and attributes come from ``run_semantic.py``, which reads these chunks
back out of Postgres.

Pre-requisites
--------------
- auto-ontology Postgres running and migrated:
      cd ../auto-ontology && uv run alembic upgrade head
- NVIDIA API key (embedding) set in the environment.
- auto-ontology checked out on ``feature/unstructured``.

Usage
-----
    uv run python unstructured/run_ingest.py
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
    args = parser.parse_args()

    # Import here so auto-ontology's sys.path addition (in config.py) has run.
    try:
        from auto_ontology.dal.session import get_engine
        from unstructured_ao.ingest import ingest_file  # auto-ontology module
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
    failed: list[Path] = []
    for pdf in pdf_paths:
        logger.info("  → %s", pdf.name)
        # A transaction per document: a PDF that fails late in a long run
        # should not roll back the ones already ingested.
        try:
            with engine.begin() as conn:
                doc_id = ingest_file(str(pdf), conn)
        except Exception:
            logger.exception("Ingest failed for %s — continuing.", pdf.name)
            failed.append(pdf)
            continue
        # ingest_file swallows an extraction failure and returns None, so the
        # return value is the only signal that this PDF produced nothing.
        if doc_id:
            doc_ids.append(doc_id)
        else:
            failed.append(pdf)

    logger.info(
        "Ingest complete: %d / %d document(s) in Postgres, %d failed.",
        len(doc_ids),
        len(pdf_paths),
        len(failed),
    )
    for pdf in failed:
        logger.warning("  failed: %s", pdf.name)
    logger.info("Next: uv run python unstructured/run_semantic.py")


if __name__ == "__main__":
    main()
