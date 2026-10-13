#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract terms and attributes from documents ``run_ingest.py`` already stored.

This is the LLM half of the pipeline.  Per document:

1. Read its chunks back from Postgres and group them into windows of
   ``EXTRACT_WINDOW_CHARS`` characters, so the whole document is read rather
   than the prefix that fits one prompt.
2. Walk the windows, extracting the term and its attributes and linking each
   attribute to the chunks that window cited (``source='llm'``).
3. Vector-search the whole document for supporting chunks no window cited
   (``source='vector'``).

It works from the chunks, not the PDFs, so it needs neither the files nor NeMo
Retriever — and it can be re-run with ``--force`` after a prompt change without
re-ingesting anything.

Cost is roughly one LLM call per window plus one per document for the term.
Each call is a few seconds of waiting on the endpoint, and the windows of a
document must run in order because every prompt carries the attributes found
so far.  Documents do not share that state, so they run concurrently and the
wall time is the longest single document rather than the sum of all of them.

Usage
-----
    uv run python unstructured/run_semantic.py
    uv run python unstructured/run_semantic.py --force         # redo processed docs
    uv run python unstructured/run_semantic.py --skip-link     # llm links only
    uv run python unstructured/run_semantic.py --workers 1     # serialise
    uv run python unstructured/run_semantic.py --document-id <id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# config.py adds auto-ontology to sys.path and sets the Ray GPU override.
import unstructured.config as config  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def process_document(document_id: str, conn, *, force: bool, link: bool) -> bool:
    """Extract and link one document. Returns False when it was skipped."""
    from unstructured_ao.dal import delete_term, get_document_chunks, get_document_term
    from unstructured_ao.extract import run_extraction
    from unstructured_ao.retrieve import link_document_attributes
    from unstructured_ao.windows import build_windows

    existing = get_document_term(conn, document_id)
    if existing and not force:
        logger.info(
            "  already has term %r — skipping (use --force to redo).", existing["name"]
        )
        return False

    # Checked before the delete below: dropping a usable term for a document
    # that turns out to have nothing to re-extract from would lose data.
    chunks = get_document_chunks(conn, document_id)
    if not chunks:
        logger.warning("  no chunks — run run_ingest.py for this document first.")
        return False

    # A document has at most one term, so a second pass has to clear the first.
    if existing:
        logger.info("  replacing term %r.", existing["name"])
        delete_term(conn, existing["id"])

    windows = build_windows(
        [c["id"] for c in chunks], [c["content"] or "" for c in chunks]
    )
    logger.info("  %d chunk(s) in %d window(s).", len(chunks), len(windows))

    run_extraction(document_id, windows, conn)
    if link:
        link_document_attributes(document_id, conn)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", help="Process a single document.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract documents that already have a term.",
    )
    parser.add_argument(
        "--skip-link",
        action="store_true",
        help="Skip the vector linking step, leaving only the llm links.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Documents to extract concurrently (default 2, 1 to serialise). "
            "Each document issues an embedding request per attribute while "
            "linking, so this costs the endpoint far more than one call at a "
            "time per worker; 4 saturated it."
        ),
    )
    args = parser.parse_args()

    try:
        from sqlalchemy import text

        from auto_ontology.dal.session import get_engine
        from unstructured_ao.dal import list_documents
    except ImportError as exc:
        logger.error(
            "Cannot import auto-ontology modules: %s\n"
            "Make sure auto-ontology is checked out at %s on branch feature/unstructured.",
            exc,
            config._AUTO_ONTOLOGY_ROOT,
        )
        sys.exit(1)

    engine = get_engine()

    if args.document_id:
        documents = [{"id": args.document_id, "name": args.document_id}]
    else:
        # Without --force, documents that already have a term are left alone,
        # so an interrupted run resumes instead of re-paying for them.
        with engine.connect() as conn:
            documents = list_documents(conn, without_term=not args.force)
            ingested = documents or list_documents(conn)
            # Chunk count stands in for how long a document will take, which
            # is what the longest-first ordering below needs.
            counts = dict(
                conn.execute(
                    text(
                        "SELECT document_id, count(*) FROM unstructured_chunk"
                        " GROUP BY document_id"
                    )
                ).all()
            )
        for doc in documents:
            doc["chunk_count"] = counts.get(doc["id"], 0)
        if not documents:
            logger.info(
                "Nothing to do — every ingested document already has a term."
                if ingested
                else "No documents ingested yet — run run_ingest.py first."
            )
            return

    logger.info("Running extraction on %d document(s) …", len(documents))

    def run_one(doc: dict) -> bool:
        # A transaction per document, so one failure does not discard the
        # extraction already paid for on the others — and so the workers do
        # not share a connection.
        with engine.begin() as conn:
            return process_document(
                doc["id"], conn, force=args.force, link=not args.skip_link
            )

    processed: list[str] = []
    failed: list[str] = []
    # Longest first: the tail of a run is one straggler document finishing its
    # windows, and starting the big ones last makes that tail as long as they are.
    documents.sort(key=lambda d: d.get("chunk_count") or 0, reverse=True)
    workers = max(1, min(args.workers, len(documents)))
    with ThreadPoolExecutor(workers) as pool:
        pending = {pool.submit(run_one, doc): doc for doc in documents}
        for i, future in enumerate(as_completed(pending), 1):
            doc = pending[future]
            try:
                done = future.result()
            except Exception:
                logger.exception("Extraction failed for %s — continuing.", doc["name"])
                failed.append(doc["name"])
                continue
            logger.info("  [%d/%d] %s", i, len(documents), doc["name"])
            if done:
                processed.append(doc["id"])

    logger.info(
        "Extraction complete: %d processed, %d skipped, %d failed.",
        len(processed),
        len(documents) - len(processed) - len(failed),
        len(failed),
    )
    for name in failed:
        logger.warning("  failed: %s", name)


if __name__ == "__main__":
    main()
