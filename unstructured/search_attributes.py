#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search the attributes ``run_semantic.py`` extracted, across every document.

Chunk search and attribute search answer different questions.  Searching
``vdb.unstructured_chunks`` returns the passages that mention something;
searching ``vdb.unstructured_attributes`` returns the concepts themselves —
each with the term and report it came from, and the chunks that support it.

``run_semantic.py`` indexes attributes as it extracts them.  ``--index``
backfills anything written before indexing existed, and is also how to rebuild
the index after an embedding model change.

Usage
-----
    uv run python unstructured/search_attributes.py --index
    uv run python unstructured/search_attributes.py "youth unemployment"
    uv run python unstructured/search_attributes.py "gender pay gap" -k 20
    uv run python unstructured/search_attributes.py "wages" --chunks
"""

from __future__ import annotations

import argparse
import logging
import sys

# config.py adds auto-ontology to sys.path and sets the Ray GPU override.
import unstructured.config as config  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_SUPPORTING_CHUNKS = """
    SELECT k.page_number, l.source, left(k.content, 160) AS preview
      FROM unstructured_attribute__chunk l
      JOIN unstructured_chunk k ON k.id = l.chunk_id
     WHERE l.attribute_id = :attribute_id
     ORDER BY l.source DESC, k.page_number
     LIMIT 3
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="What to search for.")
    parser.add_argument("-k", type=int, default=10, help="Hits to return.")
    parser.add_argument("--document-id", help="Restrict to one document.")
    parser.add_argument(
        "--attribute-type", help="Restrict to one attribute type, e.g. measure."
    )
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Show the chunks supporting each hit.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index attributes missing from the vector store, then exit.",
    )
    args = parser.parse_args()

    try:
        from auto_ontology.dal.session import get_engine
        from unstructured_ao.retrieve import index_missing_attributes, search_attributes
    except ImportError as exc:
        logger.error(
            "Cannot import auto-ontology modules: %s\nIs the checkout at %s?",
            exc,
            config._AUTO_ONTOLOGY_ROOT,
        )
        sys.exit(1)

    if args.index:
        added = index_missing_attributes()
        logger.info("Indexed %d attribute(s).", added)
        return

    if not args.query:
        parser.error("a query is required unless --index is given")

    hits = search_attributes(
        args.query,
        k=args.k,
        document_id=args.document_id,
        attribute_type=args.attribute_type,
    )
    if not hits:
        logger.info(
            "No attributes matched. Has run_semantic.py run, and is the index "
            "built (--index)?"
        )
        return

    from sqlalchemy import text

    engine = get_engine() if args.chunks else None
    print(f"\n{len(hits)} attribute(s) for {args.query!r}:\n")
    for hit in hits:
        print(f"  [{hit['score']:.3f}] {hit['name']}  ({hit['attribute_type']})")
        print(f"          term: {hit['term']}   doc: {hit['document'][:60]}")
        if hit["description"]:
            print(f"          {hit['description'][:150]}")
        if engine is not None:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(_SUPPORTING_CHUNKS), {"attribute_id": hit["id"]}
                ).mappings()
                for row in rows:
                    preview = " ".join((row["preview"] or "").split())
                    print(
                        f"            └ p{row['page_number']} "
                        f"[{row['source']}] {preview[:110]}"
                    )
        print()


if __name__ == "__main__":
    main()
