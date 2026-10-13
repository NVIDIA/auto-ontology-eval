#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the two mechanisms that link attributes to chunks.

Extraction links the chunks it cited (``source='llm'``) and the retrieve step
links the chunks that embed closest (``source='vector'``).  There is no gold
label for attribute → chunk links — the ViDoRe qrels grade query → page, which
``eval_retrieval.py`` measures — so this reports how the two mechanisms relate
rather than which is right.

Reading the counts
------------------
The link upsert keeps the vector distance on a row the extractor also cited,
so the three populations are distinguishable:

    llm only      source='llm'    and score IS NULL      cited, not retrieved
    both          source='llm'    and score IS NOT NULL  cited and retrieved
    vector only   source='vector'                        retrieved, not cited

"Both" is the agreement rate: how often vector search finds a chunk the model
demonstrably read.  A low rate means the embedding is missing real evidence.
The score percentiles show whether agreed links sit at shorter distances than
vector-only ones — if they do, that gap is where a link threshold belongs.

Usage
-----
    uv run python unstructured/eval_links.py
    uv run python unstructured/eval_links.py --document-id <id>
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys

# config.py adds auto-ontology to sys.path and sets the Ray GPU override.
import unstructured.config as config  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_LINKS_SQL = """
SELECT l.source,
       l.score,
       a.id   AS attribute_id,
       a.name AS attribute_name,
       d.id   AS document_id
FROM unstructured_attribute__chunk l
JOIN unstructured_attribute a ON a.id = l.attribute_id
JOIN unstructured_term      t ON t.id = a.term_id
JOIN unstructured_document  d ON d.id = t.document_id
-- Cast required: a parameter used only in IS NULL has no inferrable type.
WHERE (CAST(:document_id AS TEXT) IS NULL OR d.id = CAST(:document_id AS TEXT))
"""

_ATTRIBUTES_SQL = """
SELECT a.id, d.id AS document_id
FROM unstructured_attribute a
JOIN unstructured_term     t ON t.id = a.term_id
JOIN unstructured_document d ON d.id = t.document_id
WHERE (CAST(:document_id AS TEXT) IS NULL OR d.id = CAST(:document_id AS TEXT))
"""


def _percentiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    quantiles = [
        ordered[0],
        statistics.median(ordered),
        ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
    ]
    return "  ".join(f"{q:.3f}" for q in quantiles)


def report(document_id: str | None = None) -> dict:
    from sqlalchemy import text

    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        links = (
            conn.execute(text(_LINKS_SQL), {"document_id": document_id})
            .mappings()
            .all()
        )
        attributes = (
            conn.execute(text(_ATTRIBUTES_SQL), {"document_id": document_id})
            .mappings()
            .all()
        )

    if not attributes:
        logger.error(
            "No attributes found%s — run run_ingest.py first.",
            f" for document {document_id}" if document_id else "",
        )
        sys.exit(1)

    llm_only = [r for r in links if r["source"] == "llm" and r["score"] is None]
    both = [r for r in links if r["source"] == "llm" and r["score"] is not None]
    vector_only = [r for r in links if r["source"] == "vector"]

    cited = {r["attribute_id"] for r in links if r["source"] == "llm"}
    uncited = [a for a in attributes if a["id"] not in cited]
    documents = {a["document_id"] for a in attributes}

    retrieved = len(both) + len(vector_only)
    agreement = len(both) / retrieved if retrieved else 0.0

    print("\n=== Attribute → chunk links by source ===")
    print(f"Documents          : {len(documents)}")
    print(f"Attributes         : {len(attributes)}")
    print(f"Links              : {len(links)}")
    print(f"  llm only         : {len(llm_only)}")
    print(f"  both             : {len(both)}")
    print(f"  vector only      : {len(vector_only)}")
    print(f"Agreement          : {agreement:.3f}  (both / all retrieved)")
    print(
        f"Attributes w/o llm link: {len(uncited)} "
        f"({len(uncited) / len(attributes):.1%}) — vector links only"
    )
    print("\nScore percentiles (min / median / p90), lower is closer:")
    print(f"  both             : {_percentiles([float(r['score']) for r in both])}")
    print(
        f"  vector only      : {_percentiles([float(r['score']) for r in vector_only])}"
    )

    return {
        "documents": len(documents),
        "attributes": len(attributes),
        "llm_only": len(llm_only),
        "both": len(both),
        "vector_only": len(vector_only),
        "agreement": agreement,
        "attributes_without_llm_link": len(uncited),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-id", help="Report on a single document instead of all."
    )
    args = parser.parse_args()
    report(args.document_id)


if __name__ == "__main__":
    main()
