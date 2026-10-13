#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Our pipeline: attribute search, followed to the chunks, then an answer.

    query → attribute vectors → unstructured_attribute__chunk → chunks → answer

Attributes are the concepts ``run_semantic.py`` extracted, and each one is
already grounded in the chunks that support it — the LLM-cited ones from
extraction plus the retrieved ones from linking.  So a query matches a
*concept* and inherits that concept's evidence, rather than matching passage
text directly.  Whether that indirection helps or hurts is what this measures,
against the plain chunk search in ``benchmark/retrieve_generate.py``.

Only the retrieval step above is local to this file.  Everything after it —
generator, prompt, token budget, reasoning cap, pacing, page-key arithmetic,
graded qrels, scoring tiers, report — is ``pipeline.py``, shared with the
baseline so that retrieval is the only difference between the two reports.

Tier 3, the LLM judge, is a separate pass in ``judge.py``.  Until it has run,
the report carries the retrieval metrics, answer-in-context and token F1, with
the judge score ``n/a``.

Prerequisites
-------------
- ``run_ingest.py`` and ``run_semantic.py`` have populated Postgres
- attribute embeddings exist (``run_semantic.py`` indexes as it extracts)

Usage
-----
    python -m unstructured.attr_retrieve_generate
    python -m unstructured.attr_retrieve_generate --limit 10
    python -m unstructured.attr_retrieve_generate --report-only

Answers and searches are appended as they land, so an interrupted run resumes
where it stopped instead of paying for the same queries twice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Loads the .env and sets the Ray GPU override before any nemo_retriever
# import, and carries every setting this run shares with the baseline.
from unstructured import pipeline

# Puts auto-ontology's unstructured_ao on sys.path and points POSTGRES_* at the
# database the semantic stage wrote to.
import unstructured.config  # noqa: F401

LABEL = "attributes"
OTHER_LABEL = "chunks"

#: An LLM-cited chunk is evidence the model demonstrably read, so it outranks a
#: chunk that merely embedded close to the attribute.
_SOURCE_RANK = {"llm": 0, "vector": 1}

_LINKED_CHUNKS_SQL = """
SELECT l.attribute_id,
       l.source,
       l.score,
       k.content,
       k.page_number,
       d.name AS document_name
  FROM unstructured_attribute__chunk l
  JOIN unstructured_chunk    k ON k.id = l.chunk_id
  JOIN unstructured_document d ON d.id = k.document_id
 WHERE l.attribute_id = ANY(:attribute_ids)
"""


class AttributeRetriever:
    """Attribute vector search in Postgres, followed to the linked chunks."""

    name = LABEL
    description = "query → attributes → linked chunks"

    def setup(self) -> None:
        from auto_ontology.dal.session import get_engine
        from unstructured_ao.vectorstore import get_attribute_store

        self._store = get_attribute_store()
        self._engine = get_engine()

    def _attribute_ids(self, query: str) -> list[str]:
        """Attribute ids matching *query*, best first, across every document."""
        hits = self._store.similarity_search_with_score(
            query=query, k=pipeline.ATTRIBUTE_K
        )
        ids = []
        for doc, _score in hits:
            attribute_id = getattr(doc, "id", None) or doc.metadata.get("id")
            if attribute_id:
                ids.append(str(attribute_id))
        return ids

    def search(self, query: str) -> tuple[list[str], list[str]]:
        """``(context_chunks, ranked_page_keys)`` for *query*.

        Chunks are ordered by the rank of the attribute that produced them,
        then by LLM-cited before vector-linked, then by distance.  A chunk
        reached through several attributes keeps its best position, and pages
        are de-duplicated the same way, so a page ranks by its strongest chunk.
        """
        from sqlalchemy import bindparam, text

        attribute_ids = self._attribute_ids(query)
        if not attribute_ids:
            return [], []

        statement = text(_LINKED_CHUNKS_SQL).bindparams(
            bindparam("attribute_ids", value=attribute_ids, expanding=False)
        )
        with self._engine.connect() as conn:
            rows = (
                conn.execute(statement, {"attribute_ids": attribute_ids})
                .mappings()
                .all()
            )

        rank_of = {attribute_id: i for i, attribute_id in enumerate(attribute_ids)}
        ordered = sorted(
            rows,
            key=lambda r: (
                rank_of.get(r["attribute_id"], len(attribute_ids)),
                _SOURCE_RANK.get(r["source"], 2),
                float(r["score"]) if r["score"] is not None else 0.0,
            ),
        )

        # Take the best CHUNK_CAP distinct chunks and derive the page ranking
        # from those, which is what chunk search does with its CHUNK_K hits.
        # Capping pages instead would let this path rank more pages than the
        # baseline ever saw off the same budget.
        chunks: list[str] = []
        pages: list[str] = []
        seen_content: set[str] = set()
        seen_pages: set[str] = set()
        for row in ordered:
            content = row["content"] or ""
            if not content or content in seen_content:
                continue
            if len(seen_content) >= pipeline.CHUNK_CAP:
                break
            seen_content.add(content)
            chunks.append(content)

            if row["page_number"] is None:
                continue
            key = pipeline.page_key(
                Path(row["document_name"]).stem,
                int(row["page_number"]) + pipeline.PAGE_OFFSET,
            )
            if key not in seen_pages:
                seen_pages.add(key)
                pages.append(key)

        return chunks[: pipeline.CONTEXT_K], pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use only the first N queries.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-score the saved answers without calling any model.",
    )
    args = parser.parse_args()

    pipeline.require_api_key()
    results = pipeline.Results(pipeline.ATTRIBUTE_RESULTS)

    queries = pipeline.load_queries(args.limit)
    qrels = pipeline.load_qrels()
    print(f"{len(queries)} {pipeline.LANGUAGE} queries, {len(qrels)} with gold pages")

    if args.report_only:
        saved = results.answers()
        records = [saved[q["query_id"]] for q in queries if q["query_id"] in saved]
        if not records:
            raise SystemExit(f"Nothing saved in {results.answers_jsonl}")
    else:
        retriever = AttributeRetriever()
        print(f"\nRETRIEVE + ANSWER — {retriever.description}")
        records = pipeline.run_pipeline(queries, retriever, results)

    pipeline.report_and_compare(
        records,
        qrels,
        results,
        LABEL,
        OTHER_LABEL,
        pipeline.CHUNK_RESULTS / "report.json",
    )


if __name__ == "__main__":
    main()
