#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark baseline: LanceDB chunk search, then answer from those chunks.

    query → chunk vectors (LanceDB) → chunks → answer

This is the reference the attribute pipeline is measured against, and it is
deliberately plain: embed the query, take the closest chunks, hand the top few
to the generator.  It reads the LanceDB table ``ingest.py`` wrote and touches
nothing of auto-ontology's Postgres.

Only the retrieval step above is local to this file.  Everything after it —
generator, prompt, token budget, reasoning cap, pacing, page-key arithmetic,
graded qrels, scoring tiers, report — is ``unstructured/pipeline.py``, shared
with ``unstructured/eval_attributes.py`` so that retrieval is the only
difference between the two reports.

Tier 3, the LLM judge, is a separate pass in ``unstructured/judge.py``. Until it
has run, the report carries the retrieval metrics, answer-in-context and token
F1, with the judge score ``n/a``.

Prerequisite (already done — lancedb/ ships with the repo):
    python ingest.py data/vidore_v3_hr/pdfs/*.pdf

Usage
-----
    python -m unstructured.benchmark.retrieve_generate
    python -m unstructured.benchmark.retrieve_generate --limit 10
    python -m unstructured.benchmark.retrieve_generate --report-only

Answers and searches are appended as they land, so an interrupted run resumes
where it stopped instead of paying for the same queries twice.
"""

from __future__ import annotations

import argparse

# Loads the .env and sets the Ray GPU override before any nemo_retriever
# import, and carries every setting the run shares with the attribute side.
from unstructured import pipeline
from unstructured.benchmark import config

LABEL = "chunks"
OTHER_LABEL = "attributes"


class ChunkRetriever:
    """Chunk vector search in LanceDB."""

    name = LABEL
    description = "query → chunk vectors → chunks"

    def setup(self) -> None:
        from nemo_retriever.graph.retriever import Retriever

        if not config.has_ingested_data():
            raise SystemExit(
                f"No ingested documents at {config.LANCEDB_URI}. "
                "Run `python ingest.py data/vidore_v3_hr/pdfs/*.pdf` first."
            )
        self._retriever = Retriever(
            top_k=pipeline.CHUNK_K,
            vdb_kwargs={"uri": config.LANCEDB_URI, "table_name": config.TABLE_NAME},
        )

    def search(self, query: str) -> tuple[list[str], list[str]]:
        """``(context_chunks, ranked_page_keys)`` for *query*.

        Pages are de-duplicated in rank order, so a page ranks by its best
        chunk — several of the 30 hits routinely land on the same page.
        """
        hits = self._retriever.query(query)
        ranked = [
            pipeline.page_key(h["pdf_basename"], h["page_number"] + pipeline.PAGE_OFFSET)
            for h in hits
        ]
        return (
            [h.get("text", "") for h in hits[: pipeline.CONTEXT_K]],
            list(dict.fromkeys(ranked)),
        )


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
    results = pipeline.Results(pipeline.CHUNK_RESULTS)

    queries = pipeline.load_queries(args.limit)
    qrels = pipeline.load_qrels()
    print(f"{len(queries)} {pipeline.LANGUAGE} queries, {len(qrels)} with gold pages")

    if args.report_only:
        saved = results.answers()
        records = [saved[q["query_id"]] for q in queries if q["query_id"] in saved]
        if not records:
            raise SystemExit(f"Nothing saved in {results.answers_jsonl}")
    else:
        retriever = ChunkRetriever()
        print(f"\nRETRIEVE + ANSWER — {retriever.description}")
        records = pipeline.run_pipeline(queries, retriever, results)

    pipeline.report_and_compare(
        records,
        qrels,
        results,
        LABEL,
        OTHER_LABEL,
        pipeline.ATTRIBUTE_RESULTS / "report.json",
    )


if __name__ == "__main__":
    main()
