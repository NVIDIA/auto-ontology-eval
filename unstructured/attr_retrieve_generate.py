#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Our pipeline, measured: attribute search, followed to the chunks, answered.

    query → attribute vectors → unstructured_attribute__chunk → chunks → answer

The retrieval itself is ``unstructured_ao.search.attribute_search`` and the
ontology block in front of it is ``unstructured_ao.ontology``; the answer is
``unstructured_ao.answer``. None of that is here, because none of it is
measurement: it is what the product does, and a harness that owned it could
not be replaced without taking the product with it.

What is here is the harness's side of the contract. It turns the library's
passages into the page keys this benchmark scores against — NeMo numbers pages
from one and the qrels from zero, which is a correction only an evaluation
needs — chooses the experiment's knobs, and hands the rest to ``pipeline.py``,
shared with the baseline so that retrieval is the only difference between the
two reports.

Whether reaching passages through concepts helps or hurts is the question, and
the control is the plain chunk search in ``benchmark/retrieve_generate.py``.

Tier 3, the LLM judge, is a separate pass in ``judge.py``. Until it has run,
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

# Loads the .env, sets the Ray GPU override before any nemo_retriever import,
# puts the library on sys.path, and carries every setting this run shares with
# the baseline.
from unstructured import pipeline
from unstructured_ao import search

LABEL = "attributes"
OTHER_LABEL = "chunks"

#: Order the retrieved chunks by question-to-chunk similarity instead of by
#: interleaving the matched indicators.  Off by default; ``--rerank`` sets it.
#:
#: Round-robin puts indicator #1's nearest chunk at position 1 and indicator
#: #4's at position 4, so a chunk carrying the answer lands fourth whenever its
#: indicator ranked fourth, however good the chunk is.  Measured over 72
#: queries, the chunk holding the most of the gold answer sits at median
#: position 4 here against median 2 for chunk search, over contexts whose gold
#: content is otherwise within three points of each other: the chunks are
#: equally good and arrive in a worse order.
#:
#: Ordering them the way chunk search does is a deliberate borrowing.  It
#: isolates whether indicators *select* worse chunks or merely *present* them
#: worse, which nothing measured so far can tell apart.  It is a diagnostic
#: rather than a design: a pipeline that needs the baseline's ranking to order
#: its own output has conceded the ranking.
RERANK = False

#: Whether the ontology note goes in front of the context.  ``--no-preamble``
#: withholds it, so a rerank run measures ordering and nothing else.
PREAMBLE = True

#: An upper bound on the corpus, used as the rerank query's ``top_k``.  The two
#: stores hold 1801 chunks each; asking for more costs nothing and survives a
#: re-ingest that adds a few.
CORPUS_CHUNKS = 4000


class ChunkRanker:
    """Where the baseline's chunk search would put each chunk for a question.

    A measuring instrument, not a retrieval step, which is why it is here and
    not in the library: it reads the *control's* store — the LanceDB table the
    published baseline was built from, which exists only in this repo — and
    exists only to answer whether indicators select worse chunks or merely
    present them worse.  The library takes its output as an ``order`` and knows
    nothing about where the ranking came from.

    That store holds a vector for every one of the 1801 chunks Postgres has,
    ingested from the same chunking, so a chunk can be looked up by its text
    and every candidate gets a position without embedding a single passage.
    """

    def __init__(self) -> None:
        from nemo_retriever.graph.retriever import Retriever

        from unstructured.benchmark import config

        # Comfortably more than the corpus holds, so every candidate the
        # attribute side offers gets a position: one left unranked would sort
        # to the end on a technicality rather than on similarity.
        self._retriever = Retriever(
            top_k=CORPUS_CHUNKS,
            embed_kwargs=config.embed_kwargs(),
            vdb_kwargs={"uri": config.LANCEDB_URI, "table_name": config.TABLE_NAME},
        )

    def positions(self, query: str) -> dict[str, int]:
        """``{chunk text: rank}``, best first, for *query*.

        Keyed on the joinable form: the ranking comes from one store and the
        chunks being ranked come from the other, and a fifth of them differ by
        a doubled space.
        """
        hits = self._retriever.query(query)
        return {
            search.joinable(hit.get("text", "")): rank
            for rank, hit in enumerate(hits)
            if hit.get("text")
        }


class AttributeRetriever:
    """The harness's side of the contract with ``attribute_search``.

    No retrieval happens here. What is left once the search moved to the
    library is a translation: the library answers in documents and page
    numbers, and this benchmark is scored in page keys counted from zero, which
    is a correction nothing outside an evaluation has any reason to apply. The
    harness calls this slot a retriever, so the name stays.
    """

    name = LABEL
    description = "query → attributes → linked chunks"

    def setup(self) -> None:
        self._ranker = ChunkRanker() if RERANK else None

    def search(self, query: str) -> tuple[list[str], list[str], dict]:
        """``(context_chunks, ranked_page_keys, annotations)`` for *query*.

        ``annotations`` carries what the ontology knows and the chunk text does
        not: which attribute each chunk was reached through, and a note naming
        the rival readings when the corpus states one indicator several times.
        It is kept out of ``chunks`` so tiers 1 and 2 score the same material
        on both sides.
        """
        found = search.attribute_search(
            query,
            attribute_k=pipeline.ATTRIBUTE_K,
            context_k=pipeline.CONTEXT_K,
            # The page ranking is derived from this many distinct chunks, which
            # matches the baseline's CHUNK_K: capping pages instead would let
            # this path rank more pages than the baseline ever saw off the same
            # budget.
            chunk_cap=pipeline.CHUNK_CAP,
            order=self._ranker.positions(query) if self._ranker else None,
        )
        if not found.passages:
            return [], [], {}

        pages = [
            pipeline.page_key(Path(document).stem, page + pipeline.PAGE_OFFSET)
            for document, page in found.ranked_pages
        ]

        annotations = {"labels": found.labels}
        if PREAMBLE and found.note:
            annotations["preamble"] = found.note
        return found.chunks, pages, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use only the first N queries.")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Order the retrieved chunks by question-to-chunk similarity "
        "instead of interleaving the matched indicators.",
    )
    parser.add_argument(
        "--no-preamble",
        action="store_true",
        help="Withhold the ontology note, leaving the context bare.",
    )
    parser.add_argument(
        "--tag",
        help="Suffix for the results directory, so a variant's answers do not "
        "overwrite the measured baseline's.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-score the saved answers without calling any model.",
    )
    args = parser.parse_args()

    global RERANK, PREAMBLE
    RERANK = args.rerank
    PREAMBLE = not args.no_preamble

    pipeline.require_api_key()
    directory = pipeline.ATTRIBUTE_RESULTS
    if args.tag:
        directory = directory.with_name(f"{directory.name}-{args.tag}")
    results = pipeline.Results(directory)

    queries = pipeline.load_queries(args.limit)
    qrels = pipeline.load_qrels()
    print(
        f"{len(queries)} {pipeline.LANGUAGE} queries, {len(qrels)} with gold pages, "
        f"rerank {'ON' if RERANK else 'off'}, "
        f"ontology note {'on' if PREAMBLE else 'OFF'}"
    )

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
