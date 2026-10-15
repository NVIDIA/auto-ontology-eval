# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default route, measured: chunk search, then the ontology around it.

    query → chunks → unstructured_attribute__chunk → attributes → families, relations

The route itself is ``unstructured_ao.ask`` — this is the harness that scores
it. The opposite direction, concepts first, has been measured and is capped:
only chunks linked to the indicators a question happens to match can ever be
delivered, which is 72.5% of what chunk search delivers, and ordering those
candidates by question similarity already extracts all of it — on every query
tried, every reachable chunk from chunk search's top ten came back in the
reranked top ten. The missing quarter is not mis-ranked, it is unreachable, so
no retrieval change on that side can recover it.

This starts where chunk search ends. The context is the baseline's context,
built by the baseline's retriever against the baseline's store, and the
ontology is gathered around those chunks afterwards — attribute names, link
sources and scores, families, relations.

Parity came first and was established before anything was added: stripped of
its labels this delivered the baseline's chunks in the baseline's order,
verified position by position on 20 queries, and scored within the generator's
own sampling variance of it — 0.6250 against 0.5875 on contexts that were
byte-identical, which is what that variance looks like. The chunk baseline is
therefore the control for everything here, differing in the annotation and
nothing else.

Each passage is labelled with the indicators it is a reading of. The note —
how those indicators relate — is gathered on every row and withheld, which is
the next experiment rather than an oversight.
"""

import argparse

from unstructured import pipeline
from unstructured_ao import search

LABEL = "chunks + ontology"
OTHER_LABEL = "chunks"

#: Whether the ontology note goes in front of the context. Off: it is gathered
#: and recorded on every row, and the generator does not see it.
#:
#: Separate from the labels, which are always attached, because they are two
#: different claims — a label says what this passage is a reading of, the note
#: says how the readings relate — and because measuring them together would
#: leave neither attributable. Turning it on is the next experiment.
NOTE = False


class ChunkFirstRetriever:
    """Chunk search, then everything the ontology knows about the result."""

    name = LABEL
    description = "query → chunks → attributes of those chunks"

    def setup(self) -> None:
        from nemo_retriever.graph.retriever import Retriever

        from unstructured.benchmark import config

        # The control's own retriever, store and top_k, which is the one search
        # this harness still performs itself and the reason it does: parity with
        # the baseline is then by construction rather than by coincidence.
        # Postgres chunk search could stand in for it — both clients return a
        # byte-identical query vector for the same question, and the chunk
        # vectors agree — but then parity would rest on two paths staying in
        # step, and this rests on there being one path. Everything after the
        # search is the library's.
        self._retriever = Retriever(
            top_k=pipeline.CHUNK_K,
            embed_kwargs=config.embed_kwargs(),
            vdb_kwargs={"uri": config.LANCEDB_URI, "table_name": config.TABLE_NAME},
        )

    def search(self, query: str) -> tuple[list[str], list[str], dict]:
        hits = self._retriever.query(query)
        chunks = [hit.get("text", "") for hit in hits[: pipeline.CONTEXT_K]]
        # Ranked over every hit, not the delivered ones: the baseline scores
        # retrieval over CHUNK_K pages while answering from CONTEXT_K chunks,
        # and parity means matching both halves of that.
        pages = list(
            dict.fromkeys(
                pipeline.page_key(
                    hit["pdf_basename"], hit["page_number"] + pipeline.PAGE_OFFSET
                )
                for hit in hits
            )
        )

        found = search.chunk_ontology(chunks)
        annotations: dict = {
            "labels": ["; ".join(names) for names in found.labels],
            # Recorded, not rendered. ``build_context`` reads ``labels`` and
            # ``preamble`` and nothing else, so these two keys are inert by
            # construction and not by anyone remembering to leave them alone.
            "attributes": found.links,
            "ontology": found.note,
        }
        if NOTE and found.note:
            annotations["preamble"] = found.note
        return chunks, pages, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Use only the first N queries.")
    parser.add_argument(
        "--note",
        action="store_true",
        help="Also put the gathered relation note in front of the context.",
    )
    parser.add_argument(
        "--tag",
        help="Suffix for the results directory, so a variant does not "
        "overwrite the parity run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Queries in flight at once. Lower it when resuming: every "
        "unanswered query searches at once, and the embedding endpoint "
        "answers that burst with 429s until the retries give up — 54 of 58 "
        "rows were lost that way, all of which retrieved fine one at a time. "
        "Generation is capped at GEN_RPM either way, so this costs little.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-score the saved answers without calling any model.",
    )
    args = parser.parse_args()

    global NOTE
    NOTE = args.note
    if args.workers:
        pipeline.MAX_WORKERS = args.workers

    pipeline.require_api_key()
    directory = pipeline.CHUNK_FIRST_RESULTS
    if args.tag:
        directory = directory.with_name(f"{directory.name}-{args.tag}")
    results = pipeline.Results(directory)

    queries = pipeline.load_queries(args.limit)
    qrels = pipeline.load_qrels()
    print(
        f"{len(queries)} {pipeline.LANGUAGE} queries, {len(qrels)} with gold pages, "
        f"labels on, ontology note {'ON' if NOTE else 'withheld'}"
    )

    if args.report_only:
        saved = results.answers()
        records = [saved[q["query_id"]] for q in queries if q["query_id"] in saved]
        if not records:
            raise SystemExit(f"Nothing saved in {results.answers_jsonl}")
    else:
        retriever = ChunkFirstRetriever()
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

    gathered = [r for r in records if r.get("attributes")]
    if gathered:
        per_row = [len({link["attribute_id"] for link in r["attributes"]}) for r in gathered]
        noted = sum(1 for r in records if r.get("ontology"))
        print(
            f"\nontology gathered on {len(gathered)}/{len(records)} rows: "
            f"{sum(per_row) / len(per_row):.1f} indicators per context, "
            f"relation note non-empty on {noted}"
        )


if __name__ == "__main__":
    main()
