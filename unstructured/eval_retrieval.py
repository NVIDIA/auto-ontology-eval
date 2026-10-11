#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate chunk retrieval quality against the ViDoRe v3 HR benchmark.

For each query in ``queries_en.csv``:
  1. Embed the query using the NVIDIA embedding model.
  2. Search ``unstructured_chunks`` pgvector collection for top-k chunks.
  3. Match retrieved chunk ``(document_name, page_number)`` against gold pages.
  4. Compute recall@k, precision@k, nDCG@k.

Gold page format in the CSV:
  ``"filename_without_ext p.14 | filename_without_ext p.60"``

Retrieved chunk metadata fields (set at ingest time):
  ``source``: basename of the PDF (without extension, matching gold file names)
  ``page_number``: integer page number

Usage
-----
    uv run python unstructured/eval_retrieval.py
    uv run python unstructured/eval_retrieval.py --k 5 --k 10 --k 20
    uv run python unstructured/eval_retrieval.py --output results/unstructured_retrieval.csv
    uv run python unstructured/eval_retrieval.py --max-queries 20   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# config.py adds auto-ontology to sys.path and sets the Ray GPU override.
import unstructured.config as config  # noqa: F401
from unstructured.metrics import aggregate, ndcg_at_k, precision_at_k, recall_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gold-page parsing
# ---------------------------------------------------------------------------


def parse_gold_pages(gold_pages_str: str) -> set[str]:
    """Return a set of canonical ``"stem:page"`` strings from the CSV cell.

    Input  : ``"foo_doc-KE001 p.14 | foo_doc-KE001 p.60 | bar_doc p.3"``
    Returns: ``{"foo_doc-KE001:14", "foo_doc-KE001:60", "bar_doc:3"}``
    """
    result: set[str] = set()
    for entry in gold_pages_str.split("|"):
        entry = entry.strip()
        if not entry:
            continue
        # Format: "<stem> p.<n>"
        if " p." in entry:
            stem, page_part = entry.rsplit(" p.", 1)
            try:
                result.add(f"{stem.strip()}:{int(page_part.strip())}")
            except ValueError:
                logger.debug("Could not parse page number in %r", entry)
        else:
            # No page qualifier — treat whole entry as a document-level match.
            result.add(entry)
    return result


def chunk_to_key(metadata: dict) -> str | None:
    """Map a retrieved chunk's metadata to a canonical ``"stem:page"`` key."""
    source = metadata.get("source", "")
    page = metadata.get("page_number")
    if not source:
        return None
    stem = Path(source).stem if "." in source else source
    if page is not None:
        try:
            return f"{stem}:{int(page)}"
        except (ValueError, TypeError):
            pass
    return stem  # document-level match when page unknown


# ---------------------------------------------------------------------------
# PGVectorStore search
# ---------------------------------------------------------------------------


def build_vector_store():
    """Return the PGVectorStore for the ``unstructured_chunks`` collection."""
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from langchain_postgres import PGVectorStore

    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.utils.model_config import resolve

    embeddings = NVIDIAEmbeddings(
        model=resolve("EMBED", "MODEL"),
        api_key=resolve("EMBED", "API_KEY"),
        base_url=resolve("EMBED", "ENDPOINT"),
    )
    return PGVectorStore.from_params(
        embedding=embeddings,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
        collection_name=config.CHUNK_COLLECTION,
        connection=get_postgres_connection_string(),
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def load_queries(max_queries: int | None = None) -> list[dict]:
    if not config.QUERIES_CSV.exists():
        raise FileNotFoundError(f"Queries CSV not found: {config.QUERIES_CSV}")
    with config.QUERIES_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_queries:
        rows = rows[:max_queries]
    return rows


def run_eval(
    ks: list[int],
    max_queries: int | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run retrieval eval and return aggregated metrics."""
    try:
        vdb = build_vector_store()
    except Exception as exc:
        logger.error(
            "Cannot connect to PGVectorStore: %s\n"
            "Ensure the auto-ontology Postgres is running and documents are ingested "
            "(run run_ingest.py first).",
            exc,
        )
        sys.exit(1)

    queries = load_queries(max_queries)
    logger.info("Running retrieval eval on %d query(ies) at k=%s …", len(queries), ks)

    max_k = max(ks)
    per_query_rows: list[dict] = []

    for i, row in enumerate(queries):
        query_id = row["query_id"]
        query_text = row["query"]
        gold_pages_str = row.get("gold_pages", "")
        relevant: set[str] = (
            parse_gold_pages(gold_pages_str) if gold_pages_str else set()
        )

        try:
            hits = vdb.similarity_search_with_score(query=query_text, k=max_k)
        except Exception as exc:
            logger.warning("Search failed for query %s: %s", query_id, exc)
            hits = []

        retrieved_keys: list[str] = []
        for doc, _score in hits:
            key = chunk_to_key(doc.metadata)
            if key:
                retrieved_keys.append(key)

        metrics: dict = {"query_id": query_id}
        for k in ks:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_keys, relevant, k)
            metrics[f"precision@{k}"] = precision_at_k(retrieved_keys, relevant, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_keys, relevant, k)
        per_query_rows.append(metrics)

        if (i + 1) % 10 == 0:
            logger.info("  %d / %d queries done …", i + 1, len(queries))

    aggregated = aggregate(per_query_rows)

    # --- Print summary ---
    print("\n=== ViDoRe v3 HR — Chunk Retrieval Eval ===")
    print(f"Queries evaluated : {len(per_query_rows)}")
    for k in ks:
        print(
            f"  k={k:2d}  recall={aggregated.get(f'recall@{k}', 0):.3f}  "
            f"precision={aggregated.get(f'precision@{k}', 0):.3f}  "
            f"nDCG={aggregated.get(f'ndcg@{k}', 0):.3f}"
        )

    # --- Write CSV ---
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["query_id"] + [
            f"{m}@{k}" for m in ("recall", "precision", "ndcg") for k in ks
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in per_query_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        # Also write aggregated metrics as JSON beside the CSV.
        agg_path = output_path.with_suffix(".summary.json")
        agg_path.write_text(json.dumps(aggregated, indent=2))
        logger.info("Results written to %s", output_path)
        logger.info("Aggregated metrics → %s", agg_path)

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        dest="ks",
        default=[],
        help="Rank cutoff (can be repeated, e.g. --k 5 --k 10). Default: 5 10.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit the number of queries evaluated (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write per-query metrics to this CSV path (default: print to stdout only).",
    )
    args = parser.parse_args()

    ks = args.ks or [5, 10]
    run_eval(ks=ks, max_queries=args.max_queries, output_path=args.output)


if __name__ == "__main__":
    main()
