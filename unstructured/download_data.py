#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the ViDoRe v3 HR benchmark into ``unstructured/data/vidore_v3_hr/``.

Source: https://huggingface.co/datasets/vidore/vidore_v3_hr (CC-BY-4.0, ungated).

Produces the layout the rest of this module expects::

    data/vidore_v3_hr/
      pdfs/*.pdf              14 European Commission HR/labour reports
      queries_en.csv          query_id, query, gold_pages
      qrels/qrels_en.csv      query_id, doc_id, page_number, score

``gold_pages`` is the ``"<doc_id> p.<page> | <doc_id> p.<page>"`` form that
``eval_retrieval.parse_gold_pages`` reads.  Page numbers are **0-indexed**,
matching both the dataset's ``page_number_in_doc`` and the ``page_number``
NeMo Retriever puts on each extracted chunk.

The upstream ``corpus`` split is ~440 MB because it carries a page image per
row; we only need three small columns from it, so it is read with a parquet
column projection over HTTP range requests rather than downloaded whole.

Usage
-----
    uv run python unstructured/download_data.py
    uv run python unstructured/download_data.py --force   # re-download
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from unstructured import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

REPO_ID = "vidore/vidore_v3_hr"
#: Only English queries are evaluated; the dataset also ships five translations.
LANGUAGE = "english"


def _download_pdfs(force: bool) -> list[Path]:
    """Fetch every ``pdfs/*.pdf`` file from the dataset repo."""
    from huggingface_hub import hf_hub_download, list_repo_files

    names = [
        f
        for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith("pdfs/") and f.endswith(".pdf")
    ]
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for name in sorted(names):
        target = config.PDF_DIR / Path(name).name
        if target.exists() and not force:
            logger.info("  have %s", target.name)
            paths.append(target)
            continue
        logger.info("  get  %s", target.name)
        cached = hf_hub_download(REPO_ID, name, repo_type="dataset")
        target.write_bytes(Path(cached).read_bytes())
        paths.append(target)
    return paths


def _read_split(split: str, columns: list[str]) -> list[dict]:
    """Read ``<split>/test-*.parquet`` from the Hub, projecting *columns* only."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    remote = f"datasets/{REPO_ID}/{split}/test-00000-of-00001.parquet"
    with fs.open(remote, "rb") as handle:
        return pq.read_table(handle, columns=columns).to_pylist()


def _build_tables() -> tuple[list[dict], list[dict]]:
    """Return ``(query_rows, qrel_rows)`` for the English subset."""
    logger.info("Reading queries …")
    queries = _read_split("queries", ["query_id", "query", "language"])
    english = {r["query_id"]: r["query"] for r in queries if r["language"] == LANGUAGE}
    logger.info("  %d English queries of %d total", len(english), len(queries))

    logger.info("Reading corpus page index …")
    corpus = _read_split("corpus", ["corpus_id", "doc_id", "page_number_in_doc"])
    page_of = {
        r["corpus_id"]: (r["doc_id"], int(r["page_number_in_doc"])) for r in corpus
    }
    logger.info(
        "  %d pages across %d documents",
        len(page_of),
        len({d for d, _ in page_of.values()}),
    )

    logger.info("Reading qrels …")
    qrels = _read_split("qrels", ["query_id", "corpus_id", "score"])

    gold: dict[str, list[tuple[str, int]]] = {}
    qrel_rows: list[dict] = []
    for r in qrels:
        qid = r["query_id"]
        if qid not in english:
            continue
        located = page_of.get(r["corpus_id"])
        if located is None:
            logger.warning("qrel references unknown corpus_id %s", r["corpus_id"])
            continue
        doc_id, page = located
        gold.setdefault(qid, []).append((doc_id, page))
        qrel_rows.append(
            {
                "query_id": qid,
                "doc_id": doc_id,
                "page_number": page,
                "score": int(r["score"]),
            }
        )

    query_rows = [
        {
            "query_id": qid,
            "query": text,
            "gold_pages": " | ".join(
                f"{doc_id} p.{page}" for doc_id, page in sorted(set(gold.get(qid, [])))
            ),
        }
        for qid, text in english.items()
    ]
    query_rows.sort(key=lambda r: int(r["query_id"]))
    return query_rows, qrel_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d row(s) → %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download PDFs that already exist."
    )
    parser.add_argument(
        "--pdfs-only", action="store_true", help="Skip rebuilding the query CSVs."
    )
    args = parser.parse_args()

    logger.info("Downloading PDFs → %s", config.PDF_DIR)
    pdfs = _download_pdfs(args.force)
    logger.info("%d PDF(s) ready.", len(pdfs))

    if args.pdfs_only:
        return

    query_rows, qrel_rows = _build_tables()
    _write_csv(config.QUERIES_CSV, query_rows, ["query_id", "query", "gold_pages"])
    _write_csv(
        config.QRELS_DIR / "qrels_en.csv",
        qrel_rows,
        ["query_id", "doc_id", "page_number", "score"],
    )

    unjudged = sum(1 for r in query_rows if not r["gold_pages"])
    if unjudged:
        logger.warning("%d query(ies) have no gold pages.", unjudged)


if __name__ == "__main__":
    main()
