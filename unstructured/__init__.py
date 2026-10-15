# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark for the auto-ontology unstructured pipeline.

Scores retrieval and answering against ViDoRe v3 HR:
  https://huggingface.co/datasets/vidore/vidore_v3_hr

What it measures lives in ``unstructured_ao`` — the retrieval, the ontology and
the answering are the product's, and a harness that owned them could not be
replaced without taking them with it.  What is here is the corpus, the scoring
and the arms.

Setting up, once
----------------
1. ``download_data.py`` — fetch the 14 PDFs and build ``queries_en.csv``.
2. ``run_ingest.py`` — chunks and embeddings into Postgres + pgvector.
3. ``run_semantic.py`` — terms and attributes, by LLM, per document.
4. ``unstructured_ao.relate`` (in the library) — how those attributes stand to
   one another, which is what the annotations come from.

Measuring
---------
    benchmark/retrieve_generate.py   chunk search — the control
    chunk_first.py                   the default route: chunks, then ontology
    attr_retrieve_generate.py        the other direction: concepts, then chunks
    judge.py                         tier 3, as its own pass over the answers

``pipeline.py`` holds everything the arms share, so retrieval is the only
variable between them, and ``config.py`` the environment they all need found
before anything connects.  ``scripts/`` is untracked scratch — one experiment
per file, none of it imported from here.
"""
