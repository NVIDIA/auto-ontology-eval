# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval scripts for the auto-ontology unstructured document pipeline.

Evaluates chunk retrieval quality against the ViDoRe v3 HR benchmark:
  https://huggingface.co/datasets/vidore/vidore_v3_hr

Steps
-----
1. ``download_data.py`` — fetch the 14 PDFs and build ``queries_en.csv``.
2. ``run_ingest.py``  — run auto-ontology's ingest pipeline on the PDFs so
   chunks land in Postgres + pgvector.
3. ``eval_retrieval.py``  — for each benchmark query do a vector similarity
   search and score against the gold pages in queries_en.csv.
"""
