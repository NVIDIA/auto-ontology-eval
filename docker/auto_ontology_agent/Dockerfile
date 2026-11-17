# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Auto Ontology agent's sandbox image: Auto Ontology and this repository,
# the BIRD dev databases, and a catalog Postgres with BIRD dev ingested and
# compiled by this repository's own pipeline (main.py --skip-eval).
#
# Build it with scripts/build_agent_image.sh, which seeds BIRD dev first and
# passes the model keys as a build secret: they reach the catalog build only,
# never a layer.

FROM pgvector/pgvector:pg17 AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /opt/auto-ontology-eval
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY main.py ./
COPY ontology_sql_eval ./ontology_sql_eval
COPY scripts ./scripts
COPY inputs ./inputs
COPY datasets/bird ./datasets/bird
RUN uv sync --frozen --no-dev

# The catalog's Postgres. PGDATA is outside the base image's VOLUME, so the
# catalog built below stays in the image.
ENV PGDATA=/var/lib/postgresql/catalog \
    POSTGRES_HOST=localhost \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=catalog \
    POSTGRES_DATABASE=auto_ontology
COPY docker/auto_ontology_agent/build-catalog ./

# The catalog, in a stage of its own: `--target runtime` builds everything else
# without the model keys.
FROM runtime AS catalog
RUN --mount=type=secret,id=models,required=true ./build-catalog /run/secrets/models

# The scripts an episode runs, last, so editing them does not rebuild the
# catalog.
COPY docker/auto_ontology_agent/start-catalog docker/auto_ontology_agent/run-episode ./
