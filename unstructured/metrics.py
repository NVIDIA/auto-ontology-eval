# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval metrics for the ViDoRe benchmark.

All functions work on unordered or ordered lists and return values in [0, 1].
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved."""
    if not relevant:
        return 1.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / k


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance)."""
    if not relevant:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(retrieved[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_query: list[dict]) -> dict:
    """Aggregate per-query metric dicts into a summary."""
    if not per_query:
        return {}
    keys = [k for k in per_query[0] if k != "query_id"]
    return {k: mean([q[k] for q in per_query]) for k in keys}
