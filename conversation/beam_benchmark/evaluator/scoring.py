# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How question scores become ability means and the overall, with no model.

Split out from ``judge.py`` so the parts that decide what a number means can be
tested against hand-computed values, instead of only against whatever a stubbed
judge was told to say.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

#: The ten memory abilities BEAM asks about.
ABILITIES: tuple[str, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def mean(values: Iterable[float]) -> float:
    """Mean of *values*, or 0.0 when there are none."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def ability_means(
    scored: Iterable[tuple[str, str, float]],
) -> dict[str, float]:
    """Per-ability score from ``(conversation_id, ability, score)`` triples.

    Averages within a conversation first, then across conversations. With BEAM's
    two questions per ability
    per conversation that is the same as a flat mean over questions; it differs
    only when a conversation is missing a question, and then this is the one
    that keeps every conversation weighted equally instead of quietly letting
    the fuller transcripts count for more.

    Abilities with no questions are absent from the result rather than present
    as zero. A partial run should look partial.
    """
    by_conversation: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for conversation_id, ability, score in scored:
        by_conversation[ability][conversation_id].append(score)

    return {
        ability: mean(mean(scores) for scores in conversations.values())
        for ability, conversations in by_conversation.items()
    }


def overall(scored: Iterable[tuple[str, str, float]]) -> float:
    """The headline number: the mean of the per-ability means.

    With two questions per ability this equals cognee's flat mean over
    questions. The ten abilities carry very different difficulty, so a run that
    covers only some of them is not comparable to the published figure no
    matter how this is computed. Check the keys of ``ability_means`` before
    quoting it.
    """
    return mean(ability_means(scored).values())
