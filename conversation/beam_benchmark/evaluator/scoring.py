# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arithmetic BEAM's scores are made of, with no model in the loop.

Split out from ``judge.py`` so the parts that decide what a number means can be
tested against hand-computed values, instead of only against whatever a stubbed
judge was told to say.

Nothing here is novel; it reproduces ``src/evaluation/compute_metrics.py`` and
``src/evaluation/report_results.py``. Where the two disagree with the paper, the
code wins, because the code is what produced the published table.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

#: The ten memory abilities, in the order ``report_results.py`` lists them.
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

#: The one ability whose score depends on whose implementation you follow.
#: BEAM's own ``compute_metrics`` scores it by rank correlation; cognee scores
#: it by nuggets like the rest and reports rank correlation separately. See
#: ``ordering_scores`` and ``judge.score_question``.
ORDERING_ABILITY = "event_ordering"


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b between two equal-length rank vectors.

    Tau-b rather than tau-a because the rank vectors are full of ties: every
    event missing from a sequence is parked on the same sentinel rank, so a
    coefficient that ignores ties would be dominated by them.

    Returns 0.0 when either vector is constant, where the coefficient is
    undefined -- ``scipy.stats.kendalltau`` returns NaN there, and upstream
    adds that straight into a running total, so one such question would turn a
    whole ability's score into NaN. It cannot arise from BEAM's own data (the
    smallest event-ordering rubric has three distinct events), so this is a
    guard rather than a behaviour difference.
    """
    if len(x) != len(y):
        raise ValueError(f"rank vectors differ in length: {len(x)} vs {len(y)}")

    n = len(x)
    concordant_minus_discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            # A pair tied in both counts towards *both* denominators, not
            # neither: tau-b's n1 and n2 are every tied pair in x and in y, and
            # the two sets overlap.
            if dx == 0:
                ties_x += 1
            if dy == 0:
                ties_y += 1
            if dx and dy:
                concordant_minus_discordant += 1 if (dx > 0) == (dy > 0) else -1

    pairs = n * (n - 1) // 2
    denominator = (pairs - ties_x) * (pairs - ties_y)
    if denominator <= 0:
        return 0.0
    return concordant_minus_discordant / math.sqrt(denominator)


def _ranks(sequence: Sequence[str], universe: Sequence[str]) -> list[int]:
    """Position of each *universe* item in *sequence*, 1-based.

    Items absent from *sequence* all share one sentinel rank past the end, which
    is what makes this a tie-heavy vector.
    """
    absent = len(universe) + 1
    position = {item: index + 1 for index, item in enumerate(sequence)}
    return [position.get(item, absent) for item in universe]


def ordering_scores(
    reference: Sequence[str], system: Sequence[str]
) -> dict[str, float]:
    """Score one event-ordering answer, given *system* already aligned.

    *system* must have been put into the rubric's vocabulary first -- see
    ``judge.align_events`` -- otherwise set intersection finds nothing and
    everything here reads zero.

    The reported number is ``tau_norm``, not ``final_score``. The paper
    describes event ordering as Kendall tau-b and ``compute_metrics`` computes
    ``final_score = tau_norm * f1``, but ``report_results.py`` aggregates bare
    ``tau_norm``, so that is what the published table contains. Both are
    returned; a caller that wants the paper's prose can use ``final_score``, and
    get a different, lower number.

    None of them is what cognee's headline figure contains. Everything here is
    still computed under ``flavour="cognee"`` and reported as ``kendall_tau``
    beside the rubric score, which is how their registry treats it; it is just
    not the question's score. See the module docstring of ``judge``.

    ``tau_norm`` maps tau-b from [-1, 1] onto [0, 1], so 0.5 means no rank
    correlation at all. Published event-ordering scores sit at 0.15-0.27, i.e.
    *below* chance: the sentinel rank puts everything a response failed to
    mention at the bottom of one vector and everything it invented at the bottom
    of the other, which reads as anti-correlation. Scoring near 0.5 on this
    metric is already an improvement.
    """
    reference = list(reference)
    system = list(system)

    matched = set(reference) & set(system)
    true_positive = len(matched)
    false_positive = len([e for e in system if e not in reference])
    false_negative = len([e for e in reference if e not in system])

    precision = true_positive / (true_positive + false_positive) if system else 0.0
    recall = true_positive / (true_positive + false_negative) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    universe = list(dict.fromkeys(reference + system))
    tau_b = kendall_tau_b(_ranks(reference, universe), _ranks(system, universe))
    tau_norm = (tau_b + 1) / 2

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tau_norm": tau_norm,
        "final_score": tau_norm * f1,
    }


def mean(values: Iterable[float]) -> float:
    """Mean of *values*, or 0.0 when there are none."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def ability_means(
    scored: Iterable[tuple[str, str, float]],
) -> dict[str, float]:
    """Per-ability score from ``(conversation_id, ability, score)`` triples.

    Averages within a conversation first, then across conversations, which is
    the order ``report_results.py`` uses. With BEAM's two questions per ability
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

    Averaging abilities rather than questions is deliberate upstream. The ten
    abilities carry very different difficulty -- abstention runs near 0.8 while
    contradiction resolution is near 0.03 -- so a run that covers only some of
    them is not comparable to the published figure no matter how this is
    computed. Check the keys of ``ability_means`` before quoting it.
    """
    return mean(ability_means(scored).values())
