# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAM's rubric judge, ported from the published implementation.

``judge`` holds the model-facing half, ``scoring`` the arithmetic. See the
module docstrings for which upstream quirks are reproduced on purpose, and
``../README.md`` for what the resulting numbers are comparable to.
"""

from .judge import (
    Ask,
    JudgeError,
    QuestionResult,
    Verdict,
    align_events,
    cognee_judge_nugget,
    events_equivalent,
    judge_nugget,
    make_ask,
    parse_verdict,
    score_question,
    snap,
    with_retry,
)
from .scoring import (
    ABILITIES,
    ORDERING_ABILITY,
    ability_means,
    kendall_tau_b,
    ordering_scores,
    overall,
)

__all__ = [
    "ABILITIES",
    "ORDERING_ABILITY",
    "Ask",
    "JudgeError",
    "QuestionResult",
    "Verdict",
    "ability_means",
    "align_events",
    "cognee_judge_nugget",
    "events_equivalent",
    "judge_nugget",
    "kendall_tau_b",
    "make_ask",
    "ordering_scores",
    "overall",
    "parse_verdict",
    "score_question",
    "snap",
    "with_retry",
]
