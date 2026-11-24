# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognee's BEAM rubric judge, ported from their published implementation.

``judge`` holds the model-facing half, ``scoring`` the arithmetic. See the
module docstrings for which upstream quirks are reproduced on purpose.
"""

from .judge import (
    Ask,
    JudgeError,
    QuestionResult,
    Verdict,
    cognee_judge_nugget,
    make_ask,
    parse_verdict,
    score_question,
    snap,
    with_retry,
)
from .scoring import (
    ABILITIES,
    ability_means,
    overall,
)

__all__ = [
    "ABILITIES",
    "Ask",
    "JudgeError",
    "QuestionResult",
    "Verdict",
    "ability_means",
    "cognee_judge_nugget",
    "make_ask",
    "overall",
    "parse_verdict",
    "score_question",
    "snap",
    "with_retry",
]
