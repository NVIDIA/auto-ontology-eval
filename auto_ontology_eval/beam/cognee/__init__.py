# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything that makes our BEAM number comparable to cognee's published one.

``judge`` holds the model-facing half of their rubric judge, ``scoring`` the
arithmetic, ``prompts`` the loader. See the module docstrings for which
upstream quirks are reproduced on purpose.

``vendor/`` is upstream's own material, byte-for-byte and not ours to edit: the
judging prompt, the per-ability answer prompts :mod:`..reader` is driven by, and
the published results :mod:`..reference` joins onto our rows.
``tests/conversation/test_vendored_prompts.py`` pins each file by sha256,
because a prompt edited by accident would move our score while still looking
like a comparison.

The judge code sits here rather than in a directory of its own so that it is
beside the prompt it sends -- the two are one instrument, and splitting them
across trees only made it look like there were two. The copyright boundary is
``vendor/``, one level down, so that it can still be stated as a directory.
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
