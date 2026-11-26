# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognee's BEAM judging prompt, as published.

The text lives in ``vendor/judge/cognee_judge.txt`` rather than in a string
literal here, so a formatter or an editor cannot quietly rewrite it: a change
to the judge's wording moves the scores and leaves no trace of why. It is an
upstream artifact, from ``cognee/eval_framework/beam/eval/metrics/beam_rubric.py``,
and not ours to improve.
"""

from __future__ import annotations

from pathlib import Path

#: Every judging prompt, in one place to diff if upstream revises it. The
#: reader's prompts are beside them, in ``vendor/answer/``: everything under
#: ``vendor/`` is upstream's and byte-for-byte, while this module reading it is
#: ours. That is the whole reason for the extra level -- the licence applies to
#: a directory, and a directory it shares with our code cannot say so cleanly.
_JUDGE_PROMPTS = Path(__file__).resolve().parent / "vendor" / "judge"

#: Scores one rubric nugget against one response, given the question.
#: Upstream writes the placeholders as ``str.format`` fields; they are written
#: here as ``<question>``, ``<rubric_item>`` and ``<llm_response>`` and filled
#: by ``str.replace``, because the prompt's output-format block has literal
#: braces. The rendered prompt is otherwise identical to theirs.
COGNEE_JUDGE = (_JUDGE_PROMPTS / "cognee_judge.txt").read_text(encoding="utf-8")

#: The system message Cognee pairs with it.
COGNEE_JUDGE_SYSTEM = "You are an evaluation judge. Return only valid JSON."


def cognee_judge_prompt(*, question: str, nugget: str, response: str) -> str:
    """Fill in the judge prompt."""
    return (
        COGNEE_JUDGE.replace("<question>", question)
        .replace("<rubric_item>", nugget)
        .replace("<llm_response>", response)
    )
