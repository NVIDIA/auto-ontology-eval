# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAM's judging prompts, as published.

The text lives in ``../upstream/*.txt`` rather than in string literals here.
Several of the lines that carry trailing whitespace end a bullet, and a trailing
space before a newline changes how the text tokenizes -- so a formatter or an
editor quietly stripping it would move the scores by a hair and leave no trace
of why. Text files are not something ``ruff format`` rewrites.

Keeping them as files also keeps the provenance obvious: these are upstream
artifacts under BEAM's MIT licence, copied from

- ``src/prompts.py::unified_llm_judge_base_prompt`` -> ``judge.txt``
- ``src/evaluation/compute_metrics.py::llm_equivalence`` -> ``equivalence_*.txt``

and they are not ours to improve. Anything we would rather word differently
belongs in a prompt of our own, under a different name, scored separately.

The odd indentation in the equivalence prompts and the typo in
"exaplanation" are upstream's. They are reproduced because the comparison to
the published 0.323 only means something if the judge is the published one.
"""

from __future__ import annotations

from pathlib import Path

#: Shared with ``../reader.py``: one directory for every verbatim BEAM prompt,
#: so there is a single place to diff if upstream revises them.
_UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"

#: Scores one rubric nugget against one response. Placeholders are
#: ``<rubric_item>`` and ``<llm_response>``, substituted by ``str.replace`` the
#: way upstream does it -- the prompt contains literal ``{`` and ``}`` in its
#: output-format block, so ``str.format`` is not an option.
JUDGE = (_UPSTREAM / "judge.txt").read_text(encoding="utf-8")

#: Decides whether two event descriptions name the same event. Used only by
#: event ordering, to align a response's events with the rubric's before the
#: ranks are compared.
EQUIVALENCE_SYSTEM = (_UPSTREAM / "equivalence_system.txt").read_text(encoding="utf-8")

#: The user half of the same exchange. Placeholders are ``{first_paragraph}``
#: and ``{second_paragraph}``: this one *was* an f-string upstream, so the
#: braces are substitution points rather than literals.
EQUIVALENCE_USER = (_UPSTREAM / "equivalence_user.txt").read_text(encoding="utf-8")

#: Cognee's edit of :data:`JUDGE`, from
#: ``cognee/eval_framework/beam/eval/metrics/beam_rubric.py``. Not ours and not
#: BEAM's -- a third prompt, reproduced so their published 0.79 has an
#: instrument on our side.
#:
#: The difference is one line. BEAM's prompt tells the judge "if the response
#: does not address the QUESTION, score 0.0 and stop" but never supplies a
#: question; its ``<question>`` placeholder is left unsubstituted by
#: ``src/prompts.py``. Cognee adds ``- QUESTION (what the user asked):`` to the
#: inputs block and fills it, and re-words the responsiveness section to be
#: "anchored to the QUESTION". Their source comment calls it a fix, which it is:
#: a judge asked to check on-topicness against nothing has to guess, and guesses
#: inconsistently -- we measured the same abstention answer scoring 0.0 and 1.0
#: in consecutive rounds on a one-word paraphrase.
COGNEE_JUDGE = (_UPSTREAM / "cognee_judge.txt").read_text(encoding="utf-8")

#: The system message Cognee pairs with it. BEAM sends none.
COGNEE_JUDGE_SYSTEM = "You are an evaluation judge. Return only valid JSON."


def judge_prompt(*, nugget: str, response: str) -> str:
    """Fill in the nugget-scoring prompt."""
    return JUDGE.replace("<rubric_item>", nugget).replace("<llm_response>", response)


def cognee_judge_prompt(*, question: str, nugget: str, response: str) -> str:
    """Fill in Cognee's variant, which also gets the question."""
    return (
        COGNEE_JUDGE.replace("<question>", question)
        .replace("<rubric_item>", nugget)
        .replace("<llm_response>", response)
    )


def equivalence_prompt(*, first: str, second: str) -> str:
    """Fill in the user half of the equivalence check."""
    return EQUIVALENCE_USER.format(first_paragraph=first, second_paragraph=second)
