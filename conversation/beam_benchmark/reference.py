# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognee's own answers, so a result row can carry theirs beside ours.

They publish per-question artifacts for the run behind the 0.790 headline --
the answer text, their judge's score and its reason, for each of four runs. With
those joined onto our rows, a question we lose is readable without leaving the
file: what we said, what they said, and what the rubric wanted.

Scope, and it is narrow: **one conversation**. At 100K they published
conversation 2 and nothing else, so joining this onto a run over any other
conversation attaches nothing. That is also the conversation every arm in the
README was measured on, which is why it is worth having.

``upstream/cognee_report/cognee_answers_conv2.json`` is distilled from their
four ``beam_existing_ingestion_metrics_conv1_*.json.gz`` artifacts by
``tools/extract_cognee_answers.py``. Distilled rather than vendored whole
because each artifact is 2MB unpacked, almost all of it ``retrieval_context``:
85,000 to 109,000 characters of it per question, which is interesting exactly
once and is left behind the link.

The distillation is checked rather than trusted. ``tools/extract_cognee_answers.py``
recomputes their published per-ability means, overall and tau from the rows it
keeps and refuses to write a file that does not reproduce all twelve figures.
"""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UPSTREAM = Path(__file__).parent / "upstream" / "cognee_report"
REFERENCE = UPSTREAM / "cognee_answers_conv2.json"
SUMMARY = UPSTREAM / "hybrid_completion_20_20_qa_v1_cross_run_summary.json"

#: Their artifacts are per-conversation and this is the only one at 100K.
COVERED_CONVERSATION = "2"

#: Their headline at 100K, and the number every arm here is chasing.
COGNEE_OVERALL = 0.7943749999999999

#: Their event-ordering rank correlation, which they report beside the rubric
#: score rather than as it. See the README's scoring section.
COGNEE_TAU = 0.514638501094228


@cache
def cognee_ability_means() -> dict[str, float]:
    """Their published per-ability rubric means, or empty if the file is gone.

    Read from their cross-run summary rather than recomputed from the answers,
    so the comparison column is their arithmetic and not ours.
    """
    if not SUMMARY.exists():
        return {}
    report = json.loads(SUMMARY.read_text())
    return {
        ability: stats["beam_rubric"]["mean"]
        for ability, stats in report["by_question_type"].items()
    }


@cache
def cognee_answers() -> dict[str, dict[str, Any]]:
    """Their answers keyed by question text, or empty if the file is missing.

    Keyed by the question rather than by index: ours are 1-based over the whole
    400-question file and theirs are 0-based within one conversation, so the
    two indices do not line up and matching on them would silently pair the
    wrong rows. The question strings are byte-identical across all twenty.

    Empty rather than raising, so a run without the reference file still
    produces results -- the columns are a convenience, not a dependency.
    """
    if not REFERENCE.exists():
        logger.warning("No cognee reference at %s; rows will omit it.", REFERENCE)
        return {}
    return json.loads(REFERENCE.read_text())["questions"]


def cognee_columns(
    *, conversation_id: str, question: str, run: int = 0
) -> dict[str, Any]:
    """Their answer and score for one question, as columns to merge into a row.

    *run* picks which of their four runs to show the answer from, paired with
    our round. Nineteen of their twenty questions got a different answer each
    run and five got a different score, so there is no single answer of theirs
    to show; pairing by index at least shows four distinct ones across four
    rounds instead of the same one four times.

    ``cognee_score`` is that run's. ``cognee_score_mean`` is all four, which is
    the number their published table is made of and the one to compare a pooled
    result against.

    Returns ``{}`` for a question they did not publish, which is every question
    of every conversation but one.
    """
    entry = cognee_answers().get(question.strip())
    if entry is None or str(conversation_id) != entry["conversation_id"]:
        return {}

    runs = entry["runs"]
    theirs = runs[run % len(runs)]
    return {
        "cognee_answer": theirs["answer"],
        "cognee_score": theirs["score"],
        "cognee_reason": theirs["reason"],
        "cognee_score_mean": sum(r["score"] for r in runs) / len(runs),
        "cognee_run": theirs["run"],
        "cognee_context_chars": theirs["context_chars"],
        "golden_answer": entry["golden_answer"],
    }
