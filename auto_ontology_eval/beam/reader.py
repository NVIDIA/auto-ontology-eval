# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer one BEAM question from retrieved turns, with cognee's reader prompts.

Used by the ``vector`` arm only; the agent arm is its own reader.

Context assembly follows BEAM's upstream RAG baseline, including the budget. Hits are concatenated
in the order given until the next one would not fit, and once one is skipped the
rest still get their turn -- so a single enormous turn does not end the context,
it is just stepped over. That is a slightly odd rule, but it is the published
one, and it is the reason a run is not sensitive to one pathological turn.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

#: Under ``cognee/`` with the rest of their material, kept apart from our own
#: code so what may not be edited is obvious from the path.
_ANSWER_PROMPTS = Path(__file__).resolve().parent / "cognee" / "vendor" / "answer"

#: Upstream's ``reader_max_tokens``. Applied the way upstream applies it, as a
#: character budget of ``3.5 * tokens`` -- their ``len(context) // 3.5 < budget``
#: is a character count divided by an assumed characters-per-token, not a
#: tokenizer. Reproduced rather than replaced with a real tokenizer so the
#: amount of context a reader sees matches the published runs.
READER_MAX_TOKENS = 29_000

#: Characters per token in that estimate.
_CHARS_PER_TOKEN = 3.5


#: Cognee's reader, byte-exact. Their per-ability prompts come from
#: ``beam/report_artifacts/qa_prompts/`` and the user template from
#: ``infrastructure/llm/prompts/context_for_question.txt``.
#:
#: Their retriever has two slots: the per-ability file is the system prompt,
#: in place of ``answer_simple_question.txt``, and the template is the user
#: message.
#:
#: Selecting the prompt by ability reads the benchmark's own label at inference
#: time, which their report is explicit about (§5: outside a labelled benchmark
#: the type "would need to be inferred from the question itself").
_COGNEE_QA = _ANSWER_PROMPTS

#: The ability-keyed system prompts. ``default`` is their unrouted fallback.
_ABILITY_FILES = "context_for_question"

#: Their user template. Placeholders are Jinja, which is how they render it;
#: substituted literally here because there is nothing else in the template.
COGNEE_USER = (_COGNEE_QA / "context_for_question.txt").read_text(encoding="utf-8")


def cognee_system_prompts() -> dict[str, str]:
    """Load the per-ability prompts, keyed by ability name.

    Upstream's text, byte-exact, with nothing appended. Anything added here
    would have to be tuned on some questions, and BEAM's conversations are also
    its test set: a note written against a rubric item makes the score on that
    conversation a measure of the note.
    """
    return {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(_COGNEE_QA.glob("*.txt"))
        if path.stem != _ABILITY_FILES
    }


class Reader(Protocol):
    """Send one prompt, optionally under a system prompt."""

    def __call__(self, prompt: str, *, system: str | None = None) -> str: ...


def make_reader(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> Reader:
    """Build a :class:`Reader` backed by a real endpoint.

    The budget is generous for the same reason as the judge's: a reasoning
    model spends it on hidden reasoning tokens before emitting any answer, and
    returns an empty string rather than an error when it runs out. An empty
    answer scores 0 on every nugget, so it would read as a retrieval failure.

    Defaults to ``REASONING_MODEL``. Whichever is used is recorded next to the
    score, because the reader is part of the measurement.
    """
    from auto_ontology.utils.llm_invoke import get_llm_client
    from langchain_core.messages import HumanMessage, SystemMessage

    from .cognee.judge import with_retry

    client = get_llm_client(model=model, temperature=temperature, max_tokens=max_tokens)

    def read(prompt: str, *, system: str | None = None) -> str:
        messages: list[object] = [HumanMessage(content=prompt)]
        if system is not None:
            messages.insert(0, SystemMessage(content=system))
        return str(with_retry(lambda: client.invoke(messages)).content)

    return read


def resolve_model(model: str | None = None) -> str:
    """The model id :func:`make_reader` will actually use.

    Recorded next to a score rather than the name of the variable it came from:
    ``REASONING_MODEL`` is reassigned often enough that a months-old result file
    saying only "REASONING_MODEL" cannot be compared to anything.
    """
    return model or os.environ.get("REASONING_MODEL") or "<REASONING_MODEL unset>"


def build_context(chunks: Sequence[str], *, max_tokens: int = READER_MAX_TOKENS) -> str:
    """Concatenate *chunks* up to the reader's character budget.

    Returns the joined text. Chunks that would overflow are skipped rather than
    truncated: half a turn in the context is worse than none, since the reader
    is told to answer only from what it is given and cannot tell that a sentence
    was cut.
    """
    budget = max_tokens * _CHARS_PER_TOKEN
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) < budget:
            context += chunk
    return context


def turn_text(hit: object) -> str:
    """Render one retrieved turn the way upstream renders a pair.

    ``USER:`` / ``ASSISTANT:`` rather than our ingest-time header. The reader is
    answering, not searching, so the ordinal metadata that helps the embedding
    find a turn is noise here -- except the session and turn numbers, which some
    abilities (event ordering, temporal reasoning) genuinely need to answer at
    all. Those are kept on a line of their own so they cannot be mistaken for
    something the user said.
    """
    parts = [f"[turn {hit.turn_index}"]  # type: ignore[attr-defined]
    if hit.session_index is not None:  # type: ignore[attr-defined]
        parts.append(f", session {hit.session_index}")  # type: ignore[attr-defined]
    if hit.time_anchor:  # type: ignore[attr-defined]
        parts.append(f", {hit.time_anchor}")  # type: ignore[attr-defined]
    parts.append("]")
    header = "".join(parts)
    return (
        f"{header}\n"
        f"USER: {hit.question}\n\n"  # type: ignore[attr-defined]
        f"ASSISTANT: {hit.answer}\n\n"  # type: ignore[attr-defined]
    )


def answer(
    question: str,
    hits: Sequence[object],
    read: Reader,
    *,
    max_tokens: int = READER_MAX_TOKENS,
    system: str | None = None,
) -> tuple[str, str]:
    """Answer *question* from *hits*, returning ``(answer, context)``.

    The context comes back as well so a saved result can be read without
    re-running retrieval -- a score that looks wrong is almost always a
    retrieval problem, and the context is the evidence.

    The user message is cognee's template; *system* is the per-ability prompt
    from :func:`cognee_system_prompts`.
    """
    context = build_context([turn_text(hit) for hit in hits], max_tokens=max_tokens)
    if not context:
        logger.warning("no context retrieved for: %s", question[:80])
    prompt = COGNEE_USER.replace("{{ question }}", question).replace(
        "{{ context }}", context
    )
    return read(prompt, system=system).strip(), context
