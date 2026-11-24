# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer one BEAM question from retrieved turns.

The prompt is upstream's ``answer_generation_for_rag``, verbatim (see
``evaluator/prompts.py`` for why it is a text file). It is strict about using
only the context and nothing the model already knows, which is the whole point:
a question BEAM asks about a transcript must be answered from that transcript,
or the score measures the reader's world knowledge instead of our retrieval.

Context assembly is upstream's too, including the budget. Hits are concatenated
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

_ANSWER_PROMPTS = Path(__file__).resolve().parent / "prompts" / "answer"

#: Placeholders are ``<context>`` and ``<question>``, substituted with
#: ``str.replace`` as upstream does.
READER = (_ANSWER_PROMPTS / "beam_answer.txt").read_text(encoding="utf-8")

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
#: Theirs *replaces* BEAM's prompt rather than wrapping it: their retriever has
#: its own two slots, and the per-ability file is dropped into the system slot
#: in place of ``answer_simple_question.txt``. BEAM's reader prompt appears
#: nowhere in their pipeline. Sending both is not a closer approximation of
#: their setup, it is a third thing -- "be direct and concise" and "be thorough
#: and specific, do not omit any significant topic" are contradictory
#: instructions, and GPT-5 resolved them towards the first.
#:
#: Selecting the prompt by ability reads the benchmark's own label at inference
#: time, which their report is explicit about (§5: outside a labelled benchmark
#: the type "would need to be inferred from the question itself").
_COGNEE_QA = _ANSWER_PROMPTS / "cognee"

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

    Defaults to ``REASONING_MODEL``. The paper's own readers were
    Llama-4-Maverick, GPT-4.1-nano, Gemini-2.0-flash and Qwen2.5-32B; matching
    one of them is not required for our comparison, but whichever is used has to
    be stated next to the score, because the reader moved the published figure
    by more than the retrieval method did (0.269 to 0.323 across models on the
    same RAG setup).
    """
    from auto_ontology.utils.llm_invoke import get_llm_client
    from langchain_core.messages import HumanMessage, SystemMessage

    from .evaluator.judge import with_retry

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
    style: str = "beam",
) -> tuple[str, str]:
    """Answer *question* from *hits*, returning ``(answer, context)``.

    The context comes back as well so a saved result can be read without
    re-running retrieval -- a score that looks wrong is almost always a
    retrieval problem, and the context is the evidence.

    *style* picks whose reader this is. ``beam`` sends the paper's prompt and
    nothing else. ``cognee`` sends their bare user template instead, with the
    per-ability prompt as *system* -- not BEAM's prompt plus theirs, because
    that combination is neither party's and its instructions conflict.

    """
    context = build_context([turn_text(hit) for hit in hits], max_tokens=max_tokens)
    if not context:
        logger.warning("no context retrieved for: %s", question[:80])
    if style == "cognee":
        prompt = COGNEE_USER.replace("{{ question }}", question).replace(
            "{{ context }}", context
        )
    elif style == "beam":
        prompt = READER.replace("<context>", context).replace("<question>", question)
    else:
        raise ValueError(f"unknown reader style {style!r}")
    return read(prompt, system=system).strip(), context
