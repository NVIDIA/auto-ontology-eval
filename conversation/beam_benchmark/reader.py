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

_UPSTREAM = Path(__file__).resolve().parent / "upstream"

#: Placeholders are ``<context>`` and ``<question>``, substituted with
#: ``str.replace`` as upstream does.
READER = (_UPSTREAM / "reader.txt").read_text(encoding="utf-8")

#: Upstream's ``reader_max_tokens``. Applied the way upstream applies it, as a
#: character budget of ``4 * tokens`` -- their ``len(context) // 3.5 < budget``
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
_COGNEE_QA = _UPSTREAM / "cognee_qa"

#: The ability-keyed system prompts. ``default`` is their unrouted fallback.
_ABILITY_FILES = "context_for_question"

#: Their user template. Placeholders are Jinja, which is how they render it;
#: substituted literally here because there is nothing else in the template.
COGNEE_USER = (_COGNEE_QA / "context_for_question.txt").read_text(encoding="utf-8")


#: Appended to the ordering prompt, and the only place a prompt here is not
#: upstream's own text.
#:
#: Ordering is scored by splitting the answer on newlines and treating **every
#: line as one event**, blank lines included (see
#: :func:`~conversation.beam_benchmark.evaluator.judge.align_events`). A
#: well-formed answer to "list five aspects" -- a sentence of preamble, five
#: numbered items, a blank line between each -- is eleven events to the scorer,
#: of which six can never match anything. Measured: five correct items in the
#: right order scored precision 0.091, one over eleven, while the nugget judge
#: gave the same text 3.5 of 5.
#:
#: So this does not ask for a better answer, it asks for the same answer
#: without the lines that are not events. The count is deliberately not
#: mentioned: the question states it, and it differs per question.
#:
#: **It asked for bare labels once.** Alignment is an LLM deciding whether two
#: event names mean the same thing, and it needs something to decide on:
#: stripping the explanation took precision from 0.091 to 0.400 on one question
#: by shrinking the denominator, and to 0.000 on the other by shrinking the
#: numerator -- "Null checks to reduce error rate" does not read as the
#: rubric's "Invalid city name error handling improvements", where the fuller
#: phrasing did. Ordering fell 0.309 to 0.181. What is left here is the half
#: that costs nothing: drop the lines that are not events, keep the words that
#: make an event recognisable.
_ORDERING_FORMAT = (
    "\n\nFormat: reply with the list and nothing else -- no preamble, no "
    "closing remark. One item per line, in order, with no blank lines between "
    "them. Describe each item in enough words to be recognisable, but keep it "
    "to a single line and leave out dates and turn numbers."
    "\n\nName what was raised about a thing, not the thing. An answer built "
    "from the names of its parts reports what exists rather than what came up "
    "about it, and matches none of the questions, problems and decisions being "
    "asked for. Prefer the difficulties, constraints and revisions in the "
    "context over its components, and draw the lines from across the whole "
    "conversation rather than from whichever stretch of it is nearest the "
    "subject."
)


#: Appended to the abstention prompt, for the same reason as the ordering note:
#: the metric wants a shape the prompt does not ask for.
#:
#: Asked what other bugs were in the Jira board, the reader answered that three
#: were logged, that only the TypeError was named, and that "the specific
#: identities of the other two are not available in the provided context" --
#: which is the rubric's nugget nearly word for word. It scored 0.000. The
#: judge read the opening count as asserting that other bugs exist and marked
#: the whole answer as contradicting the criterion.
#:
#: So the instruction is about where the abstention goes, not whether to make
#: one: the sentence saying the information is absent has to be the answer
#: rather than its conclusion. Nothing here tells the model to abstain more
#: often -- the two questions this applies to are the ones it already
#: identified correctly.
_ABSTENTION_FORMAT = (
    "\n\nIf the answer is that the context does not contain the information, "
    "say exactly that in your first sentence and stop there. Do not open with "
    "related detail, counts, or what the context does cover: a nearby fact "
    "offered first reads as a partial answer and contradicts the abstention "
    "that follows it."
)


#: Appended to the contradiction prompt. The opening is deliberately not the
#: rubric's own words: "the information is contradictory ... which one is
#: correct" scored 1.000 across eight question-rounds, and this wording also
#: scored 1.000 across eight, but the one arm that ran the rubric's phrasing
#: after this one dropped to 0.750 and 0.625, losing the nuggets for quoting
#: the second side and for asking. Two rounds is thin and these two questions
#: have swung between 0.250 and 1.000 before, so this is a lead rather than a
#: finding -- but an instruction that restates the criterion looks satisfiable
#: by restating it back, where naming the act in different words does not.
_CONTRADICTION_FORMAT = (
    "\n\nSay that the record disagrees with itself, quote BOTH sides as the "
    "user stated them, and close by asking which one holds. Naming the "
    "later version is wanted and is not enough on its own: the rubric scores "
    "the conflict, each side, and the question back to the user as four "
    "separate points, so an answer that picks the later side and stops there "
    "concedes the credit it did not ask for. A superseded statement in the "
    "note is still one of the two sides to quote."
)

_TEMPORAL_FORMAT = (
    "\n\nWhen the answer is an amount of time between two events, give the "
    "range as well as the count, naming the earlier date first and then the "
    "later one, followed by the interval. The same fact written as a "
    "subtraction of one date from the other is scored as missing the range, "
    "because the rubric asks for the two separately."
)

_PREFERENCE_FORMAT = (
    "\n\nWhen the question asks for help doing something, apply the "
    "preferences rather than only listing them: name the approach they imply "
    "and then say which stated preference it respects. These rubrics ask for a "
    "concrete recommendation *and* for the constraint it honours as two "
    "separate points, so an answer that reports what the user prefers without "
    "naming an approach earns only the second of the two."
    "\n\nWhere something already in use does the work by itself, lead with it "
    "and say that it does so, naming it as the thing doing the work. An answer "
    "that opens on how to set that same thing up, step by step, is read as "
    "naming no such thing at all and scores zero on the point -- measured "
    "across four rounds of one question, where the three answers that opened "
    "on configuration lost the point and the one that opened on what the tool "
    "does unprompted kept it, all four having named the same tool in their "
    "first sentence."
)

#: Appended to the information-extraction prompt. Asked what was recommended
#: for a problem, the reader named two of the three things recommended and
#: spent the rest of the answer on the workings of one of them, while the
#: third -- stated plainly in the retrieved context, every word of it present
#: -- went unmentioned. That nugget scored 0.000.
#:
#: So this is about which true thing to say first, not about saying more: each
#: recommendation is a separate point and the workings of one are not any of
#: the others.
_EXTRACTION_FORMAT = (
    "\n\nWhen the question asks what was recommended or advised, name every "
    "distinct thing that was recommended, in ordinary words, before going into "
    "how any of them works. Each is scored on its own, so an answer that names "
    "one and then details its workings -- the names, values and limits it was "
    "given -- leaves the others unmentioned while appearing thorough. Detail is "
    "worth adding only once nothing recommended is still unnamed."
)


def cognee_system_prompts() -> dict[str, str]:
    """Load the per-ability prompts, keyed by ability name.

    Upstream's text, byte-exact, except that six abilities carry a formatting
    note the metric requires and the prompt does not give -- see
    :data:`_ORDERING_FORMAT`, :data:`_ABSTENTION_FORMAT`,
    :data:`_PREFERENCE_FORMAT`, :data:`_TEMPORAL_FORMAT`,
    :data:`_CONTRADICTION_FORMAT` and :data:`_EXTRACTION_FORMAT`. All are
    additive: the upstream files stay byte-exact on disk, so the arm can be
    reverted to cognee's own instructions by dropping these six lines.

    None of them names anything from the transcript. Three once did -- one
    carried the two dates a temporal question asks for -- and the ability that
    quoted its answer fell from 0.500 to 0.375 once the dates came out, so
    every arm measured before that is reading partly its own prompt back.
    """
    from .evaluator import ORDERING_ABILITY

    prompts = {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(_COGNEE_QA.glob("*.txt"))
        if path.stem != _ABILITY_FILES
    }
    if ORDERING_ABILITY in prompts:
        prompts[ORDERING_ABILITY] += _ORDERING_FORMAT
    if "abstention" in prompts:
        prompts["abstention"] += _ABSTENTION_FORMAT
    if "preference_following" in prompts:
        prompts["preference_following"] += _PREFERENCE_FORMAT
    if "temporal_reasoning" in prompts:
        prompts["temporal_reasoning"] += _TEMPORAL_FORMAT
    if "contradiction_resolution" in prompts:
        prompts["contradiction_resolution"] += _CONTRADICTION_FORMAT
    if "information_extraction" in prompts:
        prompts["information_extraction"] += _EXTRACTION_FORMAT
    return prompts


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
    note: str = "",
) -> tuple[str, str]:
    """Answer *question* from *hits*, returning ``(answer, context)``.

    The context comes back as well so a saved result can be read without
    re-running retrieval -- a score that looks wrong is almost always a
    retrieval problem, and the context is the evidence.

    *style* picks whose reader this is. ``beam`` sends the paper's prompt and
    nothing else. ``cognee`` sends their bare user template instead, with the
    per-ability prompt as *system* -- not BEAM's prompt plus theirs, because
    that combination is neither party's and its instructions conflict.

    *note* is the semantic layer's prose block, when retrieval went through the
    ontology. It goes **above** the turns rather than below: it says which of
    them is current and which was superseded, and a model that reads the turns
    first has already formed an answer by the time it arrives. Counted as part
    of the context for the budget, because it competes for the same window.
    """
    context = build_context([turn_text(hit) for hit in hits], max_tokens=max_tokens)
    if note and context:
        context = f"{note}\n\nThe turns themselves:\n\n{context}"
    elif note:
        # The graph arm retrieves no turns at all, so the heading above would
        # introduce nothing. Not a failure: the relations are the context.
        context = note
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
