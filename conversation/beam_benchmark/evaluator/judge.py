# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEAM's LLM judge, ported.

Scores one answer at a time against its rubric, the way the paper does: one
model call per nugget, each returning 0.0, 0.5 or 1.0, averaged into the
question's score.

Event ordering is the one ability the two reference implementations disagree
about, and ``flavour`` picks between them:

``flavour="beam"``
    The question's score is a normalized Kendall tau-b over events aligned by
    an LLM equivalence check (see ``scoring.ordering_scores``). Its nugget
    verdicts are computed and kept but do not count.
``flavour="cognee"``
    The question's score is the nugget mean, exactly as for the other nine
    abilities, and tau is reported beside it as a second metric. Their
    ``beam_rubric.py`` has no tau in it and no special case for this ability;
    ``kendall_tau.py`` is a separate metric in the same registry.

The difference is not cosmetic. Scored as tau, our event ordering reads 0.188
against cognee's published 0.787 -- but their 0.787 is a nugget mean, and their
tau for the same answers is 0.515. Compared like against like our nugget mean
is 0.838. Mixing the two understates a run by roughly 0.06 overall, which on a
0.790 target is most of the distance that looks like it is missing.

The point of porting it rather than writing our own is comparability. The
published best baseline at 100K is 0.323; a number of ours only sits next to it
if it came out of the same judge, so the prompts in ``upstream/`` are verbatim
and the quirks below are reproduced rather than fixed.

Known quirks, all upstream's:

- **The judge never sees the question.** The prompt's responsiveness rules talk
  about "the QUESTION" and instruct a 0.0 when the response "does not address"
  it, but the prompt has no placeholder for it and ``compute_metrics`` never
  passes one. The judge decides responsiveness from the nugget alone. This bites
  abstention hardest, where the whole task is recognising that nothing was
  asked that the transcript can answer.
- **Blank lines in an event-ordering answer count against it.** The answer is
  split on newlines with no filtering, so a blank line between paragraphs
  becomes an event that matches no rubric item and lands as a false positive.
- **Nuggets are judged independently and in isolation.** There is no
  cross-nugget consistency check, so a response can be scored 1.0 on two
  nuggets that contradict each other.

Judging is model calls, so it is the expensive half of a run. At 100K the
rubrics hold 1,051 nuggets across 400 questions, plus up to ``len(rubric)`` x
``len(answer lines)`` equivalence calls for each of the 40 event-ordering
questions.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .prompts import (
    COGNEE_JUDGE_SYSTEM,
    EQUIVALENCE_SYSTEM,
    cognee_judge_prompt,
    equivalence_prompt,
    judge_prompt,
)
from .scoring import ORDERING_ABILITY, mean, ordering_scores

logger = logging.getLogger(__name__)

#: What the paper judged with, under the name ``REASONING_ENDPOINT`` serves it
#: by -- upstream's bare ``gpt-4.1-mini`` is not a routable id here, and the
#: ``azure/`` prefix is also what sends it through ``ChatOpenAI`` rather than
#: ``ChatNVIDIA``, which matters for a judge whose whole output is JSON.
#:
#: Overridable via ``BEAM_JUDGE_MODEL``, but a run judged by anything else is
#: not on the same instrument as the published 0.323 and should not be reported
#: beside it as though it were.
DEFAULT_JUDGE_MODEL = "azure/openai/gpt-4.1-mini"

#: Upstream admits only these. A model that answers 0.75 is answering a question
#: the prompt did not ask, but it is still inside the scale, so it is taken at
#: face value the way upstream's bare ``float(...)`` takes it. Anything outside
#: [0, 1] is treated as a parse failure instead.
_VALID_RANGE = (0.0, 1.0)

_FENCE = re.compile(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", re.DOTALL)
_OBJECT = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)
_SCORE = re.compile(r'"score"\s*:\s*"?([0-9]*\.?[0-9]+)', re.IGNORECASE)


class JudgeError(RuntimeError):
    """A nugget could not be scored.

    Raised rather than scored 0.0 on purpose. An unparseable judge reply is a
    missing measurement, and folding it in as "no compliance" biases a run
    downwards in exactly the place nobody looks.
    """


class Ask(Protocol):
    """Send one prompt, get the raw reply text.

    Narrow by design: tests pass a plain function, and nothing in this module
    needs streaming, tools, or message history.
    """

    def __call__(self, prompt: str, *, system: str | None = None) -> str: ...


@dataclass(frozen=True)
class Verdict:
    """One nugget's result."""

    nugget: str
    score: float
    reason: str


@dataclass(frozen=True)
class QuestionResult:
    """One question's result.

    *score* is the number the published table averages: the mean of *verdicts*,
    except for event ordering under ``flavour="beam"``, where it is
    ``ordering["tau_norm"]``.

    *ordering* is populated for every event-ordering question whichever flavour
    asked for it, and *verdicts* likewise. Both numbers always exist; the
    flavour decides only which one is called the score. That is what makes a
    finished run re-readable against either published table without judging it
    again.
    """

    conversation_id: str
    ability: str
    question: str
    response: str
    score: float
    verdicts: tuple[Verdict, ...]
    ordering: dict[str, float] | None = None

    def as_row(self) -> tuple[str, str, float]:
        """The triple ``scoring.ability_means`` and ``scoring.overall`` take."""
        return (self.conversation_id, self.ability, self.score)


def make_ask(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> Ask:
    """Build an :class:`Ask` backed by a real endpoint.

    Defaults to ``BEAM_JUDGE_MODEL`` then :data:`DEFAULT_JUDGE_MODEL`, at
    temperature 0 as the paper ran it. Credentials come from the shared
    ``REASONING_*`` variables via Auto Ontology's client builder, so
    ``ontology_sql_eval.env.load_env()`` has to have run first.

    The budget is far above what a verdict needs (a score and a sentence, well
    under 200 tokens) because a reasoning judge spends the same allowance on
    hidden reasoning tokens first and returns an empty string when it runs out
    -- which arrives as an unparseable reply, not as an error. ``gpt-5`` did
    exactly that on roughly 1 in 20 questions at 1024.

    Imported lazily: the test suite stubs ``auto_ontology`` out entirely, and
    the scoring half of this module has no use for it.
    """
    from auto_ontology.utils.llm_invoke import get_llm_client
    from langchain_core.messages import HumanMessage, SystemMessage

    name = model or os.environ.get("BEAM_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    client = get_llm_client(model=name, temperature=temperature, max_tokens=max_tokens)

    def ask(prompt: str, *, system: str | None = None) -> str:
        messages = [HumanMessage(content=prompt)]
        if system is not None:
            messages.insert(0, SystemMessage(content=system))
        return str(with_retry(lambda: client.invoke(messages)).content)

    return ask


#: Attempts per model call, and the base for exponential backoff between them.
TRANSPORT_ATTEMPTS = 4
_BACKOFF_SECONDS = 2.0


def with_retry(call: Callable[[], Any]) -> Any:
    """Retry *call* through transient endpoint failures.

    At the transport layer rather than around a verdict, because a dropped
    connection is not a judgement and must not become one. That distinction is
    load-bearing for the ``cognee`` flavour, which by design turns any exception
    into a real 0.0: without this, an endpoint blip is indistinguishable from a
    response that failed its rubric, and a long run quietly accumulates zeros.
    We measured 17 such zeros in one 20-question round during an outage.

    Only connection-shaped failures are retried. A refusal, a context-length
    error or a bad model name will fail identically four times, so retrying
    them just delays the report.
    """
    delay = _BACKOFF_SECONDS
    for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            if not _is_transient(exc) or attempt == TRANSPORT_ATTEMPTS:
                raise
            logger.warning(
                "endpoint error (attempt %d/%d), retrying in %.0fs: %s",
                attempt,
                TRANSPORT_ATTEMPTS,
                delay,
                exc,
            )
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def _is_transient(exc: BaseException) -> bool:
    """Whether *exc* is worth another attempt.

    Matched on the exception's text rather than its type: the client wraps
    httpx, httpcore and openai errors in several layers and which one surfaces
    depends on where the failure happened, so the class is not a reliable
    signal while the message consistently is.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT)


_TRANSIENT = (
    "connection error",
    "connection reset",
    "connection aborted",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "internal server error",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    # Capacity exhaustion inside the endpoint's own connector, which it returns
    # as **401 AuthenticationError** with this in the message. Matched on the
    # text rather than the status for that reason: a 401 is otherwise exactly
    # the error that must not be retried, and classifying all of them as
    # transient would turn a wrong key into sixteen wrong-key attempts.
    "too many clients",
    "too many connections",
)


def parse_verdict(text: str) -> tuple[float, str]:
    """Pull ``(score, reason)`` out of a judge reply.

    Three fallbacks, loosening in order: the reply as JSON, the first JSON
    object embedded in it, then a bare regex for the score field. Upstream
    reaches for ``json_repair`` at the second step; the regex covers the same
    ground for a two-field object without taking on the dependency, and a reply
    mangled past that point is better raised than guessed at.
    """
    stripped = text.strip()
    fenced = _FENCE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()

    for candidate in (stripped, *(m.group(1) for m in _OBJECT.finditer(stripped))):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "score" in parsed:
            return _validate(parsed["score"]), str(parsed.get("reason", ""))

    scraped = _SCORE.search(stripped)
    if scraped:
        return _validate(scraped.group(1)), ""

    raise JudgeError(f"no score in judge reply: {text[:200]!r}")


def _validate(raw: object) -> float:
    """Coerce a reported score, rejecting anything off the scale."""
    try:
        score = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise JudgeError(f"non-numeric score {raw!r}") from exc
    if not _VALID_RANGE[0] <= score <= _VALID_RANGE[1]:
        raise JudgeError(f"score {score} outside {_VALID_RANGE}")
    return score


def judge_nugget(nugget: str, response: str, ask: Ask, *, attempts: int = 3) -> Verdict:
    """Score one nugget against one response.

    Retried on a parse failure because at temperature 0 a malformed reply is
    almost always a formatting blip rather than a judgement, and a retry is far
    cheaper than discarding the question. Retried on nothing else: a refusal or
    a timeout is the endpoint's problem to surface.
    """
    prompt = judge_prompt(nugget=nugget, response=response)
    last: JudgeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            score, reason = parse_verdict(ask(prompt))
        except JudgeError as exc:
            last = exc
            logger.warning("judge reply unparseable (attempt %d): %s", attempt, exc)
            continue
        return Verdict(nugget=nugget, score=score, reason=reason)

    assert last is not None
    raise last


def snap(score: float) -> float:
    """Collapse a reported score onto Cognee's three levels.

    Theirs rounds to the nearest of 0.0/0.5/1.0 with cuts at 0.25 and 0.75.
    BEAM does not snap at all, and the prompt asks for one of the three
    anyway, so this only bites when a judge invents 0.8 or 0.3 -- rare, but it
    is part of their arithmetic and therefore part of reproducing their number.
    """
    if score >= 0.75:
        return 1.0
    return 0.5 if score >= 0.25 else 0.0


def cognee_judge_nugget(nugget: str, question: str, response: str, ask: Ask) -> Verdict:
    """Score one nugget the way Cognee scores it.

    Four deliberate differences from :func:`judge_nugget`, each matching
    ``beam_rubric.py`` rather than our preference:

    * the prompt gets the question (the point of the whole variant);
    * a system message, ``COGNEE_JUDGE_SYSTEM``, which BEAM sends none of;
    * the score is snapped to three levels by :func:`snap`;
    * **no retry, and every failure scores 0.0** instead of raising. Their
      ``_judge_criterion`` catches ``Exception`` and returns ``0.0`` with an
      ``ERROR:`` reason, and their report says failed questions were not
      retried.

    That last one is why this does not reuse our error handling. It silently
    converts an endpoint hiccup into a real zero, which is a worse default than
    raising -- an empty reply from a reasoning model out of token budget lands
    as a confident 0.0. Reproduced because scoring their arm on our error policy
    would make the number ours instead of theirs; the ``ERROR:`` prefix on the
    reason is the only trace, so count them before reading a score.
    """
    prompt = cognee_judge_prompt(question=question, nugget=nugget, response=response)
    try:
        score, reason = parse_verdict(ask(prompt, system=COGNEE_JUDGE_SYSTEM))
    except (JudgeError, Exception) as exc:  # noqa: B014 - theirs is this broad
        logger.warning("cognee judge failed for %r: %s", nugget[:60], exc)
        return Verdict(nugget=nugget, score=0.0, reason=f"ERROR: {exc}")
    return Verdict(nugget=nugget, score=snap(score), reason=reason)


def events_equivalent(first: str, second: str, ask: Ask) -> bool:
    """Whether two snippets name the same event.

    Upstream's test is ``"yes" in response.lower()``, which this keeps. It is
    looser than it looks -- a reply of "No, yesterday's build differs" contains
    "yes" -- but the classifier is told to answer in one word and tightening it
    would shift event-ordering scores away from the published ones.
    """
    reply = ask(
        equivalence_prompt(first=first, second=second), system=EQUIVALENCE_SYSTEM
    )
    return "yes" in reply.lower()


def align_events(
    reference: Sequence[str], system: Sequence[str], ask: Ask
) -> list[str]:
    """Rewrite *system*'s events in *reference*'s wording where they match.

    Greedy and order-sensitive, as upstream: each system event takes the first
    not-yet-claimed reference event it is equivalent to, so an earlier event
    that matches two rubric items consumes the earlier one. Unmatched events are
    left as they are, which is what makes them false positives downstream.

    Blank entries skip the model. A binary classifier asked whether "" and a
    rubric item describe the same event answers no, so the call only costs
    money; the blank still goes into the output and still counts against the
    answer, which is upstream's behaviour and part of why these scores are low.
    """
    claimed: set[int] = set()
    aligned: list[str] = []
    for event in system:
        matched: int | None = None
        if event.strip():
            for index, candidate in enumerate(reference):
                if index in claimed:
                    continue
                if events_equivalent(candidate, event, ask):
                    matched = index
                    break
        if matched is None:
            aligned.append(event)
        else:
            aligned.append(reference[matched])
            claimed.add(matched)
    return aligned


def score_question(
    *,
    conversation_id: str,
    ability: str,
    question: str,
    response: str,
    rubric: Sequence[str],
    ask: Ask,
    flavour: str = "beam",
) -> QuestionResult:
    """Score one answer against its rubric.

    With ``flavour="beam"`` the judge never sees *question* -- it is recorded
    only so a saved result can be read without the questions file beside it.
    With ``flavour="cognee"`` the question is sent, which is the one substantive
    difference between the two prompts; see :func:`cognee_judge_nugget`.
    """
    if not rubric:
        raise ValueError(f"{conversation_id}/{ability}: empty rubric")

    if flavour == "cognee":
        verdicts = tuple(
            cognee_judge_nugget(nugget, question, response, ask) for nugget in rubric
        )
    elif flavour == "beam":
        verdicts = tuple(judge_nugget(nugget, response, ask) for nugget in rubric)
    else:
        raise ValueError(f"unknown judge flavour {flavour!r}")

    ordering: dict[str, float] | None = None
    if ability == ORDERING_ABILITY:
        # Upstream splits raw, keeping blank lines. See align_events.
        aligned = align_events(rubric, response.split("\n"), ask)
        ordering = ordering_scores(rubric, aligned)

    # Which of the two numbers an event-ordering question contributes depends
    # on whose published table this run is being read against; see the module
    # docstring. Tau is computed either way and reported separately.
    if ability == ORDERING_ABILITY and flavour == "beam":
        score = ordering["tau_norm"]  # type: ignore[index]
    else:
        score = mean(v.score for v in verdicts)

    return QuestionResult(
        conversation_id=conversation_id,
        ability=ability,
        question=question,
        response=response,
        score=score,
        verdicts=verdicts,
        ordering=ordering,
    )
