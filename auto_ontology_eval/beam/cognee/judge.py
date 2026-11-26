# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognee's BEAM rubric judge, ported.

Scores one answer at a time against its rubric, the way
``cognee/eval_framework/beam/eval/metrics/beam_rubric.py`` does: one model call
per nugget, given the question, the nugget and the response, each verdict
snapped to 0.0, 0.5 or 1.0 and averaged into the question's score. Every
ability is scored that way, event ordering included.

Ported rather than written fresh because a number of ours is only comparable to
cognee's published 0.794 if it came out of the same judge, so
``prompts/judge/cognee_judge.txt`` is their prompt and the quirks in
:func:`cognee_judge_nugget` are theirs.

Judging is model calls, so it is the expensive half of a run. At 100K the
rubrics hold 1,051 nuggets across 400 questions.
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

from .prompts import COGNEE_JUDGE_SYSTEM, cognee_judge_prompt
from .scoring import mean

logger = logging.getLogger(__name__)

#: Cognee's judge model, under the name ``REASONING_ENDPOINT`` serves it by.
#: The ``azure/`` prefix is also what sends it through ``ChatOpenAI`` rather
#: than ``ChatNVIDIA``, which matters for a judge whose whole output is JSON.
#:
#: Overridable via ``BEAM_JUDGE_MODEL``, but a run judged by anything else is
#: not on the same instrument as their 0.794 and should not be reported beside
#: it as though it were.
DEFAULT_JUDGE_MODEL = "azure/openai/gpt-5"

#: Upstream admits only these. A model that answers 0.75 is answering a question
#: the prompt did not ask, but it is still inside the scale, so it is taken at
#: face value before :func:`snap`. Anything outside [0, 1] is treated as a parse
#: failure instead.
_VALID_RANGE = (0.0, 1.0)

_FENCE = re.compile(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", re.DOTALL)
_OBJECT = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)
_SCORE = re.compile(r'"score"\s*:\s*"?([0-9]*\.?[0-9]+)', re.IGNORECASE)


class JudgeError(RuntimeError):
    """A judge reply could not be read as a score."""


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
    """One question's result. *score* is the mean of *verdicts*."""

    conversation_id: str
    ability: str
    question: str
    response: str
    score: float
    verdicts: tuple[Verdict, ...]

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
    temperature 0. Credentials come from the shared ``REASONING_*`` variables
    via Auto Ontology's client builder, so ``auto_ontology_eval.env.load_env()``
    has to have run first.

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
    load-bearing for :func:`cognee_judge_nugget`, which by design turns a
    failure into a real 0.0: without this, an endpoint blip is
    indistinguishable from a response that failed its rubric, and a long run
    quietly accumulates zeros. We measured 17 such zeros in one 20-question
    round during an outage.

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
    object embedded in it, then a bare regex for the score field. A reply
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


def snap(score: float) -> float:
    """Collapse a reported score onto Cognee's three levels.

    Theirs rounds to the nearest of 0.0/0.5/1.0 with cuts at 0.25 and 0.75.
    The prompt asks for one of the three anyway, so this only bites when a
    judge invents 0.8 or 0.3 -- rare, but it is part of their arithmetic and
    therefore part of reproducing their number.
    """
    if score >= 0.75:
        return 1.0
    return 0.5 if score >= 0.25 else 0.0


def cognee_judge_nugget(nugget: str, question: str, response: str, ask: Ask) -> Verdict:
    """Score one nugget the way Cognee scores it.

    Each of these matches ``beam_rubric.py`` rather than our preference:

    * the prompt gets the question;
    * a system message, ``COGNEE_JUDGE_SYSTEM``;
    * the score is snapped to three levels by :func:`snap`;
    * **no retry, and every failure scores 0.0** instead of raising. Their
      ``_judge_criterion`` catches ``Exception`` and returns ``0.0`` with an
      ``ERROR:`` reason, and their report says failed questions were not
      retried.

    That last one silently converts an endpoint hiccup into a real zero, which
    is a worse default than raising -- an empty reply from a reasoning model
    out of token budget lands as a confident 0.0. Reproduced because scoring
    on our error policy would make the number ours instead of theirs; the
    ``ERROR:`` prefix on the reason is the only trace, and the summary counts
    them as ``judge_errors``.

    Narrower than theirs in one way: only an unparseable reply or an endpoint
    failure becomes a 0.0. Theirs catches everything, which would also turn a
    bug in this harness into a judgement about the answer.
    """
    prompt = cognee_judge_prompt(question=question, nugget=nugget, response=response)
    try:
        score, reason = parse_verdict(ask(prompt, system=COGNEE_JUDGE_SYSTEM))
    except Exception as exc:  # noqa: BLE001 - classified immediately below
        if not (isinstance(exc, JudgeError) or _is_transient(exc)):
            raise
        logger.warning("cognee judge failed for %r: %s", nugget[:60], exc)
        return Verdict(nugget=nugget, score=0.0, reason=f"ERROR: {exc}")
    return Verdict(nugget=nugget, score=snap(score), reason=reason)


def score_question(
    *,
    conversation_id: str,
    ability: str,
    question: str,
    response: str,
    rubric: Sequence[str],
    ask: Ask,
) -> QuestionResult:
    """Score one answer against its rubric: the mean of its nugget verdicts."""
    if not rubric:
        raise ValueError(f"{conversation_id}/{ability}: empty rubric")

    verdicts = tuple(
        cognee_judge_nugget(nugget, question, response, ask) for nugget in rubric
    )
    return QuestionResult(
        conversation_id=conversation_id,
        ability=ability,
        question=question,
        response=response,
        score=mean(v.score for v in verdicts),
        verdicts=verdicts,
    )
