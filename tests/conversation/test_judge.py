# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge, driven by a stub instead of an endpoint.

``Ask`` is a single-method protocol precisely so these can be plain functions.
Nothing here reaches a model, so nothing here needs credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conversation.beam_benchmark.evaluator import prompts
from conversation.beam_benchmark.evaluator.judge import (
    JudgeError,
    parse_verdict,
    score_question,
)


def replying(*replies: str):
    """An ``Ask`` that returns *replies* in order, recording what it was sent."""
    remaining = list(replies)
    sent: list[tuple[str, str | None]] = []

    def ask(prompt: str, *, system: str | None = None) -> str:
        sent.append((prompt, system))
        return remaining.pop(0) if remaining else replies[-1]

    ask.sent = sent  # type: ignore[attr-defined]
    return ask


def verdict(score: float) -> str:
    return f'{{"score": {score}, "reason": "because"}}'


class TestPrompt:
    def test_substitution_leaves_the_literal_json_braces_alone(self) -> None:
        """``str.replace``, not ``str.format``.

        The output-format block contains a literal ``{"score": ...}``, which
        ``str.format`` would read as a field name and raise on.
        """
        filled = prompts.cognee_judge_prompt(
            question="the question", nugget="the nugget", response="the answer"
        )
        assert "the question" in filled
        assert "the nugget" in filled
        assert "the answer" in filled
        assert '{"score"' in filled


class TestParseVerdict:
    def test_a_bare_object(self) -> None:
        assert parse_verdict('{"score": 0.5, "reason": "partly"}') == (0.5, "partly")

    def test_a_fenced_object(self) -> None:
        text = '```json\n{"score": 1.0, "reason": "yes"}\n```'
        assert parse_verdict(text) == (1.0, "yes")

    def test_an_object_buried_in_prose(self) -> None:
        text = (
            'Here is my evaluation:\n{"score": 0.0, "reason": "no"}\nHope that helps.'
        )
        assert parse_verdict(text) == (0.0, "no")

    def test_a_score_with_no_parseable_object(self) -> None:
        """The regex fallback."""
        score, reason = parse_verdict('{"score": 0.5, "reason": "unterminated')
        assert score == 0.5
        assert reason == ""

    def test_a_string_score_is_still_a_score(self) -> None:
        assert parse_verdict('{"score": "1.0", "reason": "ok"}') == (1.0, "ok")

    def test_an_off_scale_score_is_a_failure_not_a_judgement(self) -> None:
        with pytest.raises(JudgeError, match="outside"):
            parse_verdict('{"score": 95, "reason": "percent, apparently"}')

    def test_prose_with_no_score_at_all(self) -> None:
        with pytest.raises(JudgeError, match="no score"):
            parse_verdict("The response seems fine to me.")

    def test_an_off_scale_value_inside_the_scale_passes_through(self) -> None:
        """0.75 is not on the prompt's scale but is inside [0, 1].

        Taken as reported here; :func:`snap` is what rounds it, as cognee
        does.
        """
        assert parse_verdict('{"score": 0.75, "reason": "mostly"}')[0] == 0.75


class TestScoreQuestion:
    def test_a_question_averages_its_nuggets(self) -> None:
        result = score_question(
            conversation_id="1",
            ability="information_extraction",
            question="what was the budget?",
            response="68k",
            rubric=["states 68,000", "names the currency"],
            ask=replying(verdict(1.0), verdict(0.0)),
        )
        assert result.score == pytest.approx(0.5)
        assert [v.score for v in result.verdicts] == [1.0, 0.0]

    def test_event_ordering_is_scored_by_its_nuggets_too(self) -> None:
        """Cognee's ``beam_rubric.py`` has no special case for this ability."""
        result = score_question(
            conversation_id="1",
            ability="event_ordering",
            question="in what order?",
            response="third\nsecond\nfirst",
            rubric=["first", "second", "third"],
            ask=replying(verdict(1.0)),
        )
        assert result.score == pytest.approx(1.0)
        assert len(result.verdicts) == 3

    def test_the_question_reaches_the_judge(self) -> None:
        ask = replying(verdict(1.0))
        score_question(
            conversation_id="1",
            ability="abstention",
            question="a very distinctive question string",
            response="no information available",
            rubric=["says there is no information"],
            ask=ask,
        )
        assert all("a very distinctive question string" in p for p, _ in ask.sent)

    def test_an_empty_rubric_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty rubric"):
            score_question(
                conversation_id="1",
                ability="abstention",
                question="q",
                response="r",
                rubric=[],
                ask=replying(verdict(1.0)),
            )

    def test_the_result_feeds_the_aggregators_directly(self) -> None:
        result = score_question(
            conversation_id="7",
            ability="summarization",
            question="q",
            response="r",
            rubric=["n"],
            ask=replying(verdict(0.5)),
        )
        assert result.as_row() == ("7", "summarization", 0.5)


def test_cognee_prompt_supplies_the_question() -> None:
    from conversation.beam_benchmark.evaluator.prompts import cognee_judge_prompt

    filled = cognee_judge_prompt(
        question="What bugs were logged?", nugget="states X", response="X happened"
    )
    assert "- QUESTION (what the user asked): What bugs were logged?" in filled
    assert "<question>" not in filled


def test_snap_matches_cognee_cuts() -> None:
    from conversation.beam_benchmark.evaluator.judge import snap

    assert [snap(v) for v in (0.0, 0.24, 0.25, 0.74, 0.75, 1.0)] == [
        0.0,
        0.0,
        0.5,
        0.5,
        1.0,
        1.0,
    ]


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Error code: 503 - service unavailable"),
        RuntimeError("Connection error."),
    ],
)
def test_cognee_judge_scores_endpoint_failures_zero(failure: Exception) -> None:
    """Their ``_judge_criterion`` scores an endpoint failure 0.0, so ours does."""
    from conversation.beam_benchmark.evaluator.judge import cognee_judge_nugget

    def broken(prompt: str, *, system: str | None = None) -> str:
        raise failure

    verdict = cognee_judge_nugget("states X", "What?", "X happened", broken)
    assert verdict.score == 0.0
    assert verdict.reason.startswith("ERROR:")


def test_cognee_judge_scores_an_unparseable_reply_zero() -> None:
    from conversation.beam_benchmark.evaluator.judge import cognee_judge_nugget

    verdict = cognee_judge_nugget(
        "states X", "What?", "X happened", lambda prompt, *, system=None: ""
    )
    assert verdict.score == 0.0
    assert verdict.reason.startswith("ERROR:")


def test_cognee_judge_raises_a_bug_instead_of_scoring_it() -> None:
    """A fault in this harness is not a judgement about the answer."""
    from conversation.beam_benchmark.evaluator.judge import cognee_judge_nugget

    def buggy(prompt: str, *, system: str | None = None) -> str:
        raise KeyError("nugget")

    with pytest.raises(KeyError):
        cognee_judge_nugget("states X", "What?", "X happened", buggy)


def test_cognee_judge_sends_their_system_message() -> None:
    from conversation.beam_benchmark.evaluator.judge import cognee_judge_nugget

    seen: dict[str, str | None] = {}

    def ask(prompt: str, *, system: str | None = None) -> str:
        seen["system"] = system
        return '{"score": 1.0, "reason": "ok"}'

    cognee_judge_nugget("states X", "What?", "X happened", ask)
    assert seen["system"] == "You are an evaluation judge. Return only valid JSON."


def test_with_retry_recovers_from_a_transient_error() -> None:
    from conversation.beam_benchmark.evaluator import judge as J

    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Connection error.")
        return "ok"

    J._BACKOFF_SECONDS = 0.0
    assert J.with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_with_retry_does_not_retry_a_real_failure() -> None:
    """A refusal fails the same way four times; retrying only delays it."""
    from conversation.beam_benchmark.evaluator import judge as J

    calls = {"n": 0}

    def refused() -> str:
        calls["n"] += 1
        raise ValueError("model not found: bogus/model")

    with pytest.raises(ValueError, match="model not found"):
        J.with_retry(refused)
    assert calls["n"] == 1


def test_transient_classifier_covers_the_outage_we_saw() -> None:
    from conversation.beam_benchmark.evaluator.judge import _is_transient

    assert _is_transient(RuntimeError("Connection error."))
    assert _is_transient(RuntimeError("503 Service Unavailable"))
    assert _is_transient(TimeoutError("request timed out"))
    assert not _is_transient(ValueError("context_length_exceeded"))


def test_the_reader_sends_cognees_template_under_the_ability_prompt() -> None:
    from conversation.beam_benchmark.reader import answer

    class Hit:
        turn_index, session_index, time_anchor = 1, 1, ""
        question, answer = "hi", "there"

    seen: dict[str, str | None] = {}

    def read(prompt: str, *, system: str | None = None) -> str:
        seen["prompt"], seen["system"] = prompt, system
        return "ok"

    answer("What?", [Hit()], read, system="BE THOROUGH")
    assert "The question is: `What?`" in seen["prompt"]
    assert "USER: hi" in seen["prompt"]
    assert seen["system"] == "BE THOROUGH"


def test_cognee_prompts_exclude_the_user_template() -> None:
    """``context_for_question`` is the user slot, not an ability."""
    from conversation.beam_benchmark.reader import cognee_system_prompts

    prompts = cognee_system_prompts()
    assert "context_for_question" not in prompts
    assert "default" in prompts and "summarization" in prompts
    assert len(prompts) == 11


def test_cognee_prompts_are_upstream_text_with_nothing_appended() -> None:
    """A note tuned on BEAM's rubric items would make the score measure it."""
    from conversation.beam_benchmark.reader import _COGNEE_QA, cognee_system_prompts

    for ability, prompt in cognee_system_prompts().items():
        upstream = (_COGNEE_QA / f"{ability}.txt").read_text(encoding="utf-8")
        assert prompt == upstream.strip(), ability


def test_endpoint_capacity_exhaustion_is_transient_despite_the_401() -> None:
    """The endpoint returns "too many clients" as an AuthenticationError."""
    from conversation.beam_benchmark.evaluator.judge import _is_transient

    exhausted = Exception(
        "Error code: 401 - {'error': {'message': 'Authentication Error, Error "
        "in connector: Error querying the database: FATAL: sorry, too many "
        "clients already', 'type': 'auth_error', 'code': '401'}}"
    )
    assert _is_transient(exhausted)
    # A genuine credential failure still must not be retried.
    assert not _is_transient(Exception("Error code: 401 - invalid api key"))


def test_rounds_must_all_score_the_expected_questions() -> None:
    """Pooling them would compare two question sets and call it noise."""
    from conversation.beam_benchmark.run import _require_one_question_set

    expected = ["2/abstention/0", "2/abstention/1", "2/summarization/0"]
    _require_one_question_set(
        [{"question_keys": expected}, {"question_keys": expected}], expected
    )
    # Equal counts, different questions.
    with pytest.raises(RuntimeError, match=r"r2 \(missing 1, extra 1\)"):
        _require_one_question_set(
            [
                {"question_keys": expected[:2]},
                {"question_keys": [expected[0], expected[2]]},
            ],
            expected[:2],
        )
    # The same question lost in every round still leaves the score too high.
    with pytest.raises(RuntimeError, match="missing 1"):
        _require_one_question_set(
            [{"question_keys": expected[:2]}, {"question_keys": expected[:2]}],
            expected,
        )


def test_a_failed_question_fails_the_run_after_writing_the_rest(
    tmp_path: Path,
) -> None:
    """Leaving it out would score the run over the questions that survived."""
    from conversation.beam_benchmark.run import _run_concurrently

    def work(row: dict[str, str]) -> dict[str, str]:
        if row["id"] == "bad":
            raise ValueError("boom")
        return row

    out = tmp_path / "rows.jsonl"
    rows = [{"id": "a"}, {"id": "bad"}, {"id": "b"}]
    with pytest.raises(RuntimeError, match="1 of 3"):
        _run_concurrently(rows, work, out=out, workers=2, verb="judged")
    assert sorted(json.loads(line)["id"] for line in out.read_text().splitlines()) == [
        "a",
        "b",
    ]


def _retrieve(content: str) -> dict[str, str]:
    return {"role": "tool", "name": "retrieve", "content": content}


def test_the_agent_must_only_see_its_own_conversation() -> None:
    from conversation.beam_benchmark.run import require_own_conversation

    own = _retrieve('<turn n="3" conversation="2" session="1">hi</turn>')
    require_own_conversation([own, _retrieve("no turns here")], "2")

    leaked = _retrieve('<turn n="9" conversation="7">elsewhere</turn>')
    with pytest.raises(RuntimeError, match=r"\['7'\]"):
        require_own_conversation([own, leaked], "2")


def test_a_turn_quoted_in_the_agent_answer_is_not_a_retrieval() -> None:
    """Only tool observations are what the search returned."""
    from conversation.beam_benchmark.run import require_own_conversation

    said = {"role": "ai", "content": '<turn conversation="7">'}
    require_own_conversation([said], "2")
