# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge, driven by a stub instead of an endpoint.

``Ask`` is a single-method protocol precisely so these can be plain functions.
Nothing here reaches a model, so nothing here needs credentials.
"""

from __future__ import annotations

import pytest

from conversation.beam_benchmark.evaluator import prompts
from conversation.beam_benchmark.evaluator.judge import (
    JudgeError,
    align_events,
    events_equivalent,
    judge_nugget,
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


class TestPromptsAreVerbatim:
    def test_the_judge_prompt_keeps_both_placeholders(self) -> None:
        assert "<rubric_item>" in prompts.JUDGE
        assert "<llm_response>" in prompts.JUDGE

    def test_trailing_whitespace_survived_the_copy(self) -> None:
        """The reason the prompts are text files and not string literals.

        Five lines of the judge prompt end in whitespace. A formatter or an
        editor stripping them changes how the prompt tokenizes and moves the
        scores a little, with nothing in the diff to say why -- so if this
        fails, the prompt was edited and the comparison to the published 0.323
        no longer holds.
        """
        trailing = [line for line in prompts.JUDGE.split("\n") if line != line.rstrip()]
        assert len(trailing) == 5

    def test_substitution_leaves_the_literal_json_braces_alone(self) -> None:
        """``str.replace``, not ``str.format``.

        The output-format block contains a literal ``{"score": ...}``, which
        ``str.format`` would read as a field name and raise on.
        """
        filled = prompts.judge_prompt(nugget="the nugget", response="the answer")
        assert "the nugget" in filled
        assert "the answer" in filled
        assert '"score"' in filled

    def test_the_equivalence_prompt_carries_both_snippets(self) -> None:
        filled = prompts.equivalence_prompt(first="one", second="two")
        assert "First snippet: one" in filled
        assert "Second snippet: two" in filled


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
        """The regex fallback, standing in for upstream's ``json_repair``."""
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

        Upstream's bare ``float(...)`` takes it, so this does too; silently
        rounding it to the nearest allowed value would be us scoring, not the
        judge.
        """
        assert parse_verdict('{"score": 0.75, "reason": "mostly"}')[0] == 0.75


class TestJudgeNugget:
    def test_a_clean_reply_is_scored_once(self) -> None:
        ask = replying(verdict(1.0))
        result = judge_nugget("states the budget was $68,000", "It was 68k", ask)
        assert result.score == 1.0
        assert result.nugget == "states the budget was $68,000"
        assert len(ask.sent) == 1

    def test_the_nugget_and_response_reach_the_model(self) -> None:
        ask = replying(verdict(0.0))
        judge_nugget("the nugget", "the answer", ask)
        prompt, system = ask.sent[0]
        assert "the nugget" in prompt
        assert "the answer" in prompt
        assert system is None

    def test_a_malformed_reply_is_retried(self) -> None:
        ask = replying("sorry, what?", verdict(0.5))
        assert judge_nugget("n", "r", ask).score == 0.5
        assert len(ask.sent) == 2

    def test_retries_run_out_and_raise(self) -> None:
        """Not scored 0.0.

        An unparseable judge is a missing measurement; folding it in as "no
        compliance" biases a run downwards where nobody would look for it.
        """
        ask = replying("nope", "nope", "nope")
        with pytest.raises(JudgeError):
            judge_nugget("n", "r", ask, attempts=3)
        assert len(ask.sent) == 3


class TestEventAlignment:
    def test_yes_anywhere_in_the_reply_counts_as_a_match(self) -> None:
        """Upstream's test, kept. Looser than it looks, deliberately."""
        assert events_equivalent("a", "b", replying("YES"))
        assert events_equivalent("a", "b", replying("No, yesterday differed"))
        assert not events_equivalent("a", "b", replying("NO"))

    def test_the_system_message_is_sent_with_the_pair(self) -> None:
        ask = replying("NO")
        events_equivalent("first thing", "second thing", ask)
        prompt, system = ask.sent[0]
        assert system == prompts.EQUIVALENCE_SYSTEM
        assert "first thing" in prompt and "second thing" in prompt

    def test_a_match_is_rewritten_into_the_rubric_wording(self) -> None:
        aligned = align_events(
            ["shipped the build"], ["we shipped it"], replying("YES")
        )
        assert aligned == ["shipped the build"]

    def test_an_unmatched_event_is_left_alone(self) -> None:
        aligned = align_events(["shipped the build"], ["ate lunch"], replying("NO"))
        assert aligned == ["ate lunch"]

    def test_each_rubric_event_is_claimed_at_most_once(self) -> None:
        """Greedy and order-sensitive, as upstream.

        Both answers would match both rubric items if asked; the first answer
        takes the first item, so the second has to take the second.
        """
        aligned = align_events(["first", "second"], ["one", "two"], replying("YES"))
        assert aligned == ["first", "second"]

    def test_a_blank_line_is_kept_but_costs_no_model_call(self) -> None:
        """Blank lines still count against the answer, which is the point.

        Upstream splits the answer on newlines with no filtering, so a blank
        line becomes an event matching nothing. Skipping the call for it is
        behaviour-identical -- a binary classifier does not call "" the same
        event as a rubric item -- and saves the request.
        """
        ask = replying("NO")
        aligned = align_events(["first"], ["", "  "], ask)
        assert aligned == ["", "  "]
        assert ask.sent == []


class TestScoreQuestion:
    def test_nine_abilities_average_their_nuggets(self) -> None:
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
        assert result.ordering is None

    def test_event_ordering_reports_tau_not_its_nuggets(self) -> None:
        """The whole reason this port exists.

        Every nugget scores 1.0 here, so a nugget-averaged reading would be
        1.0. The published table reads ``tau_norm``, and the answer lists the
        events backwards, so the score is 0.0.
        """
        result = score_question(
            conversation_id="1",
            ability="event_ordering",
            question="in what order?",
            response="third\nsecond\nfirst",
            rubric=["first", "second", "third"],
            ask=replying(verdict(1.0)),
        )
        assert result.score == pytest.approx(0.0)
        assert result.ordering is not None
        assert result.ordering["f1"] == pytest.approx(1.0)
        assert [v.score for v in result.verdicts] == [1.0, 1.0, 1.0]

    def test_cognee_scores_event_ordering_by_nuggets_and_keeps_tau(self) -> None:
        """The same answer, read against the other published table.

        cognee's ``beam_rubric.py`` has no tau in it and no case for this
        ability, and their cross-run summary carries ``kendall_tau`` as a
        separate metric. So the identical backwards answer that reads 0.0 above
        reads 1.0 here, with tau still computed and still 0.0 -- the two
        numbers coexist, and the flavour chooses which one is the score.
        """
        result = score_question(
            conversation_id="1",
            ability="event_ordering",
            question="in what order?",
            response="third\nsecond\nfirst",
            rubric=["first", "second", "third"],
            ask=replying(verdict(1.0)),
            flavour="cognee",
        )
        assert result.score == pytest.approx(1.0)
        assert result.ordering is not None
        assert result.ordering["tau_norm"] == pytest.approx(0.0)

    def test_the_question_is_recorded_but_never_sent(self) -> None:
        """An upstream quirk, not an oversight here.

        The judge prompt's responsiveness rules refer to "the QUESTION" but the
        prompt has no placeholder for it, so the judge rules on responsiveness
        without knowing what was asked.
        """
        ask = replying(verdict(1.0))
        result = score_question(
            conversation_id="1",
            ability="abstention",
            question="a very distinctive question string",
            response="no information available",
            rubric=["says there is no information"],
            ask=ask,
        )
        assert result.question == "a very distinctive question string"
        assert all("a very distinctive question string" not in p for p, _ in ask.sent)

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
    """The one substantive difference from BEAM's prompt."""
    from conversation.beam_benchmark.evaluator.prompts import (
        JUDGE,
        cognee_judge_prompt,
    )

    filled = cognee_judge_prompt(
        question="What bugs were logged?", nugget="states X", response="X happened"
    )
    assert "- QUESTION (what the user asked): What bugs were logged?" in filled
    assert "<question>" not in filled
    # BEAM's asks about a QUESTION it never supplies -- the defect being fixed.
    assert "does not address the QUESTION" in JUDGE
    assert "- QUESTION (what the user asked):" not in JUDGE


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


def test_cognee_judge_scores_failures_zero_rather_than_raising() -> None:
    """Their ``_judge_criterion`` swallows everything. Ours must too, here."""
    from conversation.beam_benchmark.evaluator.judge import cognee_judge_nugget

    def broken(prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("endpoint down")

    verdict = cognee_judge_nugget("states X", "What?", "X happened", broken)
    assert verdict.score == 0.0
    assert verdict.reason.startswith("ERROR:")


def test_cognee_judge_sends_their_system_message() -> None:
    from conversation.beam_benchmark.evaluator.judge import cognee_judge_nugget

    seen: dict[str, str | None] = {}

    def ask(prompt: str, *, system: str | None = None) -> str:
        seen["system"] = system
        return '{"score": 1.0, "reason": "ok"}'

    cognee_judge_nugget("states X", "What?", "X happened", ask)
    assert seen["system"] == "You are an evaluation judge. Return only valid JSON."


def test_score_question_rejects_an_unknown_flavour() -> None:
    from conversation.beam_benchmark.evaluator.judge import score_question

    with pytest.raises(ValueError, match="unknown judge flavour"):
        score_question(
            conversation_id="1",
            ability="abstention",
            question="q",
            response="r",
            rubric=["states X"],
            ask=lambda prompt, system=None: '{"score": 1.0}',
            flavour="nope",
        )


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


def test_cognee_reader_replaces_beams_prompt_rather_than_wrapping_it() -> None:
    from conversation.beam_benchmark.reader import READER, answer

    class Hit:
        turn_index, session_index, time_anchor = 1, 1, ""
        question, answer = "hi", "there"

    seen: dict[str, str | None] = {}

    def read(prompt: str, *, system: str | None = None) -> str:
        seen["prompt"], seen["system"] = prompt, system
        return "ok"

    answer("What?", [Hit()], read, system="BE THOROUGH", style="cognee")
    assert "The question is: `What?`" in seen["prompt"]
    # BEAM's conflicting instruction must not be in the stack.
    assert "Be direct and concise" not in seen["prompt"]
    assert "ONLY the information provided" not in seen["prompt"]
    assert seen["system"] == "BE THOROUGH"

    answer("What?", [Hit()], read, style="beam")
    assert "STRICT INSTRUCTIONS" in seen["prompt"]
    assert seen["system"] is None
    assert READER.strip() != ""


def test_unknown_reader_style_is_rejected() -> None:
    from conversation.beam_benchmark.reader import answer

    with pytest.raises(ValueError, match="unknown reader style"):
        answer("q", [], lambda p, system=None: "x", style="nope")


def test_cognee_prompts_exclude_the_user_template() -> None:
    """``context_for_question`` is the user slot, not an ability."""
    from conversation.beam_benchmark.reader import cognee_system_prompts

    prompts = cognee_system_prompts()
    assert "context_for_question" not in prompts
    assert "default" in prompts and "summarization" in prompts
    assert len(prompts) == 11


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


def test_rounds_with_different_question_counts_are_refused() -> None:
    """Pooling them would compare two question sets and call it noise."""
    from conversation.beam_benchmark.run import _require_one_question_set

    _require_one_question_set([{"answered": 20}, {"answered": 20}])
    with pytest.raises(RuntimeError, match="different question counts"):
        _require_one_question_set(
            [{"answered": 20}, {"answered": 18}, {"answered": 20}]
        )
