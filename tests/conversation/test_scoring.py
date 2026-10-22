# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arithmetic behind BEAM's scores.

The tau-b expectations here were cross-checked against
``scipy.stats.kendalltau(..., variant="b")`` over 15,695 random tie-heavy
vectors, agreeing to 2.2e-16. scipy is not a dependency of this repo, so the
values are pinned rather than re-derived at test time.
"""

from __future__ import annotations

import pytest

from conversation.beam_benchmark.evaluator.scoring import (
    ability_means,
    kendall_tau_b,
    ordering_scores,
    overall,
)


class TestKendallTauB:
    def test_identical_rankings_correlate_perfectly(self) -> None:
        assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_reversed_rankings_anticorrelate_perfectly(self) -> None:
        assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_one_swapped_pair_costs_a_third(self) -> None:
        # Six pairs, one discordant: (6 - 2 * 1) / 6.
        assert kendall_tau_b([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(2 / 3)

    def test_ties_shrink_the_denominator_not_the_numerator(self) -> None:
        # Six pairs: one tied in x, five concordant, none discordant. The tied
        # pair leaves the numerator at 5 but puts sqrt(5 * 6) underneath it, so
        # a sequence that never disagrees still cannot reach 1.
        assert kendall_tau_b([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(5 / (30**0.5))

    def test_a_pair_tied_in_both_counts_against_both_denominators(self) -> None:
        """The regression this exists for.

        Six pairs, three of them tied in x and one tied in y -- and that
        y-tied pair is tied in x as well. Counting it in only one of the two
        tie totals, or skipping it as tied in neither, leaves the denominator
        too large and understates tau: 0.61 against the correct 0.77.
        """
        assert kendall_tau_b([1, 1, 1, 2], [1, 1, 2, 3]) == pytest.approx(
            3 / ((6 - 3) * (6 - 1)) ** 0.5
        )

    def test_a_constant_vector_scores_zero_rather_than_nan(self) -> None:
        # scipy returns NaN here. Upstream adds that into a running total, so
        # one such question would make a whole ability NaN.
        assert kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0

    def test_mismatched_lengths_are_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            kendall_tau_b([1, 2], [1, 2, 3])


class TestOrderingScores:
    def test_the_exact_sequence_scores_one(self) -> None:
        scores = ordering_scores(["a", "b", "c"], ["a", "b", "c"])
        assert scores["tau_norm"] == pytest.approx(1.0)
        assert scores["f1"] == pytest.approx(1.0)

    def test_the_right_events_in_reverse_still_score_zero(self) -> None:
        """Recall is perfect and the score is 0. This is the metric, not a bug.

        It is also why ``f1`` is reported separately: on its own, ``tau_norm``
        cannot tell a reversed answer from an empty one.
        """
        scores = ordering_scores(["a", "b", "c"], ["c", "b", "a"])
        assert scores["tau_norm"] == pytest.approx(0.0)
        assert scores["f1"] == pytest.approx(1.0)

    def test_wholly_wrong_events_land_below_chance(self) -> None:
        """0.125, not 0.5.

        An answer sharing nothing with the rubric puts each sequence's events
        at the bottom of the other's rank vector, which reads as
        anti-correlation rather than as no information. This is the shape of
        the published 0.15-0.27 event-ordering scores.
        """
        scores = ordering_scores(["a", "b", "c"], ["x", "y", "z"])
        assert scores["tau_norm"] == pytest.approx(0.125)
        assert scores["f1"] == pytest.approx(0.0)

    def test_final_score_is_reported_but_not_the_published_number(self) -> None:
        scores = ordering_scores(["a", "b", "c"], ["a", "b", "z"])
        assert scores["final_score"] == pytest.approx(scores["tau_norm"] * scores["f1"])
        assert scores["final_score"] < scores["tau_norm"]

    def test_an_empty_answer_scores_nothing(self) -> None:
        scores = ordering_scores(["a", "b", "c"], [])
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0


class TestAggregation:
    def test_an_ability_averages_within_a_conversation_then_across(self) -> None:
        scored = [
            ("1", "summarization", 1.0),
            ("1", "summarization", 0.0),
            ("2", "summarization", 0.5),
        ]
        # Conversation 1 means 0.5, conversation 2 means 0.5.
        assert ability_means(scored)["summarization"] == pytest.approx(0.5)

    def test_a_conversation_with_more_questions_does_not_count_for_more(
        self,
    ) -> None:
        """The reason the average has two levels instead of one.

        Flat over questions this is 0.75; weighting conversations equally it is
        0.5, which is what upstream reports.
        """
        scored = [
            ("1", "abstention", 1.0),
            ("1", "abstention", 1.0),
            ("1", "abstention", 1.0),
            ("2", "abstention", 0.0),
        ]
        assert ability_means(scored)["abstention"] == pytest.approx(0.5)

    def test_abilities_with_no_questions_are_absent_not_zero(self) -> None:
        means = ability_means([("1", "abstention", 1.0)])
        assert means == {"abstention": 1.0}

    def test_the_headline_number_averages_abilities_not_questions(self) -> None:
        """A partial run over-reports if you read it as a benchmark score.

        One abstention question at 1.0 and four summarization questions at 0.0
        is 0.2 of the questions answered well and 0.5 here, because each ability
        gets one tenth of the weight regardless of how many questions carry it.
        """
        scored = [
            ("1", "abstention", 1.0),
            ("1", "summarization", 0.0),
            ("1", "summarization", 0.0),
            ("2", "summarization", 0.0),
            ("2", "summarization", 0.0),
        ]
        assert overall(scored) == pytest.approx(0.5)

    def test_nothing_scored_is_zero_rather_than_a_crash(self) -> None:
        assert overall([]) == 0.0
