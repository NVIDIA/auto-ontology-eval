# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How question scores become ability means and the overall."""

from __future__ import annotations

import pytest

from auto_ontology_eval.beam.cognee.scoring import ability_means, overall


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
