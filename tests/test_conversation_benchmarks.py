# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation benchmark registry picks a whole instrument, not parts of one."""

from __future__ import annotations

import argparse

import pytest

from conversation import cli
from conversation.beam_benchmark.evaluator.judge import DEFAULT_JUDGE_MODEL
from conversation.benchmarks import BENCHMARKS, is_conversation_benchmark, spec_for


def test_default_benchmark_is_the_one_comparable_with_the_bar() -> None:
    """``beam`` must be cognee's instrument throughout.

    0.79 is the bar and 0.794 is what it was set by, both measured with
    cognee's judge and cognee's reader prompt. Our score is only comparable
    with theirs if every part of the instrument matches; a spec that mixed
    cognee's judge with the paper's reader would produce a number comparable
    with nothing. That mixing is exactly what the old flags allowed.
    """
    spec = BENCHMARKS["beam"]
    assert spec.judge_flavour == "cognee"
    # Their report, Appendix A: gpt-5 judges. Their saved config does not name
    # the model, so it is pinned here; gpt-4.1-mini is the more lenient judge.
    assert spec.judge_model == "azure/openai/gpt-5"
    assert spec.reader_style == "cognee"
    # Cognee's four rounds. The judge is stochastic and run-to-run spread is
    # about 0.035, so a single round is a different measurement, not a cheaper
    # one.
    assert spec.rounds == 4


def test_paper_benchmark_is_the_paper_throughout() -> None:
    """``beam-paper`` is the only thing 0.323 may be compared with."""
    spec = BENCHMARKS["beam-paper"]
    assert spec.judge_flavour == "beam"
    assert spec.judge_model == DEFAULT_JUDGE_MODEL
    assert spec.reader_style == "beam"


def test_sql_datasets_are_not_conversation_benchmarks() -> None:
    assert spec_for("bird") is None
    assert not is_conversation_benchmark("beaverbench")
    assert is_conversation_benchmark("beam")


def test_own_arguments_lists_every_registered_flag() -> None:
    """The guard list cannot drift from the flags it is meant to cover.

    A flag added to the group and forgotten in ``_OWN_ARGUMENTS`` would be
    accepted on a SQL dataset and ignored, which is the failure the guard
    exists to prevent.
    """
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    assert set(vars(parser.parse_args([]))) == set(cli._OWN_ARGUMENTS)


def test_conversation_flags_are_rejected_on_a_sql_dataset() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-name")
    cli.add_arguments(parser)
    args = parser.parse_args(["--database-name", "bird", "--retrieval", "agent"])
    with pytest.raises(SystemExit):
        cli.reject_on_sql_datasets(parser, args)


def test_defaulted_conversation_flags_are_allowed_on_a_sql_dataset() -> None:
    """Only a flag the caller actually set is an error.

    Every flag in the group has a default, so an untouched namespace carries
    all of them on every SQL run; rejecting those would reject every run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-name")
    cli.add_arguments(parser)
    args = parser.parse_args(["--database-name", "bird"])
    cli.reject_on_sql_datasets(parser, args)
