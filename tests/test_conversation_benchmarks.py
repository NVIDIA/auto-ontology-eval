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


def test_beam_is_cognees_instrument() -> None:
    """Our score is only comparable with their 0.794 on their instrument."""
    assert list(BENCHMARKS) == ["beam"]
    spec = BENCHMARKS["beam"]
    # Their report, Appendix A: gpt-5 judges. Their saved config does not name
    # the model, so it is pinned here, and the judge defaults to it as well.
    assert spec.judge_model == "azure/openai/gpt-5" == DEFAULT_JUDGE_MODEL
    # Cognee's four rounds. The judge is stochastic and run-to-run spread is
    # about 0.035, so a single round is a different measurement, not a cheaper
    # one.
    assert spec.rounds == 4


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


def test_conversations_limit_the_ingest_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise targeting one conversation re-ingests and resets all twenty."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        cli, "stage_ingest", lambda **kwargs: seen.update(kwargs), raising=True
    )
    args = argparse.Namespace(
        scale="100K",
        conversations="2, 5",
        no_reset=False,
        skip_ingest=False,
        skip_eval=True,
    )
    cli.run_stages("beam", args)
    assert seen == {"scale": "100K", "ids": ["2", "5"], "reset": True}
