# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation benchmark registry picks a whole instrument, not parts of one."""

from __future__ import annotations

import argparse

import pytest

import main
from auto_ontology_eval.beam.cognee.config import BEAM
from auto_ontology_eval.beam.cognee.judge import DEFAULT_JUDGE_MODEL


def test_beam_is_cognees_instrument() -> None:
    """Our score is only comparable with their 0.794 on their instrument."""
    assert list(main.CONVERSATION_BENCHMARKS) == ["beam"]
    spec = BEAM
    # Their report, Appendix A: gpt-5 judges. Their saved config does not name
    # the model, so it is pinned here, and the judge defaults to it as well.
    assert spec.judge_model == "azure/openai/gpt-5" == DEFAULT_JUDGE_MODEL
    # Cognee's four rounds. The judge is stochastic and run-to-run spread is
    # about 0.035, so a single round is a different measurement, not a cheaper
    # one.
    assert spec.rounds == 4


def test_sql_datasets_are_not_conversation_benchmarks() -> None:
    assert main.setup_for("bird") is None
    assert not main.is_conversation_benchmark("beaverbench")
    assert main.is_conversation_benchmark("beam")


def test_own_arguments_lists_every_registered_flag() -> None:
    """The guard list cannot drift from the flags it is meant to cover.

    A flag added to the group and forgotten in ``_CONVERSATION_ARGUMENTS`` would be
    accepted on a SQL dataset and ignored, which is the failure the guard
    exists to prevent.
    """
    parser = argparse.ArgumentParser()
    main._add_conversation_arguments(parser)
    assert set(vars(parser.parse_args([]))) == set(main._CONVERSATION_ARGUMENTS)


def test_conversation_flags_are_rejected_on_a_sql_dataset() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-name")
    main._add_conversation_arguments(parser)
    args = parser.parse_args(["--database-name", "bird", "--retrieval", "agent"])
    with pytest.raises(SystemExit):
        main._reject_conversation_flags(parser, args)


def test_defaulted_conversation_flags_are_allowed_on_a_sql_dataset() -> None:
    """Only a flag the caller actually set is an error.

    Every flag in the group has a default, so an untouched namespace carries
    all of them on every SQL run; rejecting those would reject every run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-name")
    main._add_conversation_arguments(parser)
    args = parser.parse_args(["--database-name", "bird"])
    main._reject_conversation_flags(parser, args)


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> list[tuple[str, dict]]:
    """Drive the real ``main`` with the three expensive stages recorded.

    Through the command line rather than a hand-built namespace, so the flags,
    the routing and the stages are checked together. A namespace assembled here
    would pass whatever this file believed the parser produces.
    """
    calls: list[tuple[str, dict]] = []
    for name in (
        "stage_conversation_ingest",
        "stage_conversation_semantic",
        "stage_conversation_benchmark",
    ):
        monkeypatch.setattr(
            main,
            name,
            lambda _name=name, **kwargs: calls.append((_name, kwargs)),
            raising=True,
        )
    monkeypatch.setattr(
        "auto_ontology_eval.env.load_env", lambda *a, **k: None, raising=True
    )
    main.main(["--database-name", "beam", "--skip-migrate", *argv])
    return calls


def test_conversations_limit_the_ingest_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise targeting one conversation re-ingests and resets all twenty."""
    calls = _run(
        monkeypatch, "--conversations", "2, 5", "--skip-semantic", "--skip-eval"
    )
    assert calls == [
        (
            "stage_conversation_ingest",
            {"scale": "100K", "ids": ["2", "5"], "reset": True},
        )
    ]


def test_conversations_limit_the_semantic_stage_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same reason as the ingest: rebuilding twenty graphs to measure two is
    hours of model calls, and the nineteen it did not ask for are wiped first.
    """
    calls = _run(monkeypatch, "--conversations", "2, 5", "--skip-ingest", "--skip-eval")
    assert calls == [
        (
            "stage_conversation_semantic",
            {"setup": BEAM, "scale": "100K", "ids": ["2", "5"]},
        )
    ]


def test_the_three_stages_run_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest before semantic before eval, since each reads what the last wrote.

    The order is the measurement: a graph built before the turns it describes
    is a graph over the previous run's turns, and nothing downstream would say
    so.
    """
    calls = _run(monkeypatch)
    assert [name for name, _ in calls] == [
        "stage_conversation_ingest",
        "stage_conversation_semantic",
        "stage_conversation_benchmark",
    ]


def test_an_unknown_name_is_not_mistaken_for_a_sql_dataset() -> None:
    """The dispatch asks only "is this conversational?", so without this an
    unknown name takes the SQL path and spends ingest and semantic before
    anything notices there is no such dataset.
    """
    with pytest.raises(SystemExit):
        main._parse_args(["--database-name", "bidr"])


def test_unknown_name_error_names_both_pipelines() -> None:
    """A message listing only the SQL datasets is how ``beam`` stays unfindable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-name")
    args = parser.parse_args(["--database-name", "nope"])
    with pytest.raises(SystemExit):
        main._reject_unknown_dataset(parser, args)


def test_every_known_name_is_accepted() -> None:
    """Both registries, so a dataset added to one and not the other is caught."""
    from auto_ontology_eval.gym.tasks import DATASETS

    for name in [*DATASETS, *main.CONVERSATION_BENCHMARKS]:
        main._parse_args(["--database-name", name])


def test_a_sql_name_runs_no_conversation_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The routing is the guard: a SQL dataset must not reach these at all."""
    calls: list[str] = []
    for name in (
        "stage_conversation_ingest",
        "stage_conversation_semantic",
        "stage_conversation_benchmark",
    ):
        monkeypatch.setattr(
            main, name, lambda _n=name, **k: calls.append(_n), raising=True
        )
    for name in ("stage_ingest", "stage_semantic", "stage_benchmark"):
        monkeypatch.setattr(main, name, lambda *a, **k: None, raising=True)
    monkeypatch.setattr(
        "auto_ontology_eval.env.load_env", lambda *a, **k: None, raising=True
    )
    main.main(["--database-name", "bird", "--skip-migrate"])
    assert calls == []
