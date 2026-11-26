# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation benchmarks' own command line and stages.

``main.py`` runs two unrelated pipelines. The SQL one ingests a database,
compiles a semantic layer and scores exec-match through NeMo Gym; this one
ingests transcripts and scores a rubric mean over a retrieve-and-read loop.
They share a process and almost nothing else -- no schema, no connectors, no
gym, no arms.

So the conversation half lives here rather than in ``main.py``, which keeps
only the dispatch. :func:`add_arguments` registers the flags and
:func:`run_stages` runs them, and adding a conversation benchmark touches this
package alone.

What is *not* here is as deliberate: the judge, the reader, the turn budget and
the number of rounds are in :mod:`conversation.benchmarks`, keyed by benchmark
name, because they belong to a published setup rather than to a run. 0.79 is
the bar, and only an entry there can say which figure a score may be set beside.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from conversation.beam_benchmark.run import RESULTS_DIR
from conversation.benchmarks import (
    BENCHMARKS,
    BenchmarkSpec,
    is_conversation_benchmark,
    spec_for,
)

logger = logging.getLogger("pipeline")

__all__ = [
    "add_arguments",
    "benchmark_names",
    "reject_on_sql_datasets",
    "run_stages",
]

#: Every flag :func:`add_arguments` registers, by ``dest``.
#:
#: Written out rather than read back off the parser, because the only public
#: way to ask argparse what a group holds is to parse something. A test keeps
#: it honest, so a flag added above and forgotten here is a failure rather
#: than a flag that goes quietly unchecked on a SQL run.
_OWN_ARGUMENTS = ("scale", "stage", "conversations", "retrieval", "no_reset")


def benchmark_names() -> str:
    """The conversation benchmark names, for help text in either module."""
    return ", ".join(repr(name) for name in BENCHMARKS)


def _banner(step: int, title: str) -> None:
    sep = "=" * 70
    logger.info("%s", sep)
    logger.info("STAGE %d: %s", step, title)
    logger.info("%s", sep)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the conversation benchmarks' flags as their own group.

    Only the flags that vary between runs: which tier, which transcripts,
    which arm, where output lands. Everything that defines the measurement is
    in :mod:`conversation.benchmarks` instead, so that forgetting an argument
    cannot quietly score a run with an instrument no published figure was
    measured on.
    """
    group = parser.add_argument_group(
        f"Conversation benchmarks ({benchmark_names()} only)",
        "Ingest builds one memory per transcript; the benchmark stage "
        "retrieves, answers and scores with a rubric judge. Which judge, "
        "which reader, how many turns they see and how many rounds are scored "
        "are the chosen benchmark's own and are not options here -- see "
        "conversation/benchmarks.py.",
    )
    group.add_argument(
        "--scale",
        default="100K",
        choices=["100K", "500K", "1M", "10M"],
        help="Conversation length tier (default: 100K, 20 conversations / 400 "
        "questions, and the only tier with a published baseline we can match "
        "question-for-question).",
    )
    group.add_argument(
        "--stage",
        default="both",
        choices=["answer", "judge", "both"],
        help=(
            "Which half of the benchmark to run. 'answer' retrieves and reads, "
            "writing results/<scale>/<arm>_answers.jsonl; 'judge' scores "
            "that "
            "file. Split so a judge change costs no reader calls and a "
            "retrieval change costs no judge calls. Only available on a "
            "benchmark scored over a single round, since half a round is not "
            "something there is anything to repeat about (default: both)."
        ),
    )
    group.add_argument(
        "--conversations",
        default=None,
        help="Comma-separated conversation ids to restrict the run to, e.g. '2'. "
        "Defaults to the whole tier. Use it to match a published subset; it is "
        "not a sampling knob, since per-conversation scores span 0.374 here.",
    )
    group.add_argument(
        "--retrieval",
        default="vector",
        choices=["vector", "agent"],
        help="How turns are found. 'vector' is one similarity search over "
        "turn text, handed to a reader that answers in one call -- the "
        "baseline, and the arm that matches the published one-retrieval "
        "protocol. 'agent' hands the question to the ReAct agent with "
        "conversation search on, which writes its own queries, reads what "
        "comes back and may search again; it reaches the ontology through "
        "that tool. It is not comparable with the published figure, which is "
        "one retrieval and one answer, and it ignores the benchmark's reader "
        "because the agent is its own (default: vector).",
    )
    group.add_argument(
        "--no-reset",
        action="store_true",
        help=(
            "Keep each scope's existing vectors instead of clearing them before "
            "re-ingest. Only safe when the transcripts and the chunk text are "
            "both unchanged; otherwise deleted turns keep answering searches."
        ),
    )


def reject_on_sql_datasets(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Fail if a conversation flag was set for a dataset that cannot use it.

    The group is registered for every run because argparse must know its flags
    before it can parse the name that decides whether they apply. Accepting
    them silently afterwards is the part worth preventing: ``--retrieval
    agent`` on a SQL dataset reads as a request to change the arm and changes
    nothing, so the run looks like a measurement of something it is not.

    The same shape as ``--no-evidence``, which is beaverbench's alone and is
    rejected here rather than ignored downstream.
    """
    if is_conversation_benchmark(args.database_name):
        return
    used = sorted(
        f"--{name.replace('_', '-')}"
        for name in _OWN_ARGUMENTS
        if getattr(args, name, None) != parser.get_default(name)
    )
    if used:
        parser.error(
            f"{', '.join(used)} is only for {benchmark_names()}, "
            f"not {args.database_name}"
        )


def stage_ingest(
    *, scale: str, ids: Sequence[str] | None = None, reset: bool = True
) -> None:
    """Build one conversation memory per transcript.

    Replaces the SQL ``stage_ingest``: there is no source database and no
    schema, only transcripts on disk. Each conversation becomes its own memory
    scope so a question about one cannot retrieve another, which is the
    pooling the benchmark is built to catch.
    """
    _banner(1, f"BEAM ingest (conversations -> Postgres + pgvector, {scale})")
    from conversation.beam_benchmark.run import ingest_scale

    ingest_scale(scale, ids=ids, reset=reset)


def stage_benchmark(
    *,
    spec: BenchmarkSpec,
    scale: str,
    concurrency: int = 4,
    limit: int | None,
    stage: str = "both",
    conversations: Sequence[str] | None = None,
    retrieval: str = "vector",
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Answer the benchmark's probing questions, then score them.

    Two passes rather than one. *stage* selects which run: ``answer`` retrieves
    and reads, ``judge`` scores the answers file, ``both`` does them in order.
    Keeping them apart means a judge change costs no reader calls and a
    retrieval change costs no judge calls, and it leaves an answers file that
    can be read on its own when a score looks wrong. A benchmark scored over
    several rounds cannot be split this way, since half a round is not
    something there is anything to repeat about.

    *spec* carries the judge, the reader, the retrieval budget and the number
    of rounds the chosen benchmark was published with, so the result can be put
    beside a published figure rather than standing alone -- 0.79 is the bar.
    None of it is a flag, for the same reason the SQL side keeps its result
    comparison in ``exec_match.matcher_for``: scoring a run with the wrong
    instrument should not be something a command line can do.

    *conversations* restricts the run to some of the tier's transcripts. It is
    for matching a published subset, not for sampling: single-conversation
    scores range over 0.374 on this tier, so a 20-question result says more
    about which transcript was picked than about the system.
    """
    from conversation.beam_benchmark.run import (
        answer_questions,
        judge_answers,
        run_rounds,
    )

    if spec.rounds > 1:
        if stage != "both":
            raise SystemExit(
                f"{spec.name} is scored over {spec.rounds} rounds and --stage "
                f"{stage} runs half of one, which there is nothing to repeat "
                "about. Use --stage both."
            )
        _banner(3, f"BEAM, {spec.rounds} rounds ({scale})")
        run_rounds(
            scale=scale,
            rounds=spec.rounds,
            tag=retrieval,
            ids=conversations,
            top_k=spec.top_k,
            reader_model=spec.reader_model,
            judge_model=spec.judge_model,
            workers=concurrency,
            limit=limit,
            retrieval=retrieval,
            results_dir=results_dir,
        )
        return

    if stage in ("answer", "both"):
        _banner(3, f"BEAM answers (retrieve -> read, {scale})")
        answer_questions(
            scale=scale,
            ids=conversations,
            top_k=spec.top_k,
            reader_model=spec.reader_model,
            workers=concurrency,
            tag=retrieval,
            limit=limit,
            retrieval=retrieval,
            results_dir=results_dir,
        )

    if stage in ("judge", "both"):
        _banner(4, f"BEAM scoring (rubric judge, {scale})")
        judge_answers(
            scale=scale,
            tag=retrieval,
            judge_model=spec.judge_model,
            workers=concurrency,
            limit=limit if stage == "judge" else None,
            results_dir=results_dir,
        )


def run_stages(database_name: str, args: argparse.Namespace) -> None:
    """Run the conversation pipeline for *database_name*.

    The whole of what ``main.py`` does once it knows the name is a conversation
    benchmark. There is no semantic stage: nothing here has a schema to
    describe, so ``--skip-semantic`` is accepted and ignored rather than
    rejected -- a caller passing it is right, just redundant.
    """
    spec = spec_for(database_name)
    if spec is None:  # pragma: no cover - the caller checks first
        raise SystemExit(f"{database_name!r} is not a conversation benchmark")

    conversations = (
        [c.strip() for c in args.conversations.split(",") if c.strip()]
        if args.conversations
        else None
    )
    if not args.skip_ingest:
        stage_ingest(scale=args.scale, ids=conversations, reset=not args.no_reset)
    if args.skip_eval:
        return
    stage_benchmark(
        spec=spec,
        scale=args.scale,
        concurrency=args.eval_workers,
        # The arm names the output and --run-dir separates one run of it from
        # the next, as RUN_LABEL and --run-dir do on the SQL side. Nothing is
        # named by hand, so a run cannot be filed under a label that describes
        # a configuration it was not produced with.
        results_dir=Path(args.run_dir) if args.run_dir else RESULTS_DIR,
        limit=args.limit,
        stage=args.stage,
        conversations=conversations,
        retrieval=args.retrieval,
    )
