# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline entry point.

Runs the full evaluation pipeline for a dataset, in order, after first
migrating the Auto Ontology catalog if it is behind the installed Auto Ontology
(``--skip-migrate`` turns that off):

1. Ingest      source DB schema -> Postgres + pgvector (``CONNECTION_STRINGS``)
2. Semantic    compile the semantic layer (``ingestion.semantic``)
3. Benchmark   NeMo Gym: schema-only control vs the ontology-grounded agent,
               rollouts -> ``runs/{control,treatment}/<db>.jsonl``, delta printed

All stages import the Auto Ontology dependency installed by ``uv sync``::

    uv run python main.py --database-name wideworldimporters

Any stage can be skipped with ``--skip-ingest`` / ``--skip-semantic`` /
``--skip-eval`` (skipped stages don't import their deps).

The benchmark runs ``--eval-workers`` questions concurrently per arm and can be
restricted with ``--start-index`` / ``--end-index`` / ``--limit``::

    uv run python main.py --database-name bird \
        --skip-ingest --skip-semantic --limit 10 --eval-workers 2

Conversation benchmarks
-----------------------

A ``--database-name`` listed in :data:`CONVERSATION_BENCHMARKS` runs the same
three stages over transcripts instead of tables::

    uv run python main.py --database-name beam --scale 100K --eval-workers 8

Ingest writes the turns, the semantic stage builds one conversation graph per
transcript, and the eval retrieves, answers and scores -- obeying
``--skip-ingest`` / ``--skip-semantic`` / ``--skip-eval`` exactly as the SQL
side does. Both sets of stage bodies are in this module, side by side, so that
what the two pipelines share and where they diverge can be read rather than
inferred. What a benchmark is *scored* with is not here: those settings live
with the benchmark, BEAM's in ``auto_ontology_eval/beam/cognee/config.py``,
beside the judge and prompts they belong to.

What the sides do not share is the shape of the measurement: no schema, no
connectors, no arms, and no NeMo Gym. The Gym arms exist to compare two SQL
agents on exec-match, while a conversation benchmark scores a continuous rubric
mean over a retrieve-and-read loop -- which would need its own resources server,
task builder and reward function to express.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger("pipeline")

#: The ``--database-name`` values that run over transcripts rather than tables.
#:
#: Just the names, so that deciding which pipeline a run belongs to costs
#: nothing: the setup behind each one lives with its benchmark and is imported
#: only once the name is known to be here. ``beam/cognee/config.py`` reads
#: cognee's prompts off disk at import, and a BIRD run should not pay for that
#: to find out it is not BEAM.
CONVERSATION_BENCHMARKS = ("beam",)


def is_conversation_benchmark(database_name: str) -> bool:
    """Whether ``--database-name`` routes to the conversation stages."""
    return database_name in CONVERSATION_BENCHMARKS


def setup_for(database_name: str) -> "PublishedSetup | None":
    """The published setup *database_name* is measured with, or ``None``.

    Each benchmark's settings live in its own directory rather than in a
    registry here, because they are the benchmark's own: BEAM's are cognee's
    four choices, next to the judge and the prompts they go with.
    """
    if database_name == "beam":
        from auto_ontology_eval.beam.cognee.config import BEAM

        return BEAM
    return None


if TYPE_CHECKING:
    from auto_ontology_eval.beam.cognee.config import PublishedSetup

# Where each arm's rollouts go: runs/control/ and runs/treatment/.
RUN_LABEL = {"schema_only": "control", "auto_ontology": "treatment"}

# The --database-name values that route to the conversation stages instead of
# the SQL ones live in auto_ontology_eval/beam/cognee/config.py: a registry of their own
# rather than entries in gym/tasks.py's DATASETS, because every field of a
# DatasetSpec -- dialect, db_root -- is meaningless for a transcript.


def _banner(step: int, title: str) -> None:
    sep = "=" * 70
    logger.info("%s", sep)
    logger.info("STAGE %d: %s", step, title)
    logger.info("%s", sep)


def _connection_strings() -> list[str]:
    """Return non-empty entries from the comma-separated ``CONNECTION_STRINGS`` env var."""
    return [
        s.strip()
        for s in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if s.strip()
    ]


def stage_ingest(benchmark_name: str, *, reset: bool = True) -> None:
    """Ingest source schemas and apply this benchmark's enrichment artifacts."""
    _banner(1, "Ingest (source DB -> Postgres + pgvector)")
    connection_strings = _connection_strings()
    if not connection_strings:
        raise EnvironmentError(
            "CONNECTION_STRINGS is not set. Add it to your .env, e.g.:\n\n"
            "    CONNECTION_STRINGS=postgresql://user:password@host:5432/wideworldimporters"
        )

    from auto_ontology_eval.ingestion.ingest import run_ingest

    for i, connection_string in enumerate(connection_strings, start=1):
        logger.info(
            "Ingesting database %d/%d: %s",
            i,
            len(connection_strings),
            connection_string,
        )
        run_ingest(connection_string, benchmark_name, reset=reset)


def stage_semantic(benchmark_name: str) -> None:
    """Compile semantic layers and apply this benchmark's saved descriptions.

    Compilation is keyed by the *connector's* database name, which is what the
    ingest stage wrote onto the graph. That differs from the dataset folder name
    whenever a benchmark bundles differently-named databases (e.g. dataset
    ``beaverbench`` over database ``dw``), so ``--database-name`` is only used as
    a fallback when no connection strings are configured.
    """
    _banner(2, "Semantic compile (ingestion.semantic)")
    from auto_ontology_eval.ingestion.semantic import run_semantic

    connection_strings = _connection_strings()
    if connection_strings:
        from auto_ontology_eval.ingestion.ingest import database_name_for

        for i, connection_string in enumerate(connection_strings, start=1):
            database_name = database_name_for(connection_string)
            logger.info(
                "Compiling semantic layer %d/%d: %s",
                i,
                len(connection_strings),
                database_name,
            )
            run_semantic(database_name, benchmark_name)
    else:
        run_semantic(benchmark_name)


# Hydra path to the ontology server's config block. The leading "+" is required:
# gym reserves its own top-level keys with setdefault(KEY, False), so a nested
# path under one of them is "not in struct" until it is added rather than set.
_ONTOLOGY_CFG = (
    "+auto_ontology_sql_resources_server.resources_servers.auto_ontology_sql"
)


def _ontology_arm_overrides(
    *,
    concurrency: int,
    sql_examples_csv: str | None,
    sql_examples_k: int,
    sql_examples_min_score: float | None,
    value_anchors_csv: str | None,
) -> list[str]:
    """Hydra overrides for the ontology arm.

    ``max_concurrency`` is always sent. The server gates agent runs behind its
    own semaphore, which defaults to 3, so raising only ``--eval-workers``
    leaves the arm throttled at 3 with nothing to say so -- the run just takes
    longer than it should and looks like the agent is slow.
    """
    overrides = [f"{_ONTOLOGY_CFG}.max_concurrency={concurrency}"]
    if sql_examples_csv:
        overrides += [
            f"{_ONTOLOGY_CFG}.sql_examples_csv={sql_examples_csv}",
            f"{_ONTOLOGY_CFG}.sql_examples_k={sql_examples_k}",
        ]
        if sql_examples_min_score is not None:
            overrides.append(
                f"{_ONTOLOGY_CFG}.sql_examples_min_score={sql_examples_min_score}"
            )
    if value_anchors_csv:
        overrides.append(f"{_ONTOLOGY_CFG}.value_anchors_csv={value_anchors_csv}")
    return overrides


def _jsonl_task_ids(path: Path) -> set[str] | None:
    """Return task IDs from a complete JSONL file, or None when unusable."""
    if not path.is_file():
        return None
    task_ids: set[str] = set()
    try:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record.get("task_id")
                if task_id is None:
                    task_id = record.get("id")
                if task_id is None:
                    return None
                task_ids.add(str(task_id))
    except (OSError, json.JSONDecodeError):
        return None
    return task_ids or None


def _control_can_be_reused(control_output: Path, control_tasks: Path) -> bool:
    """Reuse control only when it matches the currently built task corpus."""
    output_ids = _jsonl_task_ids(control_output)
    task_ids = _jsonl_task_ids(control_tasks)
    if output_ids is None or output_ids != task_ids:
        return False

    digest = hashlib.sha256(control_tasks.read_bytes()).hexdigest()
    manifest = control_output.with_suffix(control_output.suffix + ".tasks.sha256")
    if manifest.is_file():
        return manifest.read_text().strip() == digest

    # Adopt a legacy output once when its IDs match exactly. From then on the
    # task-content digest catches changed questions, schemas, evidence, or prompts
    # even when their IDs remain unchanged.
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(digest + "\n")
    return True


def _record_control_tasks(control_output: Path, control_tasks: Path) -> None:
    digest = hashlib.sha256(control_tasks.read_bytes()).hexdigest()
    manifest = control_output.with_suffix(control_output.suffix + ".tasks.sha256")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(digest + "\n")


def stage_benchmark(
    *,
    database_name: str,
    concurrency: int = 3,
    limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    run_dir: Path | None = None,
    sql_examples_csv: str | None = None,
    sql_examples_k: int = 5,
    sql_examples_min_score: float | None = None,
    value_anchors_csv: str | None = None,
    reuse_control: bool = False,
    evidence: bool = True,
) -> None:
    """Run the NeMo Gym benchmark: both arms over the dataset, then compare.

    Replaces the old single-arm retrieval eval. That measured how well the agent
    does; this measures how much the ontology is worth, by running a schema-only
    control over the same questions and databases and reporting the delta.

    Writes rollouts to ``runs/control/`` and ``runs/treatment/``. Pass
    *run_dir* to keep each run's rollouts instead of overwriting the previous
    one.

    *sql_examples_csv* and *value_anchors_csv* attach the question-keyed prompt
    inputs to the ontology arm. The control arm cannot be given an equivalent,
    so a delta measured with them set covers those inputs **as well as** the
    ontology grounding -- which is why they are off unless asked for.

    *evidence* ``False`` leaves each question's evidence out of both arms, as
    the BEAVER leaderboard's Setting 0 does.
    """
    from auto_ontology_eval.gym.compare import compare
    from auto_ontology_eval.gym.tasks import (
        DATASETS,
        SERVER_FOR_ARM,
        build_records,
        task_file,
        write_jsonl,
    )

    _banner(3, "NeMo Gym benchmark (schema-only control vs ontology)")
    root = Path(__file__).resolve().parent
    if database_name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise SystemExit(f"No benchmark dataset {database_name!r}; known: {known}")

    for arm in SERVER_FOR_ARM:
        built = task_file(root, arm)
        # Gym's collate step writes <dataset>_metrics.json and
        # <dataset>_prepare.jsonl beside the task file and refuses to run when a
        # stale one disagrees with freshly built tasks ("Found conflicting
        # aggregate metrics"). Rebuilding the tasks without clearing these fails
        # the *next* run, not this one, which is a confusing place to land.
        data_dir = built.parent
        # Named after the task file (tasks.jsonl), not the dataset -- Gym derives
        # these from the jsonl it collates.
        for stale in (
            data_dir / "tasks_metrics.json",
            data_dir / "tasks_prepare.jsonl",
            data_dir / "tasks_metrics_conflict.json",
        ):
            if stale.exists():
                logger.info("Removing stale %s", stale.relative_to(root))
                stale.unlink()
        count = write_jsonl(
            build_records(
                DATASETS[database_name],
                root,
                arm,
                evidence=evidence,
                limit=limit,
                start_index=start_index,
                end_index=end_index,
            ),
            built,
        )
        logger.info("Built %d %s tasks -> %s", count, arm, built.relative_to(root))
        if not count:
            raise SystemExit(
                f"No {database_name!r} questions in the requested range "
                "(--start-index / --end-index / --limit)."
            )

    outputs = {}
    base = Path(run_dir) if run_dir else Path("runs")
    for arm, server in SERVER_FOR_ARM.items():
        label = RUN_LABEL[arm]
        out = base / label / f"{database_name}.jsonl"
        outputs[label] = out
        if label == "control" and reuse_control:
            control_tasks = task_file(root, "schema_only")
            if _control_can_be_reused(out, control_tasks):
                logger.info(
                    "Reusing control arm output with matching task IDs: %s", out
                )
                continue
            if out.exists():
                logger.info(
                    "Existing control output does not match current tasks; rerunning: %s",
                    out,
                )
        env = {**os.environ, "GYM_CONCURRENCY": str(concurrency)}
        if label == "treatment":
            # Gym's policy call here is discarded, but on the agent's model
            # (REASONING_MODEL) it eats the rate limit and Gym's 3 retries can
            # fail the question before the agent runs. The control arm keeps
            # GYM_MODEL: there the policy model's answer is the result.
            policy_model = os.environ.get("TREATMENT_POLICY_MODEL") or os.environ.get(
                "NON_REASONING_MODEL"
            )
            if policy_model:
                env["GYM_MODEL"] = policy_model

        cmd = ["./scripts/run_gym_arm.sh", server, str(out)]
        # The treatment arm ignores the policy model's output, so its call is
        # capped rather than left to compete for the same rate limit.
        if label == "treatment":
            cmd += ["--max-output-tokens", "1024"]
            cmd += _ontology_arm_overrides(
                concurrency=concurrency,
                sql_examples_csv=sql_examples_csv,
                sql_examples_k=sql_examples_k,
                sql_examples_min_score=sql_examples_min_score,
                value_anchors_csv=value_anchors_csv,
            )
        logger.info("Running %s arm: %s", label, " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True, env=env)
        if label == "control":
            _record_control_tasks(out, task_file(root, "schema_only"))

    logger.info("Comparing arms")
    status = compare(root / outputs["control"], root / outputs["treatment"])
    if status:
        raise SystemExit(status)


# ---------------------------------------------------------------------------
# The same three stages, built out of transcripts instead of tables.
#
# Separate functions rather than branches inside the three above: a conversation
# ingest shares no argument with a SQL one -- no connection string, no schema,
# no dialect -- and a stage that took the union of both would be a stage whose
# signature described neither.
# ---------------------------------------------------------------------------

#: Every flag :func:`_add_conversation_arguments` registers, by ``dest``.
#:
#: Written out rather than read back off the parser, because the only public
#: way to ask argparse what a group holds is to parse something. A test keeps
#: it honest, so a flag added below and forgotten here is a failure rather
#: than a flag that goes quietly unchecked on a SQL run.
_CONVERSATION_ARGUMENTS = ("scale", "conversations", "retrieval")


def benchmark_names() -> str:
    """The conversation benchmark names, for help text."""
    return ", ".join(repr(name) for name in CONVERSATION_BENCHMARKS)


def _add_conversation_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the conversation benchmarks' flags as their own group.

    Only the flags that vary between runs: which tier, which transcripts,
    which arm, where output lands. Everything that defines the measurement is
    in the benchmark's own config instead, so that forgetting an argument
    cannot quietly score a run with an instrument no published figure was
    measured on.
    """
    group = parser.add_argument_group(
        f"Conversation benchmarks ({benchmark_names()} only)",
        "Ingest writes the turns, the semantic stage builds one graph per "
        "transcript, and the eval retrieves, answers and scores with a rubric "
        "judge -- the same three stages and skip flags as a SQL run. Which "
        "judge, which reader, how many turns they see and how many rounds are "
        "scored are the chosen benchmark's own and are not options here -- see "
        "auto_ontology_eval/beam/cognee/config.py.",
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


def _reject_unknown_dataset(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Fail on a name that is neither a conversation benchmark nor a dataset.

    The dispatch below asks only whether a name is conversational, so anything
    that is not falls through to the SQL stages -- which means a typo is not a
    typo, it is a SQL run that ingests, compiles and only then discovers there
    was never a dataset by that name. Two stages and up to half an hour to
    report a misspelling.

    Checked here rather than in ``stage_benchmark``, which keeps its own copy of
    this for callers that reach it directly, and listing both pipelines because
    a message naming only the SQL datasets is how ``beam`` stays undiscoverable.
    """
    if is_conversation_benchmark(args.database_name):
        return

    from auto_ontology_eval.gym.tasks import DATASETS

    if args.database_name not in DATASETS:
        known = ", ".join(sorted([*DATASETS, *CONVERSATION_BENCHMARKS]))
        parser.error(f"unknown --database-name {args.database_name!r}; known: {known}")


def _reject_conversation_flags(
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
        for name in _CONVERSATION_ARGUMENTS
        if getattr(args, name, None) != parser.get_default(name)
    )
    if used:
        parser.error(
            f"{', '.join(used)} is only for {benchmark_names()}, "
            f"not {args.database_name}"
        )


def stage_conversation_ingest(
    *, scale: str, ids: Sequence[str] | None = None, reset: bool = True
) -> None:
    """Build one conversation memory per transcript.

    The counterpart of the SQL :func:`stage_ingest`: there is no source
    database and no schema, only transcripts on disk. Each conversation becomes
    its own memory scope so a question about one cannot retrieve another, which
    is the pooling the benchmark is built to catch.
    """
    _banner(1, f"BEAM ingest (conversations -> Postgres + pgvector, {scale})")
    from auto_ontology_eval.beam.run import ingest_scale

    ingest_scale(scale, ids=ids, reset=reset)


def stage_conversation_semantic(
    *, setup: "PublishedSetup", scale: str, ids: Sequence[str] | None = None
) -> None:
    """Build one semantic graph per transcript, over the turns just ingested.

    The counterpart of the SQL :func:`stage_semantic`, and the reason this
    pipeline has three stages rather than two: what the eval retrieves from is
    the graph, so a run that ingests and scores without this step scores
    whatever the last build left behind.

    One rebuild per conversation, each wiping its own scope first. That is the
    expensive stage here as it is on the SQL side -- tens of minutes per
    transcript -- so ``--skip-semantic`` is the flag to reach for when the
    graphs are already built and only the reading changed.
    """
    _banner(2, f"Semantic compile (conversation graphs, {setup.corpus} {scale})")
    from auto_ontology_eval.beam.run import conversation_ids
    from auto_ontology_eval.ingestion.conversation_semantic import (
        run_conversation_semantic,
    )

    targets = list(ids) if ids is not None else conversation_ids(scale)
    for position, conversation_id in enumerate(targets, start=1):
        logger.info(
            "Building graph %d/%d: conversation %s",
            position,
            len(targets),
            conversation_id,
        )
        result = run_conversation_semantic(setup.corpus, conversation_id, scale)
        logger.info(
            "conversation %s: %d nodes, %d knowledge edges, %.0fs",
            result.conversation_id,
            result.nodes,
            result.knowledge_edges,
            result.seconds,
        )


def stage_conversation_benchmark(
    *,
    setup: "PublishedSetup",
    scale: str,
    concurrency: int = 4,
    limit: int | None,
    conversations: Sequence[str] | None = None,
    retrieval: str = "vector",
    results_dir: Path | None = None,
) -> None:
    """Answer the benchmark's probing questions, then score them.

    One call per round, each retrieving, reading and judging; the score is the
    mean. ``run_rounds`` does both halves itself, including at ``rounds=1``, so
    there is nothing here to split them apart for.

    *setup* carries the judge, the reader, the retrieval budget and the number
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
    from auto_ontology_eval.beam.run import RESULTS_DIR, run_rounds

    if results_dir is None:
        results_dir = RESULTS_DIR

    _banner(3, f"BEAM, {setup.rounds} round(s) ({scale})")
    run_rounds(
        scale=scale,
        rounds=setup.rounds,
        tag=retrieval,
        ids=conversations,
        top_k=setup.top_k,
        reader_model=setup.reader_model,
        judge_model=setup.judge_model,
        workers=concurrency,
        limit=limit,
        retrieval=retrieval,
        results_dir=results_dir,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auto-ontology-eval-pipeline",
        description=(
            "Run the full evaluation pipeline: ingest -> semantic compile -> "
            "NeMo Gym benchmark."
        ),
    )
    parser.add_argument(
        "--database-name",
        required=True,
        help="Dataset / database name. "
        "Selects datasets/<name>/evaluation.json; per-question db_id routes the "
        f"connector. The conversation benchmarks are {benchmark_names()}, which "
        "take the --scale and other options below instead; each names its own "
        "judge, reader and retrieval budget in auto_ontology_eval/beam/cognee/config.py "
        "rather than taking them from here.",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Skip the ingest stage."
    )
    parser.add_argument(
        "--skip-migrate",
        action="store_true",
        help=(
            "Don't migrate the Auto Ontology catalog to the installed Auto Ontology's "
            "schema first. Use when POSTGRES_DATABASE is shared with another deployment."
        ),
    )
    parser.add_argument(
        "--skip-semantic", action="store_true", help="Skip the semantic-compile stage."
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help=("Skip the benchmark stage, e.g. to only ingest and compile."),
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help=(
            "Concurrency for each benchmark arm (default: 1). Size it against "
            "the model endpoint's rate limit, not CPU."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First question index the benchmark runs (0-based, default: 0).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Stop before this question index (default: run to the end).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many questions from the dataset.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Write benchmark rollouts under this directory instead of runs/. "
            "Use it to keep a long run's output, which runs/ overwrites."
        ),
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help=(
            "Keep existing vectors instead of clearing them before re-ingest. "
            "Applies to both pipelines: a SQL run keeps the catalog rows, a "
            "conversation run keeps each scope's turns. Only safe when the "
            "source and the chunk text are both unchanged; otherwise a dropped "
            "table or a deleted turn keeps answering searches."
        ),
    )
    _add_conversation_arguments(parser)
    parser.add_argument(
        "--reuse-control",
        action="store_true",
        help=(
            "Reuse an existing schema-only control output when its task IDs exactly "
            "match the current task set. Run the control when no matching output exists."
        ),
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help=(
            "Leave BEAVER's evidence (its domain-knowledge hints) out of both arms, "
            "as the BEAVER leaderboard's Setting 0 does. beaverbench only."
        ),
    )
    extra_prompts = parser.add_argument_group(
        "extra prompts (ontology arm only)",
        "Question-keyed inputs attached to the ontology arm's payload. The "
        "control arm cannot be given an equivalent, so a delta measured with "
        "these set covers the inputs as well as the ontology grounding -- say "
        "so when reporting it.",
    )
    extra_prompts.add_argument(
        "--sql-examples-csv",
        default=None,
        help="CSV of question-keyed SQL exemplars, e.g. "
        "inputs/extra_prompts/bird/predicted_structural_exemplars.csv",
    )
    extra_prompts.add_argument(
        "--sql-examples-k",
        type=int,
        default=5,
        help="Exemplars per question (default: 5). Ignored without --sql-examples-csv.",
    )
    extra_prompts.add_argument(
        "--sql-examples-min-score",
        type=float,
        default=None,
        help=(
            "Withhold exemplars from questions whose best match scores below "
            "this. Ignored without --sql-examples-csv."
        ),
    )
    extra_prompts.add_argument(
        "--value-anchors-csv",
        default=None,
        help="CSV of question-keyed value anchors, e.g. "
        "inputs/extra_prompts/bird/value_anchors.csv",
    )
    args = parser.parse_args(argv)
    # Checked here, not when the tasks are built: that comes after the ingest
    # and the semantic compile.
    if args.no_evidence:
        from auto_ontology_eval.gym.tasks import EVIDENCE_OPTIONAL

        if args.database_name not in EVIDENCE_OPTIONAL:
            only = ", ".join(sorted(EVIDENCE_OPTIONAL))
            parser.error(f"--no-evidence is only for {only}, not {args.database_name}")
    _reject_conversation_flags(parser, args)
    _reject_unknown_dataset(parser, args)
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    database_name = args.database_name

    from auto_ontology_eval.env import load_env

    load_env()

    if not args.skip_migrate:
        from auto_ontology_eval.catalog_migrations import ensure_current

        ensure_current()

    if is_conversation_benchmark(database_name):
        setup = setup_for(database_name)
        assert setup is not None  # CONVERSATION_BENCHMARKS and setup_for agree
        conversations = (
            [c.strip() for c in args.conversations.split(",") if c.strip()]
            if args.conversations
            else None
        )
        if not args.skip_ingest:
            stage_conversation_ingest(
                scale=args.scale, ids=conversations, reset=not args.no_reset
            )
        if not args.skip_semantic:
            stage_conversation_semantic(
                setup=setup, scale=args.scale, ids=conversations
            )
        if not args.skip_eval:
            stage_conversation_benchmark(
                setup=setup,
                scale=args.scale,
                concurrency=args.eval_workers,
                # The arm names the output and --run-dir separates one run of
                # it from the next, as RUN_LABEL and --run-dir do on the SQL
                # side. Nothing is named by hand, so a run cannot be filed
                # under a label that describes a configuration it was not
                # produced with.
                results_dir=Path(args.run_dir) if args.run_dir else None,
                limit=args.limit,
                conversations=conversations,
                retrieval=args.retrieval,
            )
        return

    if not args.skip_ingest:
        stage_ingest(database_name, reset=not args.no_reset)

    if not args.skip_semantic:
        stage_semantic(database_name)

    if not args.skip_eval:
        stage_benchmark(
            database_name=database_name,
            concurrency=args.eval_workers,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            run_dir=args.run_dir,
            sql_examples_csv=args.sql_examples_csv,
            sql_examples_k=args.sql_examples_k,
            sql_examples_min_score=args.sql_examples_min_score,
            value_anchors_csv=args.value_anchors_csv,
            reuse_control=args.reuse_control,
            evidence=not args.no_evidence,
        )


if __name__ == "__main__":
    main()
