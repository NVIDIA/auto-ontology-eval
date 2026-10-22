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

BEAM
----

``--database-name beam`` runs the conversational-memory benchmark instead.
Ingest builds one memory per conversation from
``conversation/data/beam/<scale>/``, semantic has nothing to compile and is
skipped, and the benchmark runs in two passes: answer, then judge::

    uv run python main.py --database-name beam --scale 100K --eval-workers 8

    # or one pass at a time, to re-judge without re-answering
    uv run python main.py --database-name beam --skip-ingest --stage answer
    uv run python main.py --database-name beam --skip-ingest --stage judge

It does not go through NeMo Gym. The Gym arms exist to compare two SQL agents on
exec-match; BEAM scores a continuous rubric mean over a retrieval-and-read loop,
which is a different shape of measurement and would need its own resources
server, task builder and reward function to express. Results land in
``conversation/beam_benchmark/results/<scale>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("pipeline")

# Where each arm's rollouts go: runs/control/ and runs/treatment/.
RUN_LABEL = {"schema_only": "control", "auto_ontology": "treatment"}

# --database-name value that routes to the BEAM stages instead of the SQL ones.
# A sentinel rather than an entry in gym/tasks.py's DATASETS, because every
# field of a DatasetSpec -- dialect, db_root -- is meaningless for a transcript.
BEAM = "beam"


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


def stage_ingest(benchmark_name: str) -> None:
    """Ingest source schemas and apply this benchmark's enrichment artifacts."""
    _banner(1, "Ingest (source DB -> Postgres + pgvector)")
    connection_strings = _connection_strings()
    if not connection_strings:
        raise EnvironmentError(
            "CONNECTION_STRINGS is not set. Add it to your .env, e.g.:\n\n"
            "    CONNECTION_STRINGS=postgresql://user:password@host:5432/wideworldimporters"
        )

    from ontology_sql_eval.ingestion.ingest import run_ingest

    for i, connection_string in enumerate(connection_strings, start=1):
        logger.info(
            "Ingesting database %d/%d: %s",
            i,
            len(connection_strings),
            connection_string,
        )
        run_ingest(connection_string, benchmark_name)


def stage_semantic(benchmark_name: str) -> None:
    """Compile semantic layers and apply this benchmark's saved descriptions.

    Compilation is keyed by the *connector's* database name, which is what the
    ingest stage wrote onto the graph. That differs from the dataset folder name
    whenever a benchmark bundles differently-named databases (e.g. dataset
    ``beaverbench`` over database ``dw``), so ``--database-name`` is only used as
    a fallback when no connection strings are configured.
    """
    _banner(2, "Semantic compile (ingestion.semantic)")
    from ontology_sql_eval.ingestion.semantic import run_semantic

    connection_strings = _connection_strings()
    if connection_strings:
        from ontology_sql_eval.ingestion.ingest import database_name_for

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


def stage_beam_ingest(*, scale: str, reset: bool = True) -> None:
    """Build one conversation memory per BEAM transcript.

    Replaces ``stage_ingest`` for BEAM: there is no source database and no
    schema, only transcripts on disk. Each conversation becomes its own memory
    scope so a question about one cannot retrieve another, which is the pooling
    the benchmark is built to catch.
    """
    _banner(1, f"BEAM ingest (conversations -> Postgres + pgvector, {scale})")
    from conversation.beam_benchmark.run import ingest_scale

    ingest_scale(scale, reset=reset)


def stage_beam_benchmark(
    *,
    scale: str,
    concurrency: int = 4,
    top_k: int,
    widen: int,
    reader_model: str | None,
    judge_model: str | None,
    tag: str,
    limit: int | None,
    node_k: int = 8,
    path_k: int = 10,
    node_turns: int = 10,
    stage: str = "both",
    conversations: Sequence[str] | None = None,
    rounds: int = 1,
    reader_prompt: str = "beam",
    retrieval: str = "vector",
    judge_prompt: str = "beam",
) -> None:
    """Answer BEAM's probing questions, then score them.

    Two passes rather than one. *stage* selects which run: ``answer`` retrieves
    and reads, ``judge`` scores the answers file, ``both`` does them in order.
    Keeping them apart means a judge change costs no reader calls and a
    retrieval change costs no judge calls, and it leaves an answers file that
    can be read on its own when a score looks wrong.

    Scoring uses the paper's judge by default, so the result can be put next to
    the published 0.323 -- see ``conversation/beam_benchmark/README.md`` for what
    that number is, what still differs, and which arms exist.

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

    if rounds > 1:
        if stage != "both":
            raise SystemExit(
                "--rounds needs --stage both: a round is one answer pass and one "
                "judge pass, so there is nothing to repeat about half of it."
            )
        _banner(3, f"BEAM, {rounds} rounds ({scale})")
        run_rounds(
            scale=scale,
            rounds=rounds,
            tag=tag,
            ids=conversations,
            top_k=top_k,
            widen=widen,
            node_k=node_k,
            path_k=path_k,
            node_turns=node_turns,
            reader_model=reader_model,
            judge_model=judge_model,
            workers=concurrency,
            limit=limit,
            reader_prompt=reader_prompt,
            retrieval=retrieval,
            flavour=judge_prompt,
        )
        return

    if stage in ("answer", "both"):
        _banner(3, f"BEAM answers (retrieve -> read, {scale})")
        answer_questions(
            scale=scale,
            ids=conversations,
            top_k=top_k,
            widen=widen,
            node_k=node_k,
            path_k=path_k,
            node_turns=node_turns,
            reader_model=reader_model,
            workers=concurrency,
            tag=tag,
            limit=limit,
            reader_prompt=reader_prompt,
            retrieval=retrieval,
        )

    if stage in ("judge", "both"):
        _banner(4, f"BEAM scoring (rubric judge, {scale})")
        judge_answers(
            scale=scale,
            tag=tag,
            judge_model=judge_model,
            workers=concurrency,
            limit=limit if stage == "judge" else None,
            flavour=judge_prompt,
        )


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
    """
    from ontology_sql_eval.gym.compare import compare
    from ontology_sql_eval.gym.tasks import (
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
        env = {**os.environ, "GYM_CONCURRENCY": str(concurrency)}
        cmd = ["./scripts/run_gym_arm.sh", server, str(out)]
        # The treatment arm ignores the policy model's output, so its call is
        # capped rather than left to compete for the same rate limit.
        if label == "treatment":
            cmd += ["--max-output-tokens", "16"]
            cmd += _ontology_arm_overrides(
                concurrency=concurrency,
                sql_examples_csv=sql_examples_csv,
                sql_examples_k=sql_examples_k,
                sql_examples_min_score=sql_examples_min_score,
                value_anchors_csv=value_anchors_csv,
            )
        logger.info("Running %s arm: %s", label, " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True, env=env)

    logger.info("Comparing arms")
    status = compare(root / outputs["control"], root / outputs["treatment"])
    if status:
        raise SystemExit(status)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ontology-sql-eval-pipeline",
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
        f"connector. Use {BEAM!r} for the conversational-memory benchmark, which "
        "takes the --scale and other BEAM options below instead.",
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
    beam = parser.add_argument_group(
        "BEAM (--database-name beam only)",
        "The conversational-memory benchmark. Ingest builds one memory per "
        "transcript; the benchmark stage retrieves, answers and scores with the "
        "paper's rubric judge.",
    )
    beam.add_argument(
        "--scale",
        default="100K",
        choices=["100K", "500K", "1M", "10M"],
        help="Conversation length tier (default: 100K, 20 conversations / 400 "
        "questions, and the only tier with a published baseline we can match "
        "question-for-question).",
    )
    beam.add_argument(
        "--stage",
        default="both",
        choices=["answer", "judge", "both"],
        help=(
            "Which half of the benchmark to run. 'answer' retrieves and reads, "
            "writing results/<scale>/<tag>_answers.jsonl; 'judge' scores that "
            "file. Split so a judge change costs no reader calls and a "
            "retrieval change costs no judge calls (default: both)."
        ),
    )
    beam.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Turns retrieved per question (default: 5, the paper's RAG depth).",
    )
    beam.add_argument(
        "--widen",
        type=int,
        default=0,
        help=(
            "Neighbouring turns pulled in around each hit (default: 0, as the "
            "paper). Raising it is a change to the method, not a tuning knob: "
            "it buys continuity on ordering questions and spends context on "
            "everything else."
        ),
    )
    beam.add_argument(
        "--reader-model",
        default=None,
        help="Model that answers from retrieved turns (default: REASONING_MODEL). "
        "Report it with the score -- across the paper's four readers the same "
        "RAG setup scored 0.269 to 0.323.",
    )
    beam.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Override the rubric judge. Defaults to the paper's "
            "gpt-4.1-mini; anything else is not the same instrument as 0.323."
        ),
    )
    beam.add_argument(
        "--conversations",
        default=None,
        help="Comma-separated conversation ids to restrict the run to, e.g. '2'. "
        "Defaults to the whole tier. Use it to match a published subset; it is "
        "not a sampling knob, since per-conversation scores span 0.374 here.",
    )
    beam.add_argument(
        "--judge-prompt",
        default="beam",
        choices=["beam", "cognee"],
        help="Which judge prompt to score with. 'beam' is the paper's, verbatim, "
        "and is what 0.323 was measured on -- it never shows the judge the "
        "question. 'cognee' is Cognee's edit, which supplies the question, snaps "
        "scores to 0/0.5/1 and scores failures 0.0 instead of raising. Use it "
        "only to compare against their 0.79 (default: beam).",
    )
    beam.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Repeat the whole answer-and-judge round this many times and report "
        "the mean with its standard deviation (default: 1). Matches Cognee's "
        "four 100K rounds. The spread it reports is run-to-run noise on a fixed "
        "question set, not a confidence interval over questions.",
    )
    beam.add_argument(
        "--retrieval",
        default="vector",
        choices=["vector", "semantic", "hybrid", "graph"],
        help="How turns are found. 'vector' is similarity over turn text. "
        "'semantic' matches the question against the nodes extracted from "
        "the whole transcript and reads the turns they cite, with the edges "
        "between them rendered above the context. 'hybrid' unions both. "
        "'graph' is 'semantic' with those edges written as paths rather than "
        "prose (default: vector).",
    )
    beam.add_argument(
        "--node-k",
        type=int,
        default=8,
        help="How many graph nodes the semantic arms match per question "
        "(default: 8). Ignored by --retrieval vector.",
    )
    beam.add_argument(
        "--path-k",
        type=int,
        default=10,
        help="How many one-hop paths the semantic arms match per question, "
        "whose endpoints join the node set (default: 10). A path carries both "
        "ends and the verb between them, so it reaches a question that names "
        "the relation and neither end. 0 turns the path route off.",
    )
    beam.add_argument(
        "--node-turns",
        type=int,
        default=10,
        help="How many turns the matched nodes and paths contribute between "
        "them (default: 10), nearest node first. This -- not --top-k -- is the "
        "turn budget on --retrieval semantic and graph, which never search the "
        "turn text: every turn the reader sees is a citation on a node. One "
        "chatty node can spend several slots, so a node matched sixth can have "
        "its turns cut off by the cap. 0 answers from the note alone.",
    )
    beam.add_argument(
        "--reader-prompt",
        default="beam",
        choices=["beam", "cognee"],
        help="Whose reader prompt to answer with. 'beam' is the paper's single "
        "prompt for all ten abilities. 'cognee' replaces it with their bare "
        "context template plus a per-ability system prompt picked from the "
        "question's label -- their setup, and note it reads the benchmark's own "
        "label at inference time (default: beam).",
    )
    beam.add_argument(
        "--tag",
        default="ontology",
        help="Names this run's output under results/<scale>/ (default: ontology). "
        "Use it to keep arms apart rather than overwriting one with the next.",
    )
    beam.add_argument(
        "--no-reset",
        action="store_true",
        help=(
            "Keep each scope's existing vectors instead of clearing them before "
            "re-ingest. Only safe when the transcripts and the chunk text are "
            "both unchanged; otherwise deleted turns keep answering searches."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    database_name = args.database_name

    from ontology_sql_eval.env import load_env

    load_env()

    if not args.skip_migrate:
        from ontology_sql_eval.catalog_migrations import ensure_current

        ensure_current()

    if database_name == BEAM:
        # No semantic stage: there is no schema to describe. Skipping it
        # silently rather than erroring on --skip-semantic, because a caller
        # passing it is right, just redundant.
        if not args.skip_ingest:
            stage_beam_ingest(scale=args.scale, reset=not args.no_reset)
        if not args.skip_eval:
            stage_beam_benchmark(
                scale=args.scale,
                concurrency=args.eval_workers,
                top_k=args.top_k,
                widen=args.widen,
                node_k=args.node_k,
                path_k=args.path_k,
                node_turns=args.node_turns,
                reader_model=args.reader_model,
                judge_model=args.judge_model,
                tag=args.tag,
                limit=args.limit,
                stage=args.stage,
                conversations=(
                    [c.strip() for c in args.conversations.split(",") if c.strip()]
                    if args.conversations
                    else None
                ),
                rounds=args.rounds,
                reader_prompt=args.reader_prompt,
                retrieval=args.retrieval,
                judge_prompt=args.judge_prompt,
            )
        return

    if not args.skip_ingest:
        stage_ingest(database_name)

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
        )


if __name__ == "__main__":
    main()
