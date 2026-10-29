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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("pipeline")

# Where each arm's rollouts go: runs/control/ and runs/treatment/.
RUN_LABEL = {"schema_only": "control", "auto_ontology": "treatment"}


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
        if label == "control":
            _record_control_tasks(out, task_file(root, "schema_only"))

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
        "Selects datasets/<name>/evaluation.json; per-question db_id routes the connector.",
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
        from ontology_sql_eval.gym.tasks import EVIDENCE_OPTIONAL

        if args.database_name not in EVIDENCE_OPTIONAL:
            only = ", ".join(sorted(EVIDENCE_OPTIONAL))
            parser.error(f"--no-evidence is only for {only}, not {args.database_name}")
    return args


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
            reuse_control=args.reuse_control,
            evidence=not args.no_evidence,
        )


if __name__ == "__main__":
    main()
