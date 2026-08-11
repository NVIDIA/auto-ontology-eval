# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score predicted SQL with an official Spider2 evaluation suite.

Two suites are vendored, both covering the same 135 local questions:

``snow``
    Executes predictions against Snowflake (``sf_local010``). Needs
    ``snowflake_credential.json`` and network access.
``lite``
    Executes predictions against the local ``.sqlite`` files (``local010``).
    Needs no credentials, so use it to judge SQLite eval runs.

The retrieval evaluator writes one CSV row per question. Both upstream
evaluators instead expect a directory containing ``<instance_id>.sql`` files.
This adapter performs that conversion, calls the vendored official evaluator
in-process, and adds a score column to a copy of the input CSV.

Example::

    uv run python -m ontology_sql_eval.judge.spider2 \
        --input input/spider2_gpt-5.5.csv --suite lite
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPIDER2_ROOT = REPO_ROOT / "third_party" / "Spider2"
DATASETS_DIR = REPO_ROOT / "datasets" / "spider2"
DEFAULT_INPUT = REPO_ROOT / "input" / "spider2_gpt-5.5.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


@dataclass(frozen=True)
class Suite:
    """Layout differences between the vendored Spider2-Snow and -lite suites."""

    name: str
    root: Path
    gold_config: str
    score_field: str
    # The lite suite keys everything by the bare ``local010`` instance id; our
    # CSVs use the Snowflake-style ``sf_local010`` throughout.
    strip_sf_prefix: bool
    # Suffix the evaluator appends to result_dir when writing the ids of
    # correct predictions, plus the column those ids land in.
    correct_ids_suffix: str
    correct_ids_column: str

    @property
    def suite_dir(self) -> Path:
        return self.root / "evaluation_suite"

    @property
    def gold_dir(self) -> Path:
        return self.suite_dir / "gold"

    @property
    def eval_script(self) -> Path:
        return self.suite_dir / "evaluate.py"

    @property
    def metadata_jsonl(self) -> Path:
        return self.root / f"{self.root.name}.jsonl"

    def instance_id(self, question_id: str) -> str:
        """Return the id the suite uses for *question_id*."""
        if self.strip_sf_prefix and question_id.startswith("sf_"):
            return question_id[len("sf_") :]
        return question_id

    def question_id(self, instance_id: str) -> str:
        """Inverse of :meth:`instance_id` (both suites report ``sf_``-prefixed ids)."""
        if self.strip_sf_prefix and not instance_id.startswith("sf_"):
            return f"sf_{instance_id}"
        return instance_id


SNOW_SUITE = Suite(
    name="snow",
    root=SPIDER2_ROOT / "spider2-snow",
    gold_config="spider2snow_eval.jsonl",
    score_field="spider2_snow_score",
    strip_sf_prefix=False,
    correct_ids_suffix=".csv",
    correct_ids_column="output",
)

LITE_SUITE = Suite(
    name="lite",
    root=SPIDER2_ROOT / "spider2-lite",
    gold_config="spider2lite_eval.jsonl",
    score_field="spider2_lite_score",
    strip_sf_prefix=True,
    correct_ids_suffix="-ids.csv",
    correct_ids_column="instance_id",
)

SUITES = {suite.name: suite for suite in (SNOW_SUITE, LITE_SUITE)}


def _slugify(name: str) -> str:
    """Normalize a Spider2 db name into the dataset slug (mirrors the seeders)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _gold_ids(suite: Suite) -> set[str]:
    config_path = suite.gold_dir / suite.gold_config
    return {
        suite.question_id(str(json.loads(line)["instance_id"]))
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _load_predictions(
    input_path: Path, suite: Suite
) -> tuple[list[dict[str, str]], list[str]]:
    if not input_path.is_file():
        raise SystemExit(f"Spider2 eval CSV not found: {input_path}")

    # Generated SQL routinely exceeds the 128 KB default field limit.
    csv.field_size_limit(10_000_000)
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"question_id", "returned_sql"}
        missing = required - set(fieldnames)
        if missing:
            raise SystemExit(
                f"{input_path} is missing required column(s): {sorted(missing)}"
            )
        rows = list(reader)

    question_ids = [str(row["question_id"]).strip() for row in rows]
    duplicates = sorted({qid for qid in question_ids if question_ids.count(qid) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate question_id values: {duplicates}")

    unknown = sorted(set(question_ids) - _gold_ids(suite))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} question_id(s) are not in the Spider2-{suite.name} "
            f"gold config: {unknown[:10]}"
        )
    if not rows:
        raise SystemExit(f"No prediction rows found in {input_path}")
    return rows, fieldnames


def link_local_sqlite_databases(suite: Suite) -> int:
    """Expose the seeded ``.sqlite`` files where the lite evaluator looks for them.

    The evaluator resolves ``resource/databases/<db>.sqlite`` using the ``db``
    field of ``spider2-lite.jsonl`` (``Airlines``, ``Db-IMDB``), while the seeder
    stores them as ``datasets/spider2/<slug>/<slug>.sqlite``. Symlink rather
    than copy so both layouts stay in sync. Returns the number of links created.
    """
    databases_dir = suite.root / "resource" / "databases"
    databases_dir.mkdir(parents=True, exist_ok=True)

    wanted = {
        str(row["db"])
        for line in suite.metadata_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if str(row["instance_id"]).startswith("local")
    }

    created = 0
    missing: list[str] = []
    for db_name in sorted(wanted):
        slug = _slugify(db_name)
        source = DATASETS_DIR / slug / f"{slug}.sqlite"
        if not source.is_file():
            missing.append(f"{db_name} -> {source}")
            continue
        link = databases_dir / f"{db_name}.sqlite"
        if link.exists():
            continue
        if link.is_symlink():
            link.unlink()  # dangling link from a moved dataset
        link.symlink_to(source)
        created += 1

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise SystemExit(
            "Missing SQLite database(s) for the lite evaluator:\n"
            f"{details}\nRun scripts/seed_spider2_sqlite.py first."
        )
    if created:
        logger.info("Linked %d SQLite database(s) into %s", created, databases_dir)
    return created


def _validate_suite(suite: Suite) -> None:
    required = [
        suite.eval_script,
        suite.metadata_jsonl,
        suite.gold_dir / suite.gold_config,
        suite.gold_dir / "exec_result",
    ]
    if suite is SNOW_SUITE:
        required.append(suite.suite_dir / "snowflake_credential.json")
    missing = [path for path in required if not path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            f"Official Spider2-{suite.name} evaluation files are missing:\n"
            f"{paths}\nRun scripts/seed_spider2_{suite.name}.py first."
        )
    if suite is LITE_SUITE:
        link_local_sqlite_databases(suite)


def _load_evaluator(suite: Suite) -> ModuleType:
    """Import the suite's vendored ``evaluate.py`` and cache it in ``sys.modules``.

    The suites live in hyphenated directories that are not importable packages,
    so load by path under a suite-specific name; both files are called
    ``evaluate`` and would otherwise collide. Importing executes module-level
    code, so callers must already be inside :func:`_evaluator_runtime`.
    """
    module_name = f"_spider2_{suite.name}_evaluate"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, suite.eval_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the official evaluator: {suite.eval_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A half-initialized module left behind would be returned by the cache
        # lookup above on the next attempt.
        sys.modules.pop(module_name, None)
        raise
    return module


@contextmanager
def _evaluator_runtime(suite: Suite) -> Iterator[None]:
    """Give the vendored evaluator the working directory and streams it assumes.

    Upstream is written to be run as ``python evaluate.py`` from inside its own
    suite directory, and it resolves several paths against the process working
    directory rather than ``__file__``: ``../spider2-snow.jsonl``,
    ``snowflake_credential.json`` and the ``log.txt`` it tees output into. It
    also replaces ``sys.stdout``/``sys.stderr`` at import time and never puts
    them back, which would silently swallow this process's output for the rest
    of its life.
    """
    original_cwd = Path.cwd()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    os.chdir(suite.suite_dir)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        tee = sys.stdout
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if tee is not original_stdout:
            getattr(tee, "close", lambda: None)()


def run_official_spider2_judge(
    input_path: Path,
    output_path: Path,
    *,
    suite: Suite = SNOW_SUITE,
    max_workers: int = 20,
    timeout: int = 60,
) -> tuple[int, int]:
    """Run official execution evaluation and return ``(correct, total)``."""
    _validate_suite(suite)
    rows, fieldnames = _load_predictions(input_path, suite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"spider2-{suite.name}-judge-") as tmp:
        run_root = Path(tmp)
        predictions_dir = run_root / "predictions"
        predictions_dir.mkdir()

        for row in rows:
            instance_id = suite.instance_id(str(row["question_id"]).strip())
            sql = str(row.get("returned_sql") or "").strip()
            (predictions_dir / f"{instance_id}.sql").write_text(
                sql + ("\n" if sql else ""),
                encoding="utf-8",
            )

        # Mirrors the flags the upstream ``__main__`` block would parse; only
        # these attributes are read by ``evaluate_spider2sql``.
        eval_args = argparse.Namespace(
            mode="sql",
            result_dir=str(predictions_dir),
            gold_dir=str(suite.gold_dir),
            max_workers=max_workers,
            timeout=timeout,
        )
        temp_path = run_root / "temp"
        temp_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running official Spider2-%s evaluator for %d predictions",
            suite.name,
            len(rows),
        )
        with _evaluator_runtime(suite):
            _load_evaluator(suite).evaluate_spider2sql(eval_args, temp_path)

        correct_path = predictions_dir.with_name(
            predictions_dir.name + suite.correct_ids_suffix
        )
        if not correct_path.is_file():
            raise RuntimeError(
                f"Official evaluator did not produce expected file: {correct_path}"
            )
        with correct_path.open(encoding="utf-8", newline="") as handle:
            correct_ids = {
                suite.question_id(str(row[suite.correct_ids_column]).strip())
                for row in csv.DictReader(handle)
                if row.get(suite.correct_ids_column)
            }

    output_fields = fieldnames + (
        [suite.score_field] if suite.score_field not in fieldnames else []
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            row[suite.score_field] = (
                "1" if str(row["question_id"]).strip() in correct_ids else "0"
            )
            writer.writerow(row)

    correct = len(correct_ids)
    total = len(rows)
    logger.info(
        "Official Spider2-%s score: %.4f (%d/%d). Wrote %s",
        suite.name,
        correct / total,
        correct,
        total,
        output_path,
    )
    return correct, total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scored CSV path (default: output/<input stem>_<suite>_scores.csv).",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="snow",
        help="Evaluation suite: 'snow' executes on Snowflake, 'lite' on local SQLite.",
    )
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    suite = SUITES[args.suite]
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{input_path.stem}_{suite.name}_scores.csv"
    )
    correct, total = run_official_spider2_judge(
        input_path,
        output_path,
        suite=suite,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )
    print(
        f"Official Spider2-{suite.name} score: "
        f"{correct / total:.4f} ({correct}/{total})"
    )


if __name__ == "__main__":
    main()
