# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the upstream Spider2-lite evaluation suite against our eval CSVs.

Wraps ``third_party/Spider2/spider2-lite/evaluation_suite/evaluate.py`` so
predicted SQL from ``ontology_sql_eval`` eval CSVs is scored by execution
result comparison (gold ``exec_result`` CSVs), matching the official benchmark.

See: https://github.com/xlang-ai/Spider2/tree/main/spider2-lite/evaluation_suite
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import types
from pathlib import Path

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

# Cached reference to the upstream ``evaluate.py`` module (imported lazily so
# its ``sys.stdout``-hijacking import side effect only fires when we judge).
_evaluate_module: types.ModuleType | None = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _REPO_ROOT / "third_party" / "Spider2"
_SPIDER2_LITE = _UPSTREAM_ROOT / "spider2-lite"
_EVAL_SUITE = _SPIDER2_LITE / "evaluation_suite"
_EVALUATE_PY = _EVAL_SUITE / "evaluate.py"
_GOLD_DIR = _EVAL_SUITE / "gold"
_MANIFEST_PATH = _REPO_ROOT / "datasets" / "spider2" / "manifest.json"
_SQLITE_DATABASES_DIR = _SPIDER2_LITE / "resource" / "databases"


def evaluation_suite_root() -> Path:
    return _EVAL_SUITE


def gold_dir() -> Path:
    return _GOLD_DIR


def _load_manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Spider2 manifest not found: {_MANIFEST_PATH}\n"
            "Run scripts/download_spider2.py first."
        )
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_entry(slug: str) -> dict:
    manifest = _load_manifest()
    for entry in manifest.get("databases", []):
        if entry["slug"] == slug:
            return entry
    raise KeyError(f"Unknown Spider2 slug: {slug!r}")


def question_ids_for_slug(slug: str) -> list[str]:
    """Return Spider2 ``instance_id`` values for a dataset slug."""
    return list(_manifest_entry(slug).get("question_ids", []))


def _sqlite_source_path(slug: str) -> Path:
    return _REPO_ROOT / "datasets" / "spider2" / slug / f"{slug}.sqlite"


def ensure_sqlite_symlink(slug: str) -> Path:
    """Symlink ``resource/databases/<Spider2DbName>.sqlite`` to our local file.

    ``evaluate.py`` resolves local instances via
    ``resource/databases/{metadata['db']}.sqlite``.
    """
    entry = _manifest_entry(slug)
    spider2_db_name = entry["spider2_db_name"]
    source = _sqlite_source_path(slug).resolve()
    if not source.exists():
        raise FileNotFoundError(f"SQLite database not found: {source}")

    _SQLITE_DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    link = _SQLITE_DATABASES_DIR / f"{spider2_db_name}.sqlite"
    if link.is_symlink():
        if link.resolve() == source:
            return link
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to overwrite non-symlink: {link}")

    link.symlink_to(source)
    logger.info("Linked %s -> %s", link.name, source)
    return link


def prepare_sql_submission(
    eval_csv: Path,
    submission_dir: Path,
    *,
    instance_ids: set[str] | None = None,
) -> list[str]:
    """Write ``<instance_id>.sql`` files from an eval CSV's ``returned_sql`` column."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with eval_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            instance_id = (row.get("question_id") or "").strip()
            if not instance_id:
                continue
            if instance_ids is not None and instance_id not in instance_ids:
                continue
            sql = (row.get("returned_sql") or "").strip()
            out_path = submission_dir / f"{instance_id}.sql"
            out_path.write_text(sql + ("\n" if sql and not sql.endswith("\n") else ""), encoding="utf-8")
            written.append(instance_id)

    logger.info(
        "Wrote %d SQL submission(s) to %s: %s",
        len(written),
        submission_dir,
        ", ".join(written) or "(none)",
    )
    return written


def is_snowflake_instance_id(instance_id: str) -> bool:
    """Return True for Spider2 Snowflake instance ids (``sf_bq029``, ``sf001``…)."""
    return (instance_id or "").strip().lower().startswith("sf")


def instances_are_snowflake(instance_ids: list[str] | set[str]) -> bool:
    """True when every id is a Snowflake instance (a slug maps to one dialect)."""
    ids = [i for i in instance_ids if i]
    return bool(ids) and all(is_snowflake_instance_id(i) for i in ids)


def _strip_sql_fences(sql: str) -> str:
    """Return the inner SQL from a ```sql ...``` block, else the trimmed input."""
    match = re.search(r"```sql\n(.*?)\n```", sql or "", re.DOTALL)
    return match.group(1).strip() if match else (sql or "").strip()


def _returned_sql_by_instance(eval_csv: Path) -> dict[str, str]:
    """Map ``question_id`` → ``returned_sql`` from an eval CSV."""
    mapping: dict[str, str] = {}
    with eval_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            instance_id = (row.get("question_id") or "").strip()
            if not instance_id:
                continue
            sql = (row.get("returned_sql") or "").strip()
            if sql:
                mapping[instance_id] = sql
    return mapping


def _enrich_results_pred_sql(
    results: list[dict],
    eval_csv: Path | None,
    *,
    mode: str,
) -> list[dict]:
    """Attach predicted SQL from the eval CSV when upstream omits it.

    Upstream ``evaluate_single_exec_result_instance`` always sets ``pred_sql``
    to ``None`` because ``exec_result`` mode scores pre-executed CSVs only.
    """
    if mode != "exec_result" or eval_csv is None or not eval_csv.is_file():
        return results

    sql_by_id = _returned_sql_by_instance(eval_csv)
    if not sql_by_id:
        return results

    enriched: list[dict] = []
    for item in results:
        row = dict(item)
        if not row.get("pred_sql"):
            instance_id = (row.get("instance_id") or "").strip()
            if instance_id in sql_by_id:
                row["pred_sql"] = sql_by_id[instance_id]
        enriched.append(row)
    return enriched


def snowflake_connector():
    """Return the first live Snowflake connector, or ``None`` if none configured.

    Requires ``CONNECTION_STRINGS`` (or a stored connection) pointed at the
    Spider2 Snowflake instance — see ``ontology_sql_eval.spider2_snowflake``.
    """
    from gsf.connectors import get_connectors

    for connector in get_connectors():
        if getattr(connector, "dialect", "") == "snowflake":
            return connector
    return None


def prepare_exec_result_submission(
    eval_csv: Path,
    result_dir: Path,
    *,
    instance_ids: set[str] | None = None,
    connector=None,
) -> list[str]:
    """Materialise ``<instance_id>.csv`` result tables for ``exec_result`` scoring.

    Upstream ``evaluate.py`` refuses to execute Snowflake (``sf_*``) queries in
    ``sql`` mode, so we score them in ``exec_result`` mode against pre-executed
    result tables. For each instance:

    * If ``<instance_id>.csv`` already exists in *result_dir* it is **reused**
      (cached) — Snowflake is not hit again on re-runs.
    * Otherwise the predicted SQL (the eval CSV's ``returned_sql`` column) is
      executed live against *connector* and the result saved as ``<id>.csv``.

    Rows with no predicted SQL, an execution error, or no available connector
    get an empty ``.csv`` so the instance is scored 0 rather than dropped from
    the denominator.
    """
    import pandas as pd

    result_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    cached: list[str] = []
    executed: list[str] = []
    failed: list[str] = []

    with eval_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        instance_id = (row.get("question_id") or "").strip()
        if not instance_id:
            continue
        if instance_ids is not None and instance_id not in instance_ids:
            continue

        out_path = result_dir / f"{instance_id}.csv"
        if out_path.exists() and out_path.stat().st_size > 0:
            cached.append(instance_id)
            written.append(instance_id)
            continue

        sql = _strip_sql_fences(row.get("returned_sql") or "")
        if not sql:
            out_path.write_text("", encoding="utf-8")
            failed.append(instance_id)
            continue

        if connector is None:
            logger.warning(
                "No Snowflake connector available to execute %s — scoring 0. "
                "Set CONNECTION_STRINGS to the Snowflake instance.",
                instance_id,
            )
            out_path.write_text("", encoding="utf-8")
            failed.append(instance_id)
            continue

        try:
            df = connector.execute(sql)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df.to_csv(out_path, index=False)
            executed.append(instance_id)
            written.append(instance_id)
        except Exception as exc:  # noqa: BLE001 - record + score 0, keep going
            logger.warning("Failed to execute predicted SQL for %s: %s", instance_id, exc)
            out_path.write_text("", encoding="utf-8")
            failed.append(instance_id)

    logger.info(
        "exec_result submission in %s: %d reused, %d executed, %d failed",
        result_dir,
        len(cached),
        len(executed),
        len(failed),
    )
    if failed:
        logger.warning("Instances scored 0 (no result): %s", ", ".join(failed))
    return written


def _load_evaluate_module() -> types.ModuleType:
    """Import upstream ``evaluate.py`` in-process (cached).

    ``evaluate.py`` reassigns ``sys.stdout``/``sys.stderr`` to a ``TeeOutput``
    that opens ``log.txt`` in the current directory *at import time*. We run
    that side effect inside ``_EVAL_SUITE`` and restore the original streams
    afterwards so it neither hijacks our stdout nor drops a stray log file.
    """
    global _evaluate_module
    if _evaluate_module is not None:
        return _evaluate_module

    if not _EVALUATE_PY.exists():
        raise FileNotFoundError(
            f"Spider2 evaluate.py not found: {_EVALUATE_PY}\n"
            "Run scripts/download_spider2.py to bootstrap third_party/Spider2."
        )

    spec = importlib.util.spec_from_file_location("spider2_evaluate", _EVALUATE_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluate.py spec from {_EVALUATE_PY}")
    module = importlib.util.module_from_spec(spec)

    orig_out, orig_err = sys.stdout, sys.stderr
    cwd_before = os.getcwd()
    os.chdir(_EVAL_SUITE)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd_before)
        tee = sys.stdout
        sys.stdout, sys.stderr = orig_out, orig_err
        close = getattr(tee, "close", None)
        if callable(close) and tee is not orig_out:
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.debug("Failed to close evaluate.py TeeOutput", exc_info=True)

    sys.modules["spider2_evaluate"] = module
    _evaluate_module = module
    return module


def run_spider2_evaluation(
    submission_dir: Path,
    *,
    output_dir: Path | None = None,
    max_workers: int = 4,
    mode: str = "sql",
    eval_csv: Path | None = None,
) -> dict:
    """Score a submission folder via upstream ``evaluate_spider2sql``.

    ``mode="sql"`` (default) submits ``<id>.sql`` files that upstream executes
    itself (SQLite/BigQuery). ``mode="exec_result"`` submits pre-executed
    ``<id>.csv`` result tables — required for Snowflake (``sf_*``), which
    upstream ``evaluate.py`` refuses to execute in ``sql`` mode.

    Calls the upstream evaluator **in-process** (not as a subprocess) so
    breakpoints inside ``evaluate.py`` are hit end-to-end from the
    ``Spider2 suite judge`` launch config.
    """
    if mode not in ("sql", "exec_result"):
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")

    exec_result_dir = _GOLD_DIR / "exec_result"
    if not exec_result_dir.is_dir():
        raise FileNotFoundError(
            f"Gold exec_result directory missing: {exec_result_dir}\n"
            "Re-run scripts/download_spider2.py (sparse checkout includes exec_result)."
        )

    work_dir = output_dir or (_REPO_ROOT / "output" / "spider2_eval_suite")
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = work_dir / "temp"
    results_path = work_dir / "results.json"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    evaluate_module = _load_evaluate_module()

    # Mirror the argparse.Namespace that ``evaluate_spider2sql`` expects.
    args = types.SimpleNamespace(
        mode=mode,
        result_dir=str(submission_dir),
        gold_dir=str(_GOLD_DIR),
        max_workers=max_workers,
        timeout=60,
        is_sql_debug=False,
    )
    logger.info(
        "Running Spider2 evaluation suite in-process "
        "(mode=%s, result_dir=%s, gold_dir=%s)",
        mode,
        submission_dir,
        _GOLD_DIR,
    )

    results = evaluate_module.evaluate_spider2sql(args, temp_dir)
    results = _enrich_results_pred_sql(results, eval_csv, mode=mode)

    total = len(results)
    correct = sum(int(item.get("score", 0)) for item in results)
    accuracy = (correct / total) if total else 0.0

    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mode": mode,
        "submission_dir": str(submission_dir),
        "results_path": str(results_path),
    }
    # Upstream only materialises an execution-result CSV dir in ``sql`` mode.
    if mode == "sql":
        summary["exec_result_csv_dir"] = str(Path(str(submission_dir) + "_csv"))
    logger.info(
        "Spider2 suite score: %d/%d correct (%.1f%%)",
        correct,
        total,
        accuracy * 100,
    )
    return summary


def judge_eval_csv(
    eval_csv: Path,
    slug: str,
    *,
    output_dir: Path | None = None,
    max_workers: int = 4,
) -> dict:
    """Score one eval CSV for a Spider2 database slug via the official suite."""
    if not eval_csv.exists():
        raise FileNotFoundError(f"Eval CSV not found: {eval_csv}")

    instance_ids = set(question_ids_for_slug(slug))
    snowflake = instances_are_snowflake(instance_ids)

    work_dir = output_dir or (_REPO_ROOT / "output" / "spider2_eval_suite" / slug)
    work_dir.mkdir(parents=True, exist_ok=True)

    if snowflake:
        # Upstream evaluate.py can't execute sf_* itself. We materialise result
        # tables by executing the predicted SQL live, but only when the result
        # CSV doesn't already exist — so this dir is persistent (not wiped) and
        # re-runs skip Snowflake for already-materialised instances.
        submission_dir = work_dir / "exec_results"
        submission_dir.mkdir(parents=True, exist_ok=True)
        prepare_exec_result_submission(
            eval_csv,
            submission_dir,
            instance_ids=instance_ids,
            connector=snowflake_connector(),
        )
        mode = "exec_result"
    else:
        submission_dir = work_dir / "submission"
        if submission_dir.exists():
            shutil.rmtree(submission_dir)
        submission_dir.mkdir(parents=True)
        ensure_sqlite_symlink(slug)
        prepare_sql_submission(eval_csv, submission_dir, instance_ids=instance_ids)
        mode = "sql"

    summary = run_spider2_evaluation(
        submission_dir,
        output_dir=work_dir,
        max_workers=max_workers,
        mode=mode,
        eval_csv=eval_csv,
    )
    summary["slug"] = slug
    summary["eval_csv"] = str(eval_csv)
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an eval CSV with the Spider2-lite official evaluation suite."
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Spider2 dataset slug (e.g. adventureworks).",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Eval CSV path (default: input/spider2/<slug>_<model>.csv).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Worker threads for evaluate.py (default: 4).",
    )
    return parser.parse_args(argv)


def _default_eval_csv(slug: str) -> Path:
    import os

    model_slug = os.environ.get("MODEL_NAME", "nemotron").rsplit("/", 1)[-1]
    return _REPO_ROOT / "input" / "spider2" / f"{slug}_{model_slug}.csv"


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    eval_csv = args.eval_csv or _default_eval_csv(args.slug)
    summary = judge_eval_csv(eval_csv, args.slug, max_workers=args.max_workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
