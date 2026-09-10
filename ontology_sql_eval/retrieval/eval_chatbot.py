# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the text-to-SQL agent against a chatbot evaluation JSON file.

For every entry in the JSON array (each containing ``question_id``,
``question``, ``SQL`` (expected), and ``answer_raw`` (expected user-facing
result)), this script:

1. Calls ``get_agent_response`` with the question (same path as
   ``ingest_postgres.run_retrieve``).
2. Scores the agent's SQL against the expected SQL by **executing both**
   queries against the live Postgres connector and comparing the resulting
   row sets.  Failure to execute either side yields score 0.
3. Scores the agent's answer text against ``answer_raw`` via difflib
   similarity and a normalised substring check.
4. Writes one row per question to a CSV.  Any per-question exception is
   logged, recorded in the ``error`` column, and scored 0 — execution
   continues with the next question.

Each question is iterated independently. When a question carries a ``db_id``,
it selects the matching connector from ``CONNECTION_STRINGS`` (``db_id`` equals
the connector's ``database_name``) and scopes retrieval to that database via the
connector passed in ``connectors``. Any ``evidence`` is appended to the
question. Questions without a ``db_id`` fall back to the first configured
connector.

Usage::

    uv run python -m ontology_sql_eval.retrieval.eval_chatbot
    uv run python -m ontology_sql_eval.retrieval.eval_chatbot --database-name <name>
    uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
        --database-name <name> [--input PATH] [--output PATH]

When ``--database-name`` is omitted, the dataset folder is inferred from
``CONNECTION_STRINGS`` (same source as ingest / semantic compile).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from gsf.retrieval.text_to_sql.main import get_agent_response
from gsf.retrieval.text_to_sql.state import TextToSQLPayload
from gsf.connectors import get_connectors
from gsf.utils import get_data_objects_retriever, get_semantic_objects_retriever

from ontology_sql_eval.env import load_env
from ontology_sql_eval.retrieval.scoring import (
    score_answer,
    score_sql,
    stringify_db_result,
)

load_env()

# The text-to-SQL agent stores executed-DB rows under this key on its result dict.
_DB_RESULT_KEY = "sql_response_from_db"

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_API_KEY = os.environ.get(
    "DEFAULT_MODELS_API_KEY", ""
) or os.environ.get("NVIDIA_API_KEY", "")
if not _DEFAULT_MODELS_API_KEY:
    raise EnvironmentError(
        "DEFAULT_MODELS_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export DEFAULT_MODELS_API_KEY='nvapi-...'\n\n"
        "Legacy NVIDIA_API_KEY is also supported as a fallback.\n\n"
        "Get your key at https://build.nvidia.com"
    )

# Match the chat server's wiring (gsf/server/chat/helpers.py): same retriever
# and pgvector store. Anything else here and scoring stops being apples-to-apples
# with production.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "datasets"
# Eval writes its results CSV straight into the judge's input folder so the
# judge can pick it up without a manual copy.
_INPUT_DIR = _REPO_ROOT / "input"

_DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "nemotron")


def _connection_strings() -> list[str]:
    return [
        s.strip()
        for s in os.environ.get("CONNECTION_STRINGS", "").split(",")
        if s.strip()
    ]


def _evaluation_dataset_dir_for_connection(connection_string: str) -> Path | None:
    """Return ``datasets/<name>/`` when ``evaluation.json`` exists for *connection_string*."""
    from urllib.parse import unquote, urlparse

    from ontology_sql_eval.ingestion.ingest import database_name_for

    if connection_string.startswith("sqlite:"):
        db_path = Path(unquote(urlparse(connection_string).path)).resolve()
        cur = db_path.parent
        eval_dir = _EVAL_DIR.resolve()
        while eval_dir in cur.parents:
            if (cur / "evaluation.json").exists():
                return cur
            cur = cur.parent
        return None

    candidate = _EVAL_DIR / database_name_for(connection_string)
    if (candidate / "evaluation.json").exists():
        return candidate
    return None


def dataset_name_from_env() -> str:
    """Infer the eval dataset folder name from ``CONNECTION_STRINGS``."""
    from ontology_sql_eval.ingestion.ingest import database_name_for

    connection_strings = _connection_strings()
    if not connection_strings:
        raise EnvironmentError(
            "No --database-name given and CONNECTION_STRINGS is not set. "
            "Pass --database-name, or add CONNECTION_STRINGS to your .env."
        )

    dataset_dirs = {
        ds_dir.resolve()
        for cs in connection_strings
        if (ds_dir := _evaluation_dataset_dir_for_connection(cs)) is not None
    }
    if len(dataset_dirs) == 1:
        return next(iter(dataset_dirs)).name
    if len(dataset_dirs) > 1:
        names = sorted(d.name for d in dataset_dirs)
        raise ValueError(
            f"CONNECTION_STRINGS point to multiple evaluation datasets: {names}. "
            "Pass --database-name explicitly."
        )

    if len(connection_strings) == 1:
        return database_name_for(connection_strings[0])

    raise ValueError(
        "Could not infer evaluation dataset from CONNECTION_STRINGS. "
        "Pass --database-name or ensure datasets/<name>/evaluation.json exists."
    )


def _resolve_paths(
    database_name: str | None,
    input_override: Path | None,
    output_override: Path | None,
) -> tuple[Path, Path]:
    """Derive input/output paths from *database_name* when not explicitly set.

    The eval *input* is the dataset's ``evaluation.json``; the eval *output* CSV
    is written to the repo-root ``input/`` folder (the judge's input directory),
    named ``<db>_<model>.csv`` so results from different datasets/models don't
    collide.
    """
    if input_override and output_override:
        return input_override, output_override

    model_slug = _DEFAULT_MODEL_NAME.rsplit("/", 1)[-1]
    if database_name:
        db_dir = _EVAL_DIR / database_name
        db_dir.mkdir(parents=True, exist_ok=True)
        default_input = db_dir / "evaluation.json"
        default_output = _INPUT_DIR / f"{database_name}_{model_slug}.csv"
    else:
        default_input = _EVAL_DIR / "evaluation.json"
        default_output = _INPUT_DIR / f"{model_slug}.csv"

    return input_override or default_input, output_override or default_output


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Evaluation file not found: {path}\n"
            f"Pass --input <path> or --database-name <name>."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array of questions, got {type(data).__name__}"
        )
    return data


CSV_FIELDS = [
    "row_index",
    "question_id",
    "difficulty",
    "question",
    "expected_sql",
    "returned_sql",
    "sql_text_similarity",
    "sql_exec_match",
    "expected_sql_error",
    "returned_sql_error",
    "expected_sql_result",
    "expected_answer_raw",
    "returned_answer",
    "answer_text_similarity",
    "answer_numbers_match",
    "runtime_seconds",
    "error",
]


def _print_agent_result(
    qid: Any, question: str, agent_result: Dict[str, Any] | None, expected_sql: str = ""
) -> None:
    """Pretty-print the agent result to stdout for quick visual inspection."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Question {qid}: {question}")
    print(sep)
    if expected_sql:
        print("\n  [expected_sql]")
        for line in expected_sql.splitlines():
            print(f"    {line}")
    if not agent_result:
        print("  (no result)")
        print(sep)
        return
    for key in ("sql_code", "response", "sql_response_from_db"):
        val = agent_result.get(key)
        if val is None:
            continue
        print(f"\n  [{key}]")
        for line in str(val).splitlines():
            print(f"    {line}")
    remaining = {
        k: v
        for k, v in agent_result.items()
        if k not in ("sql_code", "response", "sql_response_from_db")
    }
    if remaining:
        print("\n  [other keys]")
        for k, v in remaining.items():
            print(f"    {k}: {v}")
    print(sep)


def run_evaluation(
    input_path: Path,
    output_path: Path,
    start_index: int = 0,
    end_index: int | None = None,
) -> None:
    all_questions = _load_questions(input_path)
    questions = all_questions[start_index:end_index]
    logger.info(
        "Running questions %d–%d (%d of %d total) from %s",
        start_index,
        start_index + len(questions) - 1,
        len(questions),
        len(all_questions),
        input_path,
    )

    data_retriever = get_data_objects_retriever()
    semantic_retriever = get_semantic_objects_retriever()
    connectors = get_connectors()
    connectors_by_name: dict[str, Any] = {}
    for connector in connectors:
        name = getattr(connector, "database_name", None)
        if name:
            connectors_by_name[name] = connector

    resuming = start_index > 0 and output_path.exists()
    mode = "a" if resuming else "w"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not resuming:
            writer.writeheader()

        for idx, item in enumerate(questions, start=start_index):
            qid = item.get("question_id", idx)
            question = item.get("question", "")
            expected_sql = item.get("SQL", "")
            expected_answer = item.get("answer_raw", "")
            difficulty = item.get("difficulty", "")

            # Each question may carry its own db_id / evidence; both are
            # optional so single-dataset eval files without them still work.
            evidence = item.get("evidence", "")
            db_id = item.get("db_id", "")

            agent_question = question
            if evidence:
                agent_question = f"{question}\n\nEvidence: {evidence}"

            logger.info("[%d/%d] q%s: %s", idx + 1, len(questions), qid, question)
            if db_id:
                logger.info("  db_id=%s  evidence=%s", db_id, evidence[:120])

            row: Dict[str, Any] = {
                "row_index": idx,
                "question_id": qid,
                "difficulty": difficulty,
                "question": question,
                "expected_sql": expected_sql,
                "returned_sql": "",
                "sql_text_similarity": 0.0,
                "sql_exec_match": 0,
                "expected_sql_error": "",
                "returned_sql_error": "",
                "expected_answer_raw": expected_answer,
                "returned_answer": "",
                "answer_text_similarity": 0.0,
                "answer_numbers_match": 0,
                "runtime_seconds": "",
                "error": "",
            }

            # Route to the connector matching this question's db_id; fall back
            # to the first connector when the question is not db-scoped.
            active_connector = connectors_by_name.get(db_id) if db_id else None
            if active_connector is None:
                active_connector = connectors[0]
            active_connectors = [active_connector]

            t0 = time.perf_counter()
            try:
                payload: TextToSQLPayload = {
                    "question": agent_question,
                    "data_retriever": data_retriever,
                    "semantic_retriever": semantic_retriever,
                    "connectors": active_connectors,
                    "path_state": {},
                    "custom_prompts": "",
                    "acronyms": [],
                    "evidence": evidence,
                }
                logger.info("Running question %s", payload["question"])
                agent_result = get_agent_response(payload)
                _print_agent_result(qid, question, agent_result, expected_sql)
                returned_sql = (agent_result or {}).get("sql_code", "") or ""
                returned_db = (agent_result or {}).get(_DB_RESULT_KEY)
                returned_db_str = stringify_db_result(returned_db)

                row["returned_sql"] = returned_sql
                row["returned_answer"] = returned_db_str

                row.update(
                    score_sql(
                        active_connector, expected_sql, returned_sql, schema=db_id
                    )
                )
                row.update(score_answer(expected_answer, returned_db_str))
            except Exception as exc:
                logger.exception("Question %s failed", qid)
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["error"] += (
                    " | " + traceback.format_exc().replace("\n", " | ")[:1000]
                )
            finally:
                row["runtime_seconds"] = round(time.perf_counter() - t0, 2)
                writer.writerow(row)
                f.flush()

    logger.info("Wrote scores to %s", output_path)


def run_single_question(question: str) -> None:
    """Run a single question through the agent and print the result."""
    data_retriever = get_data_objects_retriever()
    semantic_retriever = get_semantic_objects_retriever()
    connectors = get_connectors()

    payload: TextToSQLPayload = {
        "question": question,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "connectors": connectors,
        "path_state": {},
        "custom_prompts": "",
        "acronyms": [],
    }
    t0 = time.perf_counter()
    agent_result = get_agent_response(payload)
    elapsed = round(time.perf_counter() - t0, 2)

    _print_agent_result("single", question, agent_result)
    print(f"\n  Runtime: {elapsed}s")


SINGLE_QUERY = "calculate the customer count by state province name"

START_INDEX = 0
END_INDEX = None  # None = run to the end


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-name",
        type=str,
        default=None,
        help="Dataset / database name. "
        "When omitted, inferred from CONNECTION_STRINGS (same as semantic compile). "
        "Derives input from <name>/evaluation.json and output from <name>/<model>.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSON path (overrides --database-name default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (overrides --database-name default).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        default=False,
        help="Run a single hardcoded query (edit SINGLE_QUERY in the script).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    dataset_name = args.database_name
    if not dataset_name and not (args.input and args.output):
        dataset_name = dataset_name_from_env()
        logger.info("Resolved dataset name from CONNECTION_STRINGS: %s", dataset_name)
    input_path, output_path = _resolve_paths(dataset_name, args.input, args.output)
    if args.single:
        run_single_question(SINGLE_QUERY)
    else:
        run_evaluation(
            input_path=input_path,
            output_path=output_path,
            start_index=START_INDEX,
            end_index=END_INDEX,
        )
