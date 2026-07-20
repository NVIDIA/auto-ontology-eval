# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the rerank agent against a chatbot evaluation JSON file.

This mirrors ``eval_chatbot.py`` but drives the GSF **rerank** flow
(``gsf.retrieval.rerank.main.get_agent_response``) instead of text-to-SQL.

For every entry in the JSON array (each containing ``question_id``,
``question``, ``SQL`` (expected), and ``answer_raw`` (expected user-facing
result)), this script:

1. Calls ``get_agent_response`` with a ``RerankPayload`` built from the
   question.  The rerank agent wires its own retrievers/connectors at import
   time, so the payload only carries the question (and optional path state).
2. Scores the agent's SQL against the expected SQL by **executing both**
   queries against the live Postgres connector and comparing the resulting
   row sets.  Failure to execute either side yields score 0.
3. Scores the agent's answer text against ``answer_raw`` via difflib
   similarity and a normalised substring check.
4. Writes one row per question to a CSV.  Any per-question exception is
   logged, recorded in the ``error`` column, and scored 0 — execution
   continues with the next question.

Usage::

    uv run python -m ontology_sql_eval.retrieval.eval_rerank \
        --database-name <name> [--input PATH] [--output PATH]
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

from dotenv import load_dotenv

from gsf.retrieval.rerank.main import get_agent_response
from gsf.retrieval.rerank.state import RerankPayload
from gsf.connectors import get_connectors

from ontology_sql_eval.retrieval.scoring import (
    score_answer,
    score_sql,
    stringify_db_result,
)


load_dotenv()

# The rerank agent's final user-facing answer (the array of ranked identifiers)
# lives under this key on its result dict; this is what we score and record.
_RESPONSE_KEY = "response"
# ...and the generated SQL under this key.
_SQL_KEY = "sql"

logger = logging.getLogger(__name__)

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "datasets"
# Eval writes its results CSV straight into the judge's input folder so the
# judge can pick it up without a manual copy.
_INPUT_DIR = _REPO_ROOT / "input"

_DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "nemotron")


def _resolve_paths(
    database_name: str | None,
    input_override: Path | None,
    output_override: Path | None,
) -> tuple[Path, Path]:
    """Derive input/output paths from *database_name* when not explicitly set.

    The eval *input* is the dataset's ``evaluation.json``; the eval *output* CSV
    is written to the repo-root ``input/`` folder (the judge's input directory),
    named ``<db>_rerank_<model>.csv`` so results from different
    datasets/models/flows don't collide.
    """
    if input_override and output_override:
        return input_override, output_override

    model_slug = _DEFAULT_MODEL_NAME.rsplit("/", 1)[-1]
    if database_name:
        db_dir = _EVAL_DIR / database_name
        db_dir.mkdir(parents=True, exist_ok=True)
        default_input = db_dir / "evaluation.json"
        default_output = _INPUT_DIR / f"{database_name}_rerank_{model_slug}.csv"
    else:
        default_input = _EVAL_DIR / "evaluation.json"
        default_output = _INPUT_DIR / f"rerank_{model_slug}.csv"

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
    """Print only the agent's response to stdout for quick visual inspection."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Question {qid}: {question}")
    print(sep)
    if not agent_result:
        print("  (no result)")
        print(sep)
        return
    print("\n  [response]")
    for line in str(agent_result.get("response", "")).splitlines():
        print(f"    {line}")
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

    # The rerank agent builds its own retrievers/connectors at import time; we
    # only need a connector here to execute expected/returned SQL for scoring.
    connectors = get_connectors()

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
            logger.info("[%d/%d] q%s: %s", idx + 1, len(questions), qid, question)

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

            t0 = time.perf_counter()
            try:
                payload: RerankPayload = {
                    "question": question,
                    "path_state": {},
                }
                logger.info("Running question %s", payload["question"])
                agent_result = get_agent_response(payload)
                _print_agent_result(qid, question, agent_result, expected_sql)
                returned_sql = (agent_result or {}).get(_SQL_KEY, "") or ""
                returned_response = (agent_result or {}).get(_RESPONSE_KEY)
                returned_answer_str = stringify_db_result(returned_response)

                row["returned_sql"] = returned_sql
                row["returned_answer"] = returned_answer_str

                row.update(score_sql(connectors[0], expected_sql, returned_sql))
                row.update(score_answer(expected_answer, returned_answer_str))
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
    """Run a single question through the rerank agent and print the result."""
    payload: RerankPayload = {
        "question": question,
        "path_state": {},
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
        help="Database name (e.g. wideworldimporters). "
        "Derives input from <db>/evaluation.json and output from "
        "<db>_rerank_<model>.csv.",
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
    input_path, output_path = _resolve_paths(
        args.database_name, args.input, args.output
    )
    if args.single:
        run_single_question(SINGLE_QUERY)
    else:
        run_evaluation(
            input_path=input_path,
            output_path=output_path,
            start_index=START_INDEX,
            end_index=END_INDEX,
        )
