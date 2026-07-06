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

Usage::

    uv run python -m evaluation.retrieval.eval_chatbot \
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

from gsf.retrieval.text_to_sql.main import get_agent_response
from gsf.retrieval.text_to_sql.state import TextToSQLPayload
from gsf.connectors import get_connectors
from gsf.utils import get_data_objects_retriever, get_semantic_objects_retriever

from evaluation.retrieval.scoring import (
    normalize_text,
    score_answer,
    score_sql,
    stringify_db_result,
)


load_dotenv()

# The text-to-SQL agent stores executed-DB rows under this key on its result dict.
_DB_RESULT_KEY = "sql_response_from_db"

logger = logging.getLogger(__name__)

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )

# Match the chat server's wiring (gsf/server/chat/helpers.py): same retriever
# and pgvector store. Anything else here and scoring stops being apples-to-apples
# with production.

_EVAL_DIR = Path(__file__).resolve().parents[2] / "datasets"

_DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "nemotron")


def _resolve_paths(
    database_name: str | None,
    input_override: Path | None,
    output_override: Path | None,
) -> tuple[Path, Path]:
    """Derive input/output paths from *database_name* when not explicitly set."""
    if input_override and output_override:
        return input_override, output_override

    if database_name:
        db_dir = _EVAL_DIR / database_name
        db_dir.mkdir(parents=True, exist_ok=True)
        default_input = db_dir / "evaluation.json"
        model_slug = _DEFAULT_MODEL_NAME.rsplit("/", 1)[-1]
        default_output = db_dir / f"{model_slug}.csv"
    else:
        default_input = _EVAL_DIR / "evaluation.json"
        model_slug = _DEFAULT_MODEL_NAME.rsplit("/", 1)[-1]
        default_output = _EVAL_DIR / f"{model_slug}.csv"

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
                payload: TextToSQLPayload = {
                    "question": question,
                    "data_retriever": data_retriever,
                    "semantic_retriever": semantic_retriever,
                    "connectors": connectors,
                    "path_state": {},
                    "custom_prompts": "",
                    "acronyms": [],
                }
                agent_result = get_agent_response(payload)
                _print_agent_result(qid, question, agent_result, expected_sql)
                returned_sql = (agent_result or {}).get("sql_code", "") or ""
                returned_db = (agent_result or {}).get(_DB_RESULT_KEY)
                returned_db_str = stringify_db_result(returned_db)

                row["returned_sql"] = returned_sql
                row["returned_answer"] = returned_db_str

                row.update(score_sql(connectors[0], expected_sql, returned_sql))
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


def run_evaluation_consistency(
    input_path: Path,
    output_path: Path,
    start_index: int = 0,
    end_index: int | None = None,
    runs: int = 10,
) -> None:
    """Run each question multiple times and report SQL/answer consistency."""
    all_questions = _load_questions(input_path)
    questions = all_questions[start_index:end_index]
    logger.info(
        "Consistency eval: %d questions, %d runs each, output=%s",
        len(questions),
        runs,
        output_path,
    )

    data_retriever = get_data_objects_retriever()
    semantic_retriever = get_semantic_objects_retriever()
    connectors = get_connectors()

    results: Dict[int, list] = {i: [] for i in range(len(questions))}

    for run_num in range(1, runs + 1):
        logger.info("RUN %d/%d", run_num, runs)

        for q_idx, item in enumerate(questions):
            qid = item.get("question_id", start_index + q_idx)
            question = item.get("question", "")
            logger.info("[Run %d] q%s: %s", run_num, qid, question)

            try:
                payload: TextToSQLPayload = {
                    "question": question,
                    "data_retriever": data_retriever,
                    "semantic_retriever": semantic_retriever,
                    "connectors": connectors,
                    "path_state": {},
                    "custom_prompts": "",
                    "acronyms": [],
                }
                agent_result = get_agent_response(payload)
                returned_sql = normalize_text(
                    (agent_result or {}).get("sql_code", "") or ""
                )
                returned_db = (agent_result or {}).get(_DB_RESULT_KEY)
                returned_db_str = stringify_db_result(returned_db)
            except Exception as exc:
                logger.exception("Run %d, question %s failed", run_num, qid)
                returned_sql = f"ERROR: {exc}"
                returned_db_str = ""

            results[q_idx].append({"sql": returned_sql, "answer": returned_db_str})

        logger.info("After run %d:", run_num)
        for q_idx, item in enumerate(questions):
            qid = item.get("question_id", start_index + q_idx)
            run_results = results[q_idx]
            sqls = [r["sql"] for r in run_results]
            answers = [r["answer"] for r in run_results]
            logger.info(
                "  q%s: %d runs -> %d unique SQLs, %d unique answers",
                qid,
                len(sqls),
                len(set(sqls)),
                len(set(answers)),
            )

        fieldnames = ["question_id", "question"]
        for r in range(1, run_num + 1):
            fieldnames.extend([f"sql_run_{r}", f"answer_run_{r}"])
        fieldnames.extend(["sql_consistency", "answer_consistency"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for q_idx, item in enumerate(questions):
                qid = item.get("question_id", start_index + q_idx)
                question = item.get("question", "")
                run_results = results[q_idx]

                row: Dict[str, Any] = {"question_id": qid, "question": question}

                first_sql: Dict[str, int] = {}
                first_answer: Dict[str, int] = {}

                for r_idx, r in enumerate(run_results):
                    run_col = r_idx + 1
                    sql_val = r["sql"]
                    ans_val = r["answer"]

                    if sql_val in first_sql:
                        row[f"sql_run_{run_col}"] = f"same as run {first_sql[sql_val]}"
                    else:
                        first_sql[sql_val] = run_col
                        row[f"sql_run_{run_col}"] = sql_val

                    if ans_val in first_answer:
                        row[f"answer_run_{run_col}"] = (
                            f"same as run {first_answer[ans_val]}"
                        )
                    else:
                        first_answer[ans_val] = run_col
                        row[f"answer_run_{run_col}"] = ans_val

                sql_counts = {}
                answer_counts = {}
                for r in run_results:
                    sql_counts[r["sql"]] = sql_counts.get(r["sql"], 0) + 1
                    answer_counts[r["answer"]] = answer_counts.get(r["answer"], 0) + 1
                row["sql_consistency"] = (
                    f"{max(sql_counts.values())}/{run_num}" if sql_counts else ""
                )
                row["answer_consistency"] = (
                    f"{max(answer_counts.values())}/{run_num}"
                    if answer_counts
                    else ""
                )

                writer.writerow(row)

        logger.info("Updated consistency CSV: %s (after run %d)", output_path, run_num)

    logger.info("CONSISTENCY SUMMARY (%d runs)", runs)

    for q_idx, item in enumerate(questions):
        qid = item.get("question_id", start_index + q_idx)
        question = item.get("question", "")
        run_results = results[q_idx]

        sql_counts: Dict[str, int] = {}
        answer_counts: Dict[str, int] = {}
        sql_to_answer: Dict[str, str] = {}

        for r in run_results:
            sql_counts[r["sql"]] = sql_counts.get(r["sql"], 0) + 1
            answer_counts[r["answer"]] = answer_counts.get(r["answer"], 0) + 1
            sql_to_answer[r["sql"]] = r["answer"]

        logger.info("q%s: %s", qid, question)
        for sql, count in sorted(sql_counts.items(), key=lambda x: -x[1]):
            logger.info("  SQL (%d/%d): %s", count, runs, sql[:500])
            logger.info("  Answer: %s", sql_to_answer[sql][:500])

        most_common_sql = max(sql_counts.values())
        most_common_answer = max(answer_counts.values())
        logger.info("  -> SQL consistency:    %d/%d", most_common_sql, runs)
        logger.info("  -> Answer consistency: %d/%d", most_common_answer, runs)

    logger.info("Final consistency CSV: %s", output_path)


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


SINGLE_QUERY = "What is the most frequently used GPU MODS version?"

START_INDEX = 0
END_INDEX = None  # None = run to the end
CONSISTENCY_RUNS = 10
RUN_CONSISTENCY = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-name",
        type=str,
        default=None,
        help="Database name (e.g. wideworldimporters). "
        "Derives input from <db>/evaluation.json and output from <db>/<model>.csv.",
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
        "--consistency",
        action="store_true",
        default=False,
        help="Run consistency evaluation (repeat N times and report SQL/answer stability).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs for consistency evaluation (default: 10).",
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
    elif RUN_CONSISTENCY or args.consistency:
        num_runs = args.runs if args.consistency else CONSISTENCY_RUNS
        consistency_output = output_path.with_name(
            f"{output_path.stem}_consistency.csv"
        )
        run_evaluation_consistency(
            input_path=input_path,
            output_path=consistency_output,
            start_index=START_INDEX,
            end_index=END_INDEX,
            runs=num_runs,
        )
    else:
        run_evaluation(
            input_path=input_path,
            output_path=output_path,
            start_index=START_INDEX,
            end_index=END_INDEX,
        )
