# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the text-to-SQL agent against a chatbot evaluation JSON file.

For every entry in the JSON array (each containing ``question_id``,
``question``, ``SQL`` (expected), and ``answer_raw`` (expected user-facing
result)), this script:

1. Streams the agent over the question (``stream_agent_response``, the same
   path the chat server uses), timing every graph node as it completes.
2. Scores the agent's SQL against the expected SQL by **executing both**
   queries against the live connector and comparing the resulting row sets.
   Failure to execute either side yields score 0.
3. Scores the agent's answer text against ``answer_raw`` via difflib
   similarity and a normalised substring check.
4. Writes one row per question to a CSV.  Any per-question exception is
   logged, recorded in the ``error`` column, and scored 0 — execution
   continues with the next question.

When a question carries a ``db_id``, it selects the matching connector from
``CONNECTION_STRINGS`` (``db_id`` equals the connector's ``database_name``) and
scopes retrieval to that database via the connector passed in ``connectors``.
Any ``evidence`` is passed to the agent as its own payload field on GSF builds
that support it, and appended to the question text on those that do not.
Questions without a ``db_id`` fall back to the first configured connector.

Questions are independent, so ``--workers N`` runs N of them concurrently on a
thread pool. Every run also writes a full instrumentation bundle under
``logs/<run-id>/`` (see :mod:`ontology_sql_eval.retrieval.run_logging`): a
DEBUG-level ``run.log``, one log file per question, a phase timeline, per-node
timings, and an aggregated ``summary.json``.

Usage::

    uv run python -m ontology_sql_eval.retrieval.eval_chatbot
    uv run python -m ontology_sql_eval.retrieval.eval_chatbot --database-name <name>
    uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
        --database-name <name> --limit 10 --workers 2 [--input PATH] [--output PATH]

When ``--database-name`` is omitted, the dataset folder is inferred from
``CONNECTION_STRINGS`` (same source as ingest / semantic compile).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ontology_sql_eval.env import load_env

# Must run before the GSF imports below: ``gsf.retrieval.text_to_sql.main``
# builds its LLM client at import time and raises if the credentials are not
# already in ``os.environ``. Loading .env afterwards is too late for a plain
# ``python -m`` run (VS Code masked this by injecting ``envFile`` itself).
load_env()

from gsf.retrieval.text_to_sql import main as gsf_agent_main  # noqa: E402
from gsf.retrieval.text_to_sql.main import stream_agent_response  # noqa: E402
from gsf.retrieval.text_to_sql.state import TextToSQLPayload  # noqa: E402
from gsf.connectors import get_connectors  # noqa: E402
from gsf.utils import (  # noqa: E402
    get_data_objects_retriever,
    get_semantic_objects_retriever,
)

from ontology_sql_eval.retrieval.run_logging import (  # noqa: E402
    RunLogger,
    current_question,
    http_calls,
    question_context,
    setup_run_logging,
)
from ontology_sql_eval.retrieval.scoring import (  # noqa: E402
    score_answer,
    score_sql,
    stringify_db_result,
)

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
_LOG_DIR = _REPO_ROOT / "logs"

_DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "nemotron")

# BEAVER "setting 0" runs must withhold the dataset's domain-knowledge
# evidence; the official leaderboard's end-to-end column provides only the
# question. Default keeps evidence (BIRD-style).
_INCLUDE_EVIDENCE = os.environ.get("EVAL_INCLUDE_EVIDENCE", "1") not in (
    "0",
    "false",
    "False",
)


# Whether this GSF build accepts BIRD-style evidence as its own payload field
# (added in "Evidence as parameter", #226) rather than requiring it to be glued
# onto the question text. Probed rather than assumed so one eval checkout works
# against either build — and so a GSF upgrade cannot quietly change what the
# agent is being asked without it showing up here.
_SUPPORTS_EVIDENCE_PARAM = "evidence" in getattr(
    TextToSQLPayload, "__annotations__", {}
)

# Same probe for retrieved question/SQL exemplars. Unlike evidence there is no
# fallback: gluing another database's SQL onto the question text would be worse
# than sending nothing, so on an older GSF the examples are simply dropped.
_SUPPORTS_SQL_EXAMPLES_PARAM = "sql_examples" in getattr(
    TextToSQLPayload, "__annotations__", {}
)

# Same probe for looked-up database values.
_SUPPORTS_VALUE_ANCHORS_PARAM = "value_anchors" in getattr(
    TextToSQLPayload, "__annotations__", {}
)

# How per-node timings were measured, recorded into every run summary. "phase"
# means GSF emitted explicit start/end events and each node was timed directly;
# "gap" means the older single-event stream, where a node's cost is inferred
# from the gap since the last event and consecutive repeats must be merged.
# Downstream analysis has to know which it is read: merging phase-timed entries
# would silently fuse a node's two genuine visits into one.
_node_timing_mode = "gap"

# Large agent payloads (SQL, result previews) blow past the default field limit
# when the CSV is read back for sorting.
csv.field_size_limit(sys.maxsize)


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


def _load_sql_examples(
    path: Path, k: int, min_score: float = 0.0
) -> Dict[str, List[Dict[str, str]]]:
    """Map question_id to its top-*k* retrieved question/SQL exemplars.

    Reads the CSV written by ``scripts/find_structural_exemplars.py``, which
    ranks train-set examples by question structure alone. Keys are stringified
    because the CSV always yields text while the dataset yields ints.

    Retrieval always returns something, but wording similarity is only weakly
    related to whether the structure transfers (r=0.20 against gold skeleton
    similarity on BIRD dev), so most questions get a pattern that is merely a
    verbal coincidence. ``min_score`` withholds patterns from a question whose
    best match scores below it, leaving that question to be answered under the
    general shape rules instead. The gate is per question rather than per
    exemplar so that a kept question still gets a full *k*.
    """
    if not path.exists():
        raise SystemExit(
            f"SQL examples file not found: {path}\n"
            "Generate it with: python scripts/find_structural_exemplars.py"
        )
    ranked: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    best_score: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # A truncated or hand-edited header still parses, and the rows then key
        # on nonsense instead of question ids — silently sending no examples to
        # every question. Check up front rather than debugging a null result.
        required = {"question_id", "rank", "train_question", "train_sql"}
        if min_score > 0:
            required.add("score")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path} is missing expected column(s): {sorted(missing)}.\n"
                f"Found: {reader.fieldnames}\n"
                "Regenerate it with: python scripts/find_structural_exemplars.py"
            )
        for record in reader:
            sql = (record.get("train_sql") or "").strip()
            if not sql:
                continue
            try:
                rank = int(record.get("rank") or 0)
            except ValueError:
                continue
            qid = str(record.get("question_id", ""))
            if min_score > 0:
                try:
                    score = float(record.get("score") or 0.0)
                except ValueError:
                    score = 0.0
                # Every row is read, not just the top *k*, so the gate sees the
                # question's true best score even if ranking ever changes.
                best_score[qid] = max(best_score.get(qid, 0.0), score)
            if rank > k:
                continue
            ranked.setdefault(qid, []).append(
                (
                    rank,
                    {
                        "db": (record.get("train_db") or "").strip(),
                        "question": (record.get("train_question") or "").strip(),
                        # The exemplar's own evidence shows how a stated formula
                        # was turned into SQL (casting, operand order, whether
                        # *100 was applied), which is the transferable part.
                        "evidence": (record.get("train_evidence") or "").strip(),
                        "sql": sql,
                    },
                )
            )
    # Rank order decides what the model reads first, and DictReader preserves
    # only file order, which need not be sorted.
    by_question = {
        qid: [example for _, example in sorted(pairs, key=lambda pair: pair[0])]
        for qid, pairs in ranked.items()
    }
    if min_score > 0:
        retrieved = len(by_question)
        by_question = {
            qid: examples
            for qid, examples in by_question.items()
            if best_score.get(qid, 0.0) >= min_score
        }
        logger.info(
            "SQL example gate at score >= %.2f: %d of %d questions keep their "
            "examples (%.1f%%); the rest run under the general shape rules",
            min_score,
            len(by_question),
            retrieved,
            100.0 * len(by_question) / retrieved if retrieved else 0.0,
        )
    logger.info(
        "Loaded top-%d SQL examples for %d questions from %s",
        k,
        len(by_question),
        path,
    )
    return by_question


def _load_value_anchors(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Map question_id to the database values looked up for its question.

    Reads the CSV written by ``scripts/value_index.py anchors``. Rows of kind
    "absent" carry no column and record a phrase the database does not store;
    they are kept because that negative is what stops an invented literal.
    """
    if not path.exists():
        raise SystemExit(
            f"Value anchors file not found: {path}\n"
            "Generate it with: python scripts/value_index.py build && "
            "python scripts/value_index.py anchors"
        )
    ranked: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question_id", "rank", "kind", "phrase"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path} is missing expected column(s): {sorted(missing)}.\n"
                f"Found: {reader.fieldnames}"
            )
        for record in reader:
            phrase = (record.get("phrase") or "").strip()
            if not phrase:
                continue
            try:
                rank = int(record.get("rank") or 0)
            except ValueError:
                continue
            # Absent notes last: the found values are the actionable part, and
            # rank restarts at 1 for the absent rows.
            order = rank + (1000 if (record.get("kind") or "") == "absent" else 0)
            ranked.setdefault(str(record.get("question_id", "")), []).append(
                (
                    order,
                    {
                        "kind": (record.get("kind") or "found").strip(),
                        "phrase": phrase,
                        "tbl": (record.get("tbl") or "").strip(),
                        "col": (record.get("col") or "").strip(),
                        "stored_value": (record.get("stored_value") or "").strip(),
                        "n_rows": (record.get("n_rows") or "").strip(),
                        # Present only where one phrase matched several columns
                        # of one table, which is where naming them apart helps.
                        "description": (record.get("description") or "").strip(),
                    },
                )
            )
    by_question = {
        qid: [anchor for _, anchor in sorted(pairs, key=lambda pair: pair[0])]
        for qid, pairs in ranked.items()
    }
    logger.info(
        "Loaded value anchors for %d questions from %s (%d total lines)",
        len(by_question),
        path,
        sum(len(v) for v in by_question.values()),
    )
    return by_question


def _load_prompt_extras(
    sql_examples_path: Path | None,
    sql_examples_k: int,
    sql_examples_min_score: float,
    value_anchors_path: Path | None,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    """Load whichever prompt inputs were asked for, keyed by question id.

    Asking for either against a GSF with no matching payload field is fatal:
    the flag would otherwise look honoured while nothing reached the prompt,
    and the run would be scored as though it had been.
    """
    sql_examples: Dict[str, List[Dict[str, str]]] = {}
    if sql_examples_path and sql_examples_k > 0:
        if not _SUPPORTS_SQL_EXAMPLES_PARAM:
            raise SystemExit(
                "--sql-examples was given but this GSF build has no "
                "'sql_examples' payload field; the examples would be silently "
                "dropped. Upgrade GSF or drop the flag."
            )
        sql_examples = _load_sql_examples(
            sql_examples_path, sql_examples_k, sql_examples_min_score
        )

    value_anchors: Dict[str, List[Dict[str, str]]] = {}
    if value_anchors_path:
        if not _SUPPORTS_VALUE_ANCHORS_PARAM:
            raise SystemExit(
                "--value-anchors was given but this GSF build has no "
                "'value_anchors' payload field; the anchors would be silently "
                "dropped. Upgrade GSF or drop the flag."
            )
        value_anchors = _load_value_anchors(value_anchors_path)
    return sql_examples, value_anchors


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
    # Timing / bottleneck columns. The full per-node breakdown lives in
    # logs/<run-id>/questions.jsonl; these are the headline numbers so a
    # spreadsheet sort over the results CSV already points at the slow rows.
    "agent_seconds",
    "scoring_seconds",
    "http_calls",
    "agent_nodes",
    "slowest_node",
    "slowest_node_seconds",
    "worker",
    "error",
]


def _format_agent_result(
    qid: Any, question: str, agent_result: Dict[str, Any] | None, expected_sql: str = ""
) -> str:
    """Render the agent result as one block for inspection.

    Built as a single string rather than printed line by line: under
    ``--workers > 1`` interleaved prints from several threads are unreadable,
    and one log record keeps a question's result contiguous in every log file.
    """
    sep = "=" * 80
    lines = [sep, f"  Question {qid}: {question}", sep]
    if expected_sql:
        lines.append("\n  [expected_sql]")
        lines.extend(f"    {line}" for line in expected_sql.splitlines())
    if not agent_result:
        lines += ["  (no result)", sep]
        return "\n".join(lines)
    for key in ("sql_code", "response", "sql_response_from_db"):
        val = agent_result.get(key)
        if val is None:
            continue
        lines.append(f"\n  [{key}]")
        lines.extend(f"    {line}" for line in str(val).splitlines())
    remaining = {
        k: v
        for k, v in agent_result.items()
        if k not in ("sql_code", "response", "sql_response_from_db")
    }
    if remaining:
        lines.append("\n  [other keys]")
        lines.extend(f"    {k}: {v}" for k, v in remaining.items())
    lines.append(sep)
    return "\n".join(lines)


def _run_agent_traced(
    payload: TextToSQLPayload, run_log: RunLogger | None
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """Run the agent, timing every graph node.

    Uses ``stream_agent_response`` rather than ``get_agent_response`` purely for
    the instrumentation. GSF emits ``{"type": "step", "phase": "start"|"end"}``
    around each node, so a node's cost is measured from its own start to its own
    end rather than inferred from the gap between consecutive events. That
    distinction matters: the gap method charges a node for whatever ran before
    it, and double-counts every visit because start and end both look like
    completions.

    ``phase`` is absent on older GSF builds, where every step event is a
    completion. That case falls back to gap timing with consecutive same-node
    events merged, which keeps totals right and visit counts honest.
    """
    timings: List[Dict[str, Any]] = []
    answer: Dict[str, Any] | None = None
    previous = time.perf_counter()
    previous_calls = http_calls()
    open_node: Dict[str, Any] | None = None

    def close(entry: Dict[str, Any]) -> None:
        timings.append(entry)
        if run_log is not None:
            run_log.event("agent_node", **entry)

    for event in stream_agent_response(payload):
        now = time.perf_counter()
        etype = event.get("type")

        if etype == "step":
            calls = http_calls()
            phase = event.get("phase")

            if phase in ("start", "end"):
                global _node_timing_mode
                _node_timing_mode = "phase"

            if phase == "start":
                # Stamped on the thread's context so the LLM callback --
                # which fires deeper in the stack, inside GSF -- can
                # attribute its call to the node that made it.
                _ctx = current_question()
                _ctx["node"] = event["node"]
                # When the node began, so each LLM call can report how long
                # it waited before reaching the network. The profile shows
                # the process is ~99% blocked with almost no CPU use, so any
                # growth in that gap is queueing between the eval thread, the
                # graph runner's executor, and the HTTP client -- not work.
                _ctx["node_started"] = now
                open_node = {
                    "node": event["node"],
                    "started": now,
                    "calls_at_start": calls,
                }
            elif phase == "end":
                # Fall back to the previous boundary when a start was missed,
                # so a node is never silently dropped from the timings.
                start = open_node["started"] if open_node else previous
                base_calls = (
                    open_node["calls_at_start"] if open_node else previous_calls
                )
                close(
                    {
                        "node": event["node"],
                        "seconds": round(now - start, 3),
                        "http_calls": calls - base_calls,
                        "thought": (event.get("thought") or "")[:500],
                    }
                )
                open_node = None
            else:
                # Legacy stream: one completion event per node, no phases.
                seconds = round(now - previous, 3)
                node_calls = calls - previous_calls
                thought = (event.get("thought") or "")[:500]
                if timings and timings[-1]["node"] == event["node"]:
                    merged = timings[-1]
                    merged["seconds"] = round(merged["seconds"] + seconds, 3)
                    merged["http_calls"] += node_calls
                    merged["thought"] = merged["thought"] or thought
                else:
                    close(
                        {
                            "node": event["node"],
                            "seconds": seconds,
                            "http_calls": node_calls,
                            "thought": thought,
                        }
                    )
            previous, previous_calls = now, calls

        elif etype == "sql":
            # Emitted once a node clears SQL for execution. Recorded on the
            # timeline only; the scored SQL still comes from the final answer.
            if run_log is not None:
                run_log.event("agent_sql", node=event.get("node"))

        elif etype == "result":
            answer = event["answer"]

        elif etype == "error":
            # Newer GSF names the failing node and returns a partial answer;
            # keep both so a failure says where it happened, not just that it did.
            raise RuntimeError(
                f"{event['message']} "
                f"[node={event.get('node') or 'unknown'} "
                f"type={event.get('error_type') or 'unknown'}]"
            )

    for entry in timings:
        logger.info(
            "  node %-28s %7.2fs  (%d http calls)",
            entry["node"],
            entry["seconds"],
            entry["http_calls"],
        )
    return answer, timings


def _attach_llm_recorder(run_log: RunLogger) -> None:
    """Register the per-call recorder on the agent's shared LLM client.

    ``gsf.retrieval.text_to_sql.main`` builds one client at import time and
    every node calls through it, so a single attachment covers the whole graph.
    Appending rather than replacing leaves any callbacks GSF configured itself
    in place.
    """
    client = getattr(gsf_agent_main, "llm_client", None)
    if client is None:
        logger.warning("No GSF llm_client — per-call LLM timing disabled")
        return
    existing = list(getattr(client, "callbacks", None) or [])
    if any(cb is run_log.llm_calls for cb in existing):
        return
    client.callbacks = existing + [run_log.llm_calls]

    def detach() -> None:
        """Take this run's recorder back off the shared client.

        The client is built once at GSF import time and outlives every run
        attached to it, so a recorder left behind keeps being called by the
        next one -- with its file already closed and its records landing
        nowhere. The guard above compares identity, and a second run brings a
        different recorder object, so it would not catch the stale one either.
        """
        client.callbacks = [
            cb
            for cb in (getattr(client, "callbacks", None) or [])
            if cb is not run_log.llm_calls
        ]

    run_log.on_close(detach)
    logger.info("Per-call LLM timing enabled (duration + token usage)")


def _blank_row(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "row_index": idx,
        "question_id": item.get("question_id", idx),
        "difficulty": item.get("difficulty", ""),
        "question": item.get("question", ""),
        "expected_sql": item.get("SQL", ""),
        "returned_sql": "",
        "sql_text_similarity": 0.0,
        "sql_exec_match": 0,
        "expected_sql_error": "",
        "returned_sql_error": "",
        "expected_answer_raw": item.get("answer_raw", ""),
        "returned_answer": "",
        "answer_text_similarity": 0.0,
        "answer_numbers_match": 0,
        "runtime_seconds": "",
        "agent_seconds": "",
        "scoring_seconds": "",
        "http_calls": 0,
        "agent_nodes": 0,
        "slowest_node": "",
        "slowest_node_seconds": "",
        "worker": "",
        "error": "",
    }


def _evaluate_question(
    idx: int,
    item: Dict[str, Any],
    *,
    total: int,
    retrievers: Dict[str, Any],
    connectors: List[Any],
    connectors_by_name: Dict[str, Any],
    run_log: RunLogger | None,
    sql_examples: Dict[str, List[Dict[str, str]]] | None = None,
    value_anchors: Dict[str, List[Dict[str, str]]] | None = None,
) -> Dict[str, Any]:
    """Run and score one question. Never raises: failures land in ``error``."""
    row = _blank_row(idx, item)
    qid = row["question_id"]
    question = row["question"]
    expected_sql = row["expected_sql"]
    expected_answer = row["expected_answer_raw"]

    # Each question may carry its own db_id / evidence; both are optional so
    # single-dataset eval files without them still work.
    evidence = item.get("evidence", "") if _INCLUDE_EVIDENCE else ""
    db_id = item.get("db_id", "")
    # Newer GSF takes evidence as its own payload field and uses it to refine
    # entity extraction, leaving the question text clean. Older builds have no
    # such field, so evidence only reaches the agent if it is appended to the
    # question — sending it as a parameter there would silently drop it.
    agent_question = (
        question
        if _SUPPORTS_EVIDENCE_PARAM or not evidence
        else f"{question}\n\nEvidence: {evidence}"
    )
    question_examples = (sql_examples or {}).get(str(qid)) or []
    question_anchors = (value_anchors or {}).get(str(qid)) or []

    worker = threading.current_thread().name
    row["worker"] = worker

    with question_context(qid=qid, row_index=idx, worker=worker):
        logger.info("[%d/%d] q%s: %s", idx + 1, total, qid, question)
        if db_id:
            logger.info("  db_id=%s  evidence=%s", db_id, str(evidence)[:120])
        if question_examples:
            logger.info("  sql_examples=%d", len(question_examples))
        if question_anchors:
            logger.info("  value_anchors=%d", len(question_anchors))
        if run_log is not None:
            run_log.event(
                "question_start",
                db_id=db_id,
                question=question,
                difficulty=row["difficulty"],
            )

        t0 = time.perf_counter()
        agent_seconds = scoring_seconds = 0.0
        node_timings: List[Dict[str, Any]] = []
        try:
            # A scoped question must use its matching database. Falling back
            # would execute and score the SQL against an unrelated connector.
            if db_id:
                active_connector = connectors_by_name.get(db_id)
                if active_connector is None:
                    raise ValueError(f"No connector configured for db_id={db_id!r}")
            else:
                active_connector = connectors[0]

            payload: TextToSQLPayload = {
                "question": agent_question,
                "data_retriever": retrievers["data"],
                "semantic_retriever": retrievers["semantic"],
                "connectors": [active_connector],
                "path_state": {},
                "custom_prompts": "",
                "acronyms": [],
                "shorten_answer": True,
            }
            if _SUPPORTS_EVIDENCE_PARAM:
                payload["evidence"] = evidence
            if question_examples and _SUPPORTS_SQL_EXAMPLES_PARAM:
                payload["sql_examples"] = question_examples
            if question_anchors and _SUPPORTS_VALUE_ANCHORS_PARAM:
                payload["value_anchors"] = question_anchors
            logger.info("Running question %s", payload["question"])

            t_agent = time.perf_counter()
            agent_result, node_timings = _run_agent_traced(payload, run_log)
            agent_seconds = time.perf_counter() - t_agent
            if run_log is not None:
                run_log.event("agent_done", seconds=round(agent_seconds, 3))

            logger.info(
                "%s", _format_agent_result(qid, question, agent_result, expected_sql)
            )
            returned_sql = (agent_result or {}).get("sql_code", "") or ""
            returned_db = (agent_result or {}).get(_DB_RESULT_KEY)
            returned_db_str = stringify_db_result(returned_db)

            row["returned_sql"] = returned_sql
            row["returned_answer"] = returned_db_str

            t_score = time.perf_counter()
            row.update(
                score_sql(active_connector, expected_sql, returned_sql, schema=db_id)
            )
            row.update(score_answer(expected_answer, returned_db_str))
            scoring_seconds = time.perf_counter() - t_score
            if run_log is not None:
                run_log.event(
                    "scoring_done",
                    seconds=round(scoring_seconds, 3),
                    sql_exec_match=row["sql_exec_match"],
                )
        except Exception as exc:
            logger.exception("Question %s failed", qid)
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["error"] += " | " + traceback.format_exc().replace("\n", " | ")[:1000]
        finally:
            total_seconds = time.perf_counter() - t0
            slowest = max(node_timings, key=lambda n: n["seconds"], default=None)
            row["runtime_seconds"] = round(total_seconds, 2)
            row["agent_seconds"] = round(agent_seconds, 2)
            row["scoring_seconds"] = round(scoring_seconds, 2)
            row["http_calls"] = http_calls()
            row["agent_nodes"] = len(node_timings)
            row["slowest_node"] = slowest["node"] if slowest else ""
            row["slowest_node_seconds"] = slowest["seconds"] if slowest else ""

            if run_log is not None:
                run_log.question(
                    {
                        "qid": qid,
                        "row_index": idx,
                        "db_id": db_id,
                        "difficulty": row["difficulty"],
                        "worker": worker,
                        "question": question,
                        "total_seconds": round(total_seconds, 3),
                        "agent_seconds": round(agent_seconds, 3),
                        "scoring_seconds": round(scoring_seconds, 3),
                        "http_calls": row["http_calls"],
                        "node_timings": node_timings,
                        "slowest_node": row["slowest_node"],
                        "slowest_node_seconds": row["slowest_node_seconds"],
                        "sql_exec_match": row["sql_exec_match"],
                        "returned_sql": row["returned_sql"],
                        "error": row["error"],
                    }
                )
                run_log.event(
                    "question_end",
                    seconds=round(total_seconds, 3),
                    error=bool(row["error"]),
                )
            logger.info(
                "[%d/%d] q%s done in %.2fs (agent %.2fs, scoring %.2fs, "
                "%d http calls, exec_match=%s)",
                idx + 1,
                total,
                qid,
                total_seconds,
                agent_seconds,
                scoring_seconds,
                row["http_calls"],
                row["sql_exec_match"],
            )
    return row


def _sort_csv_by_row_index(path: Path) -> None:
    """Rewrite *path* ordered by ``row_index``.

    Rows are appended as they finish so a killed run keeps its results, which
    means concurrent runs land out of order. Restoring question order at the end
    keeps the CSV diffable against a serial run.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or CSV_FIELDS)
        rows = list(reader)

    def key(row: Dict[str, str]) -> int:
        try:
            return int(row.get("row_index") or 0)
        except (TypeError, ValueError):
            return 0

    rows.sort(key=key)
    # Via a sibling temp file rather than reopening *path* with "w": truncating
    # in place puts the only copy of an expensive run's results in memory, so an
    # interrupt between the truncate and the writerows loses all of them.
    tmp = path.with_name(path.name + ".sorted.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _log_summary(summary: Dict[str, Any]) -> None:
    sep = "=" * 78
    logger.info("%s", sep)
    logger.info("  EVAL RUN SUMMARY (%s)", summary["run_id"])
    logger.info("%s", sep)
    logger.info("  Questions              : %s", summary["questions"])
    logger.info("  Errors                 : %s", summary["errors"])
    logger.info("  Wall clock             : %.2fs", summary["wall_clock_seconds"])
    logger.info(
        "  Throughput             : %s questions/min", summary["questions_per_minute"]
    )
    logger.info("  Workers requested      : %s", summary.get("workers"))
    logger.info("  Effective parallelism  : %s", summary["effective_parallelism"])
    logger.info("  Agent share of busy    : %s", summary["agent_share_of_busy_time"])
    logger.info(
        "  Model calls            : %s total (%s/question)",
        sum((summary.get("http_endpoints") or {}).values()),
        summary.get("http_calls_per_question"),
    )
    for endpoint, count in (summary.get("http_endpoints") or {}).items():
        logger.info("    %-20s %5d", endpoint, count)
    for phase, stats in (summary.get("phases") or {}).items():
        logger.info(
            "  %-22s mean %.2fs  median %.2fs  p95 %.2fs  max %.2fs",
            phase,
            stats["mean"],
            stats["median"],
            stats["p95"],
            stats["max"],
        )
    logger.info("%s", sep)
    logger.info("  SLOWEST AGENT NODES (total seconds across all questions)")
    for node, stats in list((summary.get("agent_nodes") or {}).items())[:12]:
        logger.info(
            "    %-30s total %8.2fs  n=%-4d mean %6.2fs  max %6.2fs",
            node,
            stats["total"],
            stats["count"],
            stats["mean"],
            stats["max"],
        )
    logger.info("%s", sep)


def run_evaluation(
    input_path: Path,
    output_path: Path,
    start_index: int = 0,
    end_index: int | None = None,
    workers: int = 1,
    run_log: RunLogger | None = None,
    sql_examples_path: Path | None = None,
    sql_examples_k: int = 3,
    sql_examples_min_score: float = 0.0,
    value_anchors_path: Path | None = None,
) -> None:
    """Run every selected question and write one CSV row each.

    ``workers`` questions run concurrently. The agent graph is stateless per
    invocation and the connectors hand out one connection per thread, so
    questions do not interfere; the only shared mutable state here is the CSV
    writer, guarded by a lock.
    """
    all_questions = _load_questions(input_path)
    questions = all_questions[start_index:end_index]
    if not questions:
        logger.warning("No questions selected from %s — nothing to do.", input_path)
        return

    sql_examples, value_anchors = _load_prompt_extras(
        sql_examples_path, sql_examples_k, sql_examples_min_score, value_anchors_path
    )

    workers = max(1, workers)
    logger.info(
        "Running questions %d–%d (%d of %d total) from %s with %d worker(s)",
        start_index,
        start_index + len(questions) - 1,
        len(questions),
        len(all_questions),
        input_path,
        workers,
    )

    if run_log is not None:
        _attach_llm_recorder(run_log)

    t_setup = time.perf_counter()
    retrievers = {
        "data": get_data_objects_retriever(),
        "semantic": get_semantic_objects_retriever(),
    }
    connectors = get_connectors()
    connectors_by_name: dict[str, Any] = {}
    for connector in connectors:
        name = getattr(connector, "database_name", None)
        if name:
            connectors_by_name[name] = connector
    setup_seconds = time.perf_counter() - t_setup
    logger.info(
        "Retrievers + %d connector(s) ready in %.2fs", len(connectors), setup_seconds
    )
    if run_log is not None:
        run_log.event(
            "setup_done",
            seconds=round(setup_seconds, 3),
            connectors=sorted(connectors_by_name),
        )

    resuming = start_index > 0 and output_path.exists()
    mode = "a" if resuming else "w"
    total = len(questions)
    write_lock = threading.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not resuming:
            writer.writeheader()
            f.flush()

        def emit(row: Dict[str, Any]) -> None:
            with write_lock:
                writer.writerow(row)
                f.flush()

        def work(pair: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            idx, item = pair
            row = _evaluate_question(
                idx,
                item,
                total=total,
                retrievers=retrievers,
                connectors=connectors,
                connectors_by_name=connectors_by_name,
                run_log=run_log,
                sql_examples=sql_examples,
                value_anchors=value_anchors,
            )
            emit(row)
            return row

        indexed = list(enumerate(questions, start=start_index))
        if workers == 1:
            for pair in indexed:
                work(pair)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="evalq"
            ) as pool:
                futures = [pool.submit(work, pair) for pair in indexed]
                for done, future in enumerate(as_completed(futures), start=1):
                    # work() swallows per-question errors, so a raised exception
                    # here is a bug in the driver itself — surface it, loudly.
                    future.result()
                    logger.info("Progress: %d/%d questions complete", done, total)

    if workers > 1:
        _sort_csv_by_row_index(output_path)

    if run_log is not None:
        # Recorded here rather than at the CLI so the pipeline in main.py gets
        # it too. Downstream analysis reads node_timing_mode to decide whether
        # per-node entries need merging; a summary missing it is silently
        # treated as the older format and mis-aggregated.
        run_log.note(
            node_timing_mode=_node_timing_mode,
            evidence_as_parameter=_SUPPORTS_EVIDENCE_PARAM,
            sql_examples_source=str(sql_examples_path) if sql_examples else "",
            sql_examples_k=sql_examples_k if sql_examples else 0,
            sql_examples_min_score=sql_examples_min_score,
            sql_examples_indexed=len(sql_examples),
            value_anchors_source=str(value_anchors_path) if value_anchors else "",
            value_anchors_indexed=len(value_anchors),
        )

    logger.info("Wrote scores to %s", output_path)


def run_single_question(
    question: str,
    run_log: RunLogger | None = None,
    evidence: str = "",
    sql_examples: List[Dict[str, str]] | None = None,
    value_anchors: List[Dict[str, str]] | None = None,
    connectors: List[Any] | None = None,
) -> None:
    """Run a single question through the agent and print the result.

    Everything past *question* is what the batch path sends, so a question
    debugged here is asked the way the eval asks it.
    """
    payload: TextToSQLPayload = {
        "question": (
            question
            if _SUPPORTS_EVIDENCE_PARAM or not evidence
            else f"{question}\n\nEvidence: {evidence}"
        ),
        "data_retriever": get_data_objects_retriever(),
        "semantic_retriever": get_semantic_objects_retriever(),
        "connectors": connectors if connectors is not None else get_connectors(),
        "path_state": {},
        "custom_prompts": "",
        "acronyms": [],
        "shorten_answer": True,
    }
    if _SUPPORTS_EVIDENCE_PARAM:
        payload["evidence"] = evidence
    if sql_examples and _SUPPORTS_SQL_EXAMPLES_PARAM:
        payload["sql_examples"] = sql_examples
    if value_anchors and _SUPPORTS_VALUE_ANCHORS_PARAM:
        payload["value_anchors"] = value_anchors
    with question_context(qid="single", row_index=0):
        t0 = time.perf_counter()
        agent_result, node_timings = _run_agent_traced(payload, run_log)
        elapsed = time.perf_counter() - t0
        logger.info("%s", _format_agent_result("single", question, agent_result))
        for entry in sorted(node_timings, key=lambda n: -n["seconds"]):
            logger.info(
                "  %-30s %7.2fs  (%d http calls)",
                entry["node"],
                entry["seconds"],
                entry["http_calls"],
            )
        logger.info("Runtime: %.2fs", elapsed)


def run_single_dataset_question(
    input_path: Path,
    question_id: str,
    run_log: RunLogger | None = None,
    sql_examples_path: Path | None = None,
    sql_examples_k: int = 3,
    sql_examples_min_score: float = 0.0,
    value_anchors_path: Path | None = None,
) -> None:
    """Run one question from the evaluation file, asked as the batch run asks it.

    Debugging a question in isolation is only informative if nothing about the
    request changed, so its own evidence, exemplars and anchors are loaded here
    too, and it goes to the connector for its ``db_id`` rather than to whichever
    connector happens to come first.
    """
    item = next(
        (
            question
            for question in _load_questions(input_path)
            if str(question.get("question_id", "")) == str(question_id)
        ),
        None,
    )
    if item is None:
        raise SystemExit(f"No question_id={question_id!r} in {input_path}")

    sql_examples, value_anchors = _load_prompt_extras(
        sql_examples_path, sql_examples_k, sql_examples_min_score, value_anchors_path
    )
    question = str(item.get("question", "")).strip()
    evidence = item.get("evidence", "") if _INCLUDE_EVIDENCE else ""
    db_id = item.get("db_id", "")

    connectors = get_connectors()
    if db_id:
        by_name = {
            getattr(connector, "database_name", None): connector
            for connector in connectors
        }
        if db_id not in by_name:
            raise SystemExit(f"No connector configured for db_id={db_id!r}")
        connectors = [by_name[db_id]]

    question_examples = sql_examples.get(str(question_id)) or []
    question_anchors = value_anchors.get(str(question_id)) or []
    logger.info("q%s (db_id=%s): %s", question_id, db_id or "-", question)
    if question_examples:
        logger.info("  sql_examples=%d", len(question_examples))
    if question_anchors:
        logger.info("  value_anchors=%d", len(question_anchors))

    run_single_question(
        question,
        run_log,
        evidence=evidence,
        sql_examples=question_examples,
        value_anchors=question_anchors,
        connectors=connectors,
    )


SINGLE_QUERY = "calculate the customer count by state province name"


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
        "--workers",
        type=int,
        default=1,
        help="Number of questions to run concurrently (default: 1).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First question index to run (0-based, default: 0). "
        "A non-zero value appends to an existing output CSV.",
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
        help="Run at most this many questions from --start-index "
        "(shorthand for --end-index; ignored when --end-index is given).",
    )
    parser.add_argument(
        "--sql-examples",
        type=Path,
        default=(
            Path(os.environ["SQL_EXAMPLES_CSV"])
            if os.environ.get("SQL_EXAMPLES_CSV")
            else None
        ),
        help="CSV of retrieved question/SQL exemplars to inject into the SQL "
        "generation prompt, as written by scripts/find_structural_exemplars.py. "
        "Omitted means no exemplars (default). Env: SQL_EXAMPLES_CSV.",
    )
    parser.add_argument(
        "--sql-examples-k",
        type=int,
        default=int(os.environ.get("SQL_EXAMPLES_K", "3")),
        help="How many exemplars per question to inject, by rank "
        "(default: 3). Env: SQL_EXAMPLES_K.",
    )
    parser.add_argument(
        "--sql-examples-min-score",
        type=float,
        default=float(os.environ.get("SQL_EXAMPLES_MIN_SCORE", "0")),
        help="Withhold exemplars from a question whose best retrieval score is "
        "below this, so it is answered under the general shape rules instead. "
        "0 (default) attaches exemplars to every question. Env: "
        "SQL_EXAMPLES_MIN_SCORE.",
    )
    parser.add_argument(
        "--value-anchors",
        type=Path,
        default=(
            Path(os.environ["VALUE_ANCHORS_CSV"])
            if os.environ.get("VALUE_ANCHORS_CSV")
            else None
        ),
        help="CSV of question phrases looked up against the database, as "
        "written by scripts/value_index.py anchors. Tells the model which "
        "column stores a value and its exact spelling. Omitted means no "
        "anchors (default). Env: VALUE_ANCHORS_CSV.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_LOG_DIR,
        help=f"Root directory for run logs (default: {_LOG_DIR}).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Name for this run's log directory (default: a UTC-local timestamp).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Also print DEBUG records to the console "
        "(they always go to logs/<run-id>/run.log).",
    )
    parser.add_argument(
        "--single",
        nargs="?",
        const="",
        default=None,
        metavar="QUESTION_ID",
        help="Run one question instead of the whole file. With a question_id "
        "from the evaluation file, that question is asked exactly as the batch "
        "run asks it: its own evidence, exemplars and anchors, routed to its "
        "db_id. Bare --single runs the hardcoded SINGLE_QUERY.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_log = setup_run_logging(
        args.log_dir,
        args.run_id,
        console_level=logging.DEBUG if args.verbose else logging.INFO,
    )
    try:
        dataset_name = args.database_name
        if not dataset_name and not (args.input and args.output):
            dataset_name = dataset_name_from_env()
            logger.info(
                "Resolved dataset name from CONNECTION_STRINGS: %s", dataset_name
            )
        input_path, output_path = _resolve_paths(dataset_name, args.input, args.output)

        end_index = args.end_index
        if end_index is None and args.limit is not None:
            end_index = args.start_index + args.limit

        run_log.log_environment(
            dataset=dataset_name,
            input_path=str(input_path),
            output_path=str(output_path),
            workers=args.workers,
            start_index=args.start_index,
            end_index=end_index,
            single=args.single,
            sql_examples=str(args.sql_examples or ""),
            sql_examples_k=args.sql_examples_k,
            sql_examples_min_score=args.sql_examples_min_score,
            value_anchors=str(args.value_anchors or ""),
        )
        logger.info("Logging this run to %s", run_log.dir)

        if args.single is not None:
            if args.single:
                run_single_dataset_question(
                    input_path,
                    args.single,
                    run_log,
                    sql_examples_path=args.sql_examples,
                    sql_examples_k=args.sql_examples_k,
                    sql_examples_min_score=args.sql_examples_min_score,
                    value_anchors_path=args.value_anchors,
                )
            else:
                run_single_question(SINGLE_QUERY, run_log)
        else:
            run_evaluation(
                input_path=input_path,
                output_path=output_path,
                start_index=args.start_index,
                end_index=end_index,
                workers=args.workers,
                run_log=run_log,
                sql_examples_path=args.sql_examples,
                sql_examples_k=args.sql_examples_k,
                sql_examples_min_score=args.sql_examples_min_score,
                value_anchors_path=args.value_anchors,
            )
            summary = run_log.write_summary(
                workers=args.workers,
                dataset=dataset_name,
                output_path=str(output_path),
            )
            _log_summary(summary)
            logger.info("Full run logs: %s", run_log.dir)
    finally:
        run_log.close()


if __name__ == "__main__":
    main()
