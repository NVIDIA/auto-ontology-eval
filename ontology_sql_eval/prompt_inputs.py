# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loaders for the question-keyed prompt inputs under ``prompt_inputs/``.

Both files are built by ``scripts/build_eval_inputs.py`` and keyed by
``question_id``: structural exemplars drawn from the BIRD Train corpus, and
value anchors verified against the Dev databases.

The NeMo Gym Auto Ontology resources server loads them from here.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

__all__ = ["load_sql_examples", "load_value_anchors"]


def load_sql_examples(
    path: Path, k: int, min_score: float = 0.0
) -> Dict[str, List[Dict[str, str]]]:
    """Map question_id to its top-*k* retrieved question/SQL exemplars.

    Reads the CSV written by ``scripts/build_eval_inputs.py``, which ranks
    train-set examples by how closely their SQL structure matches the question.
    Keys are stringified because the CSV always yields text while the dataset
    yields ints.

    Retrieval always returns something, and a retriever ranking on question
    wording scored only weakly on whether the structure transfers (r=0.20
    against gold skeleton similarity on BIRD dev), so most questions got a
    pattern that was merely a verbal coincidence. ``min_score`` withholds
    patterns from a question whose best match scores below it, leaving that
    question to be answered under the
    general shape rules instead. The gate is per question rather than per
    exemplar so that a kept question still gets a full *k*.
    """
    if not path.exists():
        raise SystemExit(
            f"SQL examples file not found: {path}\n"
            "The repo ships one at prompt_inputs/bird/"
            "predicted_structural_exemplars.csv; rebuild it with: "
            "python scripts/build_eval_inputs.py --only exemplars"
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
                "Regenerate it with: python scripts/build_eval_inputs.py "
                "--only exemplars"
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


def load_value_anchors(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Map question_id to the database values looked up for its question.

    Reads the CSV written by ``scripts/value_index.py anchors``. Rows of kind
    "absent" carry no column and record a phrase the database does not store;
    they are kept because that negative is what stops an invented literal.
    """
    if not path.exists():
        raise SystemExit(
            f"Value anchors file not found: {path}\n"
            "The repo ships one at prompt_inputs/bird/value_anchors.csv; "
            "rebuild it with: python scripts/build_eval_inputs.py "
            "--only anchors"
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
