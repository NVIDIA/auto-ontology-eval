# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the eval datasets into NeMo Gym task records.

Our datasets are JSON *arrays* (``datasets/<name>/evaluation.json``); Gym wants
JSONL where each line carries the prompt under ``responses_create_params`` plus
whatever extra fields the resources server's verify request declares. The record
shape here deliberately mirrors Gym's own ``bird_sql`` environment
(``question`` / ``gt_sql`` / ``sql_context`` / ``difficulty`` / ``db_id`` / ``id``)
so our numbers stay comparable to the published ones.

One JSONL is built per (dataset, arm). The arms differ only in what the prompt
contains:

* ``schema_only`` -- the full schema dump is inlined, and the model answers in
  one shot. This is the control.
* ``auto_ontology`` -- only the question travels; grounding is retrieved by Auto Ontology behind a
  tool call, so inlining a schema would defeat the comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from ontology_sql_eval.gym.ddl import (
    column_description_block,
    mysql_schema_dump,
    postgres_schema_dump,
    sqlite_schema_dump,
)

Arm = Literal["schema_only", "auto_ontology"]

# The first sentence is upstream bird_sql's prompt verbatim. The projection rule
# is added to match what Auto Ontology receives via shorten_answer=True, which appends
# "Return exactly the requested output fields and NO others" to its own prompt.
#
# Both arms need it or neither does: extra columns break the
# set(gold) == set(pred) comparison outright, so giving the instruction to only
# one arm measures projection guidance rather than ontology grounding.
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within the tags "
    '"```sql" and "```".\n'
    "Return exactly the requested output fields and NO others."
)

__all__ = ["DATASETS", "DatasetSpec", "build_records", "write_jsonl", "load_questions"]


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's questions, databases and schema live."""

    name: str
    dialect: str
    # A dataset's questions and its databases can live in different
    # directories: seed_bird.py writes evaluation.json to datasets/bird/ but the
    # databases to datasets/bird/dev/.
    db_root: str

    @property
    def evaluation_json(self) -> Path:
        return Path("datasets") / self.name / "evaluation.json"

    def sqlite_path(self, db_id: str) -> Path:
        return Path("datasets") / self.db_root / db_id / f"{db_id}.sqlite"

    def description_dir(self, db_id: str) -> Path:
        return Path("datasets") / self.db_root / db_id / "database_description"

    def mysql_dump_path(self, db_id: str) -> Path:
        return Path("datasets") / self.db_root / "dumps" / f"{db_id}.sql"


DATASETS: dict[str, DatasetSpec] = {
    # db_root is "bird/dev", not "bird": seed_bird.py writes evaluation.json to
    # datasets/bird/ but the databases to datasets/bird/dev/<db_id>/. This is
    # exactly why name and db_root are separate fields.
    "bird": DatasetSpec("bird", "sqlite", "bird/dev"),
    "fdabench": DatasetSpec("fdabench", "sqlite", "fdabench"),
    "wideworldimporters": DatasetSpec(
        "wideworldimporters", "postgres", "wideworldimporters"
    ),
    # BEAVER. Gold SQL is MySQL dialect and each db_id is its own physical
    # database. Seeded by scripts/seed_beaverbench.py; not committed.
    "beaverbench": DatasetSpec("beaverbench", "mysql", "beaverbench"),
}


def load_questions(spec: DatasetSpec, root: Path) -> list[dict[str, Any]]:
    """Read a dataset's ``evaluation.json`` (a JSON array, not JSONL)."""
    path = root / spec.evaluation_json
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def _schema_for(spec: DatasetSpec, root: Path, db_id: str, descriptions: bool) -> str:
    """Build the schema text for one database, cached by the caller."""
    if spec.dialect == "postgres":
        return postgres_schema_dump(root / "datasets" / spec.db_root / "ddl")

    if spec.dialect == "mysql":
        return mysql_schema_dump(root / spec.mysql_dump_path(db_id))

    dump = sqlite_schema_dump(root / spec.sqlite_path(db_id))
    if not descriptions:
        return dump
    block = column_description_block(root / spec.description_dir(db_id))
    return f"{dump}\n\n{block}" if block else dump


def _user_prompt(question: str, evidence: str, dialect: str, schema: str | None) -> str:
    """Assemble the user turn.

    ``schema`` is ``None`` for the Auto Ontology arm, which retrieves its own grounding.
    The dialect is always stated: unlike upstream's BIRD-only environment our
    corpora span SQLite and Postgres, and the model cannot infer which from the
    question alone.
    """
    parts = [f"### Question\n{question}"]
    if evidence:
        parts.append(f"### Evidence\n{evidence}")
    parts.append(f"Write a single {dialect} query that answers the question.")
    if schema is not None:
        parts.append(
            "The following is a SQL dump that describes the database and the "
            f"tables in it.\n{schema}"
        )
    return "\n\n".join(parts)


def build_records(
    spec: DatasetSpec,
    root: Path,
    arm: Arm,
    *,
    descriptions: bool = True,
    limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one Gym task record per question.

    ``start_index``/``end_index`` slice the evaluation set so a long run can be
    sharded or resumed; ``limit`` caps the slice, so the two compose.
    """
    questions = load_questions(spec, root)
    stop = end_index if end_index is not None else len(questions)
    questions = questions[start_index:stop]
    if limit is not None:
        questions = questions[:limit]

    schema_cache: dict[str, str] = {}
    for index, q in enumerate(questions):
        db_id = q["db_id"]
        schema: str | None = None
        if arm == "schema_only":
            if db_id not in schema_cache:
                schema_cache[db_id] = _schema_for(spec, root, db_id, descriptions)
            schema = schema_cache[db_id]

        question = q["question"]
        evidence = (q.get("evidence") or "").strip()
        # question_id is an int on BIRD/WWI but a string like "FDA0002" on
        # FDABench, so the joinable key is always rendered as text.
        #
        # Never fall back to the list index: task_id is what compare_arms.py
        # joins the two arms on, and an index shifts under --limit or any
        # reordering of evaluation.json, silently pairing a control question
        # with a *different* treatment question. A content hash is stable
        # whatever the position. Note `.get(k, default)` would not help here --
        # it returns None when the key exists with a null value.
        raw_id = q.get("question_id")
        if raw_id is None:
            digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
            raw_id = f"q{digest}"

        yield {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _user_prompt(
                            question, evidence, spec.dialect, schema
                        ),
                    },
                ]
            },
            "task_id": f"{spec.name}-{raw_id}",
            "question": question,
            "gt_sql": q["SQL"],
            "sql_context": schema or "",
            "difficulty": q.get("difficulty") or "",
            "db_id": db_id,
            "dataset": spec.name,
            "dialect": spec.dialect,
            "evidence": evidence,
            "answer_raw": q.get("answer_raw") or "",
            "id": raw_id,
        }


def write_jsonl(records: Iterator[dict[str, Any]], path: Path) -> int:
    """Write records as JSONL, returning the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n
