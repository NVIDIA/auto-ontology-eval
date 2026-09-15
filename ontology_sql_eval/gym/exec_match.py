# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-match scoring shared by both evaluation arms.

Both arms are graded by this module so that a difference in the reported number
can only come from the SQL, never from the grader.

Comparison follows the official BIRD rule -- ``set(gold) == set(pred)``, which
ignores row order *and* collapses duplicate rows -- rather than the DataFrame
shape-and-multiset comparison in :mod:`ontology_sql_eval.retrieval.scoring`.
The two disagree (a query returning ten identical rows where gold returns one
passes BIRD and fails the DataFrame rule), and matching BIRD keeps our headline
number quotable against published results.

Queries run against the database file or DSN directly, not through a GSF
connector, so the schema-only arm carries no GSF dependency at all.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

ResultSet = list[tuple[Any, ...]]

__all__ = [
    "FailureCode",
    "SqliteExecutor",
    "PostgresExecutor",
    "extract_sql",
    "has_sql_codeblock",
    "result_sets_match",
    "execute_and_compare",
]

_NO_ANSWER_FILLER = "SELECT 1"


class FailureCode(str, Enum):
    """Why a task scored zero.

    The distinction between these is the point. A run that was rate-limited into
    producing no output and a run where the model wrote genuinely wrong SQL both
    score 0.0; collapsing them lets an infrastructure failure masquerade as a
    benchmark result. ``NO_MODEL_OUTPUT`` exists specifically because that
    happened: a 66%-rate-limited bird60 run reported a plausible-looking 13.3%.
    """

    NONE = "none"
    NO_MODEL_OUTPUT = "no_model_output"
    WRONG_RESULT = "wrong_result"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_TIMEOUT = "execution_timeout"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    GOLD_EXECUTION_TIMEOUT = "gold_execution_timeout"
    UNKNOWN_ERROR = "unknown_error"


def has_sql_codeblock(text: Optional[str]) -> bool:
    """True iff the response holds a ```sql block containing at least one letter."""
    if not text:
        return False
    return bool(re.search(r"(?:```sql)(.*?[a-zA-Z].*?)(?:```)", text, flags=re.DOTALL))


def extract_sql(text: Optional[str]) -> str:
    """Pull SQL out of a fenced ```sql block.

    Mirrors NeMo Gym's ``bird_sql`` extraction so the arms stay comparable with
    published numbers, including its quirks: the *last* block wins, and comment
    stripping runs with ``DOTALL`` but not ``MULTILINE``, so a ``--`` comment
    eats the remainder. When no block is present a harmless ``SELECT 1`` filler
    is returned; it executes cleanly and matches essentially no gold query.
    """
    if not text:
        return _NO_ANSWER_FILLER

    matches = re.findall(r"(?:```sql)(.*?[a-zA-Z].*?)(?:```)", text, flags=re.DOTALL)
    if not matches:
        return _NO_ANSWER_FILLER

    ans = matches[-1]
    ans = re.sub(r"--.*?$|/\*.*?\*/", "", ans, flags=re.DOTALL)
    ans = re.sub(r"\s+", " ", ans)
    return re.sub(r"^\*\*.*\*\*", "", ans).strip()


def result_sets_match(gold: ResultSet, pred: ResultSet) -> bool:
    """BIRD's comparison: unordered set equality over row tuples."""
    try:
        return set(gold) == set(pred)
    except TypeError:
        # Unhashable cells (bytes, lists) -- fall back to a sorted repr compare.
        try:
            return sorted(map(repr, gold)) == sorted(map(repr, pred))
        except Exception:
            return False


class Executor(Protocol):
    """Runs a query for a ``db_id``.

    Returns ``(rows, status)`` where ``status`` is ``""`` on success and
    ``"timeout"`` or ``"error"`` otherwise -- the caller needs to tell a slow
    query apart from a broken one.
    """

    async def execute(
        self, db_id: str, sql: str
    ) -> tuple[Optional[ResultSet], str]: ...


def _run_sqlite(db_path: Path, sql: str) -> tuple[Optional[ResultSet], str]:
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.text_factory = lambda b: b.decode(errors="ignore")
            return conn.execute(sql).fetchall(), ""
    except Exception:
        return None, "error"


class SqliteExecutor:
    """Executes against ``<db_root>/<db_id>/<db_id>.sqlite`` (BIRD, FDABench)."""

    def __init__(
        self,
        db_root: Path,
        *,
        max_concurrency: int = 32,
        timeout_s: float = 30.0,
    ) -> None:
        self._db_root = db_root
        self._timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def db_path(self, db_id: str) -> Path:
        return self._db_root / db_id / f"{db_id}.sqlite"

    async def execute(self, db_id: str, sql: str) -> tuple[Optional[ResultSet], str]:
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_sqlite, self.db_path(db_id), sql),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError:
                return None, "timeout"


def _run_postgres(dsn: str, sql: str) -> tuple[Optional[ResultSet], str]:
    try:
        import psycopg

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall(), ""
    except Exception:
        return None, "error"


class PostgresExecutor:
    """Executes against a single Postgres DSN (WideWorldImporters).

    ``db_id`` is accepted and ignored: the dataset is one database, and keeping
    the signature identical to :class:`SqliteExecutor` lets the verifier stay
    dialect-agnostic.
    """

    def __init__(
        self,
        dsn: str,
        *,
        max_concurrency: int = 8,
        timeout_s: float = 30.0,
    ) -> None:
        self._dsn = dsn
        self._timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(self, db_id: str, sql: str) -> tuple[Optional[ResultSet], str]:
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_postgres, self._dsn, sql),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError:
                return None, "timeout"


async def execute_and_compare(
    executor: Executor,
    db_id: str,
    gold_sql: str,
    pred_sql: str,
) -> tuple[bool, Optional[str]]:
    """Run both queries and compare. Returns ``(match, error_tag)``.

    Gold is executed first: if the benchmark's own query does not run, that is a
    dataset problem and the task must be excluded from the denominator rather
    than counted as a model failure.
    """
    gold_rows, gold_status = await executor.execute(db_id, gold_sql)
    if gold_rows is None:
        return False, f"gold_sql_{gold_status or 'error'}"

    pred_rows, pred_status = await executor.execute(db_id, pred_sql)
    if pred_rows is None:
        return False, f"pred_sql_{pred_status or 'error'}"

    return result_sets_match(gold_rows, pred_rows), None
