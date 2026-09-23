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
import contextlib
import re
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import unquote, urlparse

ResultSet = list[tuple[Any, ...]]

UNUSABLE_FAILURES = frozenset({"gold_execution_error", "gold_execution_timeout"})


def drop_unusable_tasks(
    tasks: "list[list[dict]]",
) -> "tuple[list[list[dict]], int]":
    """Split out tasks whose *gold* query never ran.

    A task whose benchmark-supplied query errors or times out tells us nothing
    about the model, so counting it as a miss depresses every arm equally and
    silently understates all of them. Returns ``(usable_tasks, dropped_count)``.

    This has to be done here rather than by setting ``instance_config.mask_sample``:
    that flag is consumed by nemo-gym's token-id capture to mask samples out of
    *training* data, and ``nemo_gym.reward_profile`` -- which computes pass@k --
    does not consult it. Both are set, for their respective consumers.
    """
    usable: "list[list[dict]]" = []
    dropped = 0
    for repeats in tasks:
        kept = [r for r in repeats if r.get("failure_reason") not in UNUSABLE_FAILURES]
        if kept:
            usable.append(kept)
        elif repeats:
            dropped += 1
    return usable, dropped


__all__ = [
    "UNUSABLE_FAILURES",
    "drop_unusable_tasks",
    "FailureCode",
    "SqliteExecutor",
    "PostgresExecutor",
    "MySqlExecutor",
    "extract_sql",
    "has_sql_codeblock",
    "result_sets_match",
    "beaver_result_match",
    "matcher_for",
    "execute_and_compare",
]

_NO_ANSWER_FILLER = "SELECT 1"


class FailureCode(str, Enum):
    """Why a task scored zero.

    The distinction between these is the point. A run that was rate-limited into
    producing no output and a run where the model wrote genuinely wrong SQL both
    score 0.0; collapsing them lets an infrastructure failure masquerade as a
    benchmark result. ``NO_MODEL_OUTPUT`` exists specifically because that
    happened: a 66%-rate-limited 60-question run reported a plausible-looking 13.3%.
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


def beaver_result_match(gold: ResultSet, pred: ResultSet) -> bool:
    """BEAVER's official ex_acc comparison, expressed on raw rows.

    Deliberately *not* BIRD's rule, and the two disagree often enough to matter:
    BEAVER stringifies and strips every value before comparing, so ``1`` equals
    ``"1"`` and ``" a"`` equals ``"a"``, but ``Decimal("150.250")`` does **not**
    equal ``Decimal("150.25")``. On a MySQL corpus, where every numeric comes
    back as ``Decimal``, that last case is common rather than exotic.

    Kept in step with ``ontology_sql_eval.judge.beaver.compare_results`` by
    ``tests/test_gym_beaver_match.py`` rather than by importing it -- that module
    pulls in pandas, which the control arm's venv deliberately does not have.
    """
    gold_empty, pred_empty = not gold, not pred
    if gold_empty and pred_empty:
        # Both sides produced nothing. BEAVER scores this a match; its own
        # evaluator compensates by reporting accuracy with and without
        # empty-gold questions.
        return True
    if gold_empty or pred_empty:
        return False

    def norm(rows: ResultSet) -> "set[tuple[str, ...]]":
        return {tuple(str(v).strip() for v in row) for row in rows}

    if len(gold[0]) != len(pred[0]):
        return False
    return norm(gold) == norm(pred)


# Comparison rule per dataset. Everything defaults to BIRD's; BEAVER gets its
# own so its numbers stay comparable to the published leaderboard.
MATCHERS = {"beaverbench": beaver_result_match}


def matcher_for(dataset: str):
    """Return the result-comparison function a dataset should be scored with."""
    return MATCHERS.get(dataset, result_sets_match)


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


def _run_mysql(dsn: str, sql: str) -> tuple[Optional[ResultSet], str]:
    try:
        import mysql.connector

        with contextlib.closing(mysql.connector.connect(**_mysql_params(dsn))) as conn:
            with contextlib.closing(conn.cursor()) as cur:
                cur.execute(sql)
                return list(cur.fetchall()), ""
    except Exception:
        return None, "error"


def _mysql_params(dsn: str) -> dict[str, Any]:
    """Split a ``mysql://user:pass@host:port/db`` URI into connector kwargs."""
    parsed = urlparse(dsn)
    return {
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "database": (parsed.path or "/").lstrip("/"),
    }


class MySqlExecutor:
    """Executes against a MySQL database (BEAVER).

    BEAVER ships one physical database per ``db_id``, so the DSN is selected per
    task rather than fixed at construction: ``dw_real`` questions deliberately
    run against the ``dw`` database, which a single connection string could not
    express.
    """

    def __init__(
        self,
        dsn_for_db: "dict[str, str]",
        *,
        max_concurrency: int = 8,
        timeout_s: float = 30.0,
    ) -> None:
        self._dsn_for_db = dsn_for_db
        self._timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(self, db_id: str, sql: str) -> tuple[Optional[ResultSet], str]:
        dsn = self._dsn_for_db.get(db_id)
        if dsn is None:
            return None, "error"
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_mysql, dsn, sql),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError:
                return None, "timeout"


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
    dataset: str = "",
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

    return matcher_for(dataset)(gold_rows, pred_rows), None
