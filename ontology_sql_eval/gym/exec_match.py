# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-match scoring shared by both evaluation arms.

Both arms are graded by this module so that a difference in the reported number
can only come from the SQL, never from the grader.

Comparison follows the official BIRD rule -- ``set(gold) == set(pred)``, which
ignores row order *and* collapses duplicate rows -- rather than a stricter
shape-and-multiset comparison. The two disagree (a query returning ten
identical rows where gold returns one passes BIRD and fails a multiset rule),
and matching BIRD keeps our headline number quotable against published results.

BEAVER is the exception: it is scored with BEAVER's own official ex_acc rule
(:func:`beaver_result_match`, a 10-second query timeout, failed or missing
queries scored as empty results, and accuracy reported both with and without
empty-gold questions) so its numbers are quotable against the BEAVER
leaderboard instead.

Spider2-lite is the other: it ships gold *result* CSVs rather than gold SQL, so
its tasks carry :class:`~ontology_sql_eval.gym.spider2.GoldResults`, only the
prediction is executed, and the comparison is Spider2's own
(:mod:`ontology_sql_eval.gym.spider2`). Its BigQuery questions
(``spider2_lite_bq``) run through :class:`BigQueryExecutor`, under the byte cap
and project qualification of :mod:`ontology_sql_eval.bigquery`.

Queries run against the database file or DSN directly, not through an Auto Ontology
connector, so the schema-only arm carries no Auto Ontology dependency at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import unquote, urlparse

from ontology_sql_eval.bigquery import (
    BigQueryTarget,
    BytesCapExceeded,
    NotReadOnly,
    QueryTimeout,
    is_infrastructure_error,
    make_client,
    run_query,
)
from ontology_sql_eval.gym.spider2 import (
    GoldResults,
    Table,
    gold_is_empty,
    prediction_table,
    results_match,
    table_from_csv_text,
)

logger = logging.getLogger(__name__)

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
    "SHARED_TIMEOUT",
    "drop_unusable_tasks",
    "FailureCode",
    "GoldResults",
    "SqliteExecutor",
    "PostgresExecutor",
    "MySqlExecutor",
    "BigQueryExecutor",
    "extract_sql",
    "has_sql_codeblock",
    "result_sets_match",
    "beaver_result_match",
    "matcher_for",
    "timeout_for",
    "gold_result_tag",
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
    # BigQuery refused the prediction for scanning more than the configured
    # maximum_bytes_billed. Kept apart from EXECUTION_ERROR because it is the
    # run's budget, not the SQL, that failed: raising the cap may pass it.
    EXECUTION_BYTES_CAP = "execution_bytes_cap"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    GOLD_EXECUTION_TIMEOUT = "gold_execution_timeout"
    # Gold failed on a dataset whose official evaluator scores that as a miss
    # and keeps the task (BIRD; see SHARED_TIMEOUT). Unlike the two above, the
    # task stays in the denominator.
    GOLD_FAILURE_SCORED = "gold_failure_scored"
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

    Kept in step with ``tests/gym/beaver_reference.py``, a port of the official
    evaluator, by ``tests/gym/test_beaver_match.py`` rather than by importing
    it -- that port pulls in pandas, which the control arm's venv deliberately
    does not have.
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


# Datasets whose official evaluator counts any query that does not produce rows
# -- an error, a timeout, or no prediction at all -- as an *empty result* and
# scores it, on the gold side as well as the prediction side. So a missing or
# broken prediction against empty gold scores 1.0, and a task whose gold query
# fails stays in the denominator. Every other dataset keeps BIRD's stricter
# handling: a failed prediction is a miss and a failed gold drops the task.
# Following BEAVER here is what keeps its numbers comparable to the leaderboard.
ERRORS_AS_EMPTY = {"beaverbench"}


# Query timeout per dataset, overriding the server's configured default. BEAVER's
# official evaluator gives every query 10 seconds, and a query that finishes in
# 20 is a pass here but an empty result there -- so the leaderboard number is
# only reproducible with the leaderboard's timeout.
#
# Spider2-lite's evaluator sets no limit at all. Its slowest gold query takes
# ~11s here (local099 on db_imdb), so the default would time out a correct
# prediction that is merely a few times slower than gold. On BigQuery a query
# also waits for slots and scans terabyte tables, so it gets longer still.
QUERY_TIMEOUTS = {
    "beaverbench": 10.0,
    "spider2_lite": 120.0,
    "spider2_lite_bq": 300.0,
}

# Datasets scored the way BIRD's evaluation_ex.py scores them: the prediction and
# then the gold query run under ONE time limit, and any failure -- the gold
# query's included -- scores 0 with the task kept in the denominator. Elsewhere
# each query gets the full limit and a failed gold query excludes the task.
#
# On BIRD Mini-Dev this decides two questions. 701's gold query takes ~157s, so
# it can never be answered; officially it still counts. 518's takes ~26s on an
# idle machine, leaving the prediction a few seconds: it passes or times out
# depending on machine speed and load, as it does officially.
SHARED_TIMEOUT = {"bird"}


def timeout_for(dataset: str, default: float) -> float:
    """Return the query timeout, in seconds, a dataset should be executed with."""
    return QUERY_TIMEOUTS.get(dataset, default)


def gold_result_tag(gold_rows: ResultSet) -> str:
    """Label a task by whether its gold query returns any rows.

    Recorded on every rollout as ``gold_result`` so the verifiers can report
    accuracy over ``nonempty_gold`` tasks alone. That is the second of the two
    numbers BEAVER publishes: an empty gold result is matched by any query that
    also returns nothing -- including one that errored -- so those questions
    inflate the headline figure. (A third value, ``"failed_gold"``, comes from
    :func:`execute_and_compare` on datasets that score a failed gold as empty.)
    """
    return "nonempty_gold" if gold_rows else "empty_gold"


class Executor(Protocol):
    """Runs a query for a ``db_id``.

    Returns ``(rows, status)`` where ``status`` is ``""`` on success and
    ``"timeout"`` or ``"error"`` otherwise -- the caller needs to tell a slow
    query apart from a broken one.
    """

    #: The limit a query gets when :meth:`execute` is not given one.
    timeout_s: float

    async def execute(
        self, db_id: str, sql: str, *, timeout_s: Optional[float] = None
    ) -> tuple[Optional[ResultSet], str]: ...


def _run_sqlite(db_path: Path, sql: str) -> tuple[Optional[ResultSet], str]:
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.text_factory = lambda b: b.decode(errors="ignore")
            return conn.execute(sql).fetchall(), ""
    except Exception:
        return None, "error"


def _run_sqlite_table(
    db_path: Path, sql: str, timeout_s: float
) -> tuple[Optional[tuple[int, ResultSet]], str]:
    """Run *sql* read-only and return ``((column_count, rows), status)``.

    The column count is kept because an empty result still has columns, and
    Spider2 compares column vectors: an empty prediction matches an empty gold.
    The query is interrupted at the deadline rather than left running on a
    worker thread the caller has stopped waiting for.
    """
    deadline = time.monotonic() + timeout_s
    timed_out = False

    def past_deadline() -> int:
        nonlocal timed_out
        timed_out = time.monotonic() > deadline
        return int(timed_out)

    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except Exception:
        return None, "error"
    try:
        conn.set_progress_handler(past_deadline, 10_000)
        cursor = conn.execute(sql)
        if cursor.description is None:
            return None, "error"
        return (len(cursor.description), cursor.fetchall()), ""
    except Exception:
        return None, "timeout" if timed_out else "error"
    finally:
        conn.close()


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
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def db_path(self, db_id: str) -> Path:
        return self._db_root / db_id / f"{db_id}.sqlite"

    async def execute(
        self, db_id: str, sql: str, *, timeout_s: Optional[float] = None
    ) -> tuple[Optional[ResultSet], str]:
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_sqlite, self.db_path(db_id), sql),
                    timeout=self.timeout_s if timeout_s is None else timeout_s,
                )
            except asyncio.TimeoutError:
                return None, "timeout"

    async def execute_table(
        self, db_id: str, sql: str
    ) -> tuple[Optional[tuple[int, ResultSet]], str]:
        """Like :meth:`execute`, also returning the column count (Spider2)."""
        async with self._semaphore:
            return await asyncio.to_thread(
                _run_sqlite_table, self.db_path(db_id), sql, self.timeout_s
            )

    async def execute_prediction(
        self, db_id: str, sql: str
    ) -> tuple[Optional[Table], str]:
        """The predicted frame Spider2 compares, after its read_sql round trip."""
        result, status = await self.execute_table(db_id, sql)
        if result is None:
            return None, status
        column_count, rows = result
        return prediction_table(column_count, rows), ""


def _run_bigquery_frame(
    client: Any, target: BigQueryTarget, sql: str, timeout_s: float
):
    """Run *sql* read-only; return ``(frame, status)``. Setup failures raise."""
    try:
        rows, _ = run_query(client, target, sql, timeout_s=timeout_s, read_only=True)
        return rows.to_dataframe(create_bqstorage_client=False), ""
    except BytesCapExceeded as exc:
        logger.warning("BigQuery %s: %s", target.db_id, exc)
        return None, "bytes_cap"
    except QueryTimeout:
        return None, "timeout"
    except NotReadOnly as exc:
        logger.info("BigQuery %s: %s", target.db_id, exc)
        return None, "error"
    except Exception as exc:
        if is_infrastructure_error(exc):
            raise
        logger.info(
            "BigQuery %s query failed: %s", target.db_id, str(exc).splitlines()[0]
        )
        return None, "error"


class BigQueryExecutor:
    """Executes against the BigQuery datasets behind each ``db_id`` (Spider2-lite).

    *targets* maps ``db_id`` to its :class:`~ontology_sql_eval.bigquery.BigQueryTarget`,
    parsed from the same ``bigquery://`` CONNECTION_STRINGS the treatment
    arm's connectors use, so both arms bill the same project under the same
    cap. Every query is dry-run first and refused unless it is a ``SELECT``:
    the billing project is the user's own, and may hold writable data.

    Errors that mean the setup is broken -- no credentials, no permission to
    run jobs in the billing project, exhausted quota -- are raised rather than
    scored, so they surface as ``unknown_error`` instead of as wrong SQL.
    """

    def __init__(
        self,
        targets: "dict[str, BigQueryTarget]",
        *,
        max_concurrency: int = 8,
        timeout_s: float = 300.0,
    ) -> None:
        self._targets = targets
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._clients: dict[tuple[str, str, str, str], Any] = {}
        self._lock = threading.Lock()

    def _target(self, db_id: str) -> BigQueryTarget:
        target = self._targets.get(db_id)
        if target is None:
            raise KeyError(
                f"no bigquery:// entry for db_id {db_id!r} in CONNECTION_STRINGS; "
                f"have {sorted(self._targets)}"
            )
        return target

    @staticmethod
    def _client_key(target: BigQueryTarget) -> tuple[str, str, str, str]:
        credentials = (
            hashlib.sha256(
                json.dumps(target.credentials, sort_keys=True).encode()
            ).hexdigest()
            if target.credentials is not None
            else ""
        )
        return (
            target.billing_project,
            target.location or "",
            target.credentials_file or "",
            credentials,
        )

    def _client(self, target: BigQueryTarget) -> Any:
        key = self._client_key(target)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = make_client(target)
            return self._clients[key]

    async def _frame(self, db_id: str, sql: str, timeout_s: float):
        target = self._target(db_id)
        client = self._client(target)
        async with self._semaphore:
            return await asyncio.to_thread(
                _run_bigquery_frame, client, target, sql, timeout_s
            )

    async def execute(
        self, db_id: str, sql: str, *, timeout_s: Optional[float] = None
    ) -> tuple[Optional[ResultSet], str]:
        frame, status = await self._frame(
            db_id, sql, self.timeout_s if timeout_s is None else timeout_s
        )
        if frame is None:
            return None, status
        return [tuple(row) for row in frame.itertuples(index=False)], ""

    async def execute_prediction(
        self, db_id: str, sql: str
    ) -> tuple[Optional[Table], str]:
        """The predicted frame Spider2 compares: ``to_dataframe().to_csv()`` read back.

        An empty result comes back as status ``"empty"``: upstream's
        ``get_bigquery_sql_result`` reports it as a failed execution, so it
        scores 0 even against an empty gold.
        """
        frame, status = await self._frame(db_id, sql, self.timeout_s)
        if frame is None:
            return None, status
        if frame.empty:
            return None, "empty"
        return table_from_csv_text(frame.to_csv(index=False)), ""


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
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(
        self, db_id: str, sql: str, *, timeout_s: Optional[float] = None
    ) -> tuple[Optional[ResultSet], str]:
        dsn = self._dsn_for_db.get(db_id)
        if dsn is None:
            return None, "error"
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_mysql, dsn, sql),
                    timeout=self.timeout_s if timeout_s is None else timeout_s,
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
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(
        self, db_id: str, sql: str, *, timeout_s: Optional[float] = None
    ) -> tuple[Optional[ResultSet], str]:
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_run_postgres, self._dsn, sql),
                    timeout=self.timeout_s if timeout_s is None else timeout_s,
                )
            except asyncio.TimeoutError:
                return None, "timeout"


async def execute_and_compare(
    executor: Executor,
    db_id: str,
    gold_sql: str,
    pred_sql: Optional[str],
    dataset: str = "",
    gold: Optional[GoldResults] = None,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Run both queries and compare. Returns ``(match, error_tag, gold_result)``.

    With *gold* (Spider2-lite) there is no gold query: ``gold_sql`` is ignored,
    only the prediction runs, and it is scored against the gold result files.

    ``pred_sql=None`` means there is no prediction at all -- an empty model
    answer or an agent failure. Gold is still run: on most datasets a broken
    gold query must take the task out of the denominator rather than leave it
    there as an ordinary miss, and on BEAVER a missing prediction is scored.

    Gold is executed first. If it does not run, that is a dataset problem: the
    ``gold_sql_*`` tag returned tells the caller to exclude the task, except on
    BEAVER, which scores it (see :data:`ERRORS_AS_EMPTY`). ``gold_result`` is
    :func:`gold_result_tag` when gold ran, ``"failed_gold"`` when it failed and
    was scored as empty, and ``None`` when the task is excluded.
    """
    if gold is not None:
        return await _execute_and_compare_gold_results(executor, db_id, pred_sql, gold)

    if dataset in SHARED_TIMEOUT:
        return await _execute_and_compare_shared_timeout(
            executor, db_id, gold_sql, pred_sql, dataset
        )

    errors_as_empty = dataset in ERRORS_AS_EMPTY
    gold_rows, gold_status = await executor.execute(db_id, gold_sql)
    if gold_rows is None:
        if not errors_as_empty:
            return False, f"gold_sql_{gold_status or 'error'}", None
        gold_rows, gold_result = [], "failed_gold"
    else:
        gold_result = gold_result_tag(gold_rows)

    if pred_sql is None:
        pred_rows, pred_status = None, "missing"
    else:
        pred_rows, pred_status = await executor.execute(db_id, pred_sql)
    if pred_rows is None:
        if errors_as_empty:
            return matcher_for(dataset)(gold_rows, []), None, gold_result
        return False, f"pred_sql_{pred_status or 'error'}", gold_result

    return matcher_for(dataset)(gold_rows, pred_rows), None, gold_result


async def _execute_and_compare_gold_results(
    executor: Executor,
    db_id: str,
    pred_sql: Optional[str],
    gold: GoldResults,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Spider2-lite: run the prediction and compare it with the gold CSVs.

    Missing gold files are a seeding problem, not a wrong answer, so they take
    the task out of the denominator like a broken gold query does. A failed
    prediction is a miss, as it is officially. So is an empty BigQuery result,
    which upstream counts as a failed execution; it is reported as a wrong
    result here, since the query did run.
    """
    if not gold.paths or not all(path.is_file() for path in gold.paths):
        return False, "gold_sql_error", None
    gold_result = "empty_gold" if gold_is_empty(gold) else "nonempty_gold"

    if pred_sql is None:
        return False, "pred_sql_missing", gold_result
    execute_prediction = getattr(executor, "execute_prediction", None)
    if execute_prediction is None:
        raise TypeError(f"{type(executor).__name__} cannot score gold result files")
    pred, status = await execute_prediction(db_id, pred_sql)
    if status == "empty":
        return False, None, gold_result
    if pred is None:
        return False, f"pred_sql_{status or 'error'}", gold_result
    return results_match(pred, gold), None, gold_result


async def _execute_and_compare_shared_timeout(
    executor: Executor,
    db_id: str,
    gold_sql: str,
    pred_sql: Optional[str],
    dataset: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """BIRD's ``evaluation_ex.py``: prediction, then gold, under one time limit.

    The official ``execute_model`` wraps both queries in a single
    ``func_timeout(meta_time_out, ...)`` and scores every exception and timeout
    0, with the task kept. A failed gold query returns ``gold_sql_scored``
    rather than a ``gold_sql_*`` tag that excludes the task. Gold still runs
    after a failed prediction, with whatever time is left, so the rollout
    carries its ``gold_result``; the score is 0 either way.
    """
    budget = executor.timeout_s
    start = time.monotonic()
    if pred_sql is None:
        pred_rows, pred_status = None, "missing"
    else:
        pred_rows, pred_status = await executor.execute(
            db_id, pred_sql, timeout_s=budget
        )

    remaining = budget - (time.monotonic() - start)
    gold_rows: Optional[ResultSet] = None
    if remaining > 0:
        gold_rows, _ = await executor.execute(db_id, gold_sql, timeout_s=remaining)
    gold_result = "failed_gold" if gold_rows is None else gold_result_tag(gold_rows)

    if pred_rows is None:
        return False, f"pred_sql_{pred_status or 'error'}", gold_result
    if gold_rows is None:
        return False, "gold_sql_scored", gold_result
    return matcher_for(dataset)(gold_rows, pred_rows), None, gold_result
