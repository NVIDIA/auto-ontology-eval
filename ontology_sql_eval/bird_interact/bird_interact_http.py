"""HTTP clients for Bird ADK :6001 (user simulator) and :6002 (DB environment)."""

import asyncio
import logging

import httpx
import sqlglot
from sqlglot import exp

USER_SIM_URL = "http://localhost:6001"
DB_ENV_URL = "http://localhost:6002"
_TIMEOUT = 120.0
_ASK_USER_MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)


def _strip_catalog_qualifier(sql: str) -> str:
    """Drop the leading ``<database>.`` catalog qualifier GSF bakes into every
    table reference (e.g. ``solar_panel.public.plants`` -> ``public.plants``).

    GSF always qualifies tables with the source database name — see
    ``gsf/retrieval/text_to_sql/formatters_util.py``'s ``qualify_table()`` —
    which is correct (and required for dedup) when GSF validates the SQL
    against its own connector, since that connector is always the
    same-named base database. But ADK's grading harness executes submitted
    SQL against a differently-named per-task database
    (``f"{base_db}__{task_id}"``, see ``BIRD-Interact-ADK/shared/db_utils
    .py``'s ``create_task_db()``), so the qualifier no longer matches
    ``current_database()`` there and Postgres rejects it outright with
    "cross-database references are not implemented" — a hard Phase 1/2
    failure regardless of whether the query logic is correct.

    Uses sqlglot (parses real table-reference tokens, so it can't be
    tricked by a database name that happens to appear inside a string
    literal the way a blind string/regex replace could be) rather than
    stripping unconditionally in GSF core, since the qualifier is still
    needed there for multi-database table dedup and for GSF's own
    same-database execution/validation.

    Falls back to the original SQL, unchanged, if parsing fails — this is a
    best-effort cleanup, not a correctness gate, so a parse error here must
    never block submission of otherwise-valid SQL.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        changed = False
        for table in tree.find_all(exp.Table):
            if table.catalog:
                table.set("catalog", None)
                changed = True
        if not changed:
            return sql
        return tree.sql(dialect="postgres")
    except Exception:
        logger.warning(
            "Could not strip catalog qualifier from submitted SQL "
            "(leaving unchanged): %r",
            sql,
            exc_info=True,
        )
        return sql


async def ask_user(task_id: str, question: str) -> str:
    """POST :6001/ask → returns answer string.

    Retries once on timeout. ADK's user_simulator runs two sequential LLM
    calls per question (action-parse, then response-generate — see
    ``BIRD-Interact-ADK/user_simulator/server.py``), each with its own
    internal retry loop but no per-call timeout of its own — so under a
    slow/degraded upstream model endpoint, the two calls' combined latency
    can exceed our fixed ``_TIMEOUT`` even though neither call ever
    actually errored out server-side. A single client-side retry costs
    little in the common case and gives the request a fresh connection/
    attempt instead of permanently losing the whole session (with all its
    already-gathered clarification progress) to one slow round-trip.
    """
    last_error: Exception | None = None
    for attempt in range(_ASK_USER_MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    f"{USER_SIM_URL}/ask",
                    json={"task_id": task_id, "question": question},
                )
                r.raise_for_status()
                data = r.json()
                if "answer" not in data:
                    raise ValueError(f"Missing 'answer' key in response: {data}")
                return data["answer"]
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < _ASK_USER_MAX_ATTEMPTS - 1:
                logger.warning(
                    "ask_user timed out for task_id=%r (attempt %d/%d) — retrying",
                    task_id,
                    attempt + 1,
                    _ASK_USER_MAX_ATTEMPTS,
                )
                await asyncio.sleep(1.0)
                continue
            raise
    # Unreachable (loop always returns or raises), but keeps type-checkers happy.
    raise last_error  # type: ignore[misc]


async def submit_sql(task_id: str, sql: str) -> dict:
    """POST :6002/submit → returns full response dict."""
    sql = _strip_catalog_qualifier(sql)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            f"{DB_ENV_URL}/submit",
            json={"task_id": task_id, "sql": sql},
        )
        r.raise_for_status()
        return r.json()


async def phase_transition(task_id: str) -> None:
    """POST :6001/phase_transition."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{USER_SIM_URL}/phase_transition",
            json={"task_id": task_id},
        )
        r.raise_for_status()
