"""a-Interact bird-coin budget, mirroring the reference ADK's accounting.

The a-interact orchestrator (orchestrator/ainteract.py in the reference ADK)
hands the system agent a per-task budget and nothing else; the agent's own
callbacks (system_agent/callbacks.py) then charge every tool call against it.
Grading never checks the budget — it only bounds how many asks, schema/KB
lookups, queries and submissions the agent gets — so matching those
callbacks' rules exactly is what keeps our numbers comparable with other
a-Interact agents. ``charge`` does that, including the submission that spends
the wallet to zero.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

# Same costs as TOOL_COSTS in the reference system_agent/callbacks.py, limited
# to the tools this adapter actually calls.
TOOL_COSTS = {
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
}

# What a Phase 2 attempt needs: a submission (3), the query that checks it (1),
# and two queries to look at the data first.
PHASE2_RESERVE = 6.0

# The reference agent's MAX_MODEL_TURNS: a hard stop independent of budget.
MAX_AGENT_STEPS = 60

# Phase 1 queries and asks stop once this many coins are left. The coins stay
# in the same wallet: a submission can spend them, and phase 2 has no floor.
PHASE1_QUERY_FLOOR = 4.0


class BudgetExhausted(RuntimeError):
    """A query the remaining budget can't pay for; the reference tool refuses it too."""


@dataclass
class Budget:
    initial: float
    remaining: float
    # Auto Ontology can run queries from worker threads mid-step (see
    # MeteredConnector), so charges are serialized.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @classmethod
    def start(cls, initial: float) -> "Budget":
        return cls(initial=initial, remaining=initial)

    @property
    def exhausted(self) -> bool:
        # The reference uses -1 as its stop signal, never a real balance.
        return self.remaining < 0

    def charge(self, tool: str, reserve: float = 0.0) -> bool:
        """Deduct *tool*'s cost; False when the call is refused.

        A submission is never refused: when the balance can't cover it, or it
        spends the balance down to zero, it goes through as the agent's final
        action and the budget reads exhausted afterwards. Any other tool the
        balance can't cover is refused. *reserve* is a floor those other tools
        must leave behind; a submission ignores it.
        """
        with self._lock:
            return self._charge_unlocked(tool, reserve)

    def _charge_unlocked(self, tool: str, reserve: float = 0.0) -> bool:
        cost = TOOL_COSTS[tool]
        if tool != "submit_sql" and self.remaining - cost < reserve:
            return False
        if self.remaining < cost:
            if tool == "submit_sql":
                self.remaining = -1.0
                return True
            return False
        self.remaining -= cost
        if tool == "submit_sql" and self.remaining <= 0:
            self.remaining = -1.0
        return True

    def affordable_asks(self, reserve: float = 0.0) -> int:
        """Asks that still leave one submission, or *reserve* if that is more.

        The phase-1 floor is not stacked on top of the submission. The balance
        kept is whichever of the two is larger, which at 4 coins still pays
        for one submission.
        """
        keep = max(reserve, TOOL_COSTS["submit_sql"])
        spare = self.remaining - keep
        return max(0, int(spare // TOOL_COSTS["ask_user"]))


class MeteredConnector:
    """A connector whose queries are charged as the reference's execute_sql.

    Auto Ontology queries the database itself while it grounds values and
    validates SQL. The reference agent can only do that through execute_sql
    at 1 coin a query, so each execute() here is charged the same, and one the
    budget can't cover raises BudgetExhausted instead of running. Everything
    else (schema introspection, qualify, ...) passes through uncharged, as the
    reference charges its schema once through get_schema.

    *on_query* is told about every query, charged or refused, so the session
    can record it in its trajectory. *phase* says which a-Interact phase is
    running, so :meth:`cost_status` can tell Phase 1 to leave coins for the
    follow-up, which is paid from the same budget.
    """

    def __init__(
        self,
        base: Any,
        budget: Budget,
        on_query: Callable[[str, bool], None],
        reserve: float = 0.0,
        phase: Callable[[], int] = lambda: 1,
    ) -> None:
        self._base = base
        self._budget = budget
        self._on_query = on_query
        # Phase 1 sets this to PHASE1_QUERY_FLOOR; phase 2 clears it.
        self.reserve = reserve
        self._phase = phase

    def cost_status(self) -> str:
        """The balance and prices, for an agent that decides how many queries to run."""
        left = self._budget.remaining
        if not self.can_run_queries():
            return "BUDGET: no coins left for run_sql. Submit your best query now."
        prices = (
            f"run_sql costs {TOOL_COSTS['execute_sql']:g} coin. Submitting costs "
            f"{TOOL_COSTS['submit_sql']:g}, plus 1 for each query run to check "
            "the SQL, and a wrong submission is charged in full."
        )
        if self._phase() == 1:
            return (
                f"BUDGET: {left:g} coins left. {prices} If this answer is "
                "accepted, a follow-up question comes next and is paid from "
                f"what is left, so submit with at least {PHASE2_RESERVE:g} "
                "coins unspent when you can."
            )
        return f"BUDGET: {left:g} coins left for this follow-up. {prices}"

    def can_run_queries(self) -> bool:
        """Whether the next query would be charged rather than refused."""
        return (
            self._budget.remaining - TOOL_COSTS["execute_sql"] >= self.reserve
        )

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        charged = self._budget.charge("execute_sql", reserve=self.reserve)
        self._on_query(sql, charged)
        if not charged:
            raise BudgetExhausted(
                f"Query refused ({self._budget.remaining:.1f} remaining)."
            )
        return self._base.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def install_budget_refusal_submits_current_sql() -> None:
    """A refused phase-1 query must not be rewritten as a bad SQL statement.

    The text-to-SQL runner treats every execution error as a query to fix and
    reconstructs it, up to seven model calls. On this benchmark that loop runs
    past the run_session call timeout, so the SQL already generated never gets
    submitted. ``BudgetExhausted`` leaving ``_run_sql`` hits the runner's
    existing fallback, which submits that SQL instead.

    Tests stub ``auto_ontology.retrieval`` before this process imports the real
    package. Skipping the patch there is fine: those tests never execute SQL.
    """
    try:
        import auto_ontology.retrieval.text_to_sql.agents.sql_execution as sql_execution
    except (ModuleNotFoundError, ImportError):
        return
    _patch_run_sql_to_reraise_budget(sql_execution)


def _patch_run_sql_to_reraise_budget(module: Any) -> None:
    import inspect
    import textwrap

    if getattr(module, "_budget_refusal_installed", False):
        return
    source = textwrap.dedent(inspect.getsource(module._run_sql))
    old = "    except Exception as e:\n"
    new = "    except BudgetExhausted:\n        raise\n    except Exception as e:\n"
    if source.count(old) != 1:
        raise RuntimeError(
            "sql_execution._run_sql no longer has one 'except Exception'; "
            "the budget-refusal fallback was not installed"
        )
    # The re-raised name is resolved in that module, not in this one.
    module.BudgetExhausted = BudgetExhausted
    exec(source.replace(old, new, 1), module.__dict__)
    module._budget_refusal_installed = True
