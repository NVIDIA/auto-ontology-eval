"""a-Interact bird-coin budget, mirroring the reference ADK's accounting.

The a-interact orchestrator (orchestrator/ainteract.py in the reference ADK)
hands the system agent a per-task budget and nothing else; the agent's own
callbacks (system_agent/callbacks.py) then charge every tool call against it.
Grading never checks the budget — it only bounds how many asks, schema/KB
lookups, queries and submissions the agent gets — so matching those
callbacks' rules exactly is what keeps our numbers comparable with other
a-Interact agents.
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
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
}

# What a Phase 2 attempt needs: a submission (3), the query that checks it (1),
# and two queries to look at the data first.
PHASE2_RESERVE = 6.0

# The reference agent's MAX_MODEL_TURNS: a hard stop independent of budget.
MAX_AGENT_STEPS = 60


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

    def charge(self, tool: str) -> bool:
        """Deduct *tool*'s cost; False when the call is refused.

        A submission is never refused: when the balance can't cover it, or it
        spends the balance down to zero, it goes through as the agent's final
        action and the budget reads exhausted afterwards. Any other tool the
        balance can't cover is refused.
        """
        cost = TOOL_COSTS[tool]
        with self._lock:
            if self.remaining < cost:
                if tool == "submit_sql":
                    self.remaining = -1.0
                    return True
                return False
            self.remaining -= cost
            if tool == "submit_sql" and self.remaining <= 0:
                self.remaining = -1.0
            return True

    def affordable_asks(self) -> int:
        """Asks that still leave enough for one submission afterwards."""
        spare = self.remaining - TOOL_COSTS["submit_sql"]
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
        phase: Callable[[], int] = lambda: 1,
    ) -> None:
        self._base = base
        self._budget = budget
        self._on_query = on_query
        self._phase = phase

    def cost_status(self) -> str:
        """The balance and prices, for an agent that decides how many queries to run."""
        left = self._budget.remaining
        if left < TOOL_COSTS["execute_sql"]:
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

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        charged = self._budget.charge("execute_sql")
        self._on_query(sql, charged)
        if not charged:
            raise BudgetExhausted(
                f"Budget exhausted ({self._budget.remaining:.1f} remaining); "
                "query refused."
            )
        return self._base.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)
