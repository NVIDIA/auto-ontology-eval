"""a-Interact bird-coin budget, mirroring the reference ADK's accounting.

The a-interact orchestrator (orchestrator/ainteract.py in the reference ADK)
hands the system agent a per-task budget and nothing else; the agent's own
callbacks (system_agent/callbacks.py) then charge every tool call against it.
Grading never checks the budget — it only bounds how many asks, schema/KB
lookups and submissions the agent gets — so matching those callbacks' rules
exactly is what keeps our numbers comparable with other a-Interact agents.
"""

from __future__ import annotations

from dataclasses import dataclass

# Same costs as TOOL_COSTS in the reference system_agent/callbacks.py, limited
# to the tools this adapter actually calls.
TOOL_COSTS = {
    "get_schema": 1.0,
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
}

# The reference agent's MAX_MODEL_TURNS: a hard stop independent of budget.
MAX_AGENT_STEPS = 60


@dataclass
class Budget:
    initial: float
    remaining: float

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
