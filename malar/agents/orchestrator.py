"""Orchestrator/Planner: drives the loop; allocates budget B; explores WorldView frontiers; runs DomainSpec training campaigns and tracks coverage."""
from __future__ import annotations

from malar.agents.base import AgentContext


class Orchestrator:
    role = "Orchestrator/Planner: drives the loop; allocates budget B; explores WorldView frontiers; runs DomainSpec training campaigns and tracks coverage."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<Orchestrator role={self.role!r}>"
