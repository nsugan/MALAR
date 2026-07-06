"""Functional-Field Agent B_j: learn affordances F_j(o,t) from data+humans."""
from __future__ import annotations

from malar.agents.base import AgentContext


class FunctionalFieldAgent:
    role = "Functional-Field Agent B_j: learn affordances F_j(o,t) from data+humans."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<FunctionalFieldAgent role={self.role!r}>"
