"""Policy/Action: choose actions within the F-afforded set, ranked by R,V; justify."""
from __future__ import annotations

from malar.agents.base import AgentContext


class PolicyAgent:
    role = "Policy/Action: choose actions within the F-afforded set, ranked by R,V; justify."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<PolicyAgent role={self.role!r}>"
