"""Topology: computes phi_topo(S,t) via the topology encoder tool."""
from __future__ import annotations

from malar.agents.base import AgentContext


class TopologyAgent:
    role = "Topology: computes phi_topo(S,t) via the topology encoder tool."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<TopologyAgent role={self.role!r}>"
