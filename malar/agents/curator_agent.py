"""Memory Curator (async): novelty + budget gating, store/merge/decay, encoder re-index, correction propagation, maintains the WorldView."""
from __future__ import annotations

from malar.agents.base import AgentContext


class CuratorAgent:
    role = "Memory Curator (async): novelty + budget gating, store/merge/decay, encoder re-index, correction propagation, maintains the WorldView."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<CuratorAgent role={self.role!r}>"
