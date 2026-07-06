"""Memory (inline): per-tick retrieve + novelty read."""
from __future__ import annotations

from malar.agents.base import AgentContext


class MemoryAgent:
    role = "Memory (inline): per-tick retrieve + novelty read."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<MemoryAgent role={self.role!r}>"
