"""Perception/Ingestion: pulls a batch, updates W(t), runs F_rep."""
from __future__ import annotations

from malar.agents.base import AgentContext


class Perception:
    role = "Perception/Ingestion: pulls a batch, updates W(t), runs F_rep."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<Perception role={self.role!r}>"
