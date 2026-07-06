"""Critic/Validation: conformal + OOD gate before any memory write or action."""
from __future__ import annotations

from malar.agents.base import AgentContext


class Critic:
    role = "Critic/Validation: conformal + OOD gate before any memory write or action."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<Critic role={self.role!r}>"
