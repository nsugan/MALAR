"""Fields Coordinator: aggregate R, diffuse V, assemble the F-afforded action set; feed scorer/policy."""
from __future__ import annotations

from malar.agents.base import AgentContext


class FieldsCoordinator:
    role = "Fields Coordinator: aggregate R, diffuse V, assemble the F-afforded action set; feed scorer/policy."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<FieldsCoordinator role={self.role!r}>"
