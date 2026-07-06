"""Objective-Field Agent A_k: LLM-assisted identify, compute R^(k), learn {v_{k,c}} from data+humans, active learning + HITL."""
from __future__ import annotations

from malar.agents.base import AgentContext


class ObjectiveFieldAgent:
    role = "Objective-Field Agent A_k: LLM-assisted identify, compute R^(k), learn {v_{k,c}} from data+humans, active learning + HITL."

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"<ObjectiveFieldAgent role={self.role!r}>"
