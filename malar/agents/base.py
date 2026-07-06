"""Shared agent context. Agents DECIDE; tools COMPUTE.

Each agent wraps deterministic engine components and (optionally) the LLM gateway for
control / identification-hypotheses / explanation only. Agents read & write the shared
MALARState; they never compute embeddings or run the metric themselves.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentContext:
    engine: object              # MalarEngine
    llm: object | None = None   # LLMClient
    batch: object = None        # current WorldBatch

    @property
    def c(self):
        return self.engine.c


# Pipeline stages, in order (used for review-mode interrupts).
STAGES = [
    "perception",
    "topology",
    "identification",
    "value_learning",
    "functional_field",
    "fields_coordinator",
    "policy",
    "memory_write",
    "critic",
]
