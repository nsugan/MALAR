"""Objective registry R = {R^(1..K)}.

Each objective is a named scalar field the system optimises for (e.g. sensitivity,
specificity, early-detection, yield, defect-rate, cycle-time). Objectives are
declared by a DomainSpec; one ObjectiveFieldAgent A_k is instantiated per objective.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObjectiveSpec:
    key: str
    description: str = ""
    direction: str = "maximize"   # 'maximize' | 'minimize'
    target: float | None = None   # coverage/value target if any
    weight: float = 1.0


class ObjectiveRegistry:
    def __init__(self, specs: list[ObjectiveSpec] | None = None):
        self._specs: dict[str, ObjectiveSpec] = {}
        for s in specs or []:
            self.add(s)

    def add(self, spec: ObjectiveSpec) -> None:
        self._specs[spec.key] = spec

    def keys(self) -> list[str]:
        return list(self._specs.keys())

    def get(self, key: str) -> ObjectiveSpec:
        return self._specs[key]

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
