"""Functional fields F_j(o, t) — action possibilities (affordances) per object.

F_j(o,t) = h_j(feat(o), values(o); eta_j): for functional dimension j, decide which
actions an object affords. The region action set A(S) = union over o in O(S) of the
afforded actions. The Policy ranks a in A(S) by R, V — F generates, R/V score.

One FunctionalField is owned by a Functional-Field Agent B_j; learned from data AND
humans (sticky human affordances), with provenance via FunctionalStore.
"""
from __future__ import annotations

from dataclasses import dataclass

from malar.fields.functional_store import FunctionalStore


@dataclass
class FunctionalSpec:
    dim: str                       # functional dimension j, e.g. "response"
    actions: list[str]             # candidate actions on this dimension
    description: str = ""


class FunctionalField:
    def __init__(self, spec: FunctionalSpec, store: FunctionalStore, version: int = 1):
        self.spec = spec
        self.j = spec.dim
        self.store = store
        self.version = version     # eta_j version

    def learn_affordance(self, object_class: str, action: str, enabled: bool,
                         source: str = "data", world_ctx: str | None = None) -> bool:
        if action not in self.spec.actions:
            self.spec.actions.append(action)
        applied, _ = self.store.set(object_class, self.j, action, enabled, source, world_ctx)
        return applied

    def set_human_affordance(self, object_class: str, action: str, enabled: bool,
                             world_ctx: str | None = None) -> bool:
        return self.learn_affordance(object_class, action, enabled, source="human",
                                     world_ctx=world_ctx)

    def afforded(self, object_class: str) -> list[str]:
        """Actions this object affords along dimension j."""
        return [a for a in self.spec.actions
                if a in self.store.actions_for(object_class)]


def region_action_set(objects: list[dict], fields: list[FunctionalField]) -> list[str]:
    """A(S) = union over recognised objects of all afforded actions across F_j."""
    actions: set[str] = set()
    for o in objects:
        cls = o["class"]
        for f in fields:
            actions.update(f.afforded(cls))
    return sorted(actions)
