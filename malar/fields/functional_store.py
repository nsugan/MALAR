"""Functional/affordance store: (:Object)-[:AFFORDS {dim, action, source, version, world_ctx}]->(:Action).

In-process SOURCE OF TRUTH. Records which actions an object class affords along each
functional dimension j, with provenance. Durability: checkpointed to
`data/domains/{id}/state.json` (survives restart), and mirrored into the memory graph as
(:Object)-[:AFFORDS]->(:Action) edges by the training write path
(`DomainService.confirm`, via `MemoryGraph.upsert_affordance`) for visualisation. The
graph copy is a projection — this dict remains authoritative. (V4 fix ❷)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AffordanceRecord:
    object_class: str
    dim: str            # functional dimension j
    action: str
    enabled: bool
    source: str         # 'data' | 'human' | 'llm'
    version: int
    world_ctx: str | None = None
    sticky: bool = False


class FunctionalStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], AffordanceRecord] = {}
        self.audit: list[dict] = []

    def key(self, object_class: str, dim: str, action: str):
        return (object_class, dim, action)

    def set(self, object_class: str, dim: str, action: str, enabled: bool, source: str,
            world_ctx: str | None = None) -> tuple[bool, AffordanceRecord]:
        k = self.key(object_class, dim, action)
        existing = self._records.get(k)
        if existing and existing.sticky and source != "human":
            self.audit.append({"op": "reject_sticky", "key": k})
            return (False, existing)
        version = (existing.version + 1) if existing else 1
        rec = AffordanceRecord(object_class=object_class, dim=dim, action=action,
                               enabled=enabled, source=source, version=version,
                               world_ctx=world_ctx, sticky=(source == "human"))
        self._records[k] = rec
        self.audit.append({"op": "set", "key": k, "enabled": enabled, "source": source})
        return (True, rec)

    def actions_for(self, object_class: str) -> list[str]:
        return [r.action for (cls, dim, action), r in self._records.items()
                if cls == object_class and r.enabled]

    def all(self):
        return list(self._records.values())
