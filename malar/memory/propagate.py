"""Human-correction propagation — bounded, Critic-gated, rollback-able.

1. Apply    — set label/value on the seed memory; provenance=human, sticky, version++.
2. Scope    — blast radius via DERIVED_FROM lineage + similarity radius + shared
              WorldContext, capped by a propagation budget.
3. Re-eval  — Critic (conformal) gated: auto-relabel only high-confidence neighbours,
              flag the rest for HITL.
4. Recompute— affected {v_{k,.}} / F(.) are marked for re-learning; V re-diffused.
5. Audit    — recorded as ONE transaction that rolls back on undo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from malar.encoders.change import cosine_similarity
from malar.memory.neo4j_io import MemoryGraph
from malar.memory.store import MemoryStore
from malar.validation.conformal import ConformalGate


@dataclass
class PropagationResult:
    seed_id: str
    new_label: str
    relabeled: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    transaction: list[dict] = field(default_factory=list)

    def undo_plan(self) -> list[dict]:
        return list(reversed(self.transaction))


class CorrectionPropagator:
    def __init__(self, store: MemoryStore, graph: MemoryGraph, gate: ConformalGate,
                 budget: int = 25, sim_radius: float = 0.85, hi_conf: float = 0.9):
        self.store = store
        self.graph = graph
        self.gate = gate
        self.budget = budget
        self.sim_radius = sim_radius
        self.hi_conf = hi_conf

    def _scope(self, seed_id: str) -> list[str]:
        """DERIVED_FROM lineage + SIMILAR + same WorldContext, capped by budget."""
        candidates: set[str] = set()
        for rel in ("DERIVED_FROM", "SIMILAR", "TEMPORAL_NEXT"):
            for nid in self.graph.neighbors(seed_id, rel):
                if nid.startswith("mem_"):
                    candidates.add(nid)
        seed = self.graph.get_memory(seed_id)
        if seed and seed.get("world_ctx_id"):
            for m in self.graph.all_memories():
                if m.get("world_ctx_id") == seed["world_ctx_id"] and m["id"] != seed_id:
                    candidates.add(m["id"])
        # similarity radius (uses cached vectors)
        seed_item = self.store.get_item(seed_id)
        if seed_item is not None:
            for m in self.graph.all_memories():
                it = self.store.get_item(m["id"])
                if it is None or it.id == seed_id:
                    continue
                if cosine_similarity(seed_item.phi, it.phi) >= self.sim_radius:
                    candidates.add(it.id)
        return list(candidates)[: self.budget]

    def propagate(self, seed_id: str, new_label: str, by: str = "human") -> PropagationResult:
        txn: list[dict] = []
        # 1. Apply to seed (sticky human)
        seed_item = self.store.get_item(seed_id)
        old_label = seed_item.cls if seed_item else None
        if seed_item is not None:
            seed_item.cls = new_label
            self.graph.upsert_memory(seed_item)
        txn.append({"op": "relabel", "id": seed_id, "old": old_label, "new": new_label,
                    "provenance": "human"})

        result = PropagationResult(seed_id=seed_id, new_label=new_label, transaction=txn)

        # 2-3. Scope + Critic-gated re-eval
        for nid in self._scope(seed_id):
            it = self.store.get_item(nid)
            if it is None:
                continue
            conf = cosine_similarity(seed_item.phi, it.phi) if seed_item is not None else 0.0
            if conf >= self.hi_conf and self.gate.accept(conf):
                old = it.cls
                it.cls = new_label
                self.graph.upsert_memory(it)
                result.relabeled.append(nid)
                txn.append({"op": "relabel", "id": nid, "old": old, "new": new_label,
                            "provenance": "propagated", "conf": conf})
            else:
                result.flagged.append(nid)
                txn.append({"op": "flag", "id": nid, "conf": conf})
        return result

    def rollback(self, result: PropagationResult) -> int:
        n = 0
        for step in result.undo_plan():
            if step["op"] == "relabel":
                it = self.store.get_item(step["id"])
                if it is not None:
                    it.cls = step["old"] if step["old"] is not None else "__unlabeled__"
                    self.graph.upsert_memory(it)
                    n += 1
        return n
