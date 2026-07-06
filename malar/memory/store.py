"""Store / merge / decay primitives.

  * insert       — write a new memory to Neo4j + Qdrant + artifacts grounding.
  * reinforce    — EMA importance bump  omega <- rho*omega + (1-rho)*gain  (0<rho<1).
  * merge        — CONTRACTIVE merge of a near memory: blended vectors move a fraction
                   c<1 toward the candidate (never expands), omega EMA-bumped, tau=t.
  * decay        — multiplicative importance decay for maintenance.

Theorem guards enforced as asserts (0<rho<1, c<1).
"""
from __future__ import annotations

import numpy as np

from malar.core.config import get_settings
from malar.memory.neo4j_io import MemoryGraph
from malar.memory.qdrant_io import QdrantMemory
from malar.memory.schema import MemoryItem


class MemoryStore:
    def __init__(self, qdrant: QdrantMemory, graph: MemoryGraph, settings=None):
        self.qdrant = qdrant
        self.graph = graph
        self.s = settings or get_settings()
        self._cache: dict[str, MemoryItem] = {}   # vector cache (Qdrant is opaque for reads)

    # -- write paths ---------------------------------------------------
    def insert(self, item: MemoryItem) -> MemoryItem:
        self.qdrant.upsert(item)
        self.graph.upsert_memory(item)
        self._cache[item.id] = item
        return item

    def get_item(self, mem_id: str) -> MemoryItem | None:
        return self._cache.get(mem_id)

    def reinforce(self, mem_id: str, gain: float) -> float:
        rho = self.s.rho
        assert 0.0 < rho < 1.0, "EMA rho must satisfy 0<rho<1"
        item = self._cache.get(mem_id)
        if item is None:
            m = self.graph.get_memory(mem_id)
            if m is None:
                return 0.0
            item = MemoryItem(id=mem_id, phi=np.zeros(1), h=np.zeros(1), g=np.zeros(1),
                              omega=float(m.get("omega", 1.0)), tau=int(m.get("tau", 0)),
                              cls=m.get("class", "__unlabeled__"))
            self._cache[mem_id] = item
        item.omega = rho * item.omega + (1.0 - rho) * float(gain)
        self.graph.upsert_memory(item)
        if item.phi.size > 1:
            self.qdrant.upsert(item)
        return item.omega

    def merge(self, target_id: str, cand: MemoryItem, gain: float, t: int) -> MemoryItem | None:
        """Contractive merge: target moves a fraction c<1 toward the candidate."""
        c = self.s.merge_contraction
        assert c < 1.0, "merge must be contractive (c<1)"
        target = self._cache.get(target_id)
        if target is None:
            return None
        target.phi = (1 - c) * target.phi + c * _fit(cand.phi, target.phi.shape)
        target.h = (1 - c) * target.h + c * _fit(cand.h, target.h.shape)
        target.g = (1 - c) * target.g + c * _fit(cand.g, target.g.shape)
        rho = self.s.rho
        target.omega = rho * target.omega + (1 - rho) * float(gain)
        target.tau = t
        self.graph.upsert_memory(target)
        self.qdrant.upsert(target)
        # provenance: candidate region derived into the target
        if cand.world_ctx_id:
            self.graph.link("DERIVED_FROM", target.id, cand.world_ctx_id)
        return target

    def decay(self, mem_id: str, factor: float = 0.95) -> float:
        item = self._cache.get(mem_id)
        if item is None:
            return 0.0
        item.omega *= factor
        self.graph.upsert_memory(item)
        return item.omega

    def evict(self, mem_id: str) -> None:
        self.qdrant.delete(mem_id)
        self.graph.delete_memory(mem_id)
        self._cache.pop(mem_id, None)


def _fit(v: np.ndarray, shape) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    n = int(np.prod(shape))
    if a.size >= n:
        return a[:n].reshape(shape)
    return np.pad(a, (0, n - a.size)).reshape(shape)
