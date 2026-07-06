"""Objects = labeled memory.  o = (class c, g_o, phi_o, h_o, omega_o, tau_o).

The ObjectRegistry holds labeled objects and supports fused top-k retrieval over the
three named modalities (phi / hyper / graph). Backed in-process here; the memory
layer mirrors objects into Neo4j + Qdrant. Every object carries the WorldContext it
was identified in.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from malar.encoders.change import cosine_similarity


@dataclass
class MalarObject:
    id: str
    cls: str
    phi: np.ndarray
    h: np.ndarray
    g: np.ndarray
    omega: float = 1.0
    tau: int = 0
    world_ctx_id: str | None = None
    provenance: str = "data"      # 'data' | 'human' | 'llm'
    candidate: bool = False       # awaiting label
    meta: dict = field(default_factory=dict)


def make_object_id(cls: str, phi: np.ndarray, t: int) -> str:
    h = hashlib.sha256()
    h.update(cls.encode())
    h.update(np.asarray(phi, dtype=float).tobytes())
    h.update(str(t).encode())
    return "obj_" + h.hexdigest()[:12]


@dataclass
class RetrievalHit:
    obj: MalarObject
    score: float
    sims: dict


class ObjectRegistry:
    def __init__(self, w_phi: float = 1.0, w_h: float = 1.0, w_g: float = 1.0):
        self.objects: dict[str, MalarObject] = {}
        self.w_phi, self.w_h, self.w_g = w_phi, w_h, w_g

    def add(self, obj: MalarObject) -> MalarObject:
        self.objects[obj.id] = obj
        return obj

    def add_candidate(self, phi, h, g, t: int, world_ctx_id: str | None = None) -> MalarObject:
        oid = make_object_id("__candidate__", phi, t)
        obj = MalarObject(id=oid, cls="__candidate__", phi=np.asarray(phi), h=np.asarray(h),
                          g=np.asarray(g), tau=t, world_ctx_id=world_ctx_id, candidate=True)
        return self.add(obj)

    def label_candidate(self, obj_id: str, cls: str, by: str = "human") -> MalarObject | None:
        obj = self.objects.get(obj_id)
        if obj is None:
            return None
        obj.cls = cls
        obj.candidate = False
        obj.provenance = "human" if by == "human" else "data"
        return obj

    def _fused(self, phi, h, g, obj: MalarObject) -> tuple[float, dict]:
        s_phi = cosine_similarity(phi, obj.phi)
        s_h = cosine_similarity(h, obj.h)
        s_g = cosine_similarity(g, obj.g)
        denom = self.w_phi + self.w_h + self.w_g
        score = (self.w_phi * s_phi + self.w_h * s_h + self.w_g * s_g) / denom
        return score, {"phi": s_phi, "h": s_h, "g": s_g}

    def topk(self, phi, h, g, k: int = 5, include_candidates: bool = False) -> list[RetrievalHit]:
        hits = []
        for obj in self.objects.values():
            if obj.candidate and not include_candidates:
                continue
            score, sims = self._fused(phi, h, g, obj)
            hits.append(RetrievalHit(obj=obj, score=score, sims=sims))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:k]

    def classes(self) -> list[str]:
        return sorted({o.cls for o in self.objects.values() if not o.candidate})
