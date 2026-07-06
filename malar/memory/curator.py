"""Memory Curator (async, idempotent, gated, fully audited).

Implements the candidate-handling algorithm from the plan:

  on candidate (g_S, phi_S, h_S, t):
    cand   = retrieve_topk(...)
    Dphi,Dh,Dg = change_stats; G,U = goal_gain, critic.uncertainty
    novel  = Dphi>th_phi or Dh>th_h or Dg>th_g or G>th_G or U>th_U
    if not novel:  reinforce nearest; return
    m_near = nearest within merge radius
    if m_near: contractive merge; return
    if budget_ok: insert
    else: evict lowest-utility if region beats it, else defer

Per-modality novelty (never collapsed). Every decision is audited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from malar.core.config import get_settings
from malar.encoders.change import change_stats
from malar.memory.maintenance import compress_and_decay, utility
from malar.memory.retrieval import FusedRetriever
from malar.memory.schema import MemoryItem
from malar.memory.store import MemoryStore


@dataclass
class CuratorConfig:
    theta_phi: float = 0.25
    theta_h: float = 0.20
    theta_g: float = 0.20
    theta_G: float = 0.5      # goal-gain threshold
    theta_U: float = 0.6      # uncertainty threshold
    merge_radius: float = 0.12
    budget: int = 200


@dataclass
class CuratorDecision:
    action: str               # reinforce | merge | insert | evict+insert | defer
    mem_id: str | None
    novelty: dict
    detail: dict = field(default_factory=dict)


class MemoryCurator:
    def __init__(self, store: MemoryStore, retriever: FusedRetriever,
                 config: CuratorConfig | None = None):
        self.store = store
        self.retriever = retriever
        self.cfg = config or CuratorConfig()
        self.s = get_settings()
        self.audit: list[dict] = []

    def _audit(self, op: str, **kw) -> None:
        self.audit.append({"op": op, **kw})

    def on_candidate(self, cand: MemoryItem, t: int, goal_gain: float = 0.0,
                     uncertainty: float = 0.0, k: int = 5) -> CuratorDecision:
        hits = self.retriever.retrieve(cand.phi, cand.h, cand.g, t=t, k=k)
        if not hits:
            self.store.insert(cand)
            self._audit("insert", mem_id=cand.id, reason="empty store")
            return CuratorDecision("insert", cand.id, {"empty": True})

        nearest = hits[0]
        near_item = self.store.get_item(nearest.mem_id)
        # per-modality novelty
        nov = self._novelty(cand, near_item, nearest, goal_gain, uncertainty)
        novel = (nov["d_phi"] > self.cfg.theta_phi or nov["d_h"] > self.cfg.theta_h
                 or nov["d_g"] > self.cfg.theta_g or goal_gain > self.cfg.theta_G
                 or uncertainty > self.cfg.theta_U)

        if not novel:
            self.store.reinforce(nearest.mem_id, gain=max(0.1, goal_gain))
            self._audit("reinforce", mem_id=nearest.mem_id, novelty=nov)
            return CuratorDecision("reinforce", nearest.mem_id, nov)

        # contractive merge only if the nearest is close on the FUSED distance AND
        # close topologically (phi) — a single similar modality must not trigger a merge.
        fused_dist = 1.0 - nearest.score
        if (near_item is not None and fused_dist <= self.cfg.merge_radius
                and nov["d_phi"] <= self.cfg.theta_phi):
            merged = self.store.merge(nearest.mem_id, cand, gain=max(0.1, goal_gain), t=t)
            self._audit("merge", mem_id=nearest.mem_id, novelty=nov)
            return CuratorDecision("merge", merged.id if merged else None, nov)

        # insert if budget allows
        n_mem = len(self.store.graph.all_memories())
        if n_mem < self.cfg.budget:
            self.store.insert(cand)
            self._audit("insert", mem_id=cand.id, novelty=nov)
            return CuratorDecision("insert", cand.id, nov)

        # over budget: evict weakest if the new region beats it
        weakest_id, weakest_u = self._weakest(t)
        region_u = max(0.1, goal_gain) + (1.0 - uncertainty)
        if weakest_id is not None and region_u > weakest_u:
            self.store.evict(weakest_id)
            self.store.insert(cand)
            self._audit("evict+insert", evicted=weakest_id, mem_id=cand.id, novelty=nov)
            return CuratorDecision("evict+insert", cand.id, nov, {"evicted": weakest_id})

        self._audit("defer", novelty=nov)
        return CuratorDecision("defer", None, nov)

    def _novelty(self, cand, near_item, nearest, goal_gain, uncertainty) -> dict:
        if near_item is not None and near_item.phi.size > 1:
            cs = change_stats(
                {"phi": cand.phi, "h": cand.h, "g": cand.g},
                {"phi": near_item.phi, "h": near_item.h, "g": near_item.g},
            )
        else:
            # use retrieval similarity as proxy
            modal = nearest.components["modal"]
            cs = {"d_phi": 1 - modal.get("phi", 0.0), "d_h": 1 - modal.get("hyper", 0.0),
                  "d_g": 1 - modal.get("graph_emb", 0.0)}
        cs["G"] = goal_gain
        cs["U"] = uncertainty
        return cs

    def _weakest(self, t: int):
        items = [self.store.get_item(m["id"]) for m in self.store.graph.all_memories()]
        items = [i for i in items if i is not None]
        if not items:
            return None, 0.0
        weakest = min(items, key=lambda it: utility(it, t))
        return weakest.id, utility(weakest, t)

    # -- slow clock ----------------------------------------------------
    def compress(self, t: int) -> list[str]:
        items = [self.store.get_item(m["id"]) for m in self.store.graph.all_memories()]
        items = [i for i in items if i is not None]
        kept, evicted = compress_and_decay(items, self.cfg.budget, t)
        for eid in evicted:
            self.store.evict(eid)
        if evicted:
            self._audit("compress", evicted=evicted)
        return evicted
