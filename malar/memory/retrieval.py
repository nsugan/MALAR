"""Fused retrieval  Score(m | S, t).

Score(m|S,t) = w_phi*Sim(phi_S,phi_m) + w_h*Sim(h_S,h_m) + w_g*Sim(g_S,g_m)
               + lambda*TimeGate(t, tau_m) + mu*omega_m

Qdrant prefetch per named vector -> weighted fusion -> payload filter on tau/omega.
Structure (g_m) is hydrated from Neo4j only for the top-k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from malar.core.config import get_settings
from malar.memory.neo4j_io import MemoryGraph
from malar.memory.qdrant_io import QdrantMemory


def time_gate(t: int, tau: int, halflife: float = 50.0) -> float:
    dt = max(0, t - tau)
    return math.exp(-dt / max(1e-9, halflife))


@dataclass
class FusedHit:
    mem_id: str
    score: float
    components: dict
    payload: dict
    structure: dict | None = None


class FusedRetriever:
    def __init__(self, qdrant: QdrantMemory, graph: MemoryGraph, settings=None):
        self.qdrant = qdrant
        self.graph = graph
        self.s = settings or get_settings()

    def retrieve(self, phi, h, g, t: int, k: int = 5, prefetch: int = 20,
                 tau_min: int | None = None, omega_min: float | None = None) -> list[FusedHit]:
        named = {"phi": phi, "hyper": h, "graph_emb": g}
        per = self.qdrant.prefetch(named, limit=prefetch)

        # gather union of candidate ids with per-modality similarity
        sims: dict[str, dict] = {}
        payloads: dict[str, dict] = {}
        name_to_w = {"phi": self.s.w_phi, "hyper": self.s.w_h, "graph_emb": self.s.w_g}
        for name, rows in per.items():
            for mem_id, score, payload in rows:
                if mem_id is None:
                    continue
                sims.setdefault(mem_id, {})[name] = score
                payloads[mem_id] = payload

        hits: list[FusedHit] = []
        wsum = sum(name_to_w.values())
        for mem_id, modal in sims.items():
            payload = payloads[mem_id]
            tau = int(payload.get("tau", 0))
            omega = float(payload.get("omega", 0.0))
            if tau_min is not None and tau < tau_min:
                continue
            if omega_min is not None and omega < omega_min:
                continue
            fused_sim = sum(name_to_w[n] * modal.get(n, 0.0) for n in name_to_w) / wsum
            tg = time_gate(t, tau)
            score = fused_sim + self.s.lam_time * tg + self.s.mu_omega * omega
            hits.append(FusedHit(mem_id=mem_id, score=score,
                                 components={"sim": fused_sim, "time": tg, "omega": omega,
                                             "modal": modal},
                                 payload=payload))
        hits.sort(key=lambda x: x.score, reverse=True)
        topk = hits[:k]
        # hydrate structure from the graph only for the top-k
        for hit in topk:
            m = self.graph.get_memory(hit.mem_id)
            if m is not None:
                hit.structure = m.get("g_subgraph")
        return topk
