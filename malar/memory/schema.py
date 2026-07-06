"""Memory item schema  m = (g, phi, h, omega, tau, theta_gen) + grounding.

Stored in two keyed stores by memory_id:
  * Neo4j : (:Memory {id, tau, omega, class, cost, encoder_version}) + g_m subgraph
            + edges SIMILAR / DERIVED_FROM / TEMPORAL_NEXT / GROUNDED_IN -> WorldContext
  * Qdrant: one point per memory, named vectors phi/hyper/graph_emb, payload
            {tau, omega, class, neo4j_id, encoder_version, world_ctx_id, snapshot_id}
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MemoryItem:
    id: str
    phi: np.ndarray
    h: np.ndarray
    g: np.ndarray                 # graph_emb
    omega: float = 1.0            # importance (EMA)
    tau: int = 0                  # time of (last) update
    cls: str = "__unlabeled__"
    cost: float = 1.0
    encoder_version: str = "v1"
    world_ctx_id: str | None = None
    snapshot_id: str | None = None
    g_subgraph: dict | None = None     # structural abstraction g_m
    theta_gen: dict | None = None      # generation params
    diagram_path: str | None = None    # raw persistence diagram artifact
    spectra_path: str | None = None    # raw spectra artifact
    meta: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "tau": self.tau,
            "omega": self.omega,
            "class": self.cls,
            "neo4j_id": self.id,
            "encoder_version": self.encoder_version,
            "world_ctx_id": self.world_ctx_id,
            "snapshot_id": self.snapshot_id,
        }


def new_memory_id(phi: np.ndarray, t: int, salt: str = "") -> str:
    h = hashlib.sha256()
    h.update(np.asarray(phi, dtype=float).tobytes())
    h.update(str(t).encode())
    h.update(salt.encode())
    return "mem_" + h.hexdigest()[:14]
