"""MALARState — the shared typed state passed through the loop / LangGraph.

Holds the live world, the sampled region, the per-modality encodings, identified
objects, field values, the policy decision, and the running audit. LangGraph nodes
read/write this state; the in-process loop uses the same object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MALARState:
    t: int = 0
    mode: str = "train"                 # 'train' | 'infer'
    review_mode: bool = True
    # world
    world: Any = None                   # WorldGraph
    region: Any = None                  # Region
    region_world: Any = None            # WorldGraph (subgraph)
    snapshot_id: str | None = None
    world_ctx: Any = None               # WorldContext
    # encodings
    phi: np.ndarray | None = None
    h: np.ndarray | None = None
    g: np.ndarray | None = None
    world_emb: np.ndarray | None = None
    topo_summary: dict = field(default_factory=dict)
    diagrams: list = field(default_factory=list)
    # identification & fields
    objects: list[dict] = field(default_factory=list)   # [{class, ctx, node_idx}]
    r_values: dict = field(default_factory=dict)        # {objective_k: value}
    v_field: np.ndarray | None = None
    action_set: list[str] = field(default_factory=list)
    # decisions
    policy_decision: Any = None
    curator_decision: Any = None
    # bookkeeping
    novelty: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    # HITL / review
    pending_stage: str | None = None
    stage_outputs: dict = field(default_factory=dict)

    def log(self, stage: str, payload: dict) -> None:
        self.events.append({"t": self.t, "stage": stage, **payload})
        self.stage_outputs[stage] = payload
