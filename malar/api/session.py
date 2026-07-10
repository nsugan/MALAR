"""EngineSession — the live control-plane object the API drives.

Owns the engine, the agent graph, the active adapter and the run state. Exposes the
operations the REST endpoints map to: run / step / query / generate / curate / label,
plus mode (train/infer) and review toggles. Emits stage events to subscribers (WS).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from malar.agents.base import STAGES, AgentContext
from malar.agents.graph import MalarAgentGraph
from malar.core.loop import MalarEngine
from malar.core.state import MALARState
from malar.fields.functional_field import FunctionalSpec
from malar.fields.objectives import ObjectiveSpec
from malar.generation.generate import ReverseGenerator
from malar.world.adapters.synthetic import SyntheticAdapter


@dataclass
class RunStatus:
    mode: str = "idle"             # idle | train | infer
    running: bool = False
    paused: bool = False
    tick: int = 0
    review_mode: bool = True
    review_stages: list[str] = field(default_factory=lambda: list(STAGES))


class EngineSession:
    def __init__(self):
        self.engine: MalarEngine | None = None
        self.graph: MalarAgentGraph | None = None
        self.adapter = None
        self.status = RunStatus()
        self.subscribers: list[Callable[[dict], None]] = []
        self.history: list[MALARState] = []
        self._lock = threading.Lock()
        self.generator = ReverseGenerator()

    # -- subscriptions -------------------------------------------------
    def subscribe(self, cb: Callable[[dict], None]) -> None:
        self.subscribers.append(cb)

    def emit(self, event: dict) -> None:
        for cb in list(self.subscribers):
            try:
                cb(event)
            except Exception:
                pass

    # -- lifecycle -----------------------------------------------------
    def configure(self, domain: str = "synthetic", review_mode: bool = True,
                  budget: int = 64) -> dict:
        # Legacy single-session path. Default is the generic synthetic adapter; the raman
        # adapter is lazy-imported only if a caller explicitly asks for a "raman*" domain
        # (backward compat), so this module no longer hard-depends on domain-specific code.
        if domain.startswith("raman"):
            from malar.world.adapters.raman import RamanAdapter
            self.adapter = RamanAdapter(n_ticks=18, n_per_class=8, n_bands=128, k=6)
            objectives = [ObjectiveSpec(key="sensitivity"), ObjectiveSpec(key="specificity")]
            functionals = [FunctionalSpec(dim="response",
                                          actions=["confirm", "escalate", "watchlist"])]
            world_id = "raman_virus"
        else:
            self.adapter = SyntheticAdapter(n_ticks=12, points_per_tick=40, k=6)
            objectives = [ObjectiveSpec(key="quality")]
            functionals = [FunctionalSpec(dim="response", actions=["flag", "watchlist"])]
            world_id = "synthetic"
        self.engine = MalarEngine(self.adapter, world_id=world_id, objectives=objectives,
                                  functionals=functionals, budget=budget)
        self.generator = ReverseGenerator(self.engine.c.spectral)
        ctx = AgentContext(engine=self.engine)
        review_stages = set(STAGES) if review_mode else set()
        self.graph = MalarAgentGraph(ctx, review_stages=review_stages)
        self.status = RunStatus(mode="idle", review_mode=review_mode,
                                review_stages=list(review_stages))
        return {"domain": domain, "review_mode": review_mode, "world_id": world_id}

    def set_review(self, review_mode: bool, stages: list[str] | None = None) -> dict:
        self.status.review_mode = review_mode
        if review_mode:
            self.status.review_stages = stages if stages is not None else list(STAGES)
        else:
            self.status.review_stages = []
        if self.engine is not None:
            ctx = AgentContext(engine=self.engine)
            self.graph = MalarAgentGraph(ctx, review_stages=set(self.status.review_stages))
        return {"review_mode": review_mode, "stages": self.status.review_stages}

    # -- run / step ----------------------------------------------------
    def step(self) -> MALARState | None:
        if self.engine is None:
            self.configure()
        batches = self.adapter.batches()
        if self.status.tick >= len(batches):
            return None
        batch = batches[self.status.tick]
        ms = self.graph.run_tick(batch)
        self.status.tick += 1
        self.history.append(ms)
        self.emit({"type": "stage", "tick": ms.t, "events": ms.events,
                   "memory_size": self.engine.memory_size(),
                   "action": ms.policy_decision.action if ms.policy_decision else None})
        return ms

    def run(self, mode: str = "train", n: int | None = None) -> list[dict]:
        self.status.mode = mode
        self.status.running = True
        out = []
        batches = self.adapter.batches()
        limit = len(batches) if n is None else min(n, len(batches))
        while self.status.tick < limit and not self.status.paused:
            ms = self.step()
            if ms is None:
                break
            out.append({"tick": ms.t, "class": ms.objects[0]["class"],
                        "memory_action": ms.curator_decision.action,
                        "action": ms.policy_decision.action})
        self.status.running = False
        return out

    def pause(self) -> None:
        self.status.paused = True

    def resume(self) -> None:
        self.status.paused = False

    def stop(self) -> None:
        self.status.running = False
        self.status.paused = False

    # -- queries -------------------------------------------------------
    def query(self, k: int = 10) -> list[dict]:
        if self.engine is None:
            return []
        return [{"id": m["id"], "class": m.get("class"), "tau": m.get("tau"),
                 "omega": m.get("omega"), "world_ctx_id": m.get("world_ctx_id")}
                for m in self.engine.c.store.graph.all_memories()[:k]]

    def generate(self) -> dict:
        if not self.history:
            return {"error": "no history; run first"}
        ms = self.history[-1]
        greg = self.generator.generate(ms.topo_summary, ms.h)
        return {"n_loops": greg.n_loops, "n_points": int(len(greg.points)),
                "has_spectra": greg.spectra is not None}

    def curate(self) -> dict:
        if self.engine is None:
            return {"error": "no engine"}
        evicted = self.engine.c.curator.compress(self.status.tick)
        return {"evicted": evicted, "audit_tail": self.engine.c.curator.audit[-5:]}

    def labels_pending(self) -> list[dict]:
        if self.engine is None:
            return []
        reg = self.engine.c.registry
        return [{"id": o.id, "score": None} for o in reg.objects.values() if o.candidate]

    def label(self, obj_id: str, cls: str) -> dict:
        if self.engine is None:
            return {"error": "no engine"}
        obj = self.engine.c.registry.label_candidate(obj_id, cls, by="human")
        # correction: human value becomes sticky in the objective fields
        applied = {}
        for k, ofield in self.engine.c.objective_fields.items():
            applied[k] = ofield.set_human_value(cls, 1.0, by="human")
        return {"labeled": obj is not None, "class": cls, "value_set": applied,
                "provenance": "human"}
