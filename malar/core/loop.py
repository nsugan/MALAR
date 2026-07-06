"""core/loop.py — Algorithm 1, one MALAR timestep, wired over M1-M5.

observe -> F_rep -> sample region S -> snapshot+context -> encode (phi,h,g)
-> retrieve/novelty -> identify objects -> fields (R, V, F-action-set)
-> policy/action -> Curator (store/merge/decay). The Curator runs on its own slow
clock. Deterministic CLI over synthetic data; memory grows then stabilises.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malar.core.config import get_settings
from malar.core.state import MALARState
from malar.encoders.graph import GraphEncoder
from malar.encoders.spectral import SpectralEncoder
from malar.encoders.topology import TopologyEncoder
from malar.fields.functional_field import FunctionalField, FunctionalSpec, region_action_set
from malar.fields.functional_store import FunctionalStore
from malar.fields.objective_field import ObjectiveField
from malar.fields.objectives import ObjectiveSpec
from malar.fields.value_store import ValueStore
from malar.fields.values import diffuse_values, seed_from_objects
from malar.memory.curator import CuratorConfig, MemoryCurator
from malar.memory.neo4j_io import make_memory_graph
from malar.memory.qdrant_io import QdrantMemory
from malar.memory.retrieval import FusedRetriever
from malar.memory.schema import MemoryItem, new_memory_id
from malar.memory.store import MemoryStore
from malar.objects.identify import Identifier
from malar.objects.registry import ObjectRegistry
from malar.policy.policy import Policy
from malar.validation.conformal import ConformalGate
from malar.world.adapters.base import WorldAdapter
from malar.world.context import make_context
from malar.world.representation import f_rep
from malar.world.sampling import RegionSampler
from malar.world.snapshot import SnapshotStore


@dataclass
class EngineComponents:
    topo: TopologyEncoder
    spectral: SpectralEncoder
    graph: GraphEncoder
    sampler: RegionSampler
    snapshots: SnapshotStore
    registry: ObjectRegistry
    identifier: Identifier
    store: MemoryStore
    retriever: FusedRetriever
    curator: MemoryCurator
    policy: Policy
    objective_fields: dict
    functional_fields: list


class MalarEngine:
    def __init__(self, adapter: WorldAdapter, world_id: str = "world",
                 objectives: list[ObjectiveSpec] | None = None,
                 functionals: list[FunctionalSpec] | None = None,
                 budget: int = 64, region_budget: int = 16, llm_client=None,
                 compress_every: int = 25, domain_id: str = "default"):
        self.s = get_settings()
        self.adapter = adapter
        self.world_id = world_id
        self.domain_id = domain_id
        self.llm = llm_client
        self.compress_every = compress_every

        # Per-domain isolated artifact / snapshot dirs.
        dom_dir = self.s.data_dir / "domains" / domain_id
        art = str(dom_dir / "artifacts")
        topo = TopologyEncoder(artifacts_dir=art)
        spectral = SpectralEncoder(latent_dim=16, artifacts_dir=art)
        graph = GraphEncoder()
        sampler = RegionSampler(budget=region_budget, rng=0)
        snapshots = SnapshotStore(dom_dir / "snapshots")
        registry = ObjectRegistry(self.s.w_phi, self.s.w_h, self.s.w_g)
        gate = ConformalGate(alpha=self.s.conformal_alpha)
        identifier = Identifier(registry, gate, theta_match=0.75, llm_client=llm_client)

        # Per-domain Qdrant collection (mem__{domain_id}) and an isolated memory graph.
        qdrant = QdrantMemory(url=self.s.qdrant_url if _qdrant_reachable(self.s.qdrant_url)
                              else ":memory:",
                              collection=f"mem__{domain_id}",
                              dims={"phi": 432, "hyper": 16, "graph_emb": 48})
        graph_db = make_memory_graph(domain_id=domain_id)
        store = MemoryStore(qdrant, graph_db)
        retriever = FusedRetriever(qdrant, graph_db)
        curator = MemoryCurator(store, retriever, CuratorConfig(budget=budget))

        value_store = ValueStore()
        objective_fields = {}
        for spec in (objectives or [ObjectiveSpec(key="sensitivity")]):
            objective_fields[spec.key] = ObjectiveField(spec, value_store)
        func_store = FunctionalStore()
        functional_fields = []
        for fspec in (functionals or [FunctionalSpec(dim="response", actions=["flag", "watchlist"])]):
            functional_fields.append(FunctionalField(fspec, func_store))

        self.c = EngineComponents(
            topo=topo, spectral=spectral, graph=graph, sampler=sampler, snapshots=snapshots,
            registry=registry, identifier=identifier, store=store, retriever=retriever,
            curator=curator, policy=Policy(), objective_fields=objective_fields,
            functional_fields=functional_fields,
        )
        self.value_store = value_store
        self.func_store = func_store
        self._spectral_fit = False

    def step(self, batch, mode: str = "train") -> MALARState:
        st = MALARState(t=batch.t, mode=mode, review_mode=self.s.review_mode)
        world = self.adapter.to_world(batch)
        # F_rep is an auxiliary smoothed representation for importance/sampling; it
        # does NOT replace the geometry the encoders read.
        world.meta["rep"] = f_rep(world, steps=1)
        st.world = world
        st.log("perception", {"n_nodes": world.n_nodes, "source": batch.source})

        if not self._spectral_fit and world.n_features >= 2:
            self.c.spectral.fit(world.features)
            self._spectral_fit = True

        region = self.c.sampler.sample(world)
        rw = self.c.sampler.region_world(world, region)
        st.region, st.region_world = region, rw

        snap = self.c.snapshots.save_full(world)
        st.snapshot_id = snap.id
        ctx = make_context(self.world_id, snap.id, region, world.t, batch.source)
        st.world_ctx = ctx
        self.c.store.graph.upsert_snapshot({"id": snap.id, "t": snap.t, "hash": snap.hash})
        self.c.store.graph.upsert_world_context(ctx.to_dict())

        tr = self.c.topo.encode(rw, key=ctx.region_id)
        sr = self.c.spectral.encode(rw, key=ctx.region_id)
        gr = self.c.graph.encode(rw)
        st.phi, st.h, st.g = tr.phi, sr.h, gr.graph_emb
        st.world_emb = self.c.graph.world_embed(world)
        st.topo_summary, st.diagrams = tr.summary, tr.diagrams
        st.log("encode", {"phi_dim": len(tr.phi), "h_dim": len(sr.h), "g_dim": len(gr.graph_emb),
                          "topo": tr.summary})

        ident = self.c.identifier.identify(st.phi, st.h, st.g, st.topo_summary,
                                           world_ctx_id=ctx.id, t=world.t)
        cls = ident.cls or (batch.labels[region.indices[0]] if batch.labels else None) or "__unlabeled__"
        st.objects = [{"class": cls, "ctx": 1.0, "node_idx": region.indices[0] if region.indices else 0}]
        st.log("identify", {"matched": ident.matched, "class": cls, "method": ident.method,
                            "score": ident.score})

        for k, ofield in self.c.objective_fields.items():
            ofield.learn_from_data(cls, observed_contribution=1.0, world_ctx=ctx.id)
            R, _ = ofield.evaluate(st.objects)
            st.r_values[k] = R
        for ff in self.c.functional_fields:
            ff.learn_affordance(cls, ff.spec.actions[0], True, world_ctx=ctx.id)
        st.action_set = region_action_set(st.objects, self.c.functional_fields)

        r_seed = max(st.r_values.values()) if st.r_values else 1.0
        st.v_field = diffuse_values(rw, seed_from_objects(rw, {0: r_seed}), steps=8)
        st.log("fields", {"R": st.r_values, "action_set": st.action_set})

        r_val = max(st.r_values.values()) if st.r_values else 0.0
        v_val = float(np.max(st.v_field)) if st.v_field is not None and st.v_field.size else 0.0
        st.policy_decision = self.c.policy.decide(st.action_set, [cls], r_val, v_val)
        st.log("policy", {"action": st.policy_decision.action,
                          "rationale": st.policy_decision.rationale})

        cand = MemoryItem(id=new_memory_id(st.phi, world.t, ctx.id), phi=st.phi, h=st.h, g=st.g,
                          cls=cls, tau=world.t, world_ctx_id=ctx.id, snapshot_id=snap.id,
                          diagram_path=tr.artifact_path, spectra_path=sr.artifact_path)
        goal_gain = 1.0 - (ident.score if ident.matched else 0.0)
        uncertainty = 0.0 if ident.conformal_ok else 0.7
        st.curator_decision = self.c.curator.on_candidate(cand, t=world.t, goal_gain=goal_gain,
                                                          uncertainty=uncertainty)
        st.novelty = st.curator_decision.novelty
        st.log("memory", {"action": st.curator_decision.action,
                          "mem_id": st.curator_decision.mem_id})

        # Training mode: promote a labelled prototype into the object registry so
        # identification matures (LLM-heavy cold start -> deterministic f_k takeover).
        if mode == "train" and cls != "__unlabeled__":
            self.register_object(cls, st.phi, st.h, st.g, world.t, ctx.id)

        if world.t > 0 and world.t % self.compress_every == 0:
            evicted = self.c.curator.compress(world.t)
            st.log("compress", {"evicted": evicted})
        self.last_state = st          # latest loop state (for per-agent monitoring)
        return st

    def run(self, n: int | None = None, mode: str = "train") -> list[MALARState]:
        states = []
        for i, batch in enumerate(self.adapter.stream()):
            if n is not None and i >= n:
                break
            states.append(self.step(batch, mode=mode))
        return states

    def register_object(self, cls: str, phi, h, g, t: int, ctx_id: str | None) -> None:
        """Maintain one (updated) labelled prototype per class in the object registry.

        Keeps identification grounded in learned objects without unbounded growth: an
        existing prototype for the class is EMA-blended toward the new evidence.
        """
        from malar.objects.registry import MalarObject, make_object_id

        existing = [o for o in self.c.registry.objects.values()
                    if not o.candidate and o.cls == cls]
        if existing:
            o = existing[0]
            a = 0.7

            def _blend(old, new):
                old = np.asarray(old, dtype=float).ravel()
                new = np.asarray(new, dtype=float).ravel()
                if old.size != new.size:  # align legacy/persisted dims before EMA
                    n = max(old.size, new.size)
                    old = np.pad(old, (0, n - old.size))
                    new = np.pad(new, (0, n - new.size))
                return a * old + (1 - a) * new

            o.phi = _blend(o.phi, phi)
            o.h = _blend(o.h, h)
            o.g = _blend(o.g, g)
            o.tau = t
        else:
            oid = make_object_id(cls, phi, t)
            self.c.registry.add(MalarObject(id=oid, cls=cls, phi=np.asarray(phi),
                                            h=np.asarray(h), g=np.asarray(g), tau=t,
                                            world_ctx_id=ctx_id, provenance="data"))

    def memory_size(self) -> int:
        return len(self.c.store.graph.all_memories())


def _qdrant_reachable(url: str) -> bool:
    try:
        import httpx

        httpx.get(url + "/collections", timeout=0.3)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    import argparse

    from malar.world.adapters.synthetic import SyntheticAdapter

    ap = argparse.ArgumentParser(description="Run the MALAR loop over synthetic data.")
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--points", type=int, default=40)
    args = ap.parse_args(argv)

    adapter = SyntheticAdapter(n_ticks=args.ticks, points_per_tick=args.points, k=6)
    engine = MalarEngine(adapter, world_id="synthetic", budget=32)
    sizes = []
    for st in engine.run():
        sizes.append(engine.memory_size())
        print(f"t={st.t} class={st.objects[0]['class']:>8} "
              f"mem_action={st.curator_decision.action:>12} mem_size={sizes[-1]} "
              f"action={st.policy_decision.action}")
    print(f"final memory size: {sizes[-1]}; growth curve: {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
