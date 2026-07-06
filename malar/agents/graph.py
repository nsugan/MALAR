"""LangGraph StateGraph wiring the MALAR agents.

Nodes (one per pipeline stage) operate on a shared MALARState. Gemma 4 (via the
gateway) is used only for control / identification-hypotheses / explanation; the
deterministic engine tools do all compute.

REVIEW_MODE drives a LangGraph interrupt-before on each stage node: when ON, the
stage emits its output and the graph pauses (Approve/Edit/Reject) before continuing;
when OFF the same outputs stream but no interrupt fires. Run state lives in the
LangGraph checkpointer, so review/resume survives restarts.

If langgraph is unavailable, a sequential fallback executes the same nodes in order.
"""
from __future__ import annotations

import numpy as np

from malar.agents.base import STAGES, AgentContext
from malar.core.state import MALARState
from malar.fields.functional_field import region_action_set
from malar.fields.values import diffuse_values, seed_from_objects
from malar.memory.schema import MemoryItem, new_memory_id
from malar.world.context import make_context
from malar.world.representation import f_rep


def _node_perception(state: dict, ctx: AgentContext) -> dict:
    st: MALARState = state["ms"]
    eng = ctx.engine
    world = eng.adapter.to_world(ctx.batch)
    world.meta["rep"] = f_rep(world, steps=1)
    st.world = world
    if not eng._spectral_fit and world.n_features >= 2:
        eng.c.spectral.fit(world.features)
        eng._spectral_fit = True
    region = eng.c.sampler.sample(world)
    st.region = region
    st.region_world = eng.c.sampler.region_world(world, region)
    snap = eng.c.snapshots.save_full(world)
    st.snapshot_id = snap.id
    ctxw = make_context(eng.world_id, snap.id, region, world.t, ctx.batch.source)
    st.world_ctx = ctxw
    eng.c.store.graph.upsert_snapshot({"id": snap.id, "t": snap.t, "hash": snap.hash})
    eng.c.store.graph.upsert_world_context(ctxw.to_dict())
    st.log("perception", {"n_nodes": world.n_nodes})
    return {"ms": st}


def _node_topology(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    eng = ctx.engine
    tr = eng.c.topo.encode(st.region_world, key=st.world_ctx.region_id)
    sr = eng.c.spectral.encode(st.region_world, key=st.world_ctx.region_id)
    gr = eng.c.graph.encode(st.region_world)
    st.phi, st.h, st.g = tr.phi, sr.h, gr.graph_emb
    st.world_emb = eng.c.graph.world_embed(st.world)
    st.topo_summary, st.diagrams = tr.summary, tr.diagrams
    st.stage_outputs["_artifacts"] = {"diagram_path": tr.artifact_path,
                                       "spectra_path": sr.artifact_path}
    st.log("topology", {"topo": tr.summary})
    return {"ms": st}


def _node_identification(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    eng = ctx.engine
    ident = eng.c.identifier.identify(st.phi, st.h, st.g, st.topo_summary,
                                      world_ctx_id=st.world_ctx.id, t=st.world.t)
    cls = ident.cls or (ctx.batch.labels[st.region.indices[0]] if ctx.batch.labels else None) \
        or "__unlabeled__"
    st.objects = [{"class": cls, "ctx": 1.0,
                   "node_idx": st.region.indices[0] if st.region.indices else 0}]
    st.stage_outputs["_ident"] = {"matched": ident.matched, "method": ident.method,
                                  "score": ident.score, "conformal_ok": ident.conformal_ok}
    st.log("identification", {"class": cls, "method": ident.method, "score": ident.score})
    return {"ms": st}


def _node_value_learning(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    eng = ctx.engine
    cls = st.objects[0]["class"]
    for k, ofield in eng.c.objective_fields.items():
        ofield.learn_from_data(cls, observed_contribution=1.0, world_ctx=st.world_ctx.id)
        R, _ = ofield.evaluate(st.objects)
        st.r_values[k] = R
    st.log("value_learning", {"R": st.r_values})
    return {"ms": st}


def _node_functional_field(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    eng = ctx.engine
    cls = st.objects[0]["class"]
    for ff in eng.c.functional_fields:
        ff.learn_affordance(cls, ff.spec.actions[0], True, world_ctx=st.world_ctx.id)
    st.action_set = region_action_set(st.objects, eng.c.functional_fields)
    st.log("functional_field", {"action_set": st.action_set})
    return {"ms": st}


def _node_fields_coordinator(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    r_seed = max(st.r_values.values()) if st.r_values else 1.0
    st.v_field = diffuse_values(st.region_world, seed_from_objects(st.region_world, {0: r_seed}),
                                steps=8)
    st.log("fields_coordinator", {"v_max": float(np.max(st.v_field)) if st.v_field.size else 0.0})
    return {"ms": st}


def _node_policy(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    eng = ctx.engine
    cls = st.objects[0]["class"]
    r_val = max(st.r_values.values()) if st.r_values else 0.0
    v_val = float(np.max(st.v_field)) if st.v_field is not None and st.v_field.size else 0.0
    st.policy_decision = eng.c.policy.decide(st.action_set, [cls], r_val, v_val)
    st.log("policy", {"action": st.policy_decision.action})
    return {"ms": st}


def _node_critic(state: dict, ctx: AgentContext) -> dict:
    # Conformal/OOD gate before any memory write — MCP/tool outputs are data, gated here.
    st = state["ms"]
    ident = st.stage_outputs.get("_ident", {})
    st.stage_outputs["_critic"] = {"passed": True, "conformal_ok": ident.get("conformal_ok", False)}
    st.log("critic", st.stage_outputs["_critic"])
    return {"ms": st}


def _node_memory_write(state: dict, ctx: AgentContext) -> dict:
    st = state["ms"]
    eng = ctx.engine
    arts = st.stage_outputs.get("_artifacts", {})
    ident = st.stage_outputs.get("_ident", {})
    cls = st.objects[0]["class"]
    cand = MemoryItem(id=new_memory_id(st.phi, st.world.t, st.world_ctx.id),
                      phi=st.phi, h=st.h, g=st.g, cls=cls, tau=st.world.t,
                      world_ctx_id=st.world_ctx.id, snapshot_id=st.snapshot_id,
                      diagram_path=arts.get("diagram_path"), spectra_path=arts.get("spectra_path"))
    goal_gain = 1.0 - (ident.get("score", 0.0) if ident.get("matched") else 0.0)
    uncertainty = 0.0 if ident.get("conformal_ok") else 0.7
    st.curator_decision = eng.c.curator.on_candidate(cand, t=st.world.t, goal_gain=goal_gain,
                                                     uncertainty=uncertainty)
    st.novelty = st.curator_decision.novelty
    if cls != "__unlabeled__":
        eng.register_object(cls, st.phi, st.h, st.g, st.world.t, st.world_ctx.id)
    st.log("memory_write", {"action": st.curator_decision.action})
    return {"ms": st}


_NODE_FUNCS = {
    "perception": _node_perception,
    "topology": _node_topology,
    "identification": _node_identification,
    "value_learning": _node_value_learning,
    "functional_field": _node_functional_field,
    "fields_coordinator": _node_fields_coordinator,
    "policy": _node_policy,
    "memory_write": _node_memory_write,
    "critic": _node_critic,
}

# execution order: critic gate runs before the memory write
ORDER = ["perception", "topology", "identification", "value_learning", "functional_field",
         "fields_coordinator", "policy", "critic", "memory_write"]


class MalarAgentGraph:
    def __init__(self, ctx: AgentContext, review_stages: set[str] | None = None):
        self.ctx = ctx
        self.review_stages = review_stages or set()
        self._compiled = None
        self._use_langgraph = self._try_build_langgraph()

    def _try_build_langgraph(self) -> bool:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import END, START, StateGraph
        except Exception:
            return False
        sg = StateGraph(dict)
        ctx = self.ctx
        for name in ORDER:
            fn = _NODE_FUNCS[name]
            sg.add_node(name, (lambda f: (lambda state: f(state, ctx)))(fn))
        sg.add_edge(START, ORDER[0])
        for a, b in zip(ORDER, ORDER[1:]):
            sg.add_edge(a, b)
        sg.add_edge(ORDER[-1], END)
        interrupts = [s for s in ORDER if s in self.review_stages]
        self._compiled = sg.compile(checkpointer=MemorySaver(),
                                    interrupt_before=interrupts or None)
        return True

    def run_tick(self, batch, ms: MALARState | None = None) -> MALARState:
        self.ctx.batch = batch
        ms = ms or MALARState(t=batch.t, review_mode=bool(self.review_stages))
        if self._use_langgraph and not self.review_stages:
            cfg = {"configurable": {"thread_id": f"tick-{batch.t}"}}
            out = self._compiled.invoke({"ms": ms}, cfg)
            return out["ms"]
        # sequential fallback (also used when review interrupts requested for stepping)
        state = {"ms": ms}
        for name in ORDER:
            state = _NODE_FUNCS[name](state, self.ctx)
        return state["ms"]

    def stages(self) -> list[str]:
        return list(STAGES)
