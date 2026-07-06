import numpy as np
from malar.memory.qdrant_io import QdrantMemory
from malar.memory.neo4j_io import make_memory_graph
from malar.memory.schema import MemoryItem, new_memory_id
from malar.memory.store import MemoryStore
from malar.memory.retrieval import FusedRetriever
from malar.memory.curator import MemoryCurator, CuratorConfig
from malar.memory.propagate import CorrectionPropagator
from malar.worldview.worldview import WorldViewBuilder
from malar.validation.conformal import ConformalGate

DIMS = {"phi": 432, "hyper": 16, "graph_emb": 48}


def _cur(budget=5):
    q = QdrantMemory(url=":memory:", dims=DIMS); g = make_memory_graph(prefer_real=False)
    store = MemoryStore(q, g); ret = FusedRetriever(q, g)
    return MemoryCurator(store, ret, CuratorConfig(budget=budget)), store, g


def _cand(seed, cls, t):
    r = np.random.default_rng(seed)
    return MemoryItem(id=new_memory_id(r.normal(size=432), t, str(seed)),
                      phi=r.normal(size=432), h=r.normal(size=16), g=r.normal(size=48),
                      cls=cls, tau=t, world_ctx_id=f"ctx{seed}")


def test_stationary_halts_inserts():
    cur, store, g = _cur()
    base = _cand(0, "c0", 0)
    cur.on_candidate(base, t=0, goal_gain=0.2)
    acts = []
    for t in range(5):
        dup = MemoryItem(id=f"d{t}", phi=base.phi + np.random.default_rng(t).normal(0, 1e-4, 432),
                         h=base.h, g=base.g, cls="c0", tau=t, world_ctx_id="ctx0")
        acts.append(cur.on_candidate(dup, t=t, goal_gain=0.0).action)
    assert "insert" not in acts


def test_budget_evicts():
    cur, store, g = _cur(budget=4)
    for i in range(10):
        cur.on_candidate(_cand(i, f"c{i}", i), t=i + 20, goal_gain=0.9, uncertainty=0.9)
    assert len(g.all_memories()) <= 4
    assert any(a["op"] == "evict+insert" for a in cur.audit)


def test_propagation_rollback():
    cur, store, g = _cur()
    rng = np.random.default_rng(3); phi0 = rng.normal(size=432)
    a = MemoryItem(id="mem_a", phi=phi0, h=rng.normal(size=16), g=rng.normal(size=48),
                   cls="old", tau=1, world_ctx_id="cP")
    b = MemoryItem(id="mem_b", phi=phi0 + rng.normal(0, 1e-3, 432), h=rng.normal(size=16),
                   g=rng.normal(size=48), cls="old", tau=2, world_ctx_id="cP")
    store.insert(a); store.insert(b)
    gate = ConformalGate(floor=0.5); gate.calibrate([0.95, 0.9, 0.92])
    prop = CorrectionPropagator(store, g, gate, hi_conf=0.9, sim_radius=0.8)
    res = prop.propagate("mem_a", "corrected")
    assert "mem_b" in res.relabeled
    prop.rollback(res)
    assert store.get_item("mem_a").cls == "old"


def test_worldview_existing_ids_only():
    cur, store, g = _cur()
    store.insert(_cand(1, "x", 0))
    mid = g.all_memories()[0]["id"]
    wv = WorldViewBuilder(g).build(delta_ids=[mid, "INVENTED"])
    assert "INVENTED" not in wv.novel_ids and mid in wv.novel_ids
