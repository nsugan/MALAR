import numpy as np
from malar.memory.qdrant_io import QdrantMemory
from malar.memory.neo4j_io import make_memory_graph
from malar.memory.schema import MemoryItem, new_memory_id
from malar.memory.store import MemoryStore
from malar.memory.retrieval import FusedRetriever

DIMS = {"phi": 432, "hyper": 16, "graph_emb": 48}


def _store():
    q = QdrantMemory(url=":memory:", dims=DIMS)
    g = make_memory_graph(prefer_real=False)
    return MemoryStore(q, g), q, g


def test_store_roundtrip_and_fused_rank():
    store, q, g = _store()
    rng = np.random.default_rng(0); items = []
    for i, cls in enumerate(["a", "b", "c"]):
        it = MemoryItem(id=new_memory_id(rng.normal(size=432), i, cls),
                        phi=rng.normal(size=432), h=rng.normal(size=16), g=rng.normal(size=48),
                        cls=cls, tau=i, world_ctx_id=f"ctx{i}", snapshot_id="snapA" if i < 2 else "snapB")
        g.upsert_world_context({"id": it.world_ctx_id, "snapshot_id": it.snapshot_id,
                                "region_id": "r", "t": i, "source": "x"})
        store.insert(it); items.append(it)
    assert g.get_memory(items[0].id)["class"] == "a"
    ret = FusedRetriever(q, g)
    hits = ret.retrieve(items[1].phi, items[1].h, items[1].g, t=5, k=3)
    assert hits[0].mem_id == items[1].id
    assert len(g.memories_in_snapshot("snapA")) == 2


def test_contractive_merge_and_ema():
    store, q, g = _store()
    rng = np.random.default_rng(1)
    it = MemoryItem(id="m", phi=rng.normal(size=432), h=rng.normal(size=16), g=rng.normal(size=48),
                    cls="a", tau=0, world_ctx_id="c0")
    store.insert(it)
    w = store.reinforce("m", gain=2.0)
    assert w > 1.0
    cand = MemoryItem(id="cand", phi=it.phi + 1, h=it.h, g=it.g, world_ctx_id="c1", tau=5)
    merged = store.merge("m", cand, gain=1.0, t=5)
    assert merged.tau == 5
    assert "c1" in g.neighbors("m", "DERIVED_FROM")


def test_cold_cache_hydrates_from_qdrant_for_merge_and_reinforce():
    """After a cache miss (e.g. restart) reinforce/merge must rehydrate real vectors
    from Qdrant, not no-op or lose vectors. (V4 fix ❸)"""
    store, q, g = _store()
    rng = np.random.default_rng(7)
    it = MemoryItem(id="m", phi=rng.normal(size=432), h=rng.normal(size=16),
                    g=rng.normal(size=48), cls="a", tau=0, world_ctx_id="c0")
    store.insert(it)
    # simulate a fresh process: the durable stores keep the point, the RAM cache is empty
    store._cache.clear()
    hydrated = store.get_item("m")
    assert hydrated is not None and hydrated.phi.size == 432 and hydrated.cls == "a"

    store._cache.clear()
    w = store.reinforce("m", gain=2.0)
    assert w > 0.0  # not the 0.0 dead-path

    store._cache.clear()
    cand = MemoryItem(id="cand", phi=it.phi + 1, h=it.h, g=it.g, world_ctx_id="c1", tau=9)
    merged = store.merge("m", cand, gain=1.0, t=9)
    assert merged is not None and merged.tau == 9  # merge no longer silently drops


def test_value_and_affordance_edges_land_in_graph():
    """Learned values/affordances mirror into the graph as HAS_VALUE / AFFORDS. (V4 fix ❷)"""
    _, _, g = _store()
    g.upsert_value("a", "sensitivity", 0.9, "human", 2, world_ctx="ctx0")
    g.upsert_affordance("a", "triage", "confirm_rtpcr", "data", 1, world_ctx="ctx0")
    rels = {e[0] for e in g.edges}
    assert "HAS_VALUE" in rels and "AFFORDS" in rels
    assert "obj::a" in g.objects and "objective::sensitivity" in g.objectives
