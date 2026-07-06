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
