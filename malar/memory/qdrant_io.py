"""Qdrant I/O: one point per memory, named vectors phi / hyper / graph_emb + payload.

Uses the real qdrant-client. In production it points at QDRANT_URL (the Docker
service). For tests / offline runs pass url=":memory:" to use the client's embedded
mode — same client, same named-vector API, no server.

Collections are addressed through an **alias** so the Curator can blue-green
re-index behind it (never querying across encoder versions).
"""
from __future__ import annotations

import uuid

import numpy as np

from malar.memory.schema import MemoryItem


def _point_uuid(mem_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, mem_id))


class QdrantMemory:
    def __init__(self, url: str = ":memory:", collection: str = "malar_memory",
                 dims: dict | None = None):
        from qdrant_client import QdrantClient

        if url in (":memory:", "memory", ":mem:"):
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url)
        self.alias = collection
        self.dims = dims or {"phi": 432, "hyper": 16, "graph_emb": 48}
        self._physical = f"{collection}__v1"

    def ensure_collection(self, dims: dict | None = None, physical: str | None = None) -> str:
        from qdrant_client import models

        if dims:
            self.dims = dims
        physical = physical or self._physical
        self._physical = physical
        existing = {c.name for c in self.client.get_collections().collections}
        if physical not in existing:
            self.client.create_collection(
                physical,
                vectors_config={
                    name: models.VectorParams(size=int(size), distance=models.Distance.COSINE)
                    for name, size in self.dims.items()
                },
            )
        # point the alias at the physical collection (blue-green target)
        try:
            self.client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(collection_name=physical, alias_name=self.alias)
                    )
                ]
            )
        except Exception:
            pass  # alias may already exist
        return physical

    def upsert(self, item: MemoryItem) -> None:
        from qdrant_client import models

        self.ensure_collection()
        vec = {
            "phi": _fit(item.phi, self.dims["phi"]),
            "hyper": _fit(item.h, self.dims["hyper"]),
            "graph_emb": _fit(item.g, self.dims["graph_emb"]),
        }
        self.client.upsert(
            self._physical,
            points=[models.PointStruct(id=_point_uuid(item.id), vector=vec, payload=item.payload())],
        )

    def prefetch(self, named_query: dict, limit: int = 10) -> dict:
        """Per-named-vector prefetch. Returns {name: [(mem_id, score, payload)]}."""
        self.ensure_collection()
        out: dict[str, list] = {}
        for name, q in named_query.items():
            res = self.client.query_points(
                self.alias, query=_fit(q, self.dims[name]), using=name, limit=limit,
                with_payload=True,
            ).points
            out[name] = [(p.payload.get("neo4j_id"), float(p.score), p.payload) for p in res]
        return out

    def fetch(self, mem_id: str) -> dict | None:
        """Read one stored point back by memory id as {phi, hyper, graph_emb, payload}.

        Qdrant is otherwise write/query-only for the store layer; this gives a
        full-fidelity read path so the Curator can hydrate the in-process vector cache
        after a restart instead of losing merges/reinforcements. (V4 fix ❸)
        """
        self.ensure_collection()
        try:
            recs = self.client.retrieve(self._physical, ids=[_point_uuid(mem_id)],
                                        with_vectors=True, with_payload=True)
        except Exception:
            return None
        if not recs:
            return None
        r = recs[0]
        vec = r.vector or {}
        return {"phi": vec.get("phi"), "hyper": vec.get("hyper"),
                "graph_emb": vec.get("graph_emb"), "payload": r.payload or {}}

    def delete(self, mem_id: str) -> None:
        from qdrant_client import models

        self.client.delete(self._physical,
                           points_selector=models.PointIdsList(points=[_point_uuid(mem_id)]))

    def count(self) -> int:
        try:
            return int(self.client.count(self._physical).count)
        except Exception:
            return 0

    def scroll_all(self, with_vectors: bool = True, batch: int = 256) -> list[dict]:
        """Read every stored point as {id, class, phi, hyper, graph_emb} dicts.

        Used by the probabilistic inference layer to fit per-class likelihoods over the
        WHOLE training corpus (not just the EMA prototypes). Works on both the real
        server and the embedded ':memory:' client.
        """
        self.ensure_collection()
        out: list[dict] = []
        offset = None
        while True:
            try:
                points, offset = self.client.scroll(
                    self._physical, limit=batch, with_payload=True,
                    with_vectors=with_vectors, offset=offset)
            except Exception:
                break
            for p in points:
                vec = p.vector or {}
                out.append({
                    "id": (p.payload or {}).get("neo4j_id"),
                    "class": (p.payload or {}).get("class"),
                    "phi": vec.get("phi"), "hyper": vec.get("hyper"),
                    "graph_emb": vec.get("graph_emb"),
                })
            if offset is None or not points:
                break
        return out


def _fit(v: np.ndarray, dim: int) -> list[float]:
    a = np.asarray(v, dtype=float).ravel()
    if a.size >= dim:
        return a[:dim].tolist()
    return np.pad(a, (0, dim - a.size)).tolist()
