"""Topology encoder phi_topo(S, t).

Persistence homology of a region via ripser, turned into a fixed-length persistence
image (persim) for phi. The raw diagram is stored to data/artifacts so phi can be
re-embedded after encoder updates and exact Wasserstein distances stay available.

Deterministic tool — no LLM. Returns a feature summary (H0/H1/H2 counts +
persistence ranges) that agents may send to the LLM (never the raw vectors).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from malar.world.graph import WorldGraph


@dataclass
class TopologyResult:
    phi: np.ndarray
    diagrams: list[np.ndarray]
    summary: dict
    artifact_path: str | None = None
    meta: dict = field(default_factory=dict)


def _graph_distance_matrix(world: WorldGraph) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    A = world.adjacency.copy()
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    with np.errstate(divide="ignore"):
        cost = np.where(A > 0, 1.0 / (A + 1e-9), 0.0)
    cost[A <= 0] = 0.0
    g = sp.csr_matrix(cost)
    D = shortest_path(g, method="D", directed=False)
    finite = D[np.isfinite(D)]
    cap = (finite.max() * 1.5) if finite.size else 1.0
    D[~np.isfinite(D)] = cap
    return D


class TopologyEncoder:
    version = "topo_v1"

    def __init__(self, pixel_size: float = 0.1, n_pixels: int = 12, max_dim: int = 2,
                 artifacts_dir: str | Path | None = None, use_point_cloud: bool = True,
                 max_range: float = 2.0):
        self.pixel_size = pixel_size
        self.n_pixels = n_pixels
        self.max_dim = max_dim
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.use_point_cloud = use_point_cloud
        # Fixed birth/persistence grid so phi is ALWAYS n_pixels^2 per homology
        # dimension and comparable across regions (degenerate single-point diagrams
        # would otherwise yield an empty image).
        self.max_range = max_range

    def _compute_diagrams(self, world: WorldGraph) -> list[np.ndarray]:
        from ripser import ripser

        if world.n_nodes < 2:
            return [np.zeros((0, 2)) for _ in range(self.max_dim + 1)]
        if self.use_point_cloud and world.n_features >= 2:
            res = ripser(np.asarray(world.features, dtype=float), maxdim=self.max_dim)
        else:
            D = _graph_distance_matrix(world)
            res = ripser(D, distance_matrix=True, maxdim=self.max_dim)
        return res["dgms"]

    def _persistence_image(self, diagrams: list[np.ndarray]) -> np.ndarray:
        from persim import PersistenceImager

        px = self.max_range / self.n_pixels
        vecs = []
        for dim in range(self.max_dim + 1):
            dgm = diagrams[dim] if dim < len(diagrams) else np.zeros((0, 2))
            dgm = np.asarray(dgm, dtype=float)
            dgm = dgm[np.isfinite(dgm).all(axis=1)] if dgm.size else dgm
            target = self.n_pixels * self.n_pixels
            if dgm.shape[0] == 0:
                vecs.append(np.zeros(target))
                continue
            pimgr = PersistenceImager(
                birth_range=(0.0, self.max_range),
                pers_range=(0.0, self.max_range),
                pixel_size=px,
            )
            img = np.asarray(pimgr.transform([dgm], skew=True)[0], dtype=float).ravel()
            v = np.zeros(target)
            v[: min(target, len(img))] = img[: target]
            vecs.append(v)
        return np.concatenate(vecs)

    def _summary(self, diagrams: list[np.ndarray]) -> dict:
        out = {}
        for dim in range(min(3, len(diagrams))):
            dgm = np.asarray(diagrams[dim], dtype=float)
            dgm = dgm[np.isfinite(dgm).all(axis=1)] if dgm.size else dgm
            if dgm.shape[0] == 0:
                out[f"H{dim}"] = {"count": 0, "max_persistence": 0.0, "total_persistence": 0.0}
                continue
            pers = dgm[:, 1] - dgm[:, 0]
            out[f"H{dim}"] = {
                "count": int(dgm.shape[0]),
                "max_persistence": float(pers.max()),
                "total_persistence": float(pers.sum()),
                "birth_range": [float(dgm[:, 0].min()), float(dgm[:, 0].max())],
            }
        return out

    def _store_diagram(self, diagrams: list[np.ndarray], key: str) -> str | None:
        if self.artifacts_dir is None:
            return None
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"phi_{key}.npz"
        np.savez(path, **{f"H{d}": np.asarray(diagrams[d], dtype=float) for d in range(len(diagrams))})
        return str(path)

    def encode(self, world: WorldGraph, key: str | None = None) -> TopologyResult:
        diagrams = self._compute_diagrams(world)
        phi = self._persistence_image(diagrams)
        summary = self._summary(diagrams)
        if key is None:
            key = hashlib.sha256(world.content_hash().encode()).hexdigest()[:12]
        artifact_path = self._store_diagram(diagrams, key)
        return TopologyResult(phi=phi, diagrams=diagrams, summary=summary,
                              artifact_path=artifact_path,
                              meta={"encoder_version": self.version})


def load_diagram(path: str | Path) -> list[np.ndarray]:
    z = np.load(path)
    return [z[k] for k in sorted(z.files)]
