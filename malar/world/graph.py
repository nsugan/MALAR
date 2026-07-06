"""World object W(t) = (I, E, C, L).

I = node identities, E = edges, C = node content/features, L = the graph Laplacian
(computed lazily from the adjacency). A thin, deterministic wrapper around an
adjacency matrix + a node-feature matrix. Pure Python / numpy — no LLM here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class WorldGraph:
    """A timestamped world graph W(t).

    Attributes
    ----------
    node_ids : list[str]      identities I
    features : np.ndarray     content C, shape (n_nodes, n_features)
    adjacency : np.ndarray    weighted symmetric adjacency A (edges E), shape (n, n)
    t : int                   timestep
    meta : dict               free-form provenance
    """

    node_ids: list[str]
    features: np.ndarray
    adjacency: np.ndarray
    t: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=float)
        self.adjacency = np.asarray(self.adjacency, dtype=float)
        n = len(self.node_ids)
        if self.features.shape[0] != n:
            raise ValueError(f"features rows {self.features.shape[0]} != n_nodes {n}")
        if self.adjacency.shape != (n, n):
            raise ValueError(f"adjacency {self.adjacency.shape} != ({n},{n})")
        if not np.allclose(self.adjacency, self.adjacency.T, atol=1e-9):
            # symmetrize defensively (undirected world graph)
            self.adjacency = 0.5 * (self.adjacency + self.adjacency.T)

    # -- basic accessors -------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    def degree(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)

    def edges(self) -> list[tuple[int, int, float]]:
        out: list[tuple[int, int, float]] = []
        n = self.n_nodes
        for i in range(n):
            for j in range(i + 1, n):
                w = self.adjacency[i, j]
                if w != 0.0:
                    out.append((i, j, float(w)))
        return out

    def subgraph(self, idx: Iterable[int]) -> "WorldGraph":
        idx = list(idx)
        ids = [self.node_ids[i] for i in idx]
        feat = self.features[idx]
        adj = self.adjacency[np.ix_(idx, idx)]
        return WorldGraph(node_ids=ids, features=feat, adjacency=adj, t=self.t, meta=dict(self.meta))

    def content_hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.adjacency).tobytes())
        h.update(np.ascontiguousarray(self.features).tobytes())
        h.update(("|".join(self.node_ids)).encode())
        h.update(str(self.t).encode())
        return h.hexdigest()[:16]

    # -- mutation (perception updates) -----------------------------------
    def update_features(self, new_features: np.ndarray) -> None:
        new_features = np.asarray(new_features, dtype=float)
        if new_features.shape[0] != self.n_nodes:
            raise ValueError("feature update row mismatch")
        self.features = new_features


def knn_graph_from_points(points: np.ndarray, k: int = 5, sigma: float | None = None) -> np.ndarray:
    """Build a symmetric weighted kNN adjacency from a point cloud (Gaussian weights)."""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    diff = points[:, None, :] - points[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    if sigma is None:
        # median heuristic on off-diagonal distances
        offdiag = d2[~np.eye(n, dtype=bool)]
        med = np.median(offdiag) if offdiag.size else 1.0
        sigma = float(np.sqrt(med) + 1e-9)
    A = np.zeros((n, n))
    k = min(k, n - 1) if n > 1 else 0
    for i in range(n):
        order = np.argsort(d2[i])
        for j in order[1 : k + 1]:
            w = float(np.exp(-d2[i, j] / (2.0 * sigma * sigma)))
            A[i, j] = max(A[i, j], w)
            A[j, i] = A[i, j]
    return A
