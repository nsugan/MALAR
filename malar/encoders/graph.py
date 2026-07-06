"""Graph encoder graph_emb(S, t) and world_emb(S, t).

Produces a fixed-length structural embedding of a region (graph_emb) and of the
whole world (world_emb, used to anchor WorldContext similarity, validity-gating and
drift detection).

The embedding fuses permutation-invariant structural descriptors:
  * normalised-Laplacian spectrum (sorted, fixed length),
  * Weisfeiler-Lehman-style iterated neighbour feature aggregation (mean/std),
  * degree-distribution moments.

This is the deterministic structural encoder. The architecture is a drop-in for a
learned PyTorch Geometric GNN (same encode/world_embed contract, same
encoder_version stamping); swap the body to train one without touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from malar.world.graph import WorldGraph
from malar.world.laplacian import graph_laplacian


@dataclass
class GraphResult:
    graph_emb: np.ndarray
    summary: dict
    meta: dict = field(default_factory=dict)


class GraphEncoder:
    version = "graph_v1"

    def __init__(self, n_spectrum: int = 16, wl_iters: int = 2, emb_dim: int = 48):
        self.n_spectrum = n_spectrum
        self.wl_iters = wl_iters
        self.emb_dim = emb_dim

    def _spectrum(self, world: WorldGraph) -> np.ndarray:
        if world.n_nodes == 0:
            return np.zeros(self.n_spectrum)
        L = graph_laplacian(world.adjacency, normalized=True)
        w = np.linalg.eigvalsh(0.5 * (L + L.T))
        w = np.sort(w)
        out = np.zeros(self.n_spectrum)
        m = min(self.n_spectrum, len(w))
        out[:m] = w[:m]
        return out

    def _wl_features(self, world: WorldGraph) -> np.ndarray:
        n = world.n_nodes
        if n == 0:
            return np.zeros(4)
        X = np.asarray(world.features, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        A = world.adjacency
        deg = A.sum(axis=1, keepdims=True)
        deg_safe = np.where(deg > 0, deg, 1.0)
        H = X.copy()
        moments = []
        for _ in range(self.wl_iters):
            H = (A @ H) / deg_safe  # mean-aggregate neighbours
            moments.append(H.mean(axis=0))
            moments.append(H.std(axis=0))
        feat = np.concatenate(moments) if moments else np.zeros(1)
        return feat

    def _degree_moments(self, world: WorldGraph) -> np.ndarray:
        if world.n_nodes == 0:
            return np.zeros(4)
        d = world.degree()
        return np.array([d.mean(), d.std(), d.min(), d.max()], dtype=float)

    def _assemble(self, world: WorldGraph) -> np.ndarray:
        parts = [self._spectrum(world), self._wl_features(world), self._degree_moments(world)]
        v = np.concatenate([np.atleast_1d(p).astype(float) for p in parts])
        # project/pad to fixed emb_dim deterministically
        out = np.zeros(self.emb_dim)
        out[: min(self.emb_dim, len(v))] = v[: self.emb_dim]
        # tail wraps remaining mass so longer feature vectors aren't simply truncated
        if len(v) > self.emb_dim:
            extra = v[self.emb_dim :]
            for i, val in enumerate(extra):
                out[i % self.emb_dim] += val
        n = np.linalg.norm(out)
        return out / n if n > 0 else out

    def encode(self, world: WorldGraph) -> GraphResult:
        emb = self._assemble(world)
        summary = {
            "n_nodes": world.n_nodes,
            "n_edges": len(world.edges()),
            "mean_degree": float(world.degree().mean()) if world.n_nodes else 0.0,
            "spectral_gap": self._spectral_gap(world),
        }
        return GraphResult(graph_emb=emb, summary=summary, meta={"encoder_version": self.version})

    def _spectral_gap(self, world: WorldGraph) -> float:
        sp = self._spectrum(world)
        nz = sp[sp > 1e-9]
        return float(nz[0]) if nz.size else 0.0

    def world_embed(self, world: WorldGraph) -> np.ndarray:
        """world_emb(S,t): same descriptor over the full world; anchors context similarity."""
        return self._assemble(world)
