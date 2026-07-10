"""Synthetic generator: deterministic, structured point clouds with known topology
(blobs, circles, spirals) so encoder/loop tests have ground truth. Also doubles as
a stand-in domain for campaign smoke tests.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from malar.world.adapters.base import WorldAdapter, WorldBatch
from malar.world.graph import WorldGraph, knn_graph_from_points


def make_circle(n: int, r: float = 1.0, noise: float = 0.02, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)


def make_blob(n: int, center=(0.0, 0.0), spread: float = 0.15, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    return rng.normal(loc=center, scale=spread, size=(n, 2))


def synthetic_class_cloud(cls: str, n_dims: int = 128, n_samples: int = 16,
                          noise: float = 0.02) -> np.ndarray:
    """Deterministic, class-separable feature cloud for a GENERIC class label.

    Each class gets a stable random centroid (seeded by the class name), plus small
    per-sample noise, L2-normalised to match the encoder corpus. Domain-agnostic —
    no assumptions about spectra/pathogens/instruments. Used for the built-in synthetic
    demo path when no data folder is configured. Returns (n_samples, n_dims).
    """
    seed = abs(hash(("class", cls))) % (2**32)
    rng = np.random.default_rng(seed)
    centroid = rng.normal(0.0, 1.0, n_dims)
    centroid = np.clip(centroid, 0, None)              # non-negative, signal-like
    nrm = np.linalg.norm(centroid)
    if nrm > 0:
        centroid = centroid / nrm
    jitter = np.random.default_rng(seed + 1).normal(0.0, noise, (n_samples, n_dims))
    cloud = np.clip(centroid[None, :] + jitter, 0, None)
    row_norms = np.linalg.norm(cloud, axis=1, keepdims=True)
    return cloud / np.where(row_norms > 0, row_norms, 1.0)


class SyntheticAdapter(WorldAdapter):
    name = "synthetic"

    def __init__(self, n_ticks: int = 8, points_per_tick: int = 40, k: int = 6, seed: int = 0):
        self.n_ticks = n_ticks
        self.points_per_tick = points_per_tick
        self.k = k
        self.seed = seed

    def stream(self) -> Iterator[WorldBatch]:
        rng = np.random.default_rng(self.seed)
        shapes = ["blob", "circle"]
        for t in range(self.n_ticks):
            shape = shapes[t % len(shapes)]
            n = self.points_per_tick
            if shape == "circle":
                pts = make_circle(n, r=1.0, noise=0.03, rng=rng)
                label = "circle"
            else:
                pts = make_blob(n, center=(0.0, 0.0), spread=0.2, rng=rng)
                label = "blob"
            ids = [f"t{t}_n{i}" for i in range(n)]
            yield WorldBatch(
                t=t,
                points=pts,
                node_ids=ids,
                labels=[label] * n,
                raw={"shape": shape},
                source="synthetic",
            )

    def to_world(self, batch: WorldBatch) -> WorldGraph:
        A = knn_graph_from_points(batch.points, k=self.k)
        return WorldGraph(
            node_ids=batch.node_ids,
            features=batch.points,
            adjacency=A,
            t=batch.t,
            meta={"source": batch.source, "shape": batch.raw.get("shape")},
        )
