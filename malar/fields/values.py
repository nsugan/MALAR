"""Value fields V(S, t) + Laplacian diffusion.

R-source values (from objective fields, placed on the nodes where objects were
recognised) seed V; V then diffuses over the graph Laplacian L. V marks *where
attention matters*. Diffusion is explicit-Euler heat flow with the theorem guard
dt < 2 / lambda_max(L) enforced as an assert.
"""
from __future__ import annotations

import numpy as np

from malar.world.graph import WorldGraph
from malar.world.laplacian import graph_laplacian, lambda_max, safe_diffusion_dt


def diffuse_values(world: WorldGraph, seed: np.ndarray, steps: int = 10,
                   alpha: float = 1.0, dt_safety: float = 0.9) -> np.ndarray:
    """Diffuse a per-node seed vector over L. Returns V(S,t) per node.

    Guard: dt < 2 / lambda_max(L). Raises AssertionError if violated.
    """
    seed = np.asarray(seed, dtype=float).ravel()
    n = world.n_nodes
    if n == 0:
        return np.zeros(0)
    if seed.shape[0] != n:
        raise ValueError(f"seed len {seed.shape[0]} != n_nodes {n}")
    L = graph_laplacian(world.adjacency)
    lm = lambda_max(L)
    dt = safe_diffusion_dt(L, safety=dt_safety)
    if lm > 0:
        assert dt < 2.0 / lm + 1e-12, "diffusion dt violates dt < 2/lambda_max(L)"
    V = seed.copy()
    for _ in range(max(0, steps)):
        V = V - alpha * dt * (L @ V)
    return V


def seed_from_objects(world: WorldGraph, object_nodes: dict[int, float]) -> np.ndarray:
    """Place R-source scalar values on the nodes where objects were recognised."""
    seed = np.zeros(world.n_nodes)
    for idx, val in object_nodes.items():
        if 0 <= idx < world.n_nodes:
            seed[idx] = float(val)
    return seed


class ValueField:
    """Holds the diffused V over a region-world; queryable per node."""

    def __init__(self, world: WorldGraph, steps: int = 10, dt_safety: float = 0.9):
        self.world = world
        self.steps = steps
        self.dt_safety = dt_safety
        self.V: np.ndarray | None = None

    def compute(self, object_nodes: dict[int, float]) -> np.ndarray:
        seed = seed_from_objects(self.world, object_nodes)
        self.V = diffuse_values(self.world, seed, steps=self.steps, dt_safety=self.dt_safety)
        return self.V

    def attention(self) -> np.ndarray:
        return self.V if self.V is not None else np.zeros(self.world.n_nodes)
