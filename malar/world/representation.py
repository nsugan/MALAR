"""F_rep: node-state representation update u_i(t).

Deterministic message-passing-style smoothing of node features over the graph,
parameterised but not learned here (the learned encoders live in malar/encoders).
This produces the per-node representation the samplers and encoders consume.
"""
from __future__ import annotations

import numpy as np

from malar.world.graph import WorldGraph
from malar.world.laplacian import graph_laplacian, safe_diffusion_dt


def f_rep(world: WorldGraph, steps: int = 1, alpha: float = 0.5, dt_safety: float = 0.9) -> np.ndarray:
    """Smoothed node representation u_i(t).

    One step: u <- u - alpha * dt * L u   (heat diffusion of features),
    with dt chosen under the stability guard dt < 2/lambda_max(L).
    """
    U = np.array(world.features, dtype=float, copy=True)
    if world.n_nodes == 0:
        return U
    L = graph_laplacian(world.adjacency)
    dt = safe_diffusion_dt(L, safety=dt_safety)
    for _ in range(max(0, steps)):
        U = U - alpha * dt * (L @ U)
    return U


def node_energy(world: WorldGraph) -> np.ndarray:
    """Dirichlet energy contribution per node (used for importance / sampling)."""
    L = graph_laplacian(world.adjacency)
    F = world.features
    # diagonal of F^T L F per node ~ local smoothness; here per-node quadratic form
    LF = L @ F
    return np.einsum("ij,ij->i", F, LF)
