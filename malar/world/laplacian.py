"""Graph and Hodge Laplacians.

L  = D - A                 (graph / combinatorial Laplacian, on 0-cochains = nodes)
L0 = B1 B1^T               (Hodge 0-Laplacian; equals the graph Laplacian for unit weights)
L1 = B1^T B1 + B2 B2^T     (Hodge 1-Laplacian on edges)
L2 = B2^T B2               (Hodge 2-Laplacian on triangles, down-part)

B1 = node-edge incidence (boundary map d0)
B2 = edge-triangle incidence (boundary map d1)

All deterministic numpy. Provides lambda_max for the diffusion theorem guard
(dt < 2 / lambda_max(L)).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from malar.world.graph import WorldGraph


def graph_laplacian(adjacency: np.ndarray, normalized: bool = False) -> np.ndarray:
    A = np.asarray(adjacency, dtype=float)
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A
    if normalized:
        with np.errstate(divide="ignore"):
            dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        Dinv = np.diag(dinv)
        L = np.eye(A.shape[0]) - Dinv @ A @ Dinv
        L = 0.5 * (L + L.T)
    return L


def lambda_max(L: np.ndarray) -> float:
    if L.shape[0] == 0:
        return 0.0
    w = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(np.max(w))


def safe_diffusion_dt(L: np.ndarray, safety: float = 0.9) -> float:
    """Largest stable explicit-Euler diffusion step: dt < 2 / lambda_max(L)."""
    lm = lambda_max(L)
    if lm <= 0:
        return float(safety)  # disconnected / empty -> any step is stable
    return float(safety * 2.0 / lm)


def _edge_list(adjacency: np.ndarray, tol: float = 1e-12) -> list[tuple[int, int]]:
    n = adjacency.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(adjacency[i, j]) > tol:
                edges.append((i, j))
    return edges


def _triangles(adjacency: np.ndarray, edges: list[tuple[int, int]], tol: float = 1e-12) -> list[tuple[int, int, int]]:
    eset = {e for e in edges}
    tris = []
    n = adjacency.shape[0]
    for i, j, k in combinations(range(n), 3):
        if (i, j) in eset and (j, k) in eset and (i, k) in eset:
            tris.append((i, j, k))
    return tris


def incidence_b1(n_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Node-edge boundary map B1, shape (n_nodes, n_edges). Orientation i<j: -1 at i, +1 at j."""
    B1 = np.zeros((n_nodes, len(edges)))
    for e, (i, j) in enumerate(edges):
        B1[i, e] = -1.0
        B1[j, e] = +1.0
    return B1


def incidence_b2(edges: list[tuple[int, int]], triangles: list[tuple[int, int, int]]) -> np.ndarray:
    """Edge-triangle boundary map B2, shape (n_edges, n_triangles)."""
    edge_index = {e: idx for idx, e in enumerate(edges)}
    B2 = np.zeros((len(edges), len(triangles)))
    for t, (i, j, k) in enumerate(triangles):
        # oriented boundary of triangle (i,j,k): (j,k) - (i,k) + (i,j)
        for (a, b), sign in [((j, k), +1.0), ((i, k), -1.0), ((i, j), +1.0)]:
            key = (a, b) if a < b else (b, a)
            s = sign if a < b else -sign
            if key in edge_index:
                B2[edge_index[key], t] = s
    return B2


def hodge_laplacians(world: WorldGraph) -> dict[str, np.ndarray]:
    """Return {'L': graph L, 'L0', 'L1', 'L2', 'B1', 'B2'} for a WorldGraph."""
    A = world.adjacency
    n = world.n_nodes
    edges = _edge_list(A)
    tris = _triangles(A, edges)
    B1 = incidence_b1(n, edges)
    B2 = incidence_b2(edges, tris)

    L = graph_laplacian(A)
    L0 = B1 @ B1.T
    L1 = B1.T @ B1 + (B2 @ B2.T if B2.size else np.zeros((len(edges), len(edges))))
    L2 = B2.T @ B2 if B2.size else np.zeros((len(tris), len(tris)))
    return {"L": L, "L0": L0, "L1": L1, "L2": L2, "B1": B1, "B2": B2}


def betti_from_laplacians(lap: dict[str, np.ndarray], tol: float = 1e-8) -> dict[str, int]:
    """Combinatorial Betti numbers from null spaces of the Hodge Laplacians."""
    def nullity(M: np.ndarray) -> int:
        if M.size == 0:
            return 0
        w = np.linalg.eigvalsh(0.5 * (M + M.T))
        return int(np.sum(np.abs(w) < tol))

    return {"b0": nullity(lap["L0"]), "b1": nullity(lap["L1"]), "b2": nullity(lap["L2"])}
