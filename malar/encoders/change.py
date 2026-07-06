"""Per-modality change statistics Delta_phi, Delta_h, Delta_g.

Kept separate (never collapsed into one scalar) so novelty can fire on any modality:
  Delta_phi : Wasserstein/bottleneck on the persistence diagram (exact), or cosine on phi.
  Delta_h   : cosine/L2 distance on spectral codes.
  Delta_g   : cosine distance on graph embeddings.
"""
from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:  # align mismatched widths (e.g. legacy persisted prototypes)
        n = max(a.size, b.size)
        a = np.pad(a, (0, n - a.size))
        b = np.pad(b, (0, n - b.size))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if (na != nb) else 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_distance(a, b)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = max(a.size, b.size)
    a = np.pad(a, (0, n - a.size))
    b = np.pad(b, (0, n - b.size))
    return float(np.linalg.norm(a - b))


def diagram_wasserstein(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Exact Wasserstein distance between two persistence diagrams (persim)."""
    try:
        from persim import wasserstein

        a = _finite(dgm_a)
        b = _finite(dgm_b)
        return float(wasserstein(a, b))
    except Exception:
        # fallback: total-persistence difference
        return abs(_total_persistence(dgm_a) - _total_persistence(dgm_b))


def diagram_bottleneck(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    try:
        from persim import bottleneck

        return float(bottleneck(_finite(dgm_a), _finite(dgm_b)))
    except Exception:
        return abs(_total_persistence(dgm_a) - _total_persistence(dgm_b))


def _finite(dgm: np.ndarray) -> np.ndarray:
    d = np.asarray(dgm, dtype=float)
    if d.size == 0:
        return np.zeros((0, 2))
    return d[np.isfinite(d).all(axis=1)]


def _total_persistence(dgm: np.ndarray) -> float:
    d = _finite(dgm)
    if d.shape[0] == 0:
        return 0.0
    return float((d[:, 1] - d[:, 0]).sum())


def delta_phi(phi_s: np.ndarray, phi_m: np.ndarray,
              dgm_s: np.ndarray | None = None, dgm_m: np.ndarray | None = None) -> float:
    """Prefer exact Wasserstein on H1 diagrams when available, else cosine on phi."""
    if dgm_s is not None and dgm_m is not None:
        return diagram_wasserstein(dgm_s, dgm_m)
    return cosine_distance(phi_s, phi_m)


def delta_h(h_s: np.ndarray, h_m: np.ndarray) -> float:
    return cosine_distance(h_s, h_m)


def delta_g(g_s: np.ndarray, g_m: np.ndarray) -> float:
    return cosine_distance(g_s, g_m)


def change_stats(query: dict, neighbour: dict) -> dict:
    dphi = delta_phi(query.get("phi"), neighbour.get("phi"),
                     query.get("dgm_h1"), neighbour.get("dgm_h1"))
    dh = delta_h(query.get("h"), neighbour.get("h"))
    dg = delta_g(query.get("g"), neighbour.get("g"))
    return {"d_phi": dphi, "d_h": dh, "d_g": dg}
