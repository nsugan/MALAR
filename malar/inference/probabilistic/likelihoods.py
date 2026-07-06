"""Per-class generative likelihoods, fit over the WHOLE training corpus.

For each modality (phi = topology, h = hyperspectral, g = graph) we fit a per-class
**diagonal Gaussian with shrinkage** in that modality's feature space:

    mu_c   = mean of class-c training vectors
    var_c  = (1 - lam) * diag-var(class c) + lam * pooled-diag-var      (shrinkage)

Shrinkage toward the pooled variance keeps the estimate stable when a class has only a
few training items (the norm here, e.g. 10 spectra per virus). The log-likelihood is
reported **per-dimension averaged** so modalities of very different dimensionality
(432 vs 16 vs 48) stay comparable before they are turned into posteriors.

The corpus is read from the live per-domain Qdrant collection (all stored memories) AND
the object-registry prototypes (one EMA prototype per class). Merging guarantees full
class coverage even when the curator has compressed the vector store.
"""
from __future__ import annotations

import numpy as np

_HIDDEN = {"__candidate__", "__unlabeled__", None, ""}
_LOG2PI = float(np.log(2.0 * np.pi))


def collect_corpus(engine) -> dict[str, dict[str, np.ndarray]]:
    """Return {class: {"phi": [n,Dphi], "h": [n,Dh], "g": [n,Dg]}} over all train memories.

    Sources merged: every point in the domain's Qdrant collection (the corpus samples,
    with their variance) plus the registry's per-class EMA prototypes (guaranteeing every
    learned class is represented even if the vector store was compacted).
    """
    by_cls: dict[str, dict[str, list]] = {}

    def _add(cls, phi, h, g):
        if cls in _HIDDEN:
            return
        slot = by_cls.setdefault(cls, {"phi": [], "h": [], "g": []})
        if phi is not None:
            slot["phi"].append(np.asarray(phi, dtype=float).ravel())
        if h is not None:
            slot["h"].append(np.asarray(h, dtype=float).ravel())
        if g is not None:
            slot["g"].append(np.asarray(g, dtype=float).ravel())

    qdrant = getattr(getattr(engine, "c", None), "store", None)
    qdrant = getattr(qdrant, "qdrant", None)
    rows = []
    if qdrant is not None:
        try:
            rows = qdrant.scroll_all(with_vectors=True)
        except Exception:
            rows = []
    for r in rows:
        _add(r.get("class"), r.get("phi"), r.get("hyper"), r.get("graph_emb"))

    # Always fold in the registry prototypes (one EMA prototype per class) so that every
    # learned class is covered even when Qdrant is curator-compressed / unavailable.
    reg = getattr(getattr(engine, "c", None), "registry", None)
    if reg is not None:
        for o in reg.objects.values():
            if not getattr(o, "candidate", False):
                _add(o.cls, o.phi, o.h, o.g)

    out: dict[str, dict[str, np.ndarray]] = {}
    for cls, slot in by_cls.items():
        if not slot["phi"] and not slot["h"] and not slot["g"]:
            continue
        out[cls] = {"phi": _stack(slot["phi"]), "h": _stack(slot["h"]), "g": _stack(slot["g"])}
    return out


def _stack(rows: list) -> np.ndarray:
    """Stack 1-D vectors into [n, d], padding to the widest row (Qdrant pads to fixed
    dims while registry prototypes keep raw widths, so a class may mix the two)."""
    if not rows:
        return np.empty((0, 0))
    dim = max(r.shape[0] for r in rows)
    return np.vstack([r if r.shape[0] == dim else np.pad(r, (0, dim - r.shape[0]))
                      for r in rows])


class GaussianLikelihood:
    """Diagonal-Gaussian, shrinkage-regularised, per-class likelihood for one modality."""

    def __init__(self, shrinkage: float = 0.2, min_var: float = 1e-6):
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.min_var = float(min_var)
        self.mu: dict[str, np.ndarray] = {}
        self.var: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.dim: int = 0
        self._pooled_var: np.ndarray | None = None

    @property
    def classes(self) -> list[str]:
        return sorted(self.mu.keys())

    def fit(self, X_by_class: dict[str, np.ndarray]) -> "GaussianLikelihood":
        dim = max((X.shape[1] for X in X_by_class.values() if X.ndim == 2 and X.size), default=0)
        self.dim = int(dim)
        if dim == 0:
            return self
        all_rows = []
        fixed: dict[str, np.ndarray] = {}
        for cls, X in X_by_class.items():
            if X.ndim != 2 or X.size == 0:
                continue
            Xf = _fit_width(X, dim)
            fixed[cls] = Xf
            all_rows.append(Xf)
        if not all_rows:
            return self
        pooled = np.vstack(all_rows)
        pooled_var = np.maximum(pooled.var(axis=0), self.min_var)
        self._pooled_var = pooled_var
        lam = self.shrinkage
        for cls, Xf in fixed.items():
            n = Xf.shape[0]
            mu = Xf.mean(axis=0)
            v = Xf.var(axis=0) if n > 1 else np.zeros(dim)
            var = (1.0 - lam) * v + lam * pooled_var
            self.mu[cls] = mu
            self.var[cls] = np.maximum(var, self.min_var)
            self.counts[cls] = int(n)
        return self

    def loglik(self, x: np.ndarray) -> dict[str, float]:
        """Per-dimension averaged Gaussian log-likelihood of x under each class."""
        if self.dim == 0:
            return {c: 0.0 for c in self.mu}
        xf = _fit_width(np.asarray(x, dtype=float).reshape(1, -1), self.dim)[0]
        out = {}
        for cls in self.mu:
            mu, var = self.mu[cls], self.var[cls]
            ll = -0.5 * (((xf - mu) ** 2) / var + np.log(var) + _LOG2PI)
            out[cls] = float(ll.mean())
        return out


def _fit_width(X: np.ndarray, dim: int) -> np.ndarray:
    """Pad or trim a [n, d] array to width `dim`."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    d = X.shape[1]
    if d == dim:
        return X
    if d > dim:
        return X[:, :dim]
    return np.pad(X, ((0, 0), (0, dim - d)))
