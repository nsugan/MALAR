"""Per-modality Bayesian posteriors + product-of-experts fusion (I0 + I1).

Each modality's GaussianLikelihood gives a per-dimension log-likelihood per class. We
turn that into a categorical posterior:

    P(c | x_m) = softmax_c( loglik_m(c) / T_m + log prior(c) )

where T_m is a per-modality temperature (default 1.0; the per-dimension averaging in the
likelihood already keeps the scale comparable). The fused posterior is a weighted
**product of experts**:

    P(c | phi,h,g) proportional to prior(c) * prod_m [ P(c | x_m) / prior(c) ] ^ w_m

Disagreement between modalities widens the fused posterior (honest uncertainty); when the
modalities agree and are confident it collapses to the same class the point-estimate path
returns. `evidence[m]` is the modality's max class posterior -- a quick reliability proxy
the manager can use to down-weight a noisy modality.
"""
from __future__ import annotations

import numpy as np

from malar.inference.probabilistic.likelihoods import GaussianLikelihood, collect_corpus

_MODALITIES = ("phi", "h", "g")
_MODALITY_LABEL = {"phi": "topo", "h": "spectral", "g": "graph"}


def _softmax(d: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not d:
        return {}
    keys = list(d.keys())
    z = np.array([d[k] for k in keys], dtype=float) / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    s = e.sum()
    p = (e / s) if s > 0 else np.full(len(keys), 1.0 / len(keys))
    return {k: float(v) for k, v in zip(keys, p)}


def _entropy(p: dict[str, float]) -> float:
    v = np.array(list(p.values()), dtype=float)
    v = v[v > 0]
    return float(-(v * np.log(v)).sum()) if v.size else 0.0


def fuse_posteriors(per_modality: dict[str, dict[str, float]],
                    prior: dict[str, float],
                    weights: dict[str, float] | None = None) -> dict[str, float]:
    """Weighted product-of-experts over per-modality posteriors (keys = modality labels)."""
    classes = sorted(prior.keys())
    if not classes:
        return {}
    weights = weights or {}
    log_post = {c: np.log(max(prior.get(c, 1e-9), 1e-12)) for c in classes}
    for mlabel, post in per_modality.items():
        w = float(weights.get(mlabel, 1.0))
        if w == 0.0:
            continue
        for c in classes:
            pc = max(post.get(c, 1e-9), 1e-12)
            pri = max(prior.get(c, 1e-9), 1e-12)
            log_post[c] += w * (np.log(pc) - np.log(pri))
    return _softmax(log_post, temperature=1.0)


class BayesEnsemble:
    """Three per-modality GaussianLikelihoods fit over the whole corpus, plus fusion."""

    def __init__(self, shrinkage: float = 0.2, temperatures: dict[str, float] | None = None):
        self.shrinkage = shrinkage
        self.temps = temperatures or {"topo": 1.0, "spectral": 1.0, "graph": 1.0}
        self.lik: dict[str, GaussianLikelihood] = {}
        self.prevalence: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.classes: list[str] = []
        self.n_corpus: int = 0

    def fit(self, engine) -> "BayesEnsemble":
        corpus = collect_corpus(engine)
        self.classes = sorted(corpus.keys())
        counts: dict[str, int] = {}
        for m in _MODALITIES:
            X_by_class = {c: corpus[c][m] for c in corpus}
            self.lik[m] = GaussianLikelihood(shrinkage=self.shrinkage).fit(X_by_class)
            for c in corpus:
                n = corpus[c][m].shape[0] if corpus[c][m].ndim == 2 else 0
                counts[c] = max(counts.get(c, 0), n)
        total = sum(counts.values())
        self.counts = {c: int(counts.get(c, 0)) for c in self.classes}
        self.n_corpus = int(total)
        if total:
            self.prevalence = {c: counts.get(c, 0) / total for c in self.classes}
        else:
            n = max(len(self.classes), 1)
            self.prevalence = {c: 1.0 / n for c in self.classes}
        return self

    def is_fitted(self) -> bool:
        return bool(self.classes) and any(
            self.lik.get(m) and self.lik[m].dim for m in _MODALITIES)

    def _prior(self, mode: str = "prevalence",
               override: dict[str, float] | None = None) -> dict[str, float]:
        if override:
            s = sum(override.values()) or 1.0
            return {c: override.get(c, 0.0) / s for c in self.classes}
        if mode == "uniform" or not self.prevalence:
            n = max(len(self.classes), 1)
            return {c: 1.0 / n for c in self.classes}
        return dict(self.prevalence)

    def posteriors(self, phi, h, g, prior_mode: str = "prevalence",
                   prior_override: dict[str, float] | None = None,
                   weights: dict[str, float] | None = None) -> dict:
        """Full per-modality + fused posterior result for one input."""
        prior = self._prior(prior_mode, prior_override)
        vecs = {"phi": phi, "h": h, "g": g}
        per_modality: dict[str, dict[str, float]] = {}
        evidence: dict[str, float] = {}
        for m in _MODALITIES:
            lik = self.lik.get(m)
            label = _MODALITY_LABEL[m]
            if lik is None or not lik.dim:
                continue
            ll = lik.loglik(vecs[m])
            combined = {c: ll.get(c, -1e9) + np.log(max(prior.get(c, 1e-9), 1e-12))
                        for c in self.classes}
            post = _softmax(combined, temperature=self.temps.get(label, 1.0))
            per_modality[label] = post
            evidence[label] = max(post.values()) if post else 0.0
        fused = fuse_posteriors(per_modality, prior, weights)
        top = max(fused, key=fused.get) if fused else None
        return {
            "per_modality": per_modality,
            "fused": fused,
            "prior": prior,
            "evidence": evidence,
            "entropy": _entropy(fused),
            "top": top,
            "top_p": fused.get(top, 0.0) if top else 0.0,
            "n_corpus": self.n_corpus,
            "classes": self.classes,
        }
