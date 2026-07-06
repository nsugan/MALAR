"""Reverse world-state generation  Gen(g, phi, h, tau; theta_gen).

Given a memory's stored codes, reconstruct a PLAUSIBLE region approximating what was
learned: spectra are reconstructed via the spectral autoencoder's decoder; a point
cloud is synthesised whose topology matches the persistence summary (e.g. an H1 loop
=> a noisy circle; otherwise a blob). Used for recall visualisation and counterfactual
("what if") generation. Deterministic; seeded by the memory + theta_gen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from malar.encoders.spectral import SpectralEncoder


@dataclass
class GeneratedRegion:
    points: np.ndarray
    spectra: np.ndarray | None
    n_loops: int
    meta: dict = field(default_factory=dict)


class ReverseGenerator:
    def __init__(self, spectral: SpectralEncoder | None = None, seed: int = 0):
        self.spectral = spectral
        self.seed = seed

    def generate(self, phi_summary: dict, h: np.ndarray | None = None, n: int = 60,
                 theta_gen: dict | None = None) -> GeneratedRegion:
        rng = np.random.default_rng((theta_gen or {}).get("seed", self.seed))
        n_loops = int(phi_summary.get("H1", {}).get("count", 0)) if phi_summary else 0
        max_pers = float(phi_summary.get("H1", {}).get("max_persistence", 0.0)) if phi_summary else 0.0

        if n_loops >= 1 and max_pers > 0.1:
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
            r = 1.0
            pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
            pts = pts + rng.normal(0, 0.04, pts.shape)
        else:
            pts = rng.normal(0, 0.2, (n, 2))

        spectra = None
        if self.spectral is not None and self.spectral.fitted and h is not None:
            # decode the region-mean latent back to a representative spectrum
            z = np.asarray(h, dtype=float)[: self.spectral.components_.shape[1]]
            z = np.pad(z, (0, self.spectral.components_.shape[1] - z.size)) \
                if z.size < self.spectral.components_.shape[1] else z
            spectra = self.spectral.decode_matrix(z[None, :])

        return GeneratedRegion(points=pts, spectra=spectra, n_loops=n_loops,
                               meta={"max_persistence": max_pers})

    def counterfactual(self, phi_summary: dict, perturb: float = 0.3, **kw) -> GeneratedRegion:
        """Generate a perturbed variant (counterfactual) of the recalled region."""
        s = dict(phi_summary or {})
        h1 = dict(s.get("H1", {}))
        h1["max_persistence"] = max(0.0, h1.get("max_persistence", 0.0) * (1 - perturb))
        s["H1"] = h1
        return self.generate(s, **kw)
