"""Hyperspectral encoder h(S, t).

A spectral autoencoder over the per-node spectra of a region. Implemented as a
linear PCA autoencoder (deterministic, trainable, no GPU) with explicit encode /
decode and a reconstruction error used by the Critic. The **raw spectra** are
stored to data/artifacts so h can be recomputed after an encoder retrain
(blue-green re-index).

The encoder is fit once on a corpus, versioned, then frozen for inference. Heads
are versioned exactly like the GNN encoder (encoder_version stamps every vector).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from malar.world.graph import WorldGraph


@dataclass
class SpectralResult:
    h: np.ndarray
    recon_error: float
    artifact_path: str | None = None
    meta: dict = field(default_factory=dict)


class SpectralEncoder:
    """Linear (PCA) spectral autoencoder. h = W^T (x - mean); recon = W h + mean."""

    version = "spec_v1"

    def __init__(self, latent_dim: int = 16, artifacts_dir: str | Path | None = None):
        self.latent_dim = latent_dim
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # (n_bands, latent)
        self.fitted = False

    def fit(self, spectra: np.ndarray) -> "SpectralEncoder":
        X = np.asarray(spectra, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("fit expects (n_samples, n_bands)")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD for stable PCA
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.latent_dim, Vt.shape[0])
        comp = Vt[:k].T  # (n_bands, k)
        # ALWAYS emit exactly latent_dim columns: pad rank-deficient fits with zero
        # components so h has a fixed width across runs (prevents dim drift between
        # sessions, which would break retrieval and persisted prototypes).
        if k < self.latent_dim:
            comp = np.pad(comp, ((0, 0), (0, self.latent_dim - k)))
        self.components_ = comp  # (n_bands, latent_dim)
        self.fitted = True
        return self

    def _ensure_fit(self, X: np.ndarray) -> None:
        if not self.fitted:
            # identity-ish fallback so the pipeline runs before an explicit fit
            self.fit(X if X.shape[0] > 1 else np.vstack([X, X + 1e-6]))

    def encode_matrix(self, spectra: np.ndarray) -> np.ndarray:
        X = np.asarray(spectra, dtype=float)
        self._ensure_fit(X)
        if self.mean_ is not None and X.shape[1] != self.mean_.shape[0]:
            # align width to the fitted band count (pad/truncate) so heterogeneous
            # inputs never crash the encoder
            w = self.mean_.shape[0]
            if X.shape[1] > w:
                X = X[:, :w]
            else:
                X = np.pad(X, ((0, 0), (0, w - X.shape[1])))
        Xc = X - self.mean_
        return Xc @ self.components_  # (n, latent)

    def decode_matrix(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components_.T + self.mean_

    def recon_error(self, spectra: np.ndarray) -> float:
        X = np.asarray(spectra, dtype=float)
        Z = self.encode_matrix(X)
        Xr = self.decode_matrix(Z)
        denom = np.linalg.norm(X) + 1e-9
        return float(np.linalg.norm(X - Xr) / denom)

    def _store_spectra(self, spectra: np.ndarray, key: str) -> str | None:
        if self.artifacts_dir is None:
            return None
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"spectra_{key}.npy"
        np.save(path, np.asarray(spectra, dtype=float))
        return str(path)

    def encode(self, world: WorldGraph, key: str | None = None) -> SpectralResult:
        """Region-level h = mean of per-node latent codes (a fixed-length descriptor)."""
        X = np.asarray(world.features, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        self._ensure_fit(X)
        Z = self.encode_matrix(X)
        h = Z.mean(axis=0)
        err = self.recon_error(X)
        if key is None:
            key = world.content_hash()
        artifact_path = self._store_spectra(X, key)
        return SpectralResult(h=h, recon_error=err, artifact_path=artifact_path,
                              meta={"encoder_version": self.version})

    # -- persistence for re-index ----------------------------
    # -- persistence for re-index --------------------------------------
    def state_dict(self) -> dict:
        return {"mean": self.mean_, "components": self.components_,
                "latent_dim": self.latent_dim, "version": self.version}

    def load_state(self, state: dict) -> None:
        self.mean_ = state["mean"]
        self.components_ = state["components"]
        self.latent_dim = state["latent_dim"]
        self.fitted = True
