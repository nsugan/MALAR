"""RamanAdapter: hyperspectral Raman spectra -> similarity graph.

Each node is a spectrum (e.g. a pixel of a Raman map or one measurement). Edges
connect spectrally-similar nodes (kNN on spectral cosine distance). PH then runs on
the resulting graph -> phi; the spectral encoder consumes the raw spectra -> h.

When no real dataset is supplied, a deterministic synthetic Raman generator produces
class-specific peak patterns (pathogen signatures) so the pipeline + eval are runnable.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from malar.world.adapters.base import WorldAdapter, WorldBatch
from malar.world.graph import WorldGraph


def _spectrum(peaks: list[tuple[float, float, float]], n_bands: int, rng) -> np.ndarray:
    """Sum of Gaussian peaks (center, width, height) on a band axis + noise."""
    x = np.linspace(0, 1, n_bands)
    y = np.zeros(n_bands)
    for c, w, h in peaks:
        y += h * np.exp(-((x - c) ** 2) / (2 * w * w))
    y += rng.normal(0, 0.01, n_bands)
    y = np.clip(y, 0, None)
    norm = np.linalg.norm(y)
    return y / norm if norm > 0 else y


# class -> characteristic peaks
_CLASS_PEAKS = {
    "sars_cov_2": [(0.20, 0.02, 1.0), (0.55, 0.03, 0.6), (0.80, 0.015, 0.8)],
    "influenza_a": [(0.25, 0.02, 0.9), (0.50, 0.025, 0.7), (0.70, 0.02, 0.5)],
    "rsv": [(0.15, 0.03, 0.7), (0.60, 0.02, 0.9), (0.85, 0.02, 0.4)],
    "negative": [(0.40, 0.05, 0.3)],
}


def cosine_knn_graph(spectra: np.ndarray, k: int = 6) -> np.ndarray:
    X = np.asarray(spectra, dtype=float)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 0, norms, 1.0)
    sim = Xn @ Xn.T
    A = np.zeros((n, n))
    k = min(k, n - 1) if n > 1 else 0
    for i in range(n):
        order = np.argsort(-sim[i])
        cnt = 0
        for j in order:
            if j == i:
                continue
            A[i, j] = max(A[i, j], float(max(sim[i, j], 0.0)))
            A[j, i] = A[i, j]
            cnt += 1
            if cnt >= k:
                break
    return A


class RamanAdapter(WorldAdapter):
    name = "raman"

    def __init__(
        self,
        n_ticks: int = 6,
        n_per_class: int = 8,
        n_bands: int = 128,
        k: int = 6,
        classes: list[str] | None = None,
        seed: int = 0,
        dataset: dict | None = None,
    ):
        self.n_ticks = n_ticks
        self.n_per_class = n_per_class
        self.n_bands = n_bands
        self.k = k
        self.classes = classes or list(_CLASS_PEAKS.keys())
        self.seed = seed
        self.dataset = dataset  # optional real {spectra, labels} per tick

    def stream(self) -> Iterator[WorldBatch]:
        rng = np.random.default_rng(self.seed)
        if self.dataset is not None:
            for t, tick in enumerate(self.dataset.get("ticks", [])):
                spectra = np.asarray(tick["spectra"], dtype=float)
                labels = list(tick.get("labels", [None] * len(spectra)))
                ids = [f"raman_t{t}_n{i}" for i in range(len(spectra))]
                yield WorldBatch(t=t, points=spectra, node_ids=ids, labels=labels,
                                 raw={"n_bands": spectra.shape[1]}, source="raman_dataset")
            return
        for t in range(self.n_ticks):
            spectra = []
            labels = []
            for cls in self.classes:
                for _ in range(self.n_per_class):
                    spectra.append(_spectrum(_CLASS_PEAKS[cls], self.n_bands, rng))
                    labels.append(cls)
            spectra = np.array(spectra)
            idx = rng.permutation(len(spectra))
            spectra = spectra[idx]
            labels = [labels[i] for i in idx]
            ids = [f"raman_t{t}_n{i}" for i in range(len(spectra))]
            yield WorldBatch(t=t, points=spectra, node_ids=ids, labels=labels,
                             raw={"n_bands": self.n_bands}, source="raman_synth")

    def to_world(self, batch: WorldBatch) -> WorldGraph:
        A = cosine_knn_graph(batch.points, k=self.k)
        return WorldGraph(node_ids=batch.node_ids, features=batch.points, adjacency=A,
                          t=batch.t, meta={"source": batch.source})
