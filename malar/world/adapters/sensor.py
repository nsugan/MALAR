"""SensorTimeseriesAdapter: generic multi-sensor streams -> spatio-temporal graph.

Nodes = sensors at a tick; edges = spatial proximity (given/derived) plus a temporal
self-edge carrying the previous reading. Produces W(t) per tick.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from malar.world.adapters.base import WorldAdapter, WorldBatch
from malar.world.graph import WorldGraph, knn_graph_from_points


class SensorTimeseriesAdapter(WorldAdapter):
    name = "sensor"

    def __init__(self, n_sensors: int = 20, n_ticks: int = 10, k: int = 4, seed: int = 0,
                 positions: np.ndarray | None = None, series: np.ndarray | None = None):
        self.n_sensors = n_sensors
        self.n_ticks = n_ticks
        self.k = k
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.positions = positions if positions is not None else rng.uniform(0, 1, (n_sensors, 2))
        if series is not None:
            self.series = np.asarray(series, dtype=float)  # (n_ticks, n_sensors, d)
        else:
            base = np.sin(np.linspace(0, 4 * np.pi, n_ticks))[:, None]
            self.series = base + rng.normal(0, 0.1, (n_ticks, n_sensors, 1)) \
                + self.positions[:, :1].T[None, :, :]

    def stream(self) -> Iterator[WorldBatch]:
        for t in range(self.series.shape[0]):
            readings = self.series[t]  # (n_sensors, d)
            feats = np.concatenate([self.positions, readings], axis=1)
            ids = [f"sensor_{i}@t{t}" for i in range(readings.shape[0])]
            yield WorldBatch(t=t, points=feats, node_ids=ids,
                             labels=[None] * readings.shape[0],
                             raw={"positions": self.positions}, source="sensor")

    def to_world(self, batch: WorldBatch) -> WorldGraph:
        pos = batch.raw["positions"]
        A = knn_graph_from_points(pos, k=self.k)
        return WorldGraph(node_ids=batch.node_ids, features=batch.points, adjacency=A,
                          t=batch.t, meta={"source": batch.source})
