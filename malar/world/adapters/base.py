"""WorldAdapter: raw source -> WorldBatch -> WorldGraph W(t).

Every domain implements one WorldAdapter. Batch mode (replay) and streaming
(one WorldBatch per tick) drive the identical loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from malar.world.graph import WorldGraph


@dataclass
class WorldBatch:
    """One tick of raw observations plus optional ground-truth labels (for eval)."""

    t: int
    points: np.ndarray                 # (n, d) feature/spectra/positions
    node_ids: list[str]
    labels: list[str | None] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    source: str = "unknown"


class WorldAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def stream(self) -> Iterator[WorldBatch]:
        """Yield WorldBatch objects, one per tick."""
        raise NotImplementedError

    @abstractmethod
    def to_world(self, batch: WorldBatch) -> WorldGraph:
        """Turn a raw batch into a graph W(t)."""
        raise NotImplementedError

    def batches(self, n: int | None = None) -> list[WorldBatch]:
        out: list[WorldBatch] = []
        for i, b in enumerate(self.stream()):
            if n is not None and i >= n:
                break
            out.append(b)
        return out
