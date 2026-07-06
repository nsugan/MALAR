"""RegionSampler: pick a region S of the world under a budget B.

A region is a connected (where possible) subset of node indices. Selection is
importance-weighted (Dirichlet energy + degree) with a deterministic RNG so runs
are reproducible. Agents set the budget; the sampler computes the region.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malar.world.graph import WorldGraph
from malar.world.representation import node_energy


@dataclass
class Region:
    indices: list[int]
    seed: int
    score: float = 0.0

    def __len__(self) -> int:
        return len(self.indices)


class RegionSampler:
    def __init__(self, budget: int = 16, rng: np.random.Generator | int | None = None):
        self.budget = int(budget)
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(0 if rng is None else int(rng))

    def importance(self, world: WorldGraph) -> np.ndarray:
        if world.n_nodes == 0:
            return np.zeros(0)
        eng = np.abs(node_energy(world))
        deg = world.degree()
        imp = eng + 0.1 * deg
        s = imp.sum()
        return imp / s if s > 0 else np.ones(world.n_nodes) / world.n_nodes

    def sample(self, world: WorldGraph, budget: int | None = None) -> Region:
        """Grow a region from a high-importance seed via BFS over neighbours."""
        b = int(budget or self.budget)
        n = world.n_nodes
        if n == 0:
            return Region(indices=[], seed=-1, score=0.0)
        b = min(b, n)
        imp = self.importance(world)
        seed = int(self.rng.choice(n, p=imp))
        chosen = [seed]
        frontier = set(np.nonzero(world.adjacency[seed] > 0)[0].tolist())
        frontier.discard(seed)
        while len(chosen) < b and frontier:
            # pick highest-importance frontier node
            cand = sorted(frontier, key=lambda i: imp[i], reverse=True)[0]
            frontier.discard(cand)
            chosen.append(cand)
            for nb in np.nonzero(world.adjacency[cand] > 0)[0].tolist():
                if nb not in chosen:
                    frontier.add(nb)
        # if still short (disconnected), top up by importance
        if len(chosen) < b:
            rest = [i for i in np.argsort(-imp).tolist() if i not in chosen]
            chosen.extend(rest[: b - len(chosen)])
        score = float(imp[chosen].sum())
        return Region(indices=sorted(chosen), seed=seed, score=score)

    def region_world(self, world: WorldGraph, region: Region) -> WorldGraph:
        return world.subgraph(region.indices)
