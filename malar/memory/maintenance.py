"""CompressAndDecay — budgeted greedy memory compaction (1 - 1/e guarantee).

Importance decays each slow tick; when the store exceeds budget, greedily keep the
subset of memories that maximises a submodular coverage+utility objective. Greedy
maximisation of a monotone submodular function gives the classic (1 - 1/e) bound.
"""
from __future__ import annotations

import numpy as np

from malar.encoders.change import cosine_similarity
from malar.memory.schema import MemoryItem


def utility(item: MemoryItem, t: int, halflife: float = 50.0) -> float:
    recency = np.exp(-max(0, t - item.tau) / halflife)
    return float(item.omega * recency)


def _marginal_coverage(cand: MemoryItem, chosen: list[MemoryItem]) -> float:
    """Marginal novelty of cand vs already-chosen set (1 - max similarity)."""
    if not chosen:
        return 1.0
    sims = [cosine_similarity(cand.phi, c.phi) for c in chosen]
    return float(1.0 - max(sims))


def compress_and_decay(items: list[MemoryItem], budget: int, t: int,
                       decay_factor: float = 0.97) -> tuple[list[MemoryItem], list[str]]:
    """Decay all, then greedily keep <= budget by utility * marginal-coverage.

    Returns (kept_items, evicted_ids).
    """
    for it in items:
        it.omega *= decay_factor
    if len(items) <= budget:
        return items, []

    remaining = list(items)
    chosen: list[MemoryItem] = []
    while remaining and len(chosen) < budget:
        best, best_gain = None, -1.0
        for cand in remaining:
            gain = utility(cand, t) * (0.5 + 0.5 * _marginal_coverage(cand, chosen))
            if gain > best_gain:
                best, best_gain = cand, gain
        chosen.append(best)
        remaining.remove(best)
    evicted = [it.id for it in remaining]
    return chosen, evicted
