"""Out-of-distribution gate.

Two signals:
  * retrieval distance — if the nearest known object is too far (low max similarity),
    the input is OOD.
  * world-context validity — if world_emb is far from any trained context, a learned
    value may no longer be in-distribution (validity-gating).

Returns an OODResult with a boolean and the driving score so callers can return
"outside trained domain" and NEVER fabricate an action.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malar.encoders.change import cosine_similarity


@dataclass
class OODResult:
    ood: bool
    confidence: float        # in-distribution confidence in [0,1]
    reason: str


class OODGate:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._context_embs: list[np.ndarray] = []

    def register_context(self, world_emb: np.ndarray) -> None:
        self._context_embs.append(np.asarray(world_emb, dtype=float))

    def check_match(self, best_score: float) -> OODResult:
        ood = best_score < self.threshold
        return OODResult(ood=ood, confidence=float(best_score),
                         reason="nearest object too far" if ood else "in-distribution")

    def check_context(self, world_emb: np.ndarray) -> OODResult:
        if not self._context_embs:
            return OODResult(ood=False, confidence=1.0, reason="no registered contexts")
        sims = [cosine_similarity(world_emb, c) for c in self._context_embs]
        best = max(sims)
        ood = best < self.threshold
        return OODResult(ood=ood, confidence=float(best),
                         reason="context drift / unseen world" if ood else "context in-distribution")

    def combined(self, best_score: float, world_emb: np.ndarray | None = None) -> OODResult:
        m = self.check_match(best_score)
        if world_emb is None:
            return m
        c = self.check_context(world_emb)
        conf = min(m.confidence, c.confidence)
        ood = m.ood or c.ood
        reason = m.reason if m.ood else (c.reason if c.ood else "in-distribution")
        return OODResult(ood=ood, confidence=conf, reason=reason)
