"""Conformal gate: flag-and-defer on low confidence.

Split-conformal style: calibrate on a set of nonconformity scores (here, 1 - match
score for known-correct matches), then accept a new match only if its score exceeds
the (1-alpha) empirical quantile threshold. Before calibration it falls back to a
fixed floor so the pipeline runs cold.
"""
from __future__ import annotations

import numpy as np


class ConformalGate:
    def __init__(self, alpha: float = 0.1, floor: float = 0.5):
        self.alpha = alpha
        self.floor = floor
        self._scores: list[float] = []
        self.threshold: float = floor

    def calibrate(self, correct_match_scores: list[float]) -> float:
        """Set the acceptance threshold from scores of known-correct matches."""
        self._scores = [float(s) for s in correct_match_scores]
        if not self._scores:
            self.threshold = self.floor
            return self.threshold
        # accept scores at/above the alpha-quantile of correct matches
        q = np.quantile(self._scores, self.alpha)
        self.threshold = float(max(self.floor, q))
        return self.threshold

    def accept(self, score: float) -> bool:
        return float(score) >= self.threshold

    def p_value(self, score: float) -> float:
        """Fraction of calibration scores at or below this score (rank-based)."""
        if not self._scores:
            return 1.0 if score >= self.floor else 0.0
        arr = np.asarray(self._scores)
        return float((arr <= score).mean())
