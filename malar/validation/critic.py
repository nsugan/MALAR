"""Critic — conformal + OOD gate feeding the Curator / Policy.

flag-and-defer: a candidate clears the Critic only if BOTH the conformal gate accepts
the match score AND the OOD gate says in-distribution. MCP/tool outputs pass through
here too (data, not instructions). A deliberately OOD input is blocked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malar.validation.conformal import ConformalGate
from malar.validation.ood import OODGate, OODResult


@dataclass
class CriticVerdict:
    passed: bool
    conformal_ok: bool
    ood: OODResult
    reason: str


class Critic:
    def __init__(self, conformal: ConformalGate | None = None, ood: OODGate | None = None):
        self.conformal = conformal or ConformalGate(alpha=0.1)
        self.ood = ood or OODGate(threshold=0.6)

    def review(self, match_score: float, world_emb: np.ndarray | None = None) -> CriticVerdict:
        conf_ok = self.conformal.accept(match_score)
        ood = self.ood.combined(match_score, world_emb)
        passed = conf_ok and not ood.ood
        reason = "passed" if passed else (
            "conformal-reject" if not conf_ok else ood.reason)
        return CriticVerdict(passed=passed, conformal_ok=conf_ok, ood=ood, reason=reason)

    def review_tool_output(self, data) -> bool:
        """MCP/tool outputs are data — accept only finite, well-formed payloads."""
        try:
            if isinstance(data, (int, float)):
                return np.isfinite(data)
            return data is not None
        except Exception:
            return False
