"""Action set assembly. The afforded action set A(S) comes from the Functional fields
(F generates); the Policy ranks it by R, V (R/V score). This module just defines the
Action record and helpers to attach R/V scores to candidate actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoredAction:
    action: str
    r_score: float
    v_score: float
    objects: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.r_score + self.v_score


def score_actions(action_set: list[str], object_classes: list[str], r_value: float,
                  v_value: float) -> list[ScoredAction]:
    out = [ScoredAction(action=a, r_score=r_value, v_score=v_value, objects=list(object_classes))
           for a in action_set]
    out.sort(key=lambda x: x.total, reverse=True)
    return out
