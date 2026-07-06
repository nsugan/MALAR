"""Policy pi(a | S, t).

Chooses actions WITHIN the F-afforded set, ranked by R, V; produces a justification.
F generates the action set; R/V score it. The policy never invents actions outside
A(S). If A(S) is empty it returns a no-op with rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

from malar.policy.actions import ScoredAction, score_actions


@dataclass
class PolicyDecision:
    action: str
    rationale: str
    ranked: list[ScoredAction]
    objects: list[str]


class Policy:
    def __init__(self, r_weight: float = 1.0, v_weight: float = 1.0):
        self.r_weight = r_weight
        self.v_weight = v_weight

    def decide(self, action_set: list[str], object_classes: list[str], r_value: float,
               v_value: float) -> PolicyDecision:
        if not action_set:
            return PolicyDecision(action="no-op", objects=object_classes, ranked=[],
                                  rationale="no afforded actions for recognised objects")
        ranked = score_actions(action_set, object_classes,
                               self.r_weight * r_value, self.v_weight * v_value)
        best = ranked[0]
        rationale = (f"chose '{best.action}' from {len(action_set)} afforded actions; "
                     f"R={best.r_score:.3f}, V={best.v_score:.3f}; "
                     f"objects={object_classes}")
        return PolicyDecision(action=best.action, objects=object_classes, ranked=ranked,
                              rationale=rationale)
