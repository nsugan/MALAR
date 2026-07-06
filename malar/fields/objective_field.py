"""Per-objective value model f_k, {v_{k,c}}, head psi_k.

R^(k)(S) = sum_{o in O(S)} v_{k, class(o)} * ctx(o, S)

where v_{k,c} is the learned contribution of class c to objective k, and ctx(o,S)
is the object's contextual weight in the region (e.g. its prevalence/confidence).
Values are learned from training data AND human interventions; human values are
sticky (enforced by ValueStore). The head psi_k is versioned like the encoders.
"""
from __future__ import annotations

from dataclasses import dataclass

from malar.fields.objectives import ObjectiveSpec
from malar.fields.value_store import ValueStore


@dataclass
class RContribution:
    object_class: str
    value: float
    ctx: float
    contribution: float


class ObjectiveField:
    """Learns {v_{k,c}} for one objective k and evaluates R^(k)."""

    def __init__(self, spec: ObjectiveSpec, store: ValueStore, lr: float = 0.2, version: int = 1):
        self.spec = spec
        self.k = spec.key
        self.store = store
        self.lr = lr
        self.version = version  # psi_k version

    # -- learning ------------------------------------------------------
    def learn_from_data(self, object_class: str, observed_contribution: float, by: str = "A_k",
                        world_ctx: str | None = None) -> bool:
        """EMA-style update of v_{k,c} toward an observed contribution (data source)."""
        cur = self.store.value(object_class, self.k, default=0.0)
        new = (1 - self.lr) * cur + self.lr * float(observed_contribution)
        applied, _ = self.store.set(object_class, self.k, new, source="data", by=by,
                                    world_ctx=world_ctx)
        return applied

    def set_human_value(self, object_class: str, value: float, by: str = "human",
                        world_ctx: str | None = None) -> bool:
        applied, _ = self.store.set(object_class, self.k, value, source="human", by=by,
                                    world_ctx=world_ctx)
        return applied

    def propose_change(self, object_class: str, value: float, by: str = "A_k",
                       world_ctx: str | None = None) -> tuple[bool, dict | None]:
        """Try to set a value; if blocked by a sticky human value, return a HITL proposal."""
        applied, rec = self.store.set(object_class, self.k, value, source="data", by=by,
                                      world_ctx=world_ctx)
        if applied:
            return True, None
        proposal = {"kind": "value_change", "field_k": self.k, "object_class": object_class,
                    "current": rec.value, "proposed": value, "by": by, "world_ctx": world_ctx}
        return False, proposal

    # -- evaluation ----------------------------------------------------
    def evaluate(self, objects: list[dict]) -> tuple[float, list[RContribution]]:
        """objects: list of {'class', 'ctx'} recognised in region S. Returns R^(k)(S)."""
        contribs: list[RContribution] = []
        total = 0.0
        for o in objects:
            cls = o["class"]
            ctx = float(o.get("ctx", 1.0))
            v = self.store.value(cls, self.k, default=0.0)
            c = v * ctx
            total += c
            contribs.append(RContribution(object_class=cls, value=v, ctx=ctx, contribution=c))
        return total, contribs

    def values(self) -> dict[str, float]:
        return self.store.values_for_field(self.k)
