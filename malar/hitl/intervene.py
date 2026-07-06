"""Human-in-the-loop: active-learning queries, label requests, corrections, overrides.

The HITL queue holds items the system wants a human to act on:
  * label   — an unidentified candidate object needs a class.
  * value   — a proposed value change blocked by a sticky human value.
  * confirm — a low-confidence identification to approve/edit/reject.

Active learning ranks pending uncertainty so the Planner spends the HITL budget where
it most reduces uncertainty. Human actions are final and sticky.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

QueueKind = Literal["label", "value", "confirm", "affordance"]


@dataclass
class HITLItem:
    id: int
    kind: QueueKind
    payload: dict
    uncertainty: float = 0.0
    status: str = "pending"        # 'pending' | 'resolved' | 'rejected'
    resolution: dict | None = None
    world_ctx_id: str | None = None


class HITLQueue:
    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self.items: dict[int, HITLItem] = {}
        self.audit: list[dict] = []

    def enqueue(self, kind: QueueKind, payload: dict, uncertainty: float = 0.0,
                world_ctx_id: str | None = None) -> HITLItem:
        item = HITLItem(id=next(self._counter), kind=kind, payload=payload,
                        uncertainty=float(uncertainty), world_ctx_id=world_ctx_id)
        self.items[item.id] = item
        self.audit.append({"op": "enqueue", "id": item.id, "kind": kind})
        return item

    def pending(self) -> list[HITLItem]:
        return [it for it in self.items.values() if it.status == "pending"]

    def next_query(self) -> HITLItem | None:
        """Active learning: return the highest-uncertainty pending item."""
        pend = self.pending()
        if not pend:
            return None
        return max(pend, key=lambda it: it.uncertainty)

    def resolve(self, item_id: int, resolution: dict, accept: bool = True) -> HITLItem | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        item.status = "resolved" if accept else "rejected"
        item.resolution = resolution
        self.audit.append({"op": "resolve", "id": item_id, "accept": accept,
                           "resolution": resolution})
        return item

    def budget_remaining(self, budget: int) -> int:
        used = sum(1 for it in self.items.values() if it.status != "pending")
        return max(0, budget - used)
