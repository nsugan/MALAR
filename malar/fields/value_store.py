"""Value store: (:Object {class})-[:HAS_VALUE {field, value, source, version, by, world_ctx}]->(:Objective).

In-process source of truth with provenance + versioning. Synced to Neo4j by the
memory layer. **Human values are sticky**: agents may only *propose* changes, which
re-enter the HITL queue — they never silently overwrite a human-set value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class ValueRecord:
    object_class: str
    field_k: str          # objective key
    value: float
    source: str           # 'data' | 'human' | 'llm'
    version: int
    by: str               # agent id or human id
    world_ctx: str | None = None
    sticky: bool = False


class ValueStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ValueRecord] = {}
        self.audit: list[dict] = []

    def key(self, object_class: str, field_k: str) -> tuple[str, str]:
        return (object_class, field_k)

    def get(self, object_class: str, field_k: str) -> ValueRecord | None:
        return self._records.get(self.key(object_class, field_k))

    def value(self, object_class: str, field_k: str, default: float = 0.0) -> float:
        rec = self.get(object_class, field_k)
        return rec.value if rec else default

    def set(self, object_class: str, field_k: str, value: float, source: str, by: str,
            world_ctx: str | None = None) -> tuple[bool, ValueRecord]:
        """Set/update a value. Returns (applied, record).

        If an existing record is sticky (human-set) and the new source is not human,
        the change is REJECTED here and must be routed to the HITL queue as a proposal.
        """
        k = self.key(object_class, field_k)
        existing = self._records.get(k)
        if existing and existing.sticky and source != "human":
            self.audit.append({"op": "reject_sticky", "key": k, "proposed": value,
                               "by": by, "source": source})
            return (False, existing)
        version = (existing.version + 1) if existing else 1
        rec = ValueRecord(object_class=object_class, field_k=field_k, value=float(value),
                          source=source, version=version, by=by, world_ctx=world_ctx,
                          sticky=(source == "human"))
        self._records[k] = rec
        self.audit.append({"op": "set", "key": k, "value": value, "source": source,
                           "version": version, "by": by, "world_ctx": world_ctx})
        return (True, rec)

    def values_for_field(self, field_k: str) -> dict[str, float]:
        return {cls: rec.value for (cls, fk), rec in self._records.items() if fk == field_k}

    def all(self) -> Iterator[ValueRecord]:
        return iter(self._records.values())

    def rollback_last(self) -> bool:
        if not self.audit:
            return False
        # naive single-step rollback for the propagate transaction
        last = self.audit.pop()
        if last.get("op") == "set":
            self._records.pop(tuple(last["key"]), None)
            return True
        return False
