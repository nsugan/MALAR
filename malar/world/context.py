"""WorldContext w = (world_id, snapshot_id, region S, t, source): the provenance
anchor every learned artifact references. A context resolves back to its snapshot
and region so the exact world it was learned in is reconstructable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from malar.world.graph import WorldGraph
from malar.world.sampling import Region
from malar.world.snapshot import SnapshotStore


@dataclass
class WorldContext:
    id: str
    world_id: str
    snapshot_id: str
    region_id: str
    region_indices: list[int]
    t: int
    source: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def make_region_id(region: Region) -> str:
    raw = json.dumps(sorted(region.indices)).encode()
    return "region_" + hashlib.sha256(raw).hexdigest()[:12]


def make_context(
    world_id: str,
    snapshot_id: str,
    region: Region,
    t: int,
    source: str,
    meta: dict | None = None,
) -> WorldContext:
    rid = make_region_id(region)
    cid = "ctx_" + hashlib.sha256(f"{world_id}|{snapshot_id}|{rid}|{t}".encode()).hexdigest()[:12]
    return WorldContext(
        id=cid,
        world_id=world_id,
        snapshot_id=snapshot_id,
        region_id=rid,
        region_indices=sorted(region.indices),
        t=t,
        source=source,
        meta=meta or {},
    )


def resolve_context(ctx: WorldContext, store: SnapshotStore) -> WorldGraph:
    """Reconstruct the exact region-world a context was learned in."""
    world = store.load(ctx.snapshot_id)
    return world.subgraph(ctx.region_indices)
