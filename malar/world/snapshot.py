"""WorldSnapshot(t): persisted W(t) — periodic full + deltas — so any WorldContext
is reconstructable. Full snapshots store adjacency+features; deltas store changes.
Persistence is to data/artifacts as .npz; the graph DB stores the (:WorldSnapshot)
node id+hash (see malar/memory/neo4j_io.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from malar.world.graph import WorldGraph


@dataclass
class WorldSnapshot:
    id: str
    t: int
    hash: str
    kind: str = "full"            # 'full' | 'delta'
    parent_id: str | None = None
    meta: dict = field(default_factory=dict)


class SnapshotStore:
    """File-backed snapshot store. Deterministic ids: snap_<t>_<hash>."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "snapshots_index.json"
        self.index: dict[str, dict] = {}
        if self._index_path.exists():
            try:
                self.index = json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, ValueError):
                # corrupt index (e.g. an interrupted/partial write) -- quarantine it and
                # start fresh rather than crashing the engine on construction.
                try:
                    self._index_path.replace(self._index_path.with_suffix(".json.corrupt"))
                except Exception:
                    pass
                self.index = {}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, indent=2))
        tmp.replace(self._index_path)          # atomic write (no partial/corrupt index)

    def save_full(self, world: WorldGraph) -> WorldSnapshot:
        h = world.content_hash()
        sid = f"snap_{world.t}_{h}"
        np.savez(
            self.root / f"{sid}.npz",
            node_ids=np.array(world.node_ids, dtype=object),
            features=world.features,
            adjacency=world.adjacency,
            t=world.t,
        )
        snap = WorldSnapshot(id=sid, t=world.t, hash=h, kind="full", meta=dict(world.meta))
        self.index[sid] = {"t": snap.t, "hash": snap.hash, "kind": snap.kind, "parent_id": None}
        self._save_index()
        return snap

    def load(self, snapshot_id: str) -> WorldGraph:
        path = self.root / f"{snapshot_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"snapshot {snapshot_id} not found at {path}")
        z = np.load(path, allow_pickle=True)
        return WorldGraph(
            node_ids=list(z["node_ids"]),
            features=z["features"],
            adjacency=z["adjacency"],
            t=int(z["t"]),
        )

    def exists(self, snapshot_id: str) -> bool:
        return (self.root / f"{snapshot_id}.npz").exists()
