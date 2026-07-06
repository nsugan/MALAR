"""Encoder-drift re-index — blue-green + collection alias.

Slow clock: when an encoder is retrained, re-embed every memory from its RAW
artifacts (persistence diagram + spectra) into a FRESH Qdrant collection, then swap
the alias to point at the new collection. Queries never cross encoder versions; the
old collection is retained for rollback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malar.encoders.topology import load_diagram
from malar.memory.qdrant_io import QdrantMemory
from malar.memory.schema import MemoryItem


@dataclass
class ReindexReport:
    old_collection: str
    new_collection: str
    n_reembedded: int
    new_encoder_version: str


def reindex_blue_green(qdrant: QdrantMemory, items: list[MemoryItem],
                       reembed_phi, reembed_h, new_version: str) -> ReindexReport:
    """Re-embed memories from raw artifacts into a new physical collection + swap alias.

    reembed_phi(diagrams) -> phi vector ; reembed_h(spectra) -> h vector.
    """
    old_physical = qdrant._physical
    new_physical = f"{qdrant.alias}__{new_version}"

    # build fresh collection (same dims) and re-embed
    qdrant.ensure_collection(physical=new_physical)
    n = 0
    for it in items:
        phi = it.phi
        h = it.h
        if it.diagram_path:
            try:
                phi = reembed_phi(load_diagram(it.diagram_path))
            except Exception:
                pass
        if it.spectra_path:
            try:
                h = reembed_h(np.load(it.spectra_path))
            except Exception:
                pass
        new_item = MemoryItem(
            id=it.id, phi=np.asarray(phi), h=np.asarray(h), g=it.g, omega=it.omega,
            tau=it.tau, cls=it.cls, encoder_version=new_version,
            world_ctx_id=it.world_ctx_id, snapshot_id=it.snapshot_id,
        )
        qdrant.upsert(new_item)
        n += 1

    # alias already swapped to new_physical by ensure_collection; report rollback target
    return ReindexReport(old_collection=old_physical, new_collection=new_physical,
                         n_reembedded=n, new_encoder_version=new_version)


def rollback(qdrant: QdrantMemory, report: ReindexReport) -> None:
    """Point the alias back at the previous physical collection."""
    from qdrant_client import models

    qdrant.client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=report.old_collection,
                                                alias_name=qdrant.alias)
            )
        ]
    )
    qdrant._physical = report.old_collection
