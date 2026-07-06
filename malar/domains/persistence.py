"""Cross-process persistence of a domain's learned state.

V2 kept the object registry + value/functional stores in-process, so a trained domain was
lost on restart (only the real Qdrant volume survived). This module snapshots the learned
state to `data/domains/{id}/state.json` and restores it when the engine is rebuilt, so
predictions survive a server restart without a live training session.

Persisted: the labelled object prototypes (per-class phi/h/g), the objective value records
(sticky human values included), and the functional/affordance records. The probabilistic
inference layer refits its likelihoods from these on load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from malar.fields.functional_store import AffordanceRecord
from malar.fields.value_store import ValueRecord
from malar.objects.registry import MalarObject

STATE_FILE = "state.json"


def state_path(root: Path, did: str) -> Path:
    return root / did / STATE_FILE


def save_engine_state(engine, path: Path) -> dict:
    objects = []
    for o in engine.c.registry.objects.values():
        if getattr(o, "candidate", False):
            continue
        objects.append({
            "id": o.id, "cls": o.cls,
            "phi": np.asarray(o.phi, dtype=float).tolist(),
            "h": np.asarray(o.h, dtype=float).tolist(),
            "g": np.asarray(o.g, dtype=float).tolist(),
            "omega": float(o.omega), "tau": int(o.tau),
            "world_ctx_id": o.world_ctx_id, "provenance": o.provenance,
        })
    values = [{
        "object_class": r.object_class, "field_k": r.field_k, "value": r.value,
        "source": r.source, "version": r.version, "by": r.by,
        "world_ctx": r.world_ctx, "sticky": r.sticky,
    } for r in engine.value_store.all()]
    affordances = [{
        "object_class": r.object_class, "dim": r.dim, "action": r.action,
        "enabled": r.enabled, "source": r.source, "version": r.version,
        "world_ctx": r.world_ctx, "sticky": r.sticky,
    } for r in engine.func_store.all()]
    state = {"version": 1, "objects": objects, "values": values, "affordances": affordances}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)
    return {"objects": len(objects), "values": len(values), "affordances": len(affordances)}


def load_engine_state(engine, path: Path) -> dict:
    if not path.exists():
        return {"loaded": False}
    try:
        state = json.loads(path.read_text())
    except Exception:
        return {"loaded": False, "error": "unreadable state"}
    for od in state.get("objects", []):
        engine.c.registry.add(MalarObject(
            id=od["id"], cls=od["cls"],
            phi=np.asarray(od["phi"], dtype=float),
            h=np.asarray(od["h"], dtype=float),
            g=np.asarray(od["g"], dtype=float),
            omega=od.get("omega", 1.0), tau=od.get("tau", 0),
            world_ctx_id=od.get("world_ctx_id"),
            provenance=od.get("provenance", "data")))
    for vd in state.get("values", []):
        key = engine.value_store.key(vd["object_class"], vd["field_k"])
        engine.value_store._records[key] = ValueRecord(
            object_class=vd["object_class"], field_k=vd["field_k"], value=vd["value"],
            source=vd["source"], version=vd["version"], by=vd["by"],
            world_ctx=vd.get("world_ctx"), sticky=vd.get("sticky", False))
    for ad in state.get("affordances", []):
        key = engine.func_store.key(ad["object_class"], ad["dim"], ad["action"])
        engine.func_store._records[key] = AffordanceRecord(
            object_class=ad["object_class"], dim=ad["dim"], action=ad["action"],
            enabled=ad["enabled"], source=ad["source"], version=ad["version"],
            world_ctx=ad.get("world_ctx"), sticky=ad.get("sticky", False))
    return {"loaded": True, "objects": len(state.get("objects", [])),
            "values": len(state.get("values", [])),
            "affordances": len(state.get("affordances", []))}
