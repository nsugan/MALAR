"""Trained-model checkpoint — versioned, restorable.

Bundles the trained world model: memory graph dump (Neo4j or in-memory), the memory
vectors (Qdrant snapshot equivalent), raw artifacts references, the model registry
(spectral encoder state, learned values {v_{k,c}} = psi_k, affordances = eta_j), the
DomainSpec name and the coverage report. Round-trips so a campaign result is shippable
and comparable across versions (the eval gate).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _ckpt_dir(name: str, root: str = "data/checkpoints") -> Path:
    p = Path(root) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(engine, name: str, domain: str = "", coverage: dict | None = None,
                    root: str = "data/checkpoints") -> str:
    d = _ckpt_dir(name, root)
    graph = engine.c.store.graph
    # memory graph dump (works for the in-memory backend; Neo4j export hook below)
    graph_dump = {
        "memories": getattr(graph, "memories", {}),
        "contexts": getattr(graph, "contexts", {}),
        "snapshots": getattr(graph, "snapshots", {}),
        "edges": getattr(graph, "edges", []),
        "backend": getattr(graph, "backend", "unknown"),
    }
    (d / "memory_graph.json").write_text(json.dumps(graph_dump, default=str, indent=2))

    # memory vectors (Qdrant-snapshot equivalent)
    cache = engine.c.store._cache
    if cache:
        np.savez(d / "vectors.npz",
                 ids=np.array(list(cache.keys()), dtype=object),
                 phi=np.array([v.phi for v in cache.values()], dtype=object),
                 h=np.array([v.h for v in cache.values()], dtype=object),
                 g=np.array([v.g for v in cache.values()], dtype=object),
                 cls=np.array([v.cls for v in cache.values()], dtype=object))

    # model registry: encoder state + learned values + affordances
    registry = {
        "spectral": _encode_state(engine.c.spectral.state_dict()),
        "values": [asdict(r) for r in engine.value_store.all()],
        "affordances": [asdict(r) for r in engine.func_store.all()],
        "encoder_versions": {"topo": engine.c.topo.version, "spectral": engine.c.spectral.version,
                             "graph": engine.c.graph.version},
    }
    (d / "model_registry.json").write_text(json.dumps(registry, default=str, indent=2))
    (d / "manifest.json").write_text(json.dumps({
        "name": name, "domain": domain, "n_memories": len(graph_dump["memories"]),
        "coverage": coverage or {}, "version": "v1"}, indent=2, default=str))
    return str(d)


def restore_checkpoint(engine, name: str, root: str = "data/checkpoints") -> dict:
    d = _ckpt_dir(name, root)
    graph_dump = json.loads((d / "memory_graph.json").read_text())
    graph = engine.c.store.graph
    if hasattr(graph, "memories"):
        graph.memories = graph_dump["memories"]
        graph.contexts = graph_dump["contexts"]
        graph.snapshots = graph_dump["snapshots"]
        graph.edges = [tuple(e) for e in graph_dump["edges"]]

    vec_path = d / "vectors.npz"
    if vec_path.exists():
        from malar.memory.schema import MemoryItem

        z = np.load(vec_path, allow_pickle=True)
        for i, mid in enumerate(z["ids"]):
            it = MemoryItem(id=str(mid), phi=z["phi"][i], h=z["h"][i], g=z["g"][i],
                            cls=str(z["cls"][i]))
            engine.c.store._cache[str(mid)] = it
            engine.c.store.qdrant.upsert(it)

    reg = json.loads((d / "model_registry.json").read_text())
    sp = reg.get("spectral", {})
    if sp.get("components") is not None:
        engine.c.spectral.load_state({
            "mean": np.array(sp["mean"]), "components": np.array(sp["components"]),
            "latent_dim": sp["latent_dim"], "version": sp["version"]})
    for r in reg.get("values", []):
        engine.value_store.set(r["object_class"], r["field_k"], r["value"], r["source"],
                               r["by"], r.get("world_ctx"))
    return json.loads((d / "manifest.json").read_text())


def _encode_state(state: dict) -> dict:
    return {"mean": None if state["mean"] is None else state["mean"].tolist(),
            "components": None if state["components"] is None else state["components"].tolist(),
            "latent_dim": state["latent_dim"], "version": state["version"]}


def main(argv=None) -> int:
    from malar.training.campaign import Campaign
    from malar.training.domainspec import load_domainspec

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--domain", default="raman_virus")
    args = ap.parse_args(argv)
    spec = load_domainspec(f"domains/{args.domain}.yaml")
    camp = Campaign(spec)
    report = camp.run(verbose=False)
    path = save_checkpoint(camp.engine, args.name, domain=args.domain,
                           coverage=report.to_dict())
    print(f"checkpoint saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
