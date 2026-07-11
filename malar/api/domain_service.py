"""DomainService — per-domain operations the UI routers call.

Wraps the active domain's MalarEngine to provide: a supervised one-at-a-time training
queue (preview reps + confirm/correct/skip), unsupervised auto training, learned-
knowledge readout, world-graph extraction, and folder batch inference + comparison.
Everything is scoped to a domain_id via DomainManager (no cross-domain bleed).
"""
from __future__ import annotations

import json
import math as _math
import time
import uuid
from pathlib import Path

import numpy as np

from malar.domains.manager import DomainManager, get_manager
from malar.encoders.change import cosine_similarity
from malar.fields.functional_field import region_action_set
from malar.memory.schema import MemoryItem, new_memory_id
from malar.world.adapters.synthetic import synthetic_class_cloud
from malar.world.context import make_context
from malar.world.graph import WorldGraph, cosine_knn_graph, knn_graph_from_points

# Generic default classes for the built-in synthetic demo when a domain configures none.
_GENERIC_CLASSES = ["class_a", "class_b", "class_c"]
# Supported data-file extensions (domain-agnostic): numeric tables/arrays, images, video.
_NUMERIC_EXT = {".npy", ".npz", ".csv", ".txt", ".asc", ".spc", ".dat", ".tsv"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _kind_for(suffix: str) -> str | None:
    s = suffix.lower()
    if s in _NUMERIC_EXT:
        return "features"
    if s in _IMAGE_EXT:
        return "image"
    if s in _VIDEO_EXT:
        return "video"
    return None


def _json_safe(obj):
    """Replace non-finite floats (inf/-inf/nan) with None so responses serialize."""
    if isinstance(obj, float):
        return obj if _math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class DomainService:
    def __init__(self, manager: DomainManager | None = None, llm_client=None):
        self.dm = manager or get_manager()
        self.llm = llm_client
        self._queues: dict[str, list] = {}
        self._index: dict[str, int] = {}
        self._results: dict[str, dict] = {}
        self._last_debug: dict[str, dict] = {}
        self._queue_source: dict[str, str] = {}
        self._llm_suggestion: dict[str, dict] = {}

    # -- training queue ------------------------------------------------
    def build_queue(self, did: str, n_per_class: int = 3, replicates: int = 16,
                    from_subset: list | None = None) -> list:
        meta = self.dm.get(did)
        items = []
        loaded = self._items_from_folders(meta, replicates) if meta.data_folders else []
        if from_subset:
            loaded = self._items_from_subset(from_subset, replicates) or loaded
        self._queue_source[did] = ("folder" if loaded else
                                   ("folder_empty" if meta.data_folders else "synthetic"))
        if loaded:
            items = loaded
        elif meta.data_folders:
            # A folder is configured but resolved to no usable files — surface this
            # instead of silently training on synthetic demo data.
            items = []
        else:
            # synthetic demo items — generic, class-separable feature clouds (no
            # domain-specific assumptions). Real data comes from configured folders.
            classes = meta.classes or _GENERIC_CLASSES
            for cls in classes:
                for s in range(n_per_class):
                    pts = synthetic_class_cloud(cls, n_dims=128, n_samples=replicates)
                    items.append({"id": f"{cls}_sample{s}", "label": cls, "points": pts,
                                  "kind": "features"})
        self._queues[did] = items
        self._index[did] = 0
        return items

    def _resolve_folder(self, folder: str):
        """Resolve a configured folder to a path that exists in THIS environment.

        Handles Windows paths from the Configure tab by mapping the repo tail onto the
        container mount (/app) and a few common roots.
        """
        f = str(folder or "").strip()
        if not f:
            return None
        norm = f.replace("\\", "/").rstrip("/")
        name = norm.split("/")[-1]
        cands = [f, norm]
        if "MALAR_V2/" in norm:
            cands.append("/app/" + norm.split("MALAR_V2/", 1)[1])
        cands += ["/app/" + name, str(self.dm.s.data_dir.parent / name),
                  str(Path.cwd() / name), name]
        for c in cands:
            try:
                if c and Path(c).exists():
                    return c
            except Exception:
                continue
        return None

    def _items_from_folders(self, meta, replicates):
        items = []
        for folder in meta.data_folders:
            root = self._resolve_folder(folder)
            if not root:
                continue
            for p in sorted(Path(root).rglob("*")):
                if not p.is_file():
                    continue
                kind = _kind_for(p.suffix)
                if kind is None:
                    continue
                pts = self._load_points(p, replicates=replicates)
                if pts is None:
                    continue
                items.append({"id": f"{p.parent.name}/{p.name}", "label": p.parent.name,
                              "points": pts, "kind": kind, "path": str(p)})
        return items

    def _items_from_subset(self, subset, replicates):
        items = []
        for entry in subset:
            p = Path(entry.get("path", ""))
            pts = self._load_points(p) if p.exists() else None
            if pts is not None:
                items.append({"id": entry.get("name", p.name),
                              "label": entry.get("label") or p.parent.name,
                              "points": pts, "kind": _kind_for(p.suffix) or "features",
                              "path": str(p)})
        return items

    def _load_points(self, p: Path, n_bands: int = 128, replicates: int = 16):
        """Load ONE data file -> (replicates, n_bands) feature cloud, domain-agnostically.

        Handles numeric tables/arrays (.npy/.npz/.csv/.txt/.asc/.spc/.dat/.tsv) and,
        best-effort, images/video when an optional decoder (PIL / imageio) is installed.
        Every input is reduced to a fixed-length feature vector, L2-normalised to match
        the encoder corpus, then jittered into a small replicate cloud so the topology /
        graph encoders have non-degenerate content. Returns None if unreadable.
        """
        try:
            suffix = p.suffix.lower()
            if suffix in _IMAGE_EXT:
                vec = self._image_to_vector(p, n_bands)
            elif suffix in _VIDEO_EXT:
                return self._video_to_cloud(p, n_bands, replicates)
            elif suffix in (".npy", ".npz"):
                arr = np.load(p)
                if suffix == ".npz":
                    arr = arr[arr.files[0]]
                arr = np.asarray(arr, dtype=float)
                vec = arr.mean(axis=0) if arr.ndim == 2 else arr.ravel()
            else:
                # generic numeric text: take the value series (last numeric column, or the
                # sole column) and resample onto a fixed grid. Works for x,y signal exports
                # and single-column series alike — no instrument-specific assumptions.
                xs, ys = [], []
                for line in p.read_text(errors="ignore").splitlines():
                    nums = []
                    for t in line.replace(",", " ").split():
                        try:
                            nums.append(float(t))
                        except ValueError:
                            pass
                    if len(nums) >= 2:
                        xs.append(nums[-2])
                        ys.append(nums[-1])
                    elif len(nums) == 1:
                        xs.append(float(len(xs)))
                        ys.append(nums[0])
                if len(ys) < 4:
                    return None
                xs, ys = np.asarray(xs), np.asarray(ys)
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order]
                vec = np.interp(np.linspace(xs.min(), xs.max(), n_bands), xs, ys)
            if vec is None:
                return None
            vec = np.asarray(vec, dtype=float).ravel()
            # fit every input to a common width so all items line up for the encoders
            if vec.shape[0] != n_bands and vec.shape[0] > 1:
                vec = np.interp(np.linspace(0, 1, n_bands),
                                np.linspace(0, 1, vec.shape[0]), vec)
            nrm = np.linalg.norm(vec)
            if nrm > 0:
                vec = vec / nrm
            rng = np.random.default_rng(abs(hash(p.name)) % (2**32))
            return vec[None, :] + rng.normal(0, 0.003, (replicates, vec.shape[0]))
        except Exception:
            return None

    def _image_to_vector(self, p: Path, n_bands: int):
        """Best-effort image -> fixed-length grayscale feature vector (optional decoder)."""
        try:
            try:
                from PIL import Image
                a = np.asarray(Image.open(p).convert("L"), dtype=float)
            except Exception:
                import imageio.v3 as iio
                a = np.asarray(iio.imread(p), dtype=float)
                if a.ndim == 3:
                    a = a.mean(axis=2)
            flat = a.ravel()
            if flat.size == 0:
                return None
            return np.interp(np.linspace(0, 1, n_bands), np.linspace(0, 1, flat.size), flat)
        except Exception:
            return None

    def _video_to_cloud(self, p: Path, n_bands: int, replicates: int):
        """Best-effort video -> per-frame feature cloud (optional decoder). None if absent."""
        try:
            import imageio.v3 as iio
            frames = np.asarray(iio.imread(p, index=None), dtype=float)
            if frames.ndim == 4:
                frames = frames.mean(axis=3)
            n = frames.shape[0]
            idx = np.linspace(0, n - 1, min(replicates, n)).astype(int)
            rows = [np.interp(np.linspace(0, 1, n_bands),
                              np.linspace(0, 1, frames[i].size), frames[i].ravel())
                    for i in idx]
            cloud = np.asarray(rows, dtype=float)
            norms = np.linalg.norm(cloud, axis=1, keepdims=True)
            return cloud / np.where(norms > 0, norms, 1.0)
        except Exception:
            return None

    def _item_world(self, did: str, item: dict) -> WorldGraph:
        pts = np.asarray(item["points"], dtype=float)
        # Domain-agnostic graph build: high-dimensional feature vectors -> cosine-similarity
        # kNN; low-dimensional spatial point sets -> Euclidean kNN. No adapter-type coupling.
        if pts.shape[1] >= 8:
            A = cosine_knn_graph(pts, k=min(6, max(1, pts.shape[0] - 1)))
        else:
            A = knn_graph_from_points(pts, k=min(6, max(1, pts.shape[0] - 1)))
        return WorldGraph(node_ids=[f"{item['id']}_{i}" for i in range(pts.shape[0])],
                          features=pts, adjacency=A)

    def _ensure_corpus_fit(self, did: str, eng) -> None:
        """Fit the spectral encoder on the WHOLE queue corpus (one row per item),
        not a single item — so the PCA basis is meaningful for real data."""
        if eng._spectral_fit:
            return
        q = self._queues.get(did) or []
        if len(q) >= 2:
            corpus = np.vstack([np.asarray(it["points"], dtype=float).mean(axis=0, keepdims=True)
                                for it in q])
            eng.c.spectral.fit(corpus)
            eng._spectral_fit = True

    def next_item(self, did: str) -> dict | None:
        q = self._queues.get(did) or self.build_queue(did)
        idx = self._index.get(did, 0)
        if idx >= len(q):
            return None
        return {"index": idx, "total": len(q), "item_id": q[idx]["id"],
                "label_hint": q[idx]["label"]}

    def preview(self, did: str) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        q = self._queues.get(did) or self.build_queue(did)
        idx = self._index.get(did, 0)
        if idx >= len(q):
            return {"done": True}
        item = q[idx]
        world = self._item_world(did, item)
        self._ensure_corpus_fit(did, eng)
        tr = eng.c.topo.encode(world)
        sr = eng.c.spectral.encode(world)
        gr = eng.c.graph.encode(world)
        ident = eng.c.identifier.identify(tr.phi, sr.h, gr.graph_emb, tr.summary, t=idx)
        proposed = ident.cls or item["label"]
        # LLM-assisted identification (object + functionalities) — OPT-IN per domain so
        # training stays fast; only fires when llm_assist is enabled in Configure/Train.
        meta_la = self.dm.get(did)
        if self.llm is not None and getattr(meta_la, "llm_assist", False):
            cands = [h.obj.cls for h in eng.c.registry.topk(tr.phi, sr.h, gr.graph_emb, k=5)]
            llm_sug = self._llm_identify_item(did, item, tr.summary, cands)
        else:
            llm_sug = {"used_llm": False, "rationale": "llm-assist off (enable in Configure/Train)"}
        self._llm_suggestion[did] = llm_sug
        objectives = [{"class": proposed, "ctx": 1.0}]
        rvals = {k: of.evaluate(objectives)[0] for k, of in eng.c.objective_fields.items()}
        affs = region_action_set(objectives, eng.c.functional_fields) or \
            [f.spec.actions[0] for f in eng.c.functional_fields]
        h1 = tr.diagrams[1].tolist() if len(tr.diagrams) > 1 else []
        import numpy as _np
        self._last_debug[did] = {
            "stage": "preview", "item_id": item["id"],
            "phi_dim": int(tr.phi.size), "phi_norm": round(float(_np.linalg.norm(tr.phi)), 4),
            "h_dim": int(sr.h.size), "h_norm": round(float(_np.linalg.norm(sr.h)), 4),
            "g_dim": int(gr.graph_emb.size), "g_norm": round(float(_np.linalg.norm(gr.graph_emb)), 4),
            "topology_summary": tr.summary, "graph_summary": gr.summary,
            "identification": {"proposed": proposed, "matched": bool(ident.matched),
                               "method": ident.method, "score": round(float(ident.score), 4),
                               "conformal_ok": bool(ident.conformal_ok),
                               "rationale": ident.rationale},
            "r_values": {k: round(float(v), 4) for k, v in rvals.items()},
            "action_set": affs,
            "input_shape": list(_np.asarray(item["points"]).shape),
        }
        return {
            "done": False, "index": idx, "total": len(q), "item_id": item["id"],
            "label_hint": item["label"],
            "graph": self._graph_payload(world, limit=40),
            "topology": {"summary": tr.summary, "h1_diagram": h1},
            "hyper": {"spectrum": np.asarray(item["points"]).mean(axis=0).tolist(),
                      "latent": sr.h.tolist(), "recon_error": sr.recon_error},
            "identification": {"proposed": proposed, "matched": ident.matched,
                               "method": ident.method, "score": round(float(ident.score), 3),
                               "llm": llm_sug},
            "values": {k: round(float(v), 3) for k, v in rvals.items()},
            "affordances": affs,
        }

    def confirm(self, did: str, decision: str, corrections: dict | None = None) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        q = self._queues.get(did) or self.build_queue(did)
        idx = self._index.get(did, 0)
        if idx >= len(q):
            return {"done": True}
        item = q[idx]
        result = {"decision": decision, "item_id": item["id"]}
        if decision != "skip":
            label = (corrections or {}).get("label") or item["label"]
            world = self._item_world(did, item)
            if not eng._spectral_fit:
                eng.c.spectral.fit(world.features)
            eng._spectral_fit = True
            tr = eng.c.topo.encode(world, key=f"{did}_{item['id']}".replace("/", "_"))
            sr = eng.c.spectral.encode(world, key=f"{did}_{item['id']}".replace("/", "_"))
            gr = eng.c.graph.encode(world)
            snap = eng.c.snapshots.save_full(world)
            region = eng.c.sampler.sample(world)
            ctx = make_context(did, snap.id, region, world.t, "supervised")
            eng.c.store.graph.upsert_snapshot({"id": snap.id, "t": snap.t, "hash": snap.hash})
            eng.c.store.graph.upsert_world_context(ctx.to_dict())
            # learn values + affordances (human-sourced if a correction)
            src_human = decision == "correct"
            for k, of in eng.c.objective_fields.items():
                if src_human and corrections and "value" in corrections:
                    of.set_human_value(label, float(corrections["value"]), world_ctx=ctx.id)
                else:
                    of.learn_from_data(label, 1.0, world_ctx=ctx.id)
            for ff in eng.c.functional_fields:
                ff.learn_affordance(label, ff.spec.actions[0], True,
                                    source="human" if src_human else "data", world_ctx=ctx.id)
            # LLM-proposed affordances (functionalities) for this object class
            sug = self._llm_suggestion.get(did, {})
            for act in (sug.get("affordances") or []):
                if not isinstance(act, str):
                    continue
                ff = eng.c.functional_fields[0]
                ff.learn_affordance(label, act, True, source="llm", world_ctx=ctx.id)
            # Mirror the learned values/affordances into the memory graph as
            # (:Object)-[:HAS_VALUE]->(:Objective) / (:Object)-[:AFFORDS]->(:Action)
            # so they exist in the graph DB and render in the World Graph. Best-effort:
            # the in-process ValueStore/FunctionalStore remain the source of truth. (V4 fix ❷)
            try:
                g = eng.c.store.graph
                for ok in eng.c.objective_fields:
                    vrec = eng.value_store.get(label, ok)
                    if vrec is not None:
                        g.upsert_value(label, ok, vrec.value, vrec.source, vrec.version, ctx.id)
                for arec in eng.func_store.all():
                    if arec.object_class == label and arec.enabled:
                        g.upsert_affordance(label, arec.dim, arec.action, arec.source,
                                            arec.version, ctx.id)
            except Exception:
                pass
            cand = MemoryItem(id=new_memory_id(tr.phi, idx, ctx.id), phi=tr.phi, h=sr.h,
                              g=gr.graph_emb, cls=label, tau=idx, world_ctx_id=ctx.id,
                              snapshot_id=snap.id, diagram_path=tr.artifact_path,
                              spectra_path=sr.artifact_path)
            dec = eng.c.curator.on_candidate(cand, t=idx, goal_gain=1.0)
            eng.register_object(label, tr.phi, sr.h, gr.graph_emb, idx, ctx.id)
            # Active training agents: LLM-proposed extra agents (reasoner-driven) + the
            # modality-routed learner (diffusion for image/video, RL bandit for text).
            from malar.training.extra_agents import process_training_item
            agents_out = process_training_item(eng, self.dm.get(did), item, tr.summary,
                                               gr.graph_emb, label, ctx.id, self.llm)
            # Parent field-agents run their VALIDATED factory-agents (gated per domain);
            # outputs feed back into the fields and are stored in the graph DB.
            factory_out = []
            if getattr(self.dm.get(did), "agents_active", False):
                try:
                    from malar.agents.runner import run_validated_agents
                    agent_ctx = {"features": [list(map(float, r)) for r in world.features[:12]],
                                 "label": label, "summary": tr.summary}
                    factory_out = run_validated_agents(eng, agent_ctx, did, ctx.id, label)
                except Exception:
                    factory_out = []
            self._last_debug[did] = {
                "stage": "confirm", "item_id": item["id"], "label": label,
                "curator_action": dec.action, "curator_mem_id": dec.mem_id,
                "novelty": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                            for k, v in dec.novelty.items()},
                "world_ctx": ctx.id, "snapshot": snap.id, "provenance": result.get("provenance"),
                "training_agents": agents_out, "factory_agents": factory_out}
            result.update({"committed": True, "label": label, "memory_action": dec.action,
                           "provenance": "human" if src_human else "data",
                           "modality": agents_out.get("modality"),
                           "training_agents": agents_out, "factory_agents": factory_out})
        self._index[did] = idx + 1
        result["next_index"] = self._index[did]
        result["remaining"] = max(0, len(q) - self._index[did])
        # On finishing the supervised queue, persist learned state for restart-survival.
        if result["remaining"] == 0:
            try:
                self.dm.save_engine_state(did)
            except Exception:
                pass
        return result

    # -- learned knowledge --------------------------------------------
    def learned(self, did: str) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        vs = eng.value_store
        values: dict[str, dict] = {}
        for rec in vs.all():
            values.setdefault(rec.field_k, {})[rec.object_class] = {
                "value": round(rec.value, 3), "source": rec.source, "version": rec.version}
        affs: dict[str, list] = {}
        for rec in eng.func_store.all():
            if rec.enabled:
                affs.setdefault(rec.object_class, []).append(
                    {"dim": rec.dim, "action": rec.action, "source": rec.source})
        objectives = [{"key": of.k, "target": of.spec.target} for of in eng.c.objective_fields.values()]
        return {"objectives": objectives, "values": values, "affordances": affs,
                "classes": eng.c.registry.classes()}

    # -- world graph ---------------------------------------------------
    def graph(self, did: str, filter_type: str | None = None, limit: int = 200) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        g = eng.c.store.graph
        nodes, edges = [], []
        seen = set()
        for m in g.all_memories()[:limit]:
            nodes.append({"id": m["id"], "type": "memory", "label": m.get("class"),
                          "omega": round(float(m.get("omega", 1.0)), 2)})
            seen.add(m["id"])
        for o in eng.c.registry.objects.values():
            if not o.candidate:
                nodes.append({"id": o.id, "type": "object", "label": o.cls})
                seen.add(o.id)
        for ctx_id, ctx in getattr(g, "contexts", {}).items():
            nodes.append({"id": ctx_id, "type": "world_context", "label": ctx.get("source")})
            seen.add(ctx_id)
        # grounded knowledge nodes (V4 fix ❷): object-class / objective / action, so the
        # HAS_VALUE / AFFORDS edges below have endpoints to render against.
        for oid, o in getattr(g, "objects", {}).items():
            nodes.append({"id": oid, "type": "object_class", "label": o.get("class")})
            seen.add(oid)
        for kid, k in getattr(g, "objectives", {}).items():
            nodes.append({"id": kid, "type": "objective", "label": k.get("key")})
            seen.add(kid)
        for aid, a in getattr(g, "actions", {}).items():
            nodes.append({"id": aid, "type": "action", "label": a.get("action")})
            seen.add(aid)
        for rel, src, dst, props in getattr(g, "edges", []):
            if src in seen and dst in seen:
                if filter_type and rel != filter_type:
                    continue
                edge = {"source": src, "target": dst, "rel": rel}
                if rel == "HAS_VALUE" and props:
                    edge["value"] = props.get("value")
                edges.append(edge)
        return {"nodes": nodes[:limit], "edges": edges[: limit * 3],
                "rel_types": ["SIMILAR", "DERIVED_FROM", "TEMPORAL_NEXT", "GROUNDED_IN",
                              "OF", "HAS_VALUE", "AFFORDS"]}

    def node_detail(self, did: str, nid: str) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        g = eng.c.store.graph
        m = g.get_memory(nid)
        detail = {"id": nid}
        if m:
            detail.update({"type": "memory", "class": m.get("class"),
                           "world_ctx_id": m.get("world_ctx_id"),
                           "snapshot_id": m.get("snapshot_id"), "omega": m.get("omega")})
            it = eng.c.store.get_item(nid)
            if it is not None:
                neighbors = []
                for other in eng.c.store.graph.all_memories():
                    oit = eng.c.store.get_item(other["id"])
                    if oit is None or oit.id == nid:
                        continue
                    neighbors.append({"id": oit.id, "class": oit.cls,
                                      "sim": round(cosine_similarity(it.phi, oit.phi), 3)})
                neighbors.sort(key=lambda x: x["sim"], reverse=True)
                detail["vector_neighbors"] = neighbors[:5]
            return detail
        obj = eng.c.registry.objects.get(nid)
        if obj:
            detail.update({"type": "object", "class": obj.cls, "provenance": obj.provenance,
                           "world_ctx_id": obj.world_ctx_id})
        return detail

    def _graph_payload(self, world: WorldGraph, limit: int = 40) -> dict:
        n = min(limit, world.n_nodes)
        nodes = [{"id": world.node_ids[i], "label": str(i)} for i in range(n)]
        edges = []
        for i, j, w in world.edges():
            if i < n and j < n:
                edges.append({"source": world.node_ids[i], "target": world.node_ids[j],
                              "w": round(float(w), 2)})
        return {"nodes": nodes, "edges": edges[: limit * 3]}

    # -- folder batch inference + comparison --------------------------
    def infer_folder(self, did: str, path: str) -> dict:
        from malar.inference.service import InferenceService
        from malar.training.folder_analyzer import analyze_folder

        eng = self.dm.engine(did, llm_client=self.llm)
        svc = InferenceService(eng, llm_client=self.llm)
        rows = []
        report = analyze_folder(path) if (path and path.strip()) else None
        for it in (report.items if report else []):
            if it.kind not in ("features", "vector", "array", "image", "video",
                               "spectra", "cube"):   # spectra/cube kept for legacy reports
                continue
            pts = self._load_points(Path(it.path))
            if pts is None:
                continue
            res = svc.infer(image=pts)
            rows.append({"input_id": it.name, "label_hint": it.label,
                         "identified_object": (res.get("matched_objects") or [{}])[0].get("class"),
                         "action": res.get("action"), "confidence": res.get("confidence"),
                         "ood": res.get("ood")})
        if not rows:  # demo fallback: synthesize a small generic test set from the classes
            for cls in (self.dm.get(did).classes or _GENERIC_CLASSES):
                pts = synthetic_class_cloud(cls, n_dims=128, n_samples=16)
                res = svc.infer(image=pts)
                rows.append({"input_id": f"synthetic_{cls}", "label_hint": cls,
                             "identified_object": (res.get("matched_objects") or [{}])[0].get("class"),
                             "action": res.get("action"), "confidence": res.get("confidence"),
                             "ood": res.get("ood")})
        run_id = uuid.uuid4().hex[:8]
        out = {"run_id": run_id, "domain": did, "created": time.time(),
               "rows": rows, "folder": path}
        self._results[run_id] = out
        rdir = self.dm.root / did / "results"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"{run_id}.json").write_text(json.dumps(out, indent=2, default=str))
        self._write_csv(rdir / f"{run_id}.csv", rows)
        return {"run_id": run_id, "n_items": len(rows)}

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        import csv

        if not rows:
            path.write_text("")
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def results(self, did: str, run_id: str) -> dict:
        if run_id in self._results:
            return self._results[run_id]
        p = self.dm.root / did / "results" / f"{run_id}.json"
        if p.exists():
            return json.loads(p.read_text())
        return {"error": "run not found"}

    def compare(self, did: str, run_id: str) -> dict:
        res = self.results(did, run_id)
        rows = res.get("rows", [])
        eng = self.dm.engine(did, llm_client=self.llm)
        trained = eng.c.registry.classes()
        dist: dict[str, int] = {}
        ood = 0
        for r in rows:
            c = r.get("identified_object") or "unknown"
            dist[c] = dist.get(c, 0) + 1
            if r.get("ood"):
                ood += 1
        overlap = sorted(set(dist) & set(trained))
        return {"class_distribution": dist, "trained_classes": trained,
                "ood_rate": round(ood / max(1, len(rows)), 3),
                "overlap": overlap,
                "n_items": len(rows)}


    # -- input inspector (examine a chosen file thoroughly) -----------
    def queue_list(self, did: str) -> dict:
        q = self._queues.get(did) or self.build_queue(did)
        items = [{"index": i, "item_id": it["id"], "label": it["label"],
                  "shape": list(np.asarray(it["points"]).shape),
                  "path": it.get("path"), "kind": it.get("kind", "features")}
                 for i, it in enumerate(q)]
        src = self._queue_source.get(did, "synthetic")
        note = {"folder": "loaded from configured data folder(s)",
                "folder_empty": "a folder is configured but no readable data files were found "
                                "(check the path resolves inside the server, e.g. /app/<name>)",
                "synthetic": "no data folder configured — using built-in synthetic demo data"}.get(src, "")
        return {"items": items, "total": len(items), "source": src, "note": note}

    def inspect_input(self, did: str, index: int) -> dict:
        import numpy as np

        eng = self.dm.engine(did, llm_client=self.llm)
        q = self._queues.get(did) or self.build_queue(did)
        if index < 0 or index >= len(q):
            return {"error": "index out of range", "total": len(q)}
        item = q[index]
        pts = np.asarray(item["points"], dtype=float)
        world = self._item_world(did, item)
        self._ensure_corpus_fit(did, eng)

        # encode
        tr = eng.c.topo.encode(world)
        sr = eng.c.spectral.encode(world)
        gr = eng.c.graph.encode(world)
        ident = eng.c.identifier.identify(tr.phi, sr.h, gr.graph_emb, tr.summary, t=index)
        proposed = ident.cls or item["label"]

        # novelty preview vs nearest memory (no commit)
        novelty = None
        hits = eng.c.retriever.retrieve(tr.phi, sr.h, gr.graph_emb, t=index, k=1)
        if hits:
            near = eng.c.store.get_item(hits[0].mem_id)
            if near is not None and near.phi.size > 1:
                from malar.encoders.change import change_stats
                novelty = {k: round(float(v), 4) for k, v in change_stats(
                    {"phi": tr.phi, "h": sr.h, "g": gr.graph_emb},
                    {"phi": near.phi, "h": near.h, "g": near.g}).items()}

        # raw matrix (capped) + stats
        rmax, cmax = 32, 200
        matrix = pts[:rmax, :cmax].round(5).tolist()
        stats = {"min": round(float(pts.min()), 5), "max": round(float(pts.max()), 5),
                 "mean": round(float(pts.mean()), 5), "std": round(float(pts.std()), 5)}
        per_row_norm = [round(float(np.linalg.norm(r)), 4) for r in pts[:rmax]]

        # structure: adjacency (capped) + degrees
        n = min(world.n_nodes, rmax)
        adj = world.adjacency[:n, :n].round(4).tolist()
        degrees = [round(float(x), 4) for x in world.degree()[:n]]

        # spectral detail
        Z = eng.c.spectral.encode_matrix(pts)
        recon = eng.c.spectral.decode_matrix(Z)
        latent = Z[:rmax, :].round(4).tolist()
        mean_spec = pts.mean(axis=0)[:cmax].round(5).tolist()
        recon_mean = recon.mean(axis=0)[:cmax].round(5).tolist()

        return _json_safe({
            "file": {"item_id": item["id"], "label": item["label"], "path": item.get("path"),
                     "kind": item.get("kind", "features"), "shape": list(pts.shape),
                     "dtype": str(pts.dtype), "n_rows": int(pts.shape[0]),
                     "n_cols": int(pts.shape[1])},
            "raw": {"matrix": matrix, "rows_shown": len(matrix), "cols_shown": len(matrix[0]) if matrix else 0,
                    "stats": stats, "per_row_norm": per_row_norm},
            "spectra": {"mean": mean_spec, "samples": pts[:8, :cmax].round(5).tolist(),
                        "n_bands": int(pts.shape[1])},
            "structure": {"graph": self._graph_payload(world, limit=40),
                          "adjacency": adj, "degrees": degrees, "n_nodes": world.n_nodes,
                          "n_edges": len(world.edges())},
            "topology": {"summary": tr.summary,
                         "h0": (tr.diagrams[0].round(4).tolist() if len(tr.diagrams) > 0 else []),
                         "h1": (tr.diagrams[1].round(4).tolist() if len(tr.diagrams) > 1 else []),
                         "h2": (tr.diagrams[2].round(4).tolist() if len(tr.diagrams) > 2 else []),
                         "phi_norm": round(float(np.linalg.norm(tr.phi)), 4)},
            "spectral": {"latent": latent, "region_h": sr.h.round(4).tolist(),
                         "recon_error": round(float(sr.recon_error), 5),
                         "mean_spectrum": mean_spec, "reconstructed_mean": recon_mean},
            "graph_emb": {"vector": gr.graph_emb[:48].round(4).tolist(), "summary": gr.summary,
                          "g_norm": round(float(np.linalg.norm(gr.graph_emb)), 4)},
            "process": {"identification": {"proposed": proposed, "matched": bool(ident.matched),
                                           "method": ident.method, "score": round(float(ident.score), 4),
                                           "conformal_ok": bool(ident.conformal_ok),
                                           "rationale": ident.rationale},
                        "novelty_vs_nearest": novelty,
                        "r_values": {k: round(float(of.evaluate([{"class": proposed, "ctx": 1.0}])[0]), 4)
                                     for k, of in eng.c.objective_fields.items()},
                        "affordances": region_action_set([{"class": proposed}], eng.c.functional_fields)
                        or [f.spec.actions[0] for f in eng.c.functional_fields]},
        })

    # -- Orchestrator/Planner: LLM-derived plan from the description ---
    def plan_domain(self, did: str, apply: bool = False, folder_summary: str | None = None) -> dict:
        from malar.agents.planner import DomainPlanner

        meta = self.dm.get(did)
        # Consolidated domain + data description is embedded in the planner's query.
        plan = DomainPlanner(self.llm, context=meta.consolidated_context()).plan_domain(
            classes=meta.classes, folder_summary=folder_summary,
            adapter_type=meta.adapter_type)
        if apply:
            objectives = [{"key": o["key"], "target": o.get("target", 0.8)}
                          for o in plan.get("objectives", []) if o.get("key")]
            functionals = [{"dim": f["dim"], "actions": list(f.get("actions", []))}
                           for f in plan.get("functionals", []) if f.get("dim")]
            self.dm.update_config(did, objectives=objectives or None,
                                  functionals=functionals or None,
                                  plan=plan, extra_agents=plan.get("extra_agents", []))
        else:
            # remember the proposal even if not applied
            self.dm.update_config(did, plan=plan, extra_agents=plan.get("extra_agents", []))
        return plan

    def _llm_identify_item(self, did: str, item: dict, summary: dict, candidates: list[str]) -> dict:
        """Per-item LLM identification of object + functionalities (training)."""
        from malar.agents.planner import DomainPlanner

        meta = self.dm.get(did)
        eng = self.dm.engine(did, llm_client=self.llm)
        dims = [f.spec.dim for f in eng.c.functional_fields]
        # Consolidated domain + data description is embedded in the identify query.
        return DomainPlanner(self.llm, context=meta.consolidated_context()).suggest_object(
            feature_summary=summary, candidates=candidates, functional_dims=dims)

    # -- per-agent monitoring (the 11 MALAR agents) -------------------
    def agents_snapshot(self, did: str) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        snap = self.debug_snapshot(did)
        le = snap.get("last_event", {}) or {}
        ls = getattr(eng, "last_state", None)
        src = self._queue_source.get(did, "synthetic")
        qn = len(self._queues.get(did, []) or [])

        # group objective-field values per objective k (one A_k agent each)
        by_field: dict[str, dict] = {}
        for r in snap["value_agents"]["records"]:
            by_field.setdefault(r["objective_k"], {})[r["class"]] = r["value"]
        objective_subagents = [{"name": f"A_k[{k}]", "objective": k, "values": v,
                                "n_classes": len(v)} for k, v in by_field.items()]

        # group affordances per functional dim (one B_j agent each)
        by_dim: dict[str, dict] = {}
        for r in snap["functional_agents"]["records"]:
            by_dim.setdefault(r["dim"], {}).setdefault(r["class"], []).append(r["action"])
        functional_subagents = [{"name": f"B_j[{d}]", "dim": d,
                                 "affordances": cls, "n_classes": len(cls)}
                                for d, cls in by_dim.items()]

        def ls_field(attr, default=None):
            return getattr(ls, attr, default) if ls is not None else default

        pol = ls_field("policy_decision")
        meta = self.dm.get(did)
        plan = (meta.plan or {}) if meta else {}
        agents = {
            "orchestrator": {
                "label": "Orchestrator / Planner",
                "role": "Drives the loop; allocates budget B; LLM-plans the domain (objectives "
                        "R, functional dims F, extra agents) from the data description; tracks coverage.",
                "state": {"queue_source": src, "queue_size": qn,
                          "region_budget": eng.c.sampler.budget,
                          "compress_every": eng.compress_every,
                          "memory_size": eng.memory_size(),
                          "classes_learned": len(eng.c.registry.classes()),
                          "plan_source": plan.get("source"), "plan_used_llm": plan.get("used_llm"),
                          "plan_rationale": plan.get("rationale")},
            },
            "perception": {
                "label": "Perception / Ingestion",
                "role": "Pulls a batch, updates W(t), runs F_rep; LLM-derived processing steps "
                        "beyond the graph/topology/spectral encoders.",
                "state": {"last_input_shape": le.get("input_shape"),
                          "world_n_nodes": (ls_field("world").n_nodes if ls_field("world") else None),
                          "adapter": eng.adapter.__class__.__name__,
                          "processing_steps": plan.get("processing_steps")},
            },
            "topology": {
                "label": "Topology",
                "role": "Computes phi_topo(S,t) via persistence homology.",
                "state": {**snap["encoders"]["topology"],
                          "last_summary": le.get("topology_summary"),
                          "last_phi_norm": le.get("phi_norm", le.get("phi_norm"))},
            },
            "objective_fields": {
                "label": "Objective-Field Agents Aₖ",
                "role": "One per objective: identify objects, compute R^(k), learn {v_k,c} "
                        "from data + humans (sticky).",
                "state": {"n_objectives": len(objective_subagents)},
                "sub_agents": objective_subagents,
            },
            "functional_fields": {
                "label": "Functional-Field Agents Bⱼ",
                "role": "One per functional dim: learn affordances F_j(o,t) from data + humans.",
                "state": {"n_dims": len(functional_subagents)},
                "sub_agents": functional_subagents,
            },
            "fields_coordinator": {
                "label": "Fields Coordinator",
                "role": "Aggregate R, diffuse V, assemble the F-afforded action set.",
                "state": {"R_values": le.get("r_values") or (ls_field("r_values") or {}),
                          "action_set": (ls_field("action_set") or []),
                          "v_max": (float(__import__("numpy").max(ls.v_field))
                                    if ls is not None and ls.v_field is not None and ls.v_field.size
                                    else None)},
            },
            "memory_inline": {
                "label": "Memory (inline)",
                "role": "Per-tick retrieval + novelty read.",
                "state": {"novelty": le.get("novelty"),
                          "memory_size": eng.memory_size()},
            },
            "policy": {
                "label": "Policy / Action",
                "role": "Choose actions within the F-afforded set, ranked by R,V; justify.",
                "state": {"last_action": getattr(pol, "action", None),
                          "rationale": getattr(pol, "rationale", None),
                          "note": None if pol else "runs during auto / seamless mode"},
            },
            "generation": {
                "label": "Generation",
                "role": "Reverse world-state generation, counterfactuals.",
                "state": {"status": "available (used by recall / counterfactuals)"},
            },
            "critic": {
                "label": "Critic / Validation",
                "role": "Conformal + OOD gate before any memory write or action.",
                "state": {**snap["gates"]["conformal"],
                          "last_conformal_ok": le.get("identification", {}).get("conformal_ok")},
            },
            "curator": {
                "label": "Memory Curator (async)",
                "role": snap["curator"]["description"],
                "state": {**snap["curator"]["config"],
                          "memory_size": snap["curator"]["memory_size"]},
                "activity": snap["curator"]["audit_tail"][-8:],
            },
        }
        order = ["orchestrator", "perception", "topology", "objective_fields",
                 "functional_fields", "fields_coordinator", "memory_inline", "policy",
                 "generation", "critic", "curator"]
        for i, ea in enumerate(plan.get("extra_agents", []) or []):
            key = f"extra_{i}"
            agents[key] = {"label": ea.get("name", f"Extra agent {i+1}") + "  (LLM-proposed)",
                           "role": ea.get("role", ""), "state": {"source": "LLM plan",
                           "domain": did}}
            order.append(key)
        return _json_safe({"order": order, "agents": agents})

    # -- editable learning parameters (math / weights) ----------------
    def _engine_settings(self, eng):
        """Give the engine a PRIVATE settings copy so parameter edits are per-domain
        (not global). Idempotent: components are repointed to the copy once."""
        import copy as _copy

        if getattr(eng, "_priv_settings", None) is None:
            cp = _copy.copy(self.dm.s)               # shallow copy of the dataclass
            eng._priv_settings = cp
            eng.s = cp
            eng.c.store.s = cp
            eng.c.retriever.s = cp
            eng.c.curator.s = cp
        return eng._priv_settings

    def get_params(self, did: str) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        st = self._engine_settings(eng)
        cfg = eng.c.curator.cfg
        lr = next(iter(eng.c.objective_fields.values())).lr if eng.c.objective_fields else 0.2
        return {
            "groups": {
                "Retrieval fusion weights (Score)": {
                    "w_phi": st.w_phi, "w_h": st.w_h, "w_g": st.w_g,
                    "lam_time": st.lam_time, "mu_omega": st.mu_omega},
                "Memory / EMA": {"rho": st.rho, "merge_contraction": st.merge_contraction,
                                 "merge_radius": cfg.merge_radius},
                "Novelty thresholds (Curator)": {
                    "theta_phi": cfg.theta_phi, "theta_h": cfg.theta_h, "theta_g": cfg.theta_g,
                    "theta_G": cfg.theta_G, "theta_U": cfg.theta_U, "budget": cfg.budget},
                "Identification": {"theta_match": eng.c.identifier.theta_match,
                                   "conformal_alpha": eng.c.identifier.gate.alpha},
                "OOD / diffusion": {"ood_threshold": st.ood_threshold, "dt_safety": st.dt_safety},
                "Field learning": {"objective_lr": lr},
            },
            "descriptions": {
                "w_phi": "weight of topology similarity in fused Score",
                "w_h": "weight of spectral similarity", "w_g": "weight of graph similarity",
                "lam_time": "time-gate weight", "mu_omega": "importance (omega) weight",
                "rho": "EMA importance update (0<rho<1)",
                "merge_contraction": "contractive merge factor c (<1)",
                "merge_radius": "fused-distance radius below which a candidate merges",
                "theta_phi": "topology novelty threshold", "theta_h": "spectral novelty threshold",
                "theta_g": "graph novelty threshold", "theta_G": "goal-gain novelty threshold",
                "theta_U": "uncertainty novelty threshold", "budget": "memory budget (compaction)",
                "theta_match": "Score >= theta_match => positive identification",
                "conformal_alpha": "conformal acceptance level",
                "ood_threshold": "below this confidence => outside trained domain",
                "dt_safety": "diffusion step safety factor (dt < 2/lambda_max)",
                "objective_lr": "learning rate for {v_k,c} from data",
            },
        }

    def set_params(self, did: str, params: dict) -> dict:
        eng = self.dm.engine(did, llm_client=self.llm)
        st = self._engine_settings(eng)
        cfg = eng.c.curator.cfg
        applied = {}

        def fnum(key, default=None):
            v = params.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        for k in ("w_phi", "w_h", "w_g", "lam_time", "mu_omega", "rho",
                  "merge_contraction", "ood_threshold", "dt_safety"):
            v = fnum(k)
            if v is not None:
                setattr(st, k, v)
                applied[k] = v
        for k in ("theta_phi", "theta_h", "theta_g", "theta_G", "theta_U", "merge_radius"):
            v = fnum(k)
            if v is not None:
                setattr(cfg, k, v)
                applied[k] = v
        if params.get("budget") is not None:
            cfg.budget = int(params["budget"])
            applied["budget"] = cfg.budget
        if params.get("theta_match") is not None:
            eng.c.identifier.theta_match = fnum("theta_match")
            applied["theta_match"] = eng.c.identifier.theta_match
        if params.get("conformal_alpha") is not None:
            eng.c.identifier.gate.alpha = fnum("conformal_alpha")
            applied["conformal_alpha"] = eng.c.identifier.gate.alpha
        if params.get("objective_lr") is not None:
            lr = fnum("objective_lr")
            for of in eng.c.objective_fields.values():
                of.lr = lr
            applied["objective_lr"] = lr
        return {"applied": applied, "n": len(applied)}

    # -- debug snapshot (every aspect of the codebase) ----------------
    def debug_snapshot(self, did: str) -> dict:
        import numpy as np

        eng = self.dm.engine(did, llm_client=self.llm)
        s = eng.s

        # config / theorem-guard hyperparameters
        config = {
            "domain_id": eng.domain_id, "world_id": eng.world_id,
            "qdrant_collection": eng.c.store.qdrant.alias,
            "graph_backend": getattr(eng.c.store.graph, "backend", "?"),
            "rho_EMA": s.rho, "theta_novelty": s.theta_novelty, "eps": s.eps,
            "merge_contraction_c": s.merge_contraction, "merge_radius": s.merge_radius,
            "dt_safety": s.dt_safety,
            "fusion_weights": {"w_phi": s.w_phi, "w_h": s.w_h, "w_g": s.w_g,
                               "lambda_time": s.lam_time, "mu_omega": s.mu_omega},
            "conformal_alpha": s.conformal_alpha, "ood_threshold": s.ood_threshold,
        }

        # Perception/Topology/Encoder agents
        encoders = {
            "topology": {"version": eng.c.topo.version, "n_pixels": eng.c.topo.n_pixels,
                         "max_dim": eng.c.topo.max_dim, "max_range": eng.c.topo.max_range},
            "spectral": {"version": eng.c.spectral.version, "fitted": eng.c.spectral.fitted,
                         "latent_dim": eng.c.spectral.latent_dim,
                         "mean_shape": (None if eng.c.spectral.mean_ is None
                                        else list(eng.c.spectral.mean_.shape)),
                         "components_shape": (None if eng.c.spectral.components_ is None
                                              else list(eng.c.spectral.components_.shape))},
            "graph": {"version": eng.c.graph.version, "emb_dim": eng.c.graph.emb_dim,
                      "n_spectrum": eng.c.graph.n_spectrum, "wl_iters": eng.c.graph.wl_iters},
        }

        # Object agent — registry objects (summaries + vector norms, NEVER raw vectors)
        objects = []
        for o in eng.c.registry.objects.values():
            objects.append({"id": o.id, "class": o.cls, "candidate": o.candidate,
                            "provenance": o.provenance, "tau": o.tau,
                            "world_ctx": o.world_ctx_id, "omega": round(float(o.omega), 3),
                            "phi_norm": round(float(np.linalg.norm(o.phi)), 4),
                            "h_norm": round(float(np.linalg.norm(o.h)), 4),
                            "g_norm": round(float(np.linalg.norm(o.g)), 4)})

        # Value agents (A_k) — value_store records with provenance/version/sticky
        value_agents = []
        for rec in eng.value_store.all():
            value_agents.append({"objective_k": rec.field_k, "class": rec.object_class,
                                 "value": round(float(rec.value), 4), "source": rec.source,
                                 "version": rec.version, "sticky": rec.sticky, "by": rec.by})

        # Functional agents (B_j) — affordances
        functional_agents = []
        for rec in eng.func_store.all():
            functional_agents.append({"class": rec.object_class, "dim": rec.dim,
                                      "action": rec.action, "enabled": rec.enabled,
                                      "source": rec.source, "version": rec.version})

        # Memory items (cached vectors → norms only) + curator audit
        memories = []
        for m in eng.c.store.graph.all_memories():
            it = eng.c.store.get_item(m["id"])
            row = {"id": m["id"], "class": m.get("class"),
                   "omega": round(float(m.get("omega", 1.0)), 3), "tau": m.get("tau"),
                   "world_ctx": m.get("world_ctx_id"), "snapshot": m.get("snapshot_id")}
            if it is not None:
                row["phi_norm"] = round(float(np.linalg.norm(it.phi)), 4)
            memories.append(row)

        curator = {
            "description": "Async curator: per-modality novelty (dphi|dh|dg|G|U) -> "
                           "reinforce / merge (contractive) / insert / evict+insert / defer; "
                           "budgeted (1-1/e) compaction; bounded rollback-able propagation.",
            "config": {"theta_phi": eng.c.curator.cfg.theta_phi,
                       "theta_h": eng.c.curator.cfg.theta_h,
                       "theta_g": eng.c.curator.cfg.theta_g,
                       "theta_G": eng.c.curator.cfg.theta_G,
                       "theta_U": eng.c.curator.cfg.theta_U,
                       "merge_radius": eng.c.curator.cfg.merge_radius,
                       "budget": eng.c.curator.cfg.budget},
            "memory_size": eng.memory_size(),
            "audit_tail": eng.c.curator.audit[-12:],
        }

        gates = {
            "conformal": {"threshold": round(float(eng.c.identifier.gate.threshold), 4),
                          "calibration_n": len(eng.c.identifier.gate._scores),
                          "alpha": eng.c.identifier.gate.alpha,
                          "theta_match": eng.c.identifier.theta_match},
        }

        return _json_safe({
            "config": config, "encoders": encoders,
            "object_agent": {"description": "Identify objects from feature SUMMARY + top-k "
                             "retrieval (never raw vectors); deterministic Score + conformal "
                             "gate decide; human override final.", "objects": objects,
                             "classes": eng.c.registry.classes()},
            "value_agents": {"description": "One agent per objective R^(k); learns {v_k,c} from "
                             "data + humans; human values STICKY.", "records": value_agents},
            "functional_agents": {"description": "One agent per functional dim F_j; learns "
                                  "affordances per class.", "records": functional_agents},
            "curator": curator, "gates": gates, "memory": memories,
            "last_event": self._last_debug.get(did, {}),
        })

    # -- unsupervised auto training -----------------------------------
    def auto_train(self, did: str, n_ticks: int = 0) -> dict:
        """Unsupervised pass over the domain's REAL data (folder items) when available,
        else the synthetic adapter — identify, learn values/affordances, Curator updates
        memory + vectors; no per-item confirm. Tracks coverage."""
        eng = self.dm.engine(did, llm_client=self.llm)
        q = self._queues.get(did) or self.build_queue(did)
        source = self._queue_source.get(did, "synthetic")
        progress = []
        inserts = 0

        if source == "folder" and q:
            seen_classes = set()
            la_on = getattr(self.dm.get(did), "llm_assist", False)
            # auto-ingest every real file (all 140), commit each without confirm
            for i, item in enumerate(q):
                # one LLM identification per new class to assign functionalities (opt-in)
                if self.llm is not None and la_on and item["label"] not in seen_classes:
                    seen_classes.add(item["label"])
                    try:
                        self._llm_suggestion[did] = self._llm_identify_item(
                            did, item, {"label_hint": item["label"]},
                            eng.c.registry.classes())
                    except Exception:
                        pass
                r = self.confirm(did, "confirm")
                if r.get("memory_action") in ("insert", "evict+insert"):
                    inserts += 1
                progress.append({"tick": i, "class": r.get("label"),
                                 "memory_action": r.get("memory_action"),
                                 "memory_size": eng.memory_size()})
                if r.get("remaining", 0) == 0:
                    break
        else:
            limit = n_ticks or 18
            for i, batch in enumerate(eng.adapter.stream()):
                if i >= limit:
                    break
                st = eng.step(batch, mode="train")
                if st.curator_decision.action in ("insert", "evict+insert"):
                    inserts += 1
                progress.append({"tick": i, "class": st.objects[0]["class"],
                                 "memory_action": st.curator_decision.action,
                                 "memory_size": eng.memory_size()})

        novelty_rate = round(inserts / max(1, len(progress)), 3)
        # Persist learned state so predictions survive a server restart.
        try:
            self.dm.save_engine_state(did)
        except Exception:
            pass
        return {"ticks": len(progress), "memory_size": eng.memory_size(),
                "novelty_rate": novelty_rate, "classes": eng.c.registry.classes(),
                "source": source, "progress": progress}


_SERVICE: DomainService | None = None


def get_service() -> DomainService:
    global _SERVICE
    if _SERVICE is None:
        from malar.llm.client import LLMClient
        _SERVICE = DomainService(llm_client=LLMClient())
    return _SERVICE
