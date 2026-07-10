"""Multimodal front-end — map unknown image / video / text to phi / h / graph_emb.

This runs BEFORE identification. Each modality is reduced to the same three encoder
outputs the trained model uses, plus a world_emb for validity-gating.

  * Image  — a (n_samples, n_features) numeric matrix is treated as feature vectors and
             turned into a similarity graph directly. Otherwise Gemma 4 vision (via the
             gateway) proposes regions/descriptions, encoded at lower confidence. Then
             topology/spectral/graph encoders -> phi/h/graph_emb.
  * Video  — sample/segment frames -> per-frame features -> temporal aggregation -> encoders.
  * Text   — Gemma 4 parses the problem into a structured query + pseudo-features; if the
             gateway is unavailable, a deterministic keyword parser maps to known classes.

Confidence is attenuated for LLM-proposed (non-extracted) inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from malar.world.graph import WorldGraph, cosine_knn_graph, knn_graph_from_points


@dataclass
class FrontEndResult:
    phi: np.ndarray
    h: np.ndarray
    g: np.ndarray
    world_emb: np.ndarray
    summary: dict
    modality: str
    confidence_scale: float = 1.0       # < 1 for LLM-proposed inputs
    parsed: dict = field(default_factory=dict)


class MultimodalFrontEnd:
    def __init__(self, engine, llm_client=None):
        self.engine = engine
        self.llm = llm_client

    # -- public ----------------------------------------------------
    def process(self, text: str | None = None, image=None, video=None,
                modality: str | None = None) -> FrontEndResult:
        if image is not None:
            return self._image(image)
        if video is not None:
            return self._video(video)
        if text is not None:
            return self._text(text)
        raise ValueError("frontend.process needs one of text / image / video")

    # -- image -----------------------------------------------------
    def _encode_world(self, world: WorldGraph, modality: str, conf: float,
                      parsed: dict | None = None) -> FrontEndResult:
        tr = self.engine.c.topo.encode(world)
        sr = self.engine.c.spectral.encode(world)
        gr = self.engine.c.graph.encode(world)
        return FrontEndResult(phi=tr.phi, h=sr.h, g=gr.graph_emb,
                              world_emb=self.engine.c.graph.world_embed(world),
                              summary=tr.summary, modality=modality,
                              confidence_scale=conf, parsed=parsed or {})

    def _image(self, image) -> FrontEndResult:
        arr = np.asarray(image, dtype=float)
        # Feature matrix: (n_samples, n_features) with high-dim rows -> cosine-knn graph.
        if arr.ndim == 2 and arr.shape[1] >= 8:
            A = cosine_knn_graph(arr, k=min(6, max(1, arr.shape[0] - 1)))
            world = WorldGraph(node_ids=[f"px{i}" for i in range(arr.shape[0])],
                               features=arr, adjacency=A)
            return self._encode_world(world, "image:features", 1.0)
        # Generic 2D point set (defect ROI) -> knn graph.
        pts = arr if arr.ndim == 2 else arr.reshape(-1, 1)
        A = knn_graph_from_points(pts, k=min(6, max(1, pts.shape[0] - 1)))
        world = WorldGraph(node_ids=[f"p{i}" for i in range(pts.shape[0])],
                           features=pts, adjacency=A)
        return self._encode_world(world, "image:points", 0.8)

    def _video(self, video) -> FrontEndResult:
        frames = list(video)
        if not frames:
            raise ValueError("empty video")
        # per-frame -> aggregate by stacking frame point clouds (temporal aggregation)
        agg = np.vstack([np.asarray(f, dtype=float) for f in frames])
        res = self._image(agg)
        res.modality = "video"
        res.confidence_scale *= 0.9
        return res

    # -- text ------------------------------------------------------
    def _text(self, text: str) -> FrontEndResult:
        parsed = self._parse_text(text)
        cls = parsed.get("class")
        # pseudo-features: borrow the learned prototype's codes for the resolved class
        proto = None
        if cls:
            for o in self.engine.c.registry.objects.values():
                if not o.candidate and o.cls == cls:
                    proto = o
                    break
        if proto is not None:
            return FrontEndResult(phi=proto.phi, h=proto.h, g=proto.g,
                                  world_emb=self.engine.c.graph.world_embed_proxy(proto.g)
                                  if hasattr(self.engine.c.graph, "world_embed_proxy") else proto.g,
                                  summary={"resolved_from": "text", "class": cls},
                                  modality="text", confidence_scale=0.95, parsed=parsed)
        # no resolution -> zero pseudo-features (will trip the OOD gate)
        dims = {"phi": 432, "h": 16, "g": 48}
        return FrontEndResult(phi=np.zeros(dims["phi"]), h=np.zeros(dims["h"]),
                              g=np.zeros(dims["g"]), world_emb=np.zeros(dims["g"]),
                              summary={"resolved_from": "text", "class": None},
                              modality="text", confidence_scale=0.3, parsed=parsed)

    def _parse_text(self, text: str) -> dict:
        if self.llm is not None:
            try:
                classes = self.engine.c.registry.classes()
                # Route through the reasoner ROLE (LLM-tab route -> env default), not a
                # hardcoded alias literal, per the "provider chosen in config, never in
                # code" rule. (V4 fix: routing bypass)
                out = self.llm.reason(
                    f"Map the problem to ONE known class or UNKNOWN.\nClasses: {classes}\n"
                    f"Problem: {text}\nReply: CLASS: <class or UNKNOWN>",
                    source="frontend")
                for line in out.splitlines():
                    if "CLASS:" in line:
                        c = line.split("CLASS:", 1)[1].strip()
                        if c and c.upper() != "UNKNOWN" and c in classes:
                            return {"class": c, "method": "llm"}
            except Exception:
                pass
        # deterministic keyword fallback — whole-word match, ignore short tokens
        import re
        low = text.lower()
        words = set(re.findall(r"[a-z0-9]+", low))
        best = None
        for c in self.engine.c.registry.classes():
            cl = c.lower()
            if cl in low or cl.replace("_", " ") in low:
                return {"class": c, "method": "keyword"}
            tokens = [t for t in cl.replace("_", " ").split() if len(t) >= 3]
            hits = sum(1 for t in tokens if t in words)
            if tokens and hits == len(tokens):
                return {"class": c, "method": "keyword"}
            if hits and best is None:
                best = c
        if best is not None:
            return {"class": best, "method": "keyword-partial"}
        return {"class": None, "method": "none"}
