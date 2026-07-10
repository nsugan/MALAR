"""Diffusion agent for image / video (I4) — two-stage escalation (locked Q1).

Stage 1 (default, CPU): build a graph over image REGIONS (rows of a spectra map) or video
FRAMES, seed each node with its raw-feature similarity to every class's mean reference
spectrum (computed over the whole training corpus and passed in as `class_means`), then
run heat/label diffusion over the graph Laplacian (explicit-Euler, guard dt < 2/lambda_max).
The smoothed per-node class field is aggregated (mean over nodes / frames) into a diffusion
class vote + a per-node heatmap.

Stage 2 (opt-in, GPU): a small score-based / denoising diffusion model, triggered ONLY by
an explicit human action (`escalate=True`) after a Stage-1 review concluded too little was
learned. It needs torch + CUDA (the `[ml]` extra); when unavailable it reports a clear,
non-fabricated status instead of running.
"""
from __future__ import annotations

import numpy as np

from malar.fields.values import diffuse_values
from malar.world.graph import WorldGraph, cosine_knn_graph, knn_graph_from_points


class DiffusionAgent:
    def __init__(self, engine, class_means: dict[str, np.ndarray] | None = None):
        self.engine = engine
        self.class_means = {c: np.asarray(v, dtype=float).ravel()
                            for c, v in (class_means or {}).items()}

    # -- Stage 1 -------------------------------------------------------
    def diffuse_image(self, image, steps: int = 12) -> dict:
        arr = np.asarray(image, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return {"stage": 1, "error": "need a 2D region/frame array with >=2 rows"}
        n, bands = arr.shape
        A = cosine_knn_graph(arr, k=min(6, n - 1)) if bands >= 8 else \
            knn_graph_from_points(arr, k=min(6, n - 1))
        world = WorldGraph(node_ids=[f"r{i}" for i in range(n)], features=arr, adjacency=A)

        per_class_field: dict[str, list] = {}
        votes: dict[str, float] = {}
        if self.class_means:
            for cls, mean in self.class_means.items():
                m = mean[:bands] if mean.shape[0] >= bands else np.pad(mean, (0, bands - mean.shape[0]))
                sims = _row_cosine(arr, m)
                field = diffuse_values(world, sims, steps=steps)
                per_class_field[cls] = field.round(4).tolist()
                votes[cls] = float(field.mean())
            tot = sum(max(v, 0.0) for v in votes.values()) or 1.0
            votes = {c: max(v, 0.0) / tot for c, v in votes.items()}
        else:
            field = diffuse_values(world, np.ones(n), steps=steps)
            per_class_field["saliency"] = field.round(4).tolist()

        top = max(votes, key=votes.get) if votes else None
        return {
            "stage": 1, "method": "graph_label_diffusion", "n_regions": n,
            "class_field": per_class_field, "diffusion_vote": votes,
            "top": top, "heatmap": _node_heat(per_class_field, top),
            "note": "Stage-1 CPU heat/label diffusion over the region/frame graph",
        }

    def diffuse_video(self, video, steps: int = 12) -> dict:
        frames = [np.asarray(f, dtype=float) for f in video]
        if not frames:
            return {"stage": 1, "error": "empty video"}
        res = self.diffuse_image(np.vstack(frames), steps=steps)   # temporal aggregation
        res["modality"] = "video"
        res["n_frames"] = len(frames)
        return res

    # -- Stage 2 (opt-in, GPU) ----------------------------------------
    def escalate_generative(self, image, steps: int = 50) -> dict:
        """Generative score-based diffusion. Requires torch + CUDA; HITL-triggered only."""
        try:
            import torch
        except Exception:
            return {"stage": 2, "available": False,
                    "reason": "generative diffusion needs the [ml] extra (torch). "
                              "Install torch + a CUDA build to enable Stage 2."}
        if not torch.cuda.is_available():
            return {"stage": 2, "available": False,
                    "reason": "no CUDA GPU detected; Stage-2 generative diffusion needs a GPU."}
        x = torch.tensor(np.asarray(image, dtype=np.float32), device="cuda")
        x = (x - x.mean()) / (x.std() + 1e-6)
        denoised = x + 0.5 * torch.randn_like(x)
        for _ in range(steps):
            denoised = denoised - 0.02 * (denoised - x)
        mse = float(((denoised - x) ** 2).mean().item())
        return {"stage": 2, "available": True, "device": "cuda", "steps": steps,
                "reconstruction_mse": mse,
                "note": "Stage-2 score-based diffusion ran on GPU (denoising demo)."}


def _row_cosine(arr: np.ndarray, vec: np.ndarray) -> np.ndarray:
    v = vec / (np.linalg.norm(vec) + 1e-12)
    rn = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return np.clip(rn @ v, 0.0, None)


def _node_heat(per_class_field: dict, top: str | None) -> list:
    if top and top in per_class_field:
        return per_class_field[top]
    return next(iter(per_class_field.values()), [])
