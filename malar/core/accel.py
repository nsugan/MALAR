"""Compute acceleration helpers + an honest device report.

MALAR's per-item deterministic math (persistence homology via ripser, small numpy
encoders) is tiny — milliseconds per item — so offloading it to a GPU/NPU does not make
training faster (the host<->device transfer costs more than the compute). Training cost is
dominated by **LLM calls** (llm-assist per item + in-loop agents). So the real levers are:

  * run the local LLM (Ollama) on the GPU — already wired via `gpus: all` (the RTX 3050);
    route agent roles to malar-reasoner/fast/vision to use it instead of the cloud;
  * run the many independent in-loop agent calls CONCURRENTLY (parallel_map here);
  * the Stage-2 generative diffusion uses torch CUDA when available.

The AMD NPU (XDNA) and Radeon iGPU are NOT usable by this numpy/ripser stack without an
ONNX / Ryzen-AI / DirectML rewrite of the encoders; they are honestly not wired in.
`device_report()` states exactly what is and isn't accelerated so nothing is oversold.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def default_workers() -> int:
    try:
        return max(2, min(8, (os.cpu_count() or 4)))
    except Exception:
        return 4


def parallel_map(fn, items, workers: int | None = None):
    """Run fn over items concurrently (threads), preserving order. Used to overlap the
    independent per-agent LLM calls / sandbox runs in the training loop — the real speedup
    for agent-heavy training. Falls back to a serial map for 0/1 items."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    n = min(workers or default_workers(), len(items))
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def device_report() -> dict:
    """What compute is available and what is / isn't GPU-accelerated (no overselling)."""
    rep: dict = {
        "cpu_count": os.cpu_count(),
        "parallel_workers": default_workers(),
        "torch": False,
        "cuda": False,
        "cuda_device": None,
        "gpu_accelerated": [],
        "cpu_only": [
            "persistence homology (ripser)",
            "spectral / graph encoders (numpy)",
            "retrieval / curator / Stage-1 diffusion",
        ],
        "not_wired": [
            "AMD NPU (XDNA) — needs an ONNX + Ryzen-AI rewrite of the encoders",
            "Radeon iGPU — needs DirectML/ROCm; not beneficial for this small per-item data",
        ],
        "notes": [],
    }
    try:
        import torch  # noqa: PLC0415
        rep["torch"] = True
        if torch.cuda.is_available():
            rep["cuda"] = True
            try:
                rep["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                rep["cuda_device"] = "cuda:0"
            rep["gpu_accelerated"].append("Stage-2 generative diffusion (when escalated)")
    except Exception:
        pass
    rep["gpu_accelerated"].append(
        "local LLM inference via Ollama (RTX 3050) — when agent roles are routed to "
        "malar-reasoner / malar-fast / malar-vision")
    rep["notes"] = [
        "Per-item math is ms-scale; GPU/NPU won't speed it up.",
        "Training cost is dominated by LLM calls — route them to local Ollama (GPU) and/or "
        f"rely on the concurrent in-loop agent calls ({rep['parallel_workers']} workers).",
    ]
    return rep
