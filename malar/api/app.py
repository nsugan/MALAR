"""FastAPI app — MALAR engine + control plane.

Endpoints: /configure /status /run /step /query /generate /curate /label /labels
/coverage /infer, plus /control, /review (control.py) and /ws (ws.py).

A single in-process EngineSession backs the app. Stage events are forwarded to WS
subscribers so the UI can render the live pipeline and review cards.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from malar.api.control import build_control_router
from malar.api.domains import router as domains_router
from malar.api.fs_routes import router as fs_router
from malar.api.graph import router as graph_router
from malar.api.llm_routes import router as llm_router
from malar.api.agent_routes import router as agent_router
from malar.api.predict_routes import router as predict_router
from malar.api.results import router as results_router
from malar.api.session import EngineSession
from malar.api.train_ui import router as train_router
from malar.api.ws import manager
from malar.api.ws import router as ws_router

app = FastAPI(title="MALAR control plane", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

SESSION = EngineSession()
SESSION.subscribe(lambda ev: manager.emit_threadsafe(ev))


def get_session() -> EngineSession:
    return SESSION


def _decode_image(image_b64: str):
    """Decode a base64 numeric array (JSON list-of-lists or .npy bytes)."""
    import base64
    import io
    import json

    import numpy as np

    raw = image_b64.split(",", 1)[-1]  # strip optional data-URL prefix
    decoded = base64.b64decode(raw)
    try:
        return np.asarray(json.loads(decoded.decode("utf-8")), dtype=float)
    except Exception:
        return np.load(io.BytesIO(decoded), allow_pickle=False)


class ConfigureCmd(BaseModel):
    domain: str = "synthetic"
    review_mode: bool = True
    budget: int = 64


class LabelCmd(BaseModel):
    object_id: str
    label: str


class InferCmd(BaseModel):
    text: str | None = None
    image_b64: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "status": SESSION.status.__dict__}


@app.get("/accel")
def accel():
    """What compute is available and what is / isn't GPU-accelerated (honest report)."""
    from malar.core.accel import device_report
    return device_report()


@app.post("/configure")
def configure(cmd: ConfigureCmd):
    return SESSION.configure(domain=cmd.domain, review_mode=cmd.review_mode, budget=cmd.budget)


@app.get("/status")
def status():
    size = SESSION.engine.memory_size() if SESSION.engine else 0
    return {"status": SESSION.status.__dict__, "memory_size": size,
            "ticks_done": SESSION.status.tick}


@app.post("/run")
def run(mode: str = "train", n: int | None = None):
    return {"result": SESSION.run(mode=mode, n=n)}


@app.post("/step")
def step():
    ms = SESSION.step()
    return {"tick": SESSION.status.tick, "done": ms is None,
            "action": ms.policy_decision.action if ms else None}


@app.get("/query")
def query(k: int = 10):
    return {"memories": SESSION.query(k=k)}


@app.post("/generate")
def generate():
    return SESSION.generate()


@app.post("/curate")
def curate():
    return SESSION.curate()


@app.get("/labels")
def labels():
    return {"pending": SESSION.labels_pending()}


@app.post("/label")
def label(cmd: LabelCmd):
    return SESSION.label(cmd.object_id, cmd.label)


@app.get("/coverage")
def coverage():
    if SESSION.engine is None:
        return {"coverage": {}}
    vs = SESSION.engine.value_store
    cov = {}
    for rec in vs.all():
        cov.setdefault(rec.field_k, {})[rec.object_class] = {
            "value": rec.value, "source": rec.source, "version": rec.version}
    return {"coverage": cov, "classes": SESSION.engine.c.registry.classes()}


@app.post("/infer")
def infer(cmd: InferCmd):
    try:
        from malar.inference.service import InferenceService

        svc = InferenceService(SESSION.engine)
        image = _decode_image(cmd.image_b64) if cmd.image_b64 else None
        return svc.infer(text=cmd.text, image=image)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "ood": True, "action": None}


app.include_router(ws_router)
app.include_router(build_control_router(get_session))
# Multi-domain UI routers (per UI plan)
app.include_router(domains_router)
app.include_router(train_router)
app.include_router(graph_router)
app.include_router(results_router)
app.include_router(llm_router)
app.include_router(predict_router)
app.include_router(agent_router)
app.include_router(fs_router)
