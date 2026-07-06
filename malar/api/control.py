"""Control-plane routes: /control (start/stop/pause/mode) and /review (toggle + per-stage)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ControlCmd(BaseModel):
    action: str                      # start | stop | pause | resume | step
    mode: str | None = None          # train | infer
    n: int | None = None


class ReviewCmd(BaseModel):
    review_mode: bool
    stages: list[str] | None = None


def build_control_router(get_session):
    r = APIRouter()

    @r.post("/control")
    def control(cmd: ControlCmd):
        s = get_session()
        if cmd.action == "start":
            return {"result": s.run(mode=cmd.mode or "train", n=cmd.n), "status": s.status.__dict__}
        if cmd.action == "step":
            ms = s.step()
            return {"tick": s.status.tick, "done": ms is None}
        if cmd.action == "pause":
            s.pause()
            return {"paused": True}
        if cmd.action == "resume":
            s.resume()
            return {"paused": False}
        if cmd.action == "stop":
            s.stop()
            return {"stopped": True}
        return {"error": f"unknown action {cmd.action}"}

    @r.post("/review")
    def review(cmd: ReviewCmd):
        return get_session().set_review(cmd.review_mode, cmd.stages)

    return r
