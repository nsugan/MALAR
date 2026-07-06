"""LLM observability routes — see the queries and outputs of the LLMs.

  GET  /llm/status     -> gateway reachable? which model aliases are available
  GET  /llm/log        -> recent LLM calls (prompt + response/error + latency + source)
  POST /llm/complete   -> send a manual prompt to an alias and get the response (logged)
  POST /llm/clear      -> clear the call log
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from malar.llm.client import LLMClient, LLMUnavailable, clear_log, get_log, get_route, set_route

router = APIRouter(prefix="/llm", tags=["llm"])
_client = LLMClient()


class CompleteCmd(BaseModel):
    alias: str = "malar-reasoner"
    prompt: str
    system: str | None = None
    max_tokens: int = 512
    temperature: float = 0.2


@router.get("/status")
def status():
    try:
        models = _client.list_models()
        return {"reachable": True, "base_url": _client.s.llm_base_url, "models": models,
                "aliases": {"reasoner": _client.s.alias_reasoner, "fast": _client.s.alias_fast,
                            "vision": _client.s.alias_vision}}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "base_url": _client.s.llm_base_url, "error": str(e),
                "models": [], "aliases": {"reasoner": _client.s.alias_reasoner,
                                          "fast": _client.s.alias_fast,
                                          "vision": _client.s.alias_vision}}


@router.get("/log")
def log(limit: int = 100):
    return {"calls": get_log(limit=limit)}


@router.post("/complete")
def complete(cmd: CompleteCmd):
    try:
        out = _client.complete(cmd.alias, cmd.prompt, system=cmd.system,
                               temperature=cmd.temperature, max_tokens=cmd.max_tokens,
                               source="manual")
        return {"ok": True, "response": out}
    except LLMUnavailable as e:
        return {"ok": False, "error": str(e)}


@router.post("/clear")
def clear():
    return {"cleared": clear_log()}


class RouteCmd(BaseModel):
    reasoner: str | None = None
    fast: str | None = None
    vision: str | None = None


@router.get("/route")
def route():
    """Current agent-role -> alias routing, plus the aliases available on the gateway."""
    try:
        models = _client.list_models()
    except Exception:
        models = []
    return {"route": get_route(),
            "defaults": {"reasoner": _client.s.alias_reasoner, "fast": _client.s.alias_fast,
                         "vision": _client.s.alias_vision},
            "available": models,
            "frontier_aliases": ["malar-claude", "malar-openai", "malar-deepseek", "malar-frontier"]}


@router.put("/route")
def set_route_ep(cmd: RouteCmd):
    if cmd.reasoner is not None:
        set_route("reasoner", cmd.reasoner or None)
    if cmd.fast is not None:
        set_route("fast", cmd.fast or None)
    if cmd.vision is not None:
        set_route("vision", cmd.vision or None)
    return {"route": get_route()}
