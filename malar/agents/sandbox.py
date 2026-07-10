"""Execution of a generated GeneratedAgent.

**Capabilities are OPEN (by request):** generated agents run with FULL Python capabilities
— any import, all builtins — so an agent can carry out any task it was designed for
(compute, read/write files, call services, use any library). This is NOT a security
sandbox: only run agents you have reviewed and validated. `numpy` and `math` are
pre-injected for convenience (as `np`/`numpy`/`math`); the return value is coerced to
JSON-able types so it can be shown in the UI and stored.
"""
from __future__ import annotations

import builtins as _builtins
import math as _math

import numpy as np


def run_agent_code(code: str, ctx: dict) -> dict:
    # Full builtins + real imports — the agent may do anything it needs.
    g = {"__builtins__": _builtins.__dict__, "__name__": "generated_agent",
         "np": np, "numpy": np, "math": _math}
    try:
        exec(compile(code, "<generated_agent>", "exec"), g)  # noqa: S102 (reviewed agent)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stage": "compile", "error": str(e)[:300]}
    cls = g.get("GeneratedAgent")
    if cls is None:
        return {"ok": False, "stage": "load", "error": "no GeneratedAgent class"}
    try:
        out = cls().run(dict(ctx))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stage": "run", "error": str(e)[:300]}
    return {"ok": True, "output": _jsonable(out)}


def _jsonable(o):
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
