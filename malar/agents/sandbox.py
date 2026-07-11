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
import os
import threading

import numpy as np

# Wall-clock cap on a single agent run. Since agents run with full capabilities, a
# slow/looping one would otherwise stall the (now concurrent) training pool and, via the
# shared threadpool, the whole API. Override with MALAR_AGENT_TIMEOUT.
_AGENT_TIMEOUT = float(os.environ.get("MALAR_AGENT_TIMEOUT", "15"))


def run_agent_code(code: str, ctx: dict, timeout: float | None = None) -> dict:
    timeout = _AGENT_TIMEOUT if timeout is None else timeout
    result: dict = {}

    def _work():
        # Full builtins + real imports — the agent may do anything it needs.
        g = {"__builtins__": _builtins.__dict__, "__name__": "generated_agent",
             "np": np, "numpy": np, "math": _math}
        try:
            exec(compile(code, "<generated_agent>", "exec"), g)  # noqa: S102 (reviewed agent)
        except Exception as e:  # noqa: BLE001
            result.update(ok=False, stage="compile", error=str(e)[:300])
            return
        cls = g.get("GeneratedAgent")
        if cls is None:
            result.update(ok=False, stage="load", error="no GeneratedAgent class")
            return
        try:
            out = cls().run(dict(ctx))
            result.update(ok=True, output=_jsonable(out))
        except Exception as e:  # noqa: BLE001
            result.update(ok=False, stage="run", error=str(e)[:300])

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        # abandon the runaway (daemon) thread and return — never block the caller
        return {"ok": False, "stage": "timeout",
                "error": f"agent exceeded {timeout:.0f}s wall-clock and was abandoned"}
    return result or {"ok": False, "stage": "run", "error": "no result"}


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
