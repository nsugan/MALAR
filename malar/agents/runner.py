"""Run validated agent-factory agents from their PARENT field-agents during the MALAR loop.

When a domain has `agents_active`, each training item triggers its validated generated
agents (grouped by parent: objective Aⱼ / functional Bⱼ / value / planner). Each agent runs
in the sandbox on the item's data; its output is (a) fed BACK into the parent field so it
actually influences learning, and (b) stored in the graph DB (Neo4j / in-memory) with a
reference to the domain, world-context, class, parent and agent.

Kept fully defensive — a failing agent never breaks training.
"""
from __future__ import annotations

import time
import uuid


def parent_of(name: str, role: str) -> str:
    n = (name or "").lower()
    if n.startswith("objective_"):
        return "objective"
    if n.startswith("value_"):
        return "value"
    if "functional dimension" in (role or "").lower():
        return "functional"
    return "planner"


def _first_numeric(out) -> float | None:
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _feedback(engine, parent: str, name: str, out, label: str, ctx_id: str | None) -> str | None:
    """Route the agent's output back INTO its parent field-agent (closing the loop)."""
    v = _first_numeric(out)
    try:
        if parent == "objective" and v is not None:
            key = name[len("objective_"):] if name.startswith("objective_") else None
            of = engine.c.objective_fields.get(key) if key else None
            if of:
                of.learn_from_data(label, float(v), world_ctx=ctx_id)
                return f"objective '{key}' += {v:.3f}"
        elif parent == "value" and v is not None:
            key = name[len("value_"):] if name.startswith("value_") else None
            if key:
                engine.value_store.set(label, key, float(v), source="agent", by=name,
                                       world_ctx=ctx_id)
                return f"value '{key}' = {v:.3f}"
        elif parent == "functional" and engine.c.functional_fields:
            act = name.split("_")[-1]
            engine.c.functional_fields[0].learn_affordance(label, act, True, source="agent",
                                                           world_ctx=ctx_id)
            return f"affordance '{act}'"
    except Exception:
        return None
    return None


def run_validated_agents(engine, ctx: dict, domain_id: str, world_ctx_id: str | None,
                         label: str) -> list[dict]:
    """Run every VALIDATED agent on `ctx`, feed the output back to its parent, and store it.
    Returns a per-agent summary for the debug view. Never raises."""
    try:
        from malar.api.agent_service import get_agent_factory, get_agent_memory
        mem, fac = get_agent_memory(), get_agent_factory()
    except Exception:
        return []
    graph = getattr(getattr(engine, "c", None), "store", None)
    graph = getattr(graph, "graph", None)
    results = []
    try:
        # Only THIS domain's validated agents run in its training loop (isolation).
        agents = mem.all(domain_id=domain_id, with_code=True)
    except Exception:
        return []
    for a in agents:
        if not a.get("validated"):
            continue
        run = fac.run(a["id"], ctx)
        parent = parent_of(a["name"], a.get("role", ""))
        if not run.get("ok"):
            results.append({"agent": a["name"], "parent": parent, "ok": False,
                            "error": run.get("error") or run.get("reason")})
            continue
        out = run.get("output")
        fed = _feedback(engine, parent, a["name"], out, label, world_ctx_id)
        rec = {"id": "aout_" + uuid.uuid4().hex[:12], "agent_id": a["id"], "agent_name": a["name"],
               "parent": parent, "domain_id": domain_id, "cls": label, "output": out,
               "world_ctx_id": world_ctx_id, "fed_back": fed, "ts": time.time()}
        if graph is not None:
            try:
                graph.add_agent_output(rec)
            except Exception:
                pass
        results.append({"agent": a["name"], "parent": parent, "ok": True,
                        "output": out, "fed_back": fed})
    return results
