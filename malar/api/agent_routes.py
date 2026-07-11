"""Agent Factory routes — propose / generate / inspect / edit / test / validate / run.

  GET  /agents                          -> list stored agents (no code)
  GET  /agents/{id}                     -> full detail incl. algorithm + code
  POST /agents/generate                 {name, role, context} -> reuse-or-generate ONE
  PUT  /agents/{id}/code                {code, did?} -> save edited code (re-validate+test)
  POST /agents/{id}/test                {did? | features} -> run against real/sample data
  POST /agents/{id}/validate            {validated} -> approve after review
  POST /agents/{id}/run                 {features, label?, summary?} -> sandboxed run
  GET  /domains/{did}/agents/proposals?source=planner|functional|objective|value
  POST /domains/{did}/agents/generate-selected {agents:[{name,role}]} -> generate chosen
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from malar.api.agent_service import get_agent_factory, get_agent_memory

router = APIRouter(tags=["agents"])


class GenerateCmd(BaseModel):
    name: str
    role: str
    context: str = ""
    domain: str | None = None


class ValidateCmd(BaseModel):
    validated: bool = True


class RunCmd(BaseModel):
    features: list | None = None
    label: str | None = None
    summary: dict | None = None
    allow_unvalidated: bool = False


class SelectedCmd(BaseModel):
    agents: list = []      # [{name, role}]
    context: str = ""


class CodeCmd(BaseModel):
    code: str
    did: str | None = None


class TestCmd(BaseModel):
    did: str | None = None
    features: list | None = None
    label: str | None = None
    summary: dict | None = None


def _sample_ctx(did: str | None) -> dict | None:
    """A representative ctx from the domain's data so generated code fits the real shape."""
    if not did:
        return None
    try:
        from malar.api.domain_service import get_service
        svc = get_service()
        q = svc._queues.get(did) or svc.build_queue(did)
        if not q:
            return None
        it = q[0]
        pts = it.get("points")
        feats = [list(map(float, r)) for r in (pts[:12] if pts is not None else [])]
        return {"features": feats, "label": it.get("label"),
                "summary": {"modality": it.get("kind", "features"),
                            "n_bands": len(feats[0]) if feats else 0}}
    except Exception:
        return None


@router.post("/agents/generate")
def generate(cmd: GenerateCmd):
    return get_agent_factory().get_or_generate(cmd.name, cmd.role, cmd.context,
                                               domain_id=cmd.domain)


@router.get("/agents")
def list_agents(domain: str = "", cross: bool = False):
    """List agents for `domain` (each domain owns its own). With cross=1, also include
    other domains' VALIDATED agents (flagged cross_domain) for cross-domain inference."""
    return {"agents": get_agent_memory().all(domain_id=(domain or None), with_code=False,
                                             include_cross_domain=cross)}


@router.get("/agents/{aid}")
def get_agent(aid: str):
    a = get_agent_memory().get(aid)
    if a:
        a["embedding"] = None
    return a or {"error": "not found"}


@router.put("/agents/{aid}/code")
def update_code(aid: str, cmd: CodeCmd):
    return get_agent_factory().update_code(aid, cmd.code, sample_ctx=_sample_ctx(cmd.did))


@router.post("/agents/{aid}/test")
def test_agent(aid: str, cmd: TestCmd):
    if cmd.features is not None:
        ctx = {"features": cmd.features, "label": cmd.label, "summary": cmd.summary or {}}
    else:
        ctx = _sample_ctx(cmd.did) or {"features": [], "label": None, "summary": {}}
    return get_agent_factory().test(aid, ctx)


@router.post("/agents/{aid}/validate")
def validate_agent(aid: str, cmd: ValidateCmd):
    return get_agent_factory().validate_agent(aid, cmd.validated)


@router.delete("/agents/{aid}")
def delete_agent(aid: str):
    return {"deleted": get_agent_memory().delete(aid), "id": aid}


@router.post("/agents/{aid}/run")
def run_agent(aid: str, cmd: RunCmd):
    ctx = {"features": cmd.features or [], "label": cmd.label, "summary": cmd.summary or {}}
    return get_agent_factory().run(aid, ctx, allow_unvalidated=cmd.allow_unvalidated)


@router.get("/domains/{did}/agents/proposals")
def proposals(did: str, source: str = "planner"):
    """Fast (no-LLM) list of PROPOSED agents from a chosen source, so the user picks which to
    generate. source: planner | functional | objective | value."""
    from malar.domains.manager import get_manager
    meta = get_manager().get(did)
    if not meta:
        return {"error": "domain not found", "proposals": []}
    props: list[dict] = []
    if source == "functional":
        for f in (meta.functionals or []):
            dim = f.get("dim", "functional")
            for act in (f.get("actions") or []):
                props.append({"source": "functional", "name": f"{dim}_{act}",
                              "role": f"Perform '{act}' along functional dimension '{dim}'.",
                              "proposed": {"dim": dim, "action": act}})
    elif source == "objective":
        for o in (meta.objectives or []):
            k = o.get("key", "objective")
            props.append({"source": "objective", "name": f"objective_{k}",
                          "role": f"Assess and optimise objective '{k}' (target "
                                  f"{o.get('target')}) for recognised objects.",
                          "proposed": {"objective": k, "target": o.get("target"),
                                       "direction": o.get("direction", "maximize")}})
    elif source == "value":
        vals = _learned_values(did)
        for o in (meta.objectives or []):
            k = o.get("key", "objective")
            props.append({"source": "value", "name": f"value_{k}",
                          "role": f"Learn/assess per-class value contributions for objective '{k}'.",
                          "proposed": {"objective": k, "current_values": vals.get(k, {})}})
    else:
        for e in (meta.extra_agents or []):
            props.append({"source": "planner", "name": e.get("name", "agent"),
                          "role": e.get("role") or e.get("description", ""),
                          "proposed": {"kind": "extra_agent"}})
        for i, step in enumerate((meta.plan or {}).get("processing_steps", []) or []):
            props.append({"source": "planner", "name": f"step_{i + 1}",
                          "role": str(step), "proposed": {"kind": "processing_step"}})
    return {"domain": did, "source": source, "proposals": props}


def _learned_values(did: str) -> dict:
    try:
        from malar.api.domain_service import get_service
        eng = get_service().dm._engines.get(did)
        if eng is None:
            return {}
        out: dict = {}
        for rec in eng.value_store.all():
            out.setdefault(rec.field_k, {})[rec.object_class] = round(rec.value, 3)
        return out
    except Exception:
        return {}


@router.post("/domains/{did}/agents/generate-selected")
def generate_selected(did: str, cmd: SelectedCmd):
    """Generate ONLY the chosen agents. Feeds a real data sample into the code prompt and
    auto-tests each generated agent. The UI calls this per-agent to avoid proxy timeouts."""
    from malar.domains.manager import get_manager
    meta = get_manager().get(did)
    # Embed the consolidated domain + data description in the code/algorithm prompts.
    ctx = cmd.context or (meta.consolidated_context() if meta else "") or ""
    sample = _sample_ctx(did)
    fac = get_agent_factory()
    out = [fac.get_or_generate(name=a.get("name", "agent"), role=a.get("role", ""),
                               context=ctx, sample_ctx=sample, domain_id=did)
           for a in (cmd.agents or [])]
    return {"domain": did, "results": out}


class ActivateCmd(BaseModel):
    only_validated: bool = True
    cross_domain: bool = False    # also run OTHER domains' validated agents (opt-in)


def _parent_of(name: str, role: str) -> str:
    """Infer the parent field-agent that owns a generated agent."""
    n = (name or "").lower()
    if n.startswith("objective_"):
        return "objective"
    if n.startswith("value_"):
        return "value"
    if "functional dimension" in (role or "").lower():
        return "functional"
    return "planner"


@router.post("/domains/{did}/agents/activate")
def activate_agents(did: str, cmd: ActivateCmd):
    """Run the (validated) agents over the domain sample; route each output through its
    parent field-agent and store it in the graph (Neo4j, or in-memory fallback) with a
    reference to the domain, parent, agent and class."""
    import time
    import uuid

    from malar.api.domain_service import get_service
    svc = get_service()
    try:
        eng = svc.dm.engine(did, llm_client=svc.llm)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "activated": 0, "results": []}
    fac, mem = get_agent_factory(), get_agent_memory()
    sample = _sample_ctx(did) or {"features": [], "label": None, "summary": {}}
    graph = eng.c.store.graph
    results = []
    # This domain's agents, plus (opt-in) other domains' validated agents flagged
    # cross_domain — the orchestrator uses these together for cross-domain inference.
    for a in mem.all(domain_id=did, with_code=True, include_cross_domain=cmd.cross_domain):
        if cmd.only_validated and not a["validated"]:
            continue
        run = fac.run(a["id"], sample, allow_unvalidated=not cmd.only_validated)
        parent = _parent_of(a["name"], a["role"])
        rec = {"id": "aout_" + uuid.uuid4().hex[:12], "agent_id": a["id"],
               "agent_name": a["name"], "parent": parent, "domain_id": did,
               "agent_domain": a.get("domain_id"), "cross_domain": bool(a.get("cross_domain")),
               "cls": sample.get("label"),
               "output": run.get("output") if run.get("ok") else None,
               "ok": bool(run.get("ok")), "error": run.get("error") or run.get("reason"),
               "ts": time.time()}
        try:
            graph.add_agent_output(rec)
        except Exception:
            pass
        results.append({"agent": a["name"], "parent": parent, "ok": rec["ok"],
                        "agent_domain": a.get("domain_id"),
                        "cross_domain": bool(a.get("cross_domain")),
                        "output": run.get("output"), "error": rec["error"]})
    return {"domain": did, "activated": len(results),
            "stored_in": getattr(graph, "backend", "?"), "results": results}


@router.get("/domains/{did}/agents/outputs")
def agent_outputs_ep(did: str):
    from malar.api.domain_service import get_service
    try:
        eng = get_service().dm.engine(did)
        return {"domain": did, "backend": getattr(eng.c.store.graph, "backend", "?"),
                "outputs": eng.c.store.graph.agent_outputs(did)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "outputs": []}


class UseAgentsCmd(BaseModel):
    active: bool = True


@router.post("/domains/{did}/agents/use")
def use_agents(did: str, cmd: UseAgentsCmd):
    """Toggle whether the parent field-agents run their validated factory-agents in-loop."""
    from malar.domains.manager import get_manager
    dm = get_manager()
    m = dm.get(did)
    if not m:
        return {"error": "domain not found"}
    m.agents_active = bool(cmd.active)
    try:
        dm._save()
    except Exception:
        pass
    return {"domain": did, "agents_active": m.agents_active}


@router.get("/domains/{did}/agents/status")
def agents_status(did: str):
    from malar.domains.manager import get_manager
    m = get_manager().get(did)
    validated = len([a for a in get_agent_memory().all(domain_id=did, with_code=False)
                     if a["validated"]])
    return {"domain": did, "agents_active": bool(getattr(m, "agents_active", False)),
            "validated_agents": validated}
