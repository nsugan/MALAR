"""DomainPlanner — the Orchestrator/Planner + Perception planning brain (LLM-driven).

The planner is constructed with the CONSOLIDATED domain + data context
(DomainMeta.consolidated_context) and embeds it in EVERY query it makes — both the
domain description AND the data description are always present, never one or the other.

From that context the planner calls the LLM (via the gateway) to derive the learning plan:
  * objective fields R (what is good) with directions/targets,
  * functional dims F (the actions to carry out) with candidate actions,
  * processing steps and any EXTRA domain-specific agents needed beyond the regular
    graph / topology / spectral agents,
and, per training item, to propose the object class + the functionalities to assign.

Every LLM call goes through LLMClient (so it appears in the LLM log). When the gateway
is unavailable the planner falls back to deterministic defaults so training never
breaks — but it still records the attempt.
"""
from __future__ import annotations

import json
import re

_PLAN_SYSTEM = (
    "You are MALAR's Orchestrator/Planner. MALAR learns objects from data using "
    "topological (persistence homology), spectral, and graph encoders, then assigns "
    "objective VALUE fields R and FUNCTIONAL fields F (actions). Given a short "
    "description of a training dataset, design the learning plan. Reply with STRICT "
    "JSON only, no prose."
)

_IDENT_SYSTEM = (
    "You identify the object/class and the functional actions to assign from a "
    "topological/spectral FEATURE SUMMARY and the domain description. You never see raw "
    "vectors. Reply with STRICT JSON only."
)


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    # grab the first {...} block
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class DomainPlanner:
    def __init__(self, llm_client=None, context: str = ""):
        # `context` is the CONSOLIDATED domain + data description (see
        # DomainMeta.consolidated_context). It is embedded in EVERY query this
        # planner/orchestrator makes, so the LLM always has both.
        self.llm = llm_client
        self.context = (context or "").strip()

    def _ctx_block(self) -> str:
        """The consolidated domain+data context, prepended to every planner query."""
        if not self.context:
            return ""
        return ("=== DOMAIN & DATA CONTEXT (always applies to this request) ===\n"
                f"{self.context}\n"
                "=== END CONTEXT ===\n\n")

    # -- Orchestrator: derive the plan from the domain + data context -
    def plan_domain(self, classes: list[str] | None = None,
                    folder_summary: str | None = None, adapter_type: str = "synthetic") -> dict:
        classes = classes or []
        prompt = (
            self._ctx_block()
            + f"Known classes/labels: {classes or 'unknown'}\n"
            f"Data summary: {folder_summary or 'n/a'}\n"
            f"Modality/adapter: {adapter_type}\n\n"
            "Design the plan. Reply with JSON of this exact shape:\n"
            '{"objectives":[{"key":"snake_case","description":"...","direction":"maximize|minimize","target":0.8}],'
            '"functionals":[{"dim":"snake_case","actions":["action1","action2"]}],'
            '"processing_steps":["step1","step2"],'
            '"extra_agents":[{"name":"...","role":"what it does beyond graph/topology/spectral"}],'
            '"rationale":"one or two sentences"}'
        )
        raw = ""
        used_llm = False
        if self.llm is not None:
            try:
                raw = self.llm.reason(prompt, system=_PLAN_SYSTEM, max_tokens=700, source="planner")
                used_llm = True
            except Exception as e:  # noqa: BLE001
                raw = f"__error__:{e}"
        plan = _extract_json(raw)
        if not plan or "objectives" not in plan:
            plan = self._fallback_plan(adapter_type, classes)
            plan["source"] = "fallback"
        else:
            plan["source"] = "llm"
        plan["used_llm"] = used_llm
        plan.setdefault("objectives", [])
        plan.setdefault("functionals", [])
        plan.setdefault("processing_steps", [])
        plan.setdefault("extra_agents", [])
        plan.setdefault("rationale", "")
        return plan

    # -- per-item: identify object + functionalities -----------------
    def suggest_object(self, feature_summary: dict, candidates: list[str],
                       functional_dims: list[str]) -> dict:
        prompt = (
            self._ctx_block()
            + f"Feature summary (topology H0/H1/H2 + spectral): {feature_summary}\n"
            f"Retrieval candidate classes: {candidates or 'none yet'}\n"
            f"Functional dimensions to assign actions for: {functional_dims}\n\n"
            "Identify the most likely object class and the functional actions to assign. "
            'Reply JSON: {"class":"<best class or NEW:name>","affordances":["..."],'
            '"confidence":0.0,"rationale":"one line"}'
        )
        if self.llm is None:
            return {"class": (candidates[0] if candidates else None), "affordances": [],
                    "confidence": 0.0, "rationale": "llm disabled", "used_llm": False}
        try:
            raw = self.llm.reason(prompt, system=_IDENT_SYSTEM, max_tokens=300, source="identify")
            out = _extract_json(raw)
            out["used_llm"] = True
            return out
        except Exception as e:  # noqa: BLE001
            return {"class": (candidates[0] if candidates else None), "affordances": [],
                    "confidence": 0.0, "rationale": f"llm unavailable: {e}", "used_llm": False}

    # -- deterministic fallback (no gateway) -------------------------
    def _fallback_plan(self, adapter_type: str, classes: list[str]) -> dict:
        # Generic default preset (domain-agnostic). Legacy raman/sensor presets are kept
        # only for pre-existing domains that still declare those adapter types.
        generic = {
            "objectives": [{"key": "quality", "description": "overall quality / correctness",
                            "direction": "maximize", "target": 0.8},
                           {"key": "confidence", "description": "identification confidence",
                            "direction": "maximize", "target": 0.8}],
            "functionals": [{"dim": "response", "actions": ["flag", "watchlist", "escalate"]}],
            "processing_steps": ["persistence homology (topology)", "spectral autoencoder",
                                 "graph kNN embedding"],
            "extra_agents": [],
        }
        presets = {
            "raman": {
                "objectives": [
                    {"key": "sensitivity", "description": "true-positive detection rate",
                     "direction": "maximize", "target": 0.9},
                    {"key": "specificity", "description": "true-negative rate",
                     "direction": "maximize", "target": 0.9}],
                "functionals": [{"dim": "response",
                                 "actions": ["confirm", "escalate", "watchlist"]}],
                "processing_steps": ["parse signal", "baseline/normalise",
                                     "persistence homology", "spectral autoencoder", "graph kNN"],
                "extra_agents": [],
            },
            "sensor": {
                "objectives": [{"key": "yield", "description": "process yield",
                                "direction": "maximize", "target": 0.9},
                               {"key": "defect_rate", "description": "defect fraction",
                                "direction": "minimize", "target": 0.05}],
                "functionals": [{"dim": "response",
                                 "actions": ["adjust", "hold", "flag"]}],
                "processing_steps": ["window streams", "spatio-temporal graph", "encoders"],
                "extra_agents": [],
            },
        }
        p = presets.get(adapter_type, generic)
        p["rationale"] = "deterministic preset (LLM gateway unavailable)"
        return p
