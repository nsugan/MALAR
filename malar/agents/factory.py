"""Agent Factory — generate, store, reuse, test and run functional agents.

Records the full LLM exchange (algo/code query+response + provider) into the agent `trace`.
Robustness: the domain context is trimmed, and if a model returns an EMPTY message (some
prompts trigger this on Claude/Gemma), the call is retried once with the context dropped.
"""
from __future__ import annotations

import json
import re

from malar.agents.sandbox import run_agent_code
from malar.agents.validate import validate_code
from malar.llm.client import get_route

_INTERFACE = '''
class GeneratedAgent:
    name = "..."
    def run(self, ctx: dict) -> dict:
        """ctx keys: features (2D list of floats), label (str|None), summary (dict).
        Return a dict of findings (numbers/strings/lists only)."""
        ...
'''
_CTX_MAX = 400   # trimming the domain description avoids empty-response triggers


def _describe_sample(ctx: dict | None) -> str:
    if not ctx:
        return "ctx['features'] is a 2D list of floats; ctx['label'] is a class string; " \
               "ctx['summary'] is a dict."
    f = ctx.get("features") or []
    rows = len(f)
    cols = len(f[0]) if rows and isinstance(f[0], (list, tuple)) else "?"
    skeys = list((ctx.get("summary") or {}).keys())
    # NB: describe shape only — embedding raw numeric values in the prompt makes some models
    # (e.g. Claude) return an empty message.
    return (f"ctx['features'] is a numeric 2D list of floats with shape ~{rows} x {cols}. "
            f"ctx['label'] is a class string. ctx['summary'] has keys: {skeys}. "
            "Write code that works on exactly this shape.")


def _algo_prompt(name: str, role: str, context: str, sample: str) -> str:
    ctx = f"Domain context: {context}\n" if context else ""
    return (
        f"You are MALAR's algorithm designer. Design a concise algorithm for a functional "
        f"agent named '{name}'.\nRole: {role}\n{ctx}Data it receives: {sample}\n"
        "Give numbered steps only (inputs, the computation — use whatever libraries/methods "
        "the task needs, outputs). No code."
    )


def _code_prompt(name: str, role: str, algorithm: str, sample: str) -> str:
    return (
        "Write Python implementing EXACTLY this interface and nothing else:\n"
        f"{_INTERFACE}\n"
        f"Agent name: {name}\nRole: {role}\nData it receives: {sample}\n"
        f"Algorithm to implement:\n{algorithm}\n\n"
        "RULES: output MUST be syntactically valid Python 3 defining the class above. You MAY "
        "import and use ANY Python library you need to accomplish the task (numpy and math are "
        "pre-injected as np/math; import anything else you require). Handle empty/short inputs "
        "gracefully. run() MUST return a dict of JSON-able findings (numbers/strings/lists/dicts). "
        "Return ONLY the code — no prose, no markdown fences."
    )


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


class AgentFactory:
    def __init__(self, memory, llm):
        self.memory = memory
        self.llm = llm

    def _alias(self, role: str) -> str:
        r = get_route()
        s = getattr(self.llm, "s", None)
        if role == "algo":
            return r.get("algo") or (getattr(s, "alias_algo", None) or "malar-deepseek")
        return r.get("coder") or (getattr(s, "alias_coder", None) or "malar-claude")

    def get_or_generate(self, name: str, role: str, context: str = "",
                        sample_ctx: dict | None = None, reuse_threshold: float = 0.82,
                        domain_id: str | None = None) -> dict:
        # Reuse is scoped to this domain — each domain owns its agents (isolation).
        existing = self.memory.find_similar(role, name, threshold=reuse_threshold,
                                            only_validated=True, domain_id=domain_id)
        if existing:
            self.memory.bump_usage(existing["id"])
            return {"reused": True, "agent_id": existing["id"], "name": existing["name"],
                    "similarity": existing.get("similarity"), "validated": True}
        if self.llm is None:
            return {"reused": False, "generated": False, "error": "no LLM available"}
        sample = _describe_sample(sample_ctx)
        ctx = (context or "")[:_CTX_MAX]
        algo_alias, code_alias = self._alias("algo"), self._alias("coder")
        trace: dict = {"algo": {"alias": algo_alias, "prompt": None, "response": None},
                       "code": {"alias": code_alias, "prompt": None, "response": None}}

        # -- algorithm (with empty-response retry: drop the domain context) --
        algorithm, ap, err = self._robust(self.llm.algo,
                                          lambda c: _algo_prompt(name, role, c, sample),
                                          ctx, max_tokens=1200)
        trace["algo"]["prompt"], trace["algo"]["response"] = ap, (algorithm or f"ERROR: {err}")
        if not algorithm:
            return {"reused": False, "generated": False, "stage": "algo",
                    "error": (err or "empty response from the algorithm model"), "trace": trace}

        # -- code (empty-response retry too) --
        raw, cp, err = self._robust(self.llm.code,
                                    lambda _c: _code_prompt(name, role, algorithm, sample),
                                    ctx, max_tokens=1600)
        trace["code"]["prompt"], trace["code"]["response"] = cp, (raw or f"ERROR: {err}")
        if not raw:
            return {"reused": False, "generated": False, "stage": "code",
                    "error": (err or "empty response from the code model"), "trace": trace}

        code = _extract_code(raw)
        report = validate_code(code)
        test = run_agent_code(code, sample_ctx) if (report["ok"] and sample_ctx) else None
        aid = self.memory.add(name, role, algorithm, code,
                              provider_algo=algo_alias, provider_code=code_alias,
                              note=self._note(report, test), trace=json.dumps(trace),
                              domain_id=domain_id)
        if test is not None:
            self.memory.record_io(aid, sample_ctx, test)   # show the auto-test I/O
        return {"reused": False, "generated": True, "agent_id": aid, "name": name,
                "validation": report, "test": test, "validated": False, "trace": trace}

    def _robust(self, fn, build_prompt, ctx, **kw):
        """Call fn(prompt); if it returns empty text, retry once with the context dropped.
        Returns (text, prompt_used, error)."""
        prompt = build_prompt(ctx)
        try:
            out = (fn(prompt, **kw) or "").strip()
        except Exception as e:  # noqa: BLE001
            return "", prompt, str(e)[:200]
        if out:
            return out, prompt, None
        # empty text — some prompts trigger this; retry without the domain context
        prompt2 = build_prompt("")
        try:
            out2 = (fn(prompt2, **kw) or "").strip()
        except Exception as e:  # noqa: BLE001
            return "", prompt2, str(e)[:200]
        return out2, prompt2, (None if out2 else "model returned an empty response")

    def update_code(self, aid: str, code: str, sample_ctx: dict | None = None) -> dict:
        a = self.memory.get(aid)
        if not a:
            return {"ok": False, "error": "not found"}
        report = validate_code(code)
        test = run_agent_code(code, sample_ctx) if (report["ok"] and sample_ctx) else None
        self.memory.update_code(aid, code)
        self.memory.set_note(aid, self._note(report, test))
        if test is not None:
            self.memory.record_io(aid, sample_ctx, test)
        return {"ok": True, "id": aid, "validation": report, "test": test, "validated": False}

    def test(self, aid: str, ctx: dict) -> dict:
        a = self.memory.get(aid)
        if not a:
            return {"ok": False, "error": "not found"}
        rep = validate_code(a["code"])
        if not rep["ok"]:
            return {"ok": False, "reason": "failed validation", "issues": rep["issues"]}
        out = run_agent_code(a["code"], ctx)
        self.memory.record_io(aid, ctx, out)     # remember last I/O for verification
        return out

    def run(self, aid: str, ctx: dict, allow_unvalidated: bool = False) -> dict:
        a = self.memory.get(aid)
        if not a:
            return {"ok": False, "error": "agent not found"}
        if not a["validated"] and not allow_unvalidated:
            return {"ok": False, "reason": "agent pending review (validate it first)"}
        rep = validate_code(a["code"])
        if not rep["ok"]:
            return {"ok": False, "reason": "failed validation", "issues": rep["issues"]}
        out = run_agent_code(a["code"], ctx)
        self.memory.record_io(aid, ctx, out)     # remember last I/O for verification
        if out.get("ok"):
            self.memory.bump_usage(aid)
        return out

    def validate_agent(self, aid: str, validated: bool = True) -> dict:
        a = self.memory.get(aid)
        if not a:
            return {"ok": False, "error": "not found"}
        report = validate_code(a["code"])
        if validated and not report["ok"]:
            return {"ok": False, "error": "code fails static validation", "issues": report["issues"]}
        self.memory.mark_validated(aid, validated)
        return {"ok": True, "id": aid, "validated": validated}

    @staticmethod
    def _note(report: dict, test: dict | None) -> str:
        if not report["ok"]:
            return "invalid: " + "; ".join(report["issues"])[:180]
        if test is None:
            return "valid (untested)"
        return "valid, test ok" if test.get("ok") else "valid, TEST ERROR: " + str(test.get("error"))[:150]
