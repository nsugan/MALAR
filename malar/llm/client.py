"""LLM gateway client — a thin OpenAI-compatible client pointed at LiteLLM.

Agents call ONLY this client, BY ROLE. Each role's ALIAS is resolved as:
    runtime route (LLM tab) -> env default (settings.alias_*) -> the alias in the config.
The provider/model behind each alias lives in infra/litellm_config.yaml — never in code.
Any role can be routed to ANY alias (local Ollama or a cloud API), all through LiteLLM.

Every call (prompt + response or error + latency + caller) is recorded in a shared ring
buffer (LLM_LOG) so the Web UI can show exactly what went to the LLM and what came back.
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock

from malar.core.config import get_settings

LLM_LOG: deque = deque(maxlen=500)
_LOG_LOCK = Lock()
_SEQ = {"n": 0}


def _record(entry: dict) -> dict:
    with _LOG_LOCK:
        _SEQ["n"] += 1
        entry["id"] = _SEQ["n"]
        LLM_LOG.append(entry)
    return entry


def get_log(limit: int = 100) -> list[dict]:
    with _LOG_LOCK:
        items = list(LLM_LOG)
    return items[-limit:][::-1]


def clear_log() -> int:
    with _LOG_LOCK:
        n = len(LLM_LOG)
        LLM_LOG.clear()
    return n


# Runtime provider routing — which configured ALIAS each agent role uses. None => the env
# default (settings.alias_*). Set from the LLM tab to route any role to any provider.
_ROUTE: dict = {"reasoner": None, "fast": None, "vision": None, "coder": None, "algo": None}


def set_route(role: str, alias: str | None) -> dict:
    if role in _ROUTE:
        _ROUTE[role] = alias or None
    return dict(_ROUTE)


def get_route() -> dict:
    return dict(_ROUTE)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings=None):
        self.s = settings or get_settings()
        self._client = None

    def _ensure(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except Exception as e:  # noqa: BLE001
                raise LLMUnavailable(f"openai sdk missing: {e}")
            self._client = OpenAI(base_url=self.s.llm_base_url, api_key=self.s.llm_api_key,
                                  timeout=float(getattr(self.s, 'llm_timeout', 120.0)),
                                  max_retries=0)
        return self._client

    def list_models(self) -> list[str]:
        client = self._ensure()
        try:
            return [m.id for m in client.models.list().data]
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"gateway unreachable: {e}")

    def complete(self, alias: str, prompt: str, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048, source: str = "agent") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        try:
            client = self._ensure()
            resp = client.chat.completions.create(
                model=alias, messages=messages, temperature=temperature, max_tokens=max_tokens)
            out = resp.choices[0].message.content or ""
            _record({"ts": t0, "alias": alias, "source": source, "kind": "text",
                     "system": system, "prompt": prompt, "response": out, "ok": True,
                     "error": None, "latency_ms": int((time.time() - t0) * 1000)})
            return out
        except Exception as e:  # noqa: BLE001
            _record({"ts": t0, "alias": alias, "source": source, "kind": "text",
                     "system": system, "prompt": prompt, "response": None, "ok": False,
                     "error": str(e), "latency_ms": int((time.time() - t0) * 1000)})
            raise LLMUnavailable(f"completion failed via '{alias}': {e}")

    def vision(self, alias: str, prompt: str, image_b64: str,
               max_tokens: int = 1536, source: str = "agent") -> str:
        """Multimodal call (Gemma is multimodal). image_b64 is a data URL payload."""
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]
        t0 = time.time()
        try:
            client = self._ensure()
            resp = client.chat.completions.create(
                model=alias, messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens)
            out = resp.choices[0].message.content or ""
            _record({"ts": t0, "alias": alias, "source": source, "kind": "vision",
                     "system": None, "prompt": prompt + " [+image]", "response": out,
                     "ok": True, "error": None, "latency_ms": int((time.time() - t0) * 1000)})
            return out
        except Exception as e:  # noqa: BLE001
            _record({"ts": t0, "alias": alias, "source": source, "kind": "vision",
                     "system": None, "prompt": prompt + " [+image]", "response": None,
                     "ok": False, "error": str(e), "latency_ms": int((time.time() - t0) * 1000)})
            raise LLMUnavailable(f"vision call failed via '{alias}': {e}")

    def reason(self, prompt: str, **kw) -> str:
        return self.complete(_ROUTE["reasoner"] or self.s.alias_reasoner, prompt, **kw)

    def fast(self, prompt: str, **kw) -> str:
        return self.complete(_ROUTE["fast"] or self.s.alias_fast, prompt, **kw)

    def vision_route(self, prompt: str, image_b64: str, **kw) -> str:
        return self.vision(_ROUTE["vision"] or self.s.alias_vision, prompt, image_b64, **kw)

    def code(self, prompt: str, **kw) -> str:
        """Agent CODE generation. Alias = LLM-tab route -> env default -> any LiteLLM alias."""
        return self.complete(_ROUTE["coder"] or self.s.alias_coder, prompt, source="coder", **kw)

    def algo(self, prompt: str, **kw) -> str:
        """ALGORITHM design. Alias = LLM-tab route -> env default -> any LiteLLM alias."""
        return self.complete(_ROUTE["algo"] or self.s.alias_algo, prompt, source="algo", **kw)
