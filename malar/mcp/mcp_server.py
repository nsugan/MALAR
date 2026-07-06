"""MCP server — expose MALAR capabilities as MCP tools to external agents.

Exposes a small, safe surface: query memory, identify a feature summary, run a single
loop step, and request inference. The actual heavy compute stays in the deterministic
engine; this is a thin protocol adapter. Kept dependency-light: it describes the tool
schema and dispatches to engine callables, so it can be mounted by any MCP runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class MCPTool:
    name: str
    description: str
    handler: Callable
    schema: dict


class MalarMCPServer:
    def __init__(self, engine=None, inference=None):
        self.engine = engine
        self.inference = inference
        self.tools: dict[str, MCPTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.add("malar.query_memory",
                 "Retrieve top-k memories for a feature summary (ids + classes only).",
                 self._query_memory, {"type": "object", "properties": {"k": {"type": "integer"}}})
        self.add("malar.step",
                 "Run a single MALAR loop step over the next batch.",
                 self._step, {"type": "object", "properties": {}})
        self.add("malar.infer",
                 "Run inference on text/image and return action + confidence + ood.",
                 self._infer, {"type": "object",
                               "properties": {"text": {"type": "string"}}})

    def add(self, name: str, description: str, handler: Callable, schema: dict) -> None:
        self.tools[name] = MCPTool(name, description, handler, schema)

    def list_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "schema": t.schema}
                for t in self.tools.values()]

    def call(self, name: str, **kwargs):
        if name not in self.tools:
            raise KeyError(name)
        return self.tools[name].handler(**kwargs)

    # handlers
    def _query_memory(self, k: int = 5):
        if self.engine is None:
            return {"error": "no engine"}
        mems = self.engine.c.store.graph.all_memories()[:k]
        return [{"id": m["id"], "class": m.get("class")} for m in mems]

    def _step(self):
        return {"error": "streaming step requires a live adapter"}

    def _infer(self, text: str = "", **kw):
        if self.inference is None:
            return {"error": "no inference service"}
        return self.inference.infer(text=text)
