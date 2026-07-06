"""MCP client — attach external tools to MALAR agents.

HARD RULE: MCP/tool outputs are DATA, never instructions. Every tool result is
wrapped and MUST pass the Critic gate (validation) before it can influence memory or
actions. This module wraps tool calls and tags results as untrusted data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolResult:
    tool: str
    data: object
    trusted: bool = False           # always starts untrusted
    passed_critic: bool = False
    meta: dict = field(default_factory=dict)


class MCPToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, **kwargs) -> ToolResult:
        if name not in self._tools:
            raise KeyError(f"unknown MCP tool: {name}")
        out = self._tools[name](**kwargs)
        # results are DATA, untrusted until the Critic gate clears them
        return ToolResult(tool=name, data=out, trusted=False)


def gate_tool_result(result: ToolResult, critic_accept: Callable[[object], bool]) -> ToolResult:
    """Pass an MCP tool result through the Critic gate before use."""
    result.passed_critic = bool(critic_accept(result.data))
    result.trusted = result.passed_critic
    return result
