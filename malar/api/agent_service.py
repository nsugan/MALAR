"""Singleton wiring for the Agent Factory (generated-agent memory + factory)."""
from __future__ import annotations

from malar.agents.agent_memory import AgentMemory
from malar.agents.factory import AgentFactory
from malar.core.config import get_settings
from malar.llm.client import LLMClient

_MEM: AgentMemory | None = None
_FAC: AgentFactory | None = None


def get_agent_memory() -> AgentMemory:
    global _MEM
    if _MEM is None:
        _MEM = AgentMemory(get_settings().data_dir / "agent_memory.sqlite")
    return _MEM


def get_agent_factory() -> AgentFactory:
    global _FAC
    if _FAC is None:
        _FAC = AgentFactory(get_agent_memory(), LLMClient())
    return _FAC
