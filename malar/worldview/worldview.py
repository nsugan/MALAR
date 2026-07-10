"""WorldView(t) — grounded LLM abstraction of memory state.

Summarises active objects/regimes, novelties, exploration frontiers, an uncertainty
map and open labeling questions. Refreshed from memory **deltas** (summaries + ids,
never raw vectors). Consumed by the Orchestrator (explore) and the Curator (scope
corrections, prioritise labels).

HARD RULE: references EXISTING ids only — never invents objects. Versioned and
auditable. The LLM is optional; without it a deterministic structural summary is
produced (and still only references existing ids).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from malar.memory.neo4j_io import MemoryGraph


@dataclass
class WorldView:
    version: int
    active_object_ids: list[str]
    novel_ids: list[str]
    frontier_ids: list[str]
    open_questions: list[dict]
    uncertainty_map: dict
    narrative: str = ""
    meta: dict = field(default_factory=dict)


class WorldViewBuilder:
    def __init__(self, graph: MemoryGraph, llm_client=None):
        self.graph = graph
        self.llm = llm_client
        self.version = 0
        self.history: list[WorldView] = []

    def _valid_ids(self) -> set[str]:
        return {m["id"] for m in self.graph.all_memories()}

    def build(self, delta_ids: list[str], hitl_pending: list[dict] | None = None,
              uncertainty: dict | None = None) -> WorldView:
        self.version += 1
        valid = self._valid_ids()
        # filter to EXISTING ids only — never invent
        delta_ids = [i for i in delta_ids if i in valid]
        novel = [i for i in delta_ids]
        active = sorted(valid)[:50]
        frontier = [q.get("payload", {}).get("mem_id") for q in (hitl_pending or [])
                    if q.get("payload", {}).get("mem_id") in valid]
        questions = [{"id": q.get("id"), "kind": q.get("kind")} for q in (hitl_pending or [])]

        narrative = self._narrative(active, novel, questions)
        wv = WorldView(version=self.version, active_object_ids=active, novel_ids=novel,
                       frontier_ids=[f for f in frontier if f], open_questions=questions,
                       uncertainty_map=uncertainty or {}, narrative=narrative)
        self.history.append(wv)
        return wv

    def _narrative(self, active, novel, questions) -> str:
        base = (f"{len(active)} active memories; {len(novel)} novel this delta; "
                f"{len(questions)} open labeling questions.")
        if self.llm is None:
            return base
        prompt = ("Summarise the world state in 2 sentences for an operator. "
                  "Reference only the provided ids; do not invent objects.\n"
                  f"Active(ids): {active[:10]}\nNovel(ids): {novel[:10]}\n"
                  f"Open questions: {questions[:10]}\nFacts: {base}")
        try:
            # reasoner ROLE via the route table, not a hardcoded alias. (V4 fix)
            return self.llm.reason(prompt, source="worldview").strip() or base
        except Exception:
            return base
