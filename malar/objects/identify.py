"""LLM-assisted identification (conformal-gated).

Metric (tool): encoder computes phi/h/g; retrieval returns top-k with distances.
Semantic (LLM): the agent sends a FEATURE SUMMARY (H0/H1/H2 counts, persistence
ranges, spectral peak descriptors) + retrieval context to the gateway -> label +
candidate values + rationale. Arbitration: deterministic Score + conformal gate
decide; human override is final. Raw vectors are NEVER sent to the LLM.

LLM-heavy at cold start; as memory matures the deterministic Score (f_k) takes over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from malar.objects.registry import ObjectRegistry, RetrievalHit
from malar.validation.conformal import ConformalGate


@dataclass
class Identification:
    matched: bool
    cls: str | None
    score: float
    method: str                # 'metric' | 'llm' | 'candidate'
    conformal_ok: bool
    rationale: str = ""
    topk: list = field(default_factory=list)
    world_ctx_id: str | None = None


class Identifier:
    def __init__(self, registry: ObjectRegistry, gate: ConformalGate | None = None,
                 theta_match: float = 0.75, llm_client=None):
        self.registry = registry
        self.gate = gate or ConformalGate(alpha=0.1)
        self.theta_match = theta_match
        self.llm_client = llm_client  # optional; LLMClient

    def identify(self, phi, h, g, summary: dict, k: int = 5, world_ctx_id: str | None = None,
                 t: int = 0) -> Identification:
        hits: list[RetrievalHit] = self.registry.topk(phi, h, g, k=k)
        if not hits:
            return self._fallback_candidate(phi, h, g, t, world_ctx_id,
                                            reason="empty registry")
        best = hits[0]
        conformal_ok = self.gate.accept(best.score)

        # Strong deterministic match — metric path wins, LLM not needed.
        if best.score >= self.theta_match and conformal_ok:
            return Identification(matched=True, cls=best.obj.cls, score=best.score,
                                  method="metric", conformal_ok=True,
                                  rationale=f"metric Score={best.score:.3f} >= theta",
                                  topk=hits, world_ctx_id=world_ctx_id)

        # Cold-start / uncertain — ask the LLM for a hypothesis from the SUMMARY only.
        if self.llm_client is not None:
            label, rationale = self._llm_hypothesis(summary, hits)
            if label and label in self.registry.classes() and conformal_ok:
                # arbitration: LLM may relabel only if conformal gate also accepts
                return Identification(matched=True, cls=label, score=best.score,
                                      method="llm", conformal_ok=conformal_ok,
                                      rationale=rationale, topk=hits, world_ctx_id=world_ctx_id)

        # No confident decision -> store a candidate for human labeling.
        return self._fallback_candidate(phi, h, g, t, world_ctx_id,
                                        reason=f"uncertain (best={best.score:.3f})",
                                        topk=hits)

    def _llm_hypothesis(self, summary: dict, hits: list[RetrievalHit]):
        ctx_lines = [f"- {hit.obj.cls}: Score={hit.score:.3f}" for hit in hits]
        prompt = (
            "You identify an object from a TOPOLOGICAL/SPECTRAL feature SUMMARY and "
            "retrieval context. Choose the single best class label from the candidates, "
            "or reply UNKNOWN.\n"
            f"Feature summary: {summary}\n"
            f"Retrieval candidates:\n" + "\n".join(ctx_lines) +
            "\nReply with: LABEL: <class or UNKNOWN> ; REASON: <one line>"
        )
        try:
            out = self.llm_client.reason(prompt, source="identify")
        except Exception as e:  # noqa: BLE001
            return None, f"llm unavailable: {e}"
        label = None
        reason = out.strip()
        for line in out.splitlines():
            if "LABEL:" in line:
                label = line.split("LABEL:", 1)[1].split(";")[0].strip()
        if label and label.upper() == "UNKNOWN":
            label = None
        return label, reason

    def _fallback_candidate(self, phi, h, g, t, world_ctx_id, reason: str, topk=None):
        self.registry.add_candidate(phi, h, g, t, world_ctx_id)
        return Identification(matched=False, cls=None, score=topk[0].score if topk else 0.0,
                              method="candidate", conformal_ok=False,
                              rationale=f"candidate stored for labeling: {reason}",
                              topk=topk or [], world_ctx_id=world_ctx_id,
                              )
