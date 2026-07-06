"""Inference service — acting on unknown inputs (read-mostly), behind a hard OOD gate.

Pipeline:
  1. front-end maps image/video/text -> phi/h/graph + world_emb.
  2. identify against learned objects (retrieval + conformal), validity-gated via world_emb.
  3. look up the object's R (and diffused V) and affordances F.
  4. Policy selects from the F-afforded set ranked by R/V -> action + rationale +
     matched objects/contexts + confidence.
  5. OOD gate: low confidence / out-of-distribution -> "outside trained domain", NO action,
     defer to human; optionally queue the novel case back to training.

Returns {action, rationale, matched_objects, world_contexts, confidence, ood}. Writes
only logs + an optional novel-case queue.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from malar.fields.functional_field import region_action_set
from malar.inference.frontend import MultimodalFrontEnd
from malar.validation.ood import OODGate


@dataclass
class InferenceResult:
    action: str | None
    rationale: str
    matched_objects: list = field(default_factory=list)
    world_contexts: list = field(default_factory=list)
    confidence: float = 0.0
    ood: bool = False

    def to_dict(self) -> dict:
        return {"action": self.action, "rationale": self.rationale,
                "matched_objects": self.matched_objects, "world_contexts": self.world_contexts,
                "confidence": round(self.confidence, 3), "ood": self.ood}


class InferenceService:
    def __init__(self, engine, llm_client=None, ood_threshold: float = 0.55):
        if engine is None:
            raise RuntimeError("inference requires a trained engine (configure + train first)")
        self.engine = engine
        self.frontend = MultimodalFrontEnd(engine, llm_client)
        self.ood = OODGate(threshold=ood_threshold)
        self.novel_queue: list[dict] = []
        # register trained world contexts for validity gating
        for o in engine.c.registry.objects.values():
            if not o.candidate:
                self.ood.register_context(o.g)

    def infer(self, text: str | None = None, image=None, video=None) -> dict:
        fe = self.frontend.process(text=text, image=image, video=video)

        hits = self.engine.c.registry.topk(fe.phi, fe.h, fe.g, k=3)
        if not hits:
            return InferenceResult(action=None, rationale="no trained objects",
                                   confidence=0.0, ood=True).to_dict()
        best = hits[0]
        confidence = best.score * fe.confidence_scale

        ood = self.ood.combined(best.score, fe.g)
        if ood.ood or confidence < self.ood.threshold:
            self.novel_queue.append({"summary": fe.summary, "modality": fe.modality})
            return InferenceResult(
                action=None,
                rationale=f"outside trained domain ({ood.reason}); deferred to human",
                matched_objects=[], confidence=confidence, ood=True).to_dict()

        cls = best.obj.cls
        objects = [{"class": cls, "ctx": 1.0}]

        # R (and V proxy) for the matched class
        r_values = {}
        for k, ofield in self.engine.c.objective_fields.items():
            R, _ = ofield.evaluate(objects)
            r_values[k] = R
        r_val = max(r_values.values()) if r_values else 0.0
        v_val = r_val  # diffused-V proxy at the object site

        action_set = region_action_set(objects, self.engine.c.functional_fields)
        decision = self.engine.c.policy.decide(action_set, [cls], r_val, v_val)

        return InferenceResult(
            action=decision.action,
            rationale=(f"identified '{cls}' (score={best.score:.3f}, conf={confidence:.3f}); "
                       f"{decision.rationale}"),
            matched_objects=[{"id": best.obj.id, "class": cls, "score": round(best.score, 3)}],
            world_contexts=[best.obj.world_ctx_id] if best.obj.world_ctx_id else [],
            confidence=confidence, ood=False).to_dict()


def main(argv=None) -> int:
    from malar.training.campaign import Campaign
    from malar.training.domainspec import load_domainspec

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="raman_virus")
    ap.add_argument("--text", default="")
    ap.add_argument("--file", default="")
    args = ap.parse_args(argv)

    spec = load_domainspec(f"domains/{args.domain}.yaml")
    camp = Campaign(spec)
    camp.run(verbose=False)
    svc = InferenceService(camp.engine)

    if args.file:
        arr = np.load(args.file) if args.file.endswith(".npy") else None
        res = svc.infer(image=arr)
    else:
        res = svc.infer(text=args.text or "possible sars_cov_2 signature in sample")
    import json
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
