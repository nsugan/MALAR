"""Inference Management agent (I6 + second pass I3/I4/I5).

Orchestrates the probabilistic inference layer for one input:

    front-end (text/image/video -> phi/h/g)
      -> per-modality Bayesian posteriors (topo / spectral / graph)   [tools]
      -> product-of-experts fusion                                    [tool]
      -> MCMC posterior-predictive over {class, value, action}        [tool]
      -> cross-domain referencing (opt-in)                            [I3]
      -> diffusion field for image/video                              [I4]
      -> contextual-bandit RL policy for text/context                 [I5]
      -> OOD gate (never fabricate an action outside the domain)
      -> verdict (LLM narrates; deterministic fallback)

"Agents decide; tools compute": the LLM only chooses options and writes the verdict; every
posterior, sample, diffusion field and bandit draw is computed by deterministic tools.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from malar.fields.functional_field import region_action_set
from malar.inference.frontend import MultimodalFrontEnd
from malar.inference.probabilistic.bayes import BayesEnsemble
from malar.inference.probabilistic.crossdomain import CrossDomainReferencer
from malar.inference.probabilistic.diffusion import DiffusionAgent
from malar.inference.probabilistic.mcmc import ValuePosterior, predictive_simulation
from malar.inference.probabilistic.rl import build_from_engine
from malar.validation.ood import OODGate

AGENT_ROSTER = [
    {"id": "bayes_topo", "name": "Bayesian-Topology", "modality": "phi"},
    {"id": "bayes_spectral", "name": "Bayesian-Spectral", "modality": "h"},
    {"id": "bayes_graph", "name": "Bayesian-Graph", "modality": "g"},
    {"id": "crossdomain", "name": "Cross-Domain Reference", "modality": "phi/h/g"},
    {"id": "diffusion", "name": "Diffusion (image/video)", "modality": "image/video"},
    {"id": "rl", "name": "RL (text/context)", "modality": "text"},
    {"id": "manager", "name": "Inference Management", "modality": "all"},
]


@dataclass
class PredictOptions:
    prior_mode: str = "prevalence"
    prior_override: dict | None = None
    weights: dict | None = None
    n_samples: int = 2000
    mcmc_method: str = "mc"
    seed: int = 0
    cross_domain: bool = False
    target_domains: list = field(default_factory=list)
    narrate: bool = True


class InferenceManager:
    def __init__(self, engine, llm_client=None, ood_threshold: float = 0.55,
                 shrinkage: float = 0.2, domain_manager=None, domain_id: str | None = None,
                 class_means: dict | None = None):
        if engine is None:
            raise RuntimeError("inference requires a trained engine (configure + train first)")
        self.engine = engine
        self.llm = llm_client
        self.domain_id = domain_id
        self.frontend = MultimodalFrontEnd(engine, llm_client)
        self.ood = OODGate(threshold=ood_threshold)
        for o in engine.c.registry.objects.values():
            if not o.candidate:
                self.ood.register_context(o.g)
        self.objective_keys = list(engine.c.objective_fields.keys())
        self.bayes = BayesEnsemble(shrinkage=shrinkage).fit(engine)
        self.value_post = ValuePosterior(engine.value_store, self.objective_keys,
                                         counts=self.bayes.counts)
        self.crossref = (CrossDomainReferencer(domain_manager, domain_id)
                         if (domain_manager is not None and domain_id) else None)
        self.diffusion = DiffusionAgent(engine, class_means)
        self.bandit = build_from_engine(engine)
        self._last: dict = {}
        self._last_context = None

    # -- engine policy wrapped as action_of(cls, r_value) ----------------
    def _action_of(self, cls: str, r_value: float) -> str:
        objects = [{"class": cls, "ctx": 1.0}]
        action_set = region_action_set(objects, self.engine.c.functional_fields)
        dec = self.engine.c.policy.decide(action_set, [cls], r_value, r_value)
        return dec.action

    def refit(self) -> None:
        self.bayes.fit(self.engine)
        self.value_post = ValuePosterior(self.engine.value_store, self.objective_keys,
                                         counts=self.bayes.counts)
        self.bandit = build_from_engine(self.engine)

    def agents(self) -> list[dict]:
        roster = []
        for a in AGENT_ROSTER:
            item = dict(a)
            item["enabled"] = True
            last = self._last.get(a["id"])
            if last is not None:
                item["last"] = last
            roster.append(item)
        return roster

    # -- RL feedback (HITL reward) --------------------------------------
    def rl_feedback(self, action: str, reward: float, context=None) -> dict:
        ctx = context if context is not None else self._last_context
        if ctx is None:
            return {"ok": False, "reason": "no context available; run a prediction first"}
        ok = self.bandit.update(ctx, action, float(reward))
        return {"ok": ok, "action": action, "reward": reward, "updates": self.bandit.updates}

    def escalate_diffusion(self, image, steps: int = 50) -> dict:
        return self.diffusion.escalate_generative(image, steps=steps)

    # -- main entry ------------------------------------------------------
    def predict(self, text: str | None = None, image=None, video=None,
                options: PredictOptions | None = None) -> dict:
        opt = options or PredictOptions()
        t0 = time.time()
        if not self.bayes.is_fitted():
            return {"error": "no trained corpus to infer from (train the domain first)",
                    "ood": True}

        fe = self.frontend.process(text=text, image=image, video=video)

        hits = self.engine.c.registry.topk(fe.phi, fe.h, fe.g, k=3)
        best_score = hits[0].score if hits else 0.0
        ood = self.ood.combined(best_score, fe.world_emb)
        confidence = best_score * fe.confidence_scale
        is_ood = ood.ood or confidence < self.ood.threshold

        bres = self.bayes.posteriors(fe.phi, fe.h, fe.g, prior_mode=opt.prior_mode,
                                     prior_override=opt.prior_override, weights=opt.weights)
        for aid, mlabel in (("bayes_topo", "topo"), ("bayes_spectral", "spectral"),
                            ("bayes_graph", "graph")):
            post = bres["per_modality"].get(mlabel, {})
            top = max(post, key=post.get) if post else None
            self._last[aid] = {"top": top, "p": round(post.get(top, 0.0), 3) if top else 0.0,
                               "ts": t0}

        result = {
            "input": {"modality": fe.modality, "summary": fe.summary,
                      "confidence_scale": round(fe.confidence_scale, 3), "parsed": fe.parsed},
            "ood": is_ood, "ood_reason": ood.reason,
            "match_confidence": round(confidence, 3),
            "bayes": bres, "objectives": self.objective_keys,
        }

        # I3 — cross-domain referencing (opt-in; runs even when OOD, it can explain OOD).
        if opt.cross_domain and self.crossref is not None:
            cd = self.crossref.search(fe.phi, fe.h, fe.g, opt.target_domains)
            result["cross_domain"] = cd
            self._last["crossdomain"] = {"matches": len(cd.get("matches", [])), "ts": t0}
        else:
            result["cross_domain"] = {"enabled": False,
                                      "note": "opt-in per request (honors domain isolation)"}

        # I4 — diffusion field for image / video inputs.
        if image is not None or video is not None:
            diff = (self.diffusion.diffuse_video(video) if video is not None
                    else self.diffusion.diffuse_image(image))
            result["diffusion"] = diff
            self._last["diffusion"] = {"top": diff.get("top"), "stage": diff.get("stage"),
                                       "ts": t0}

        # I5 — contextual-bandit RL policy from the context (graph embedding).
        self._last_context = fe.g
        rl = self.bandit.act(fe.g)
        result["rl"] = rl
        self._last["rl"] = {"action": rl.get("action"), "updates": rl.get("updates"), "ts": t0}

        if is_ood:
            result["predicted_action"] = None
            result["mcmc"] = None
            result["verdict"] = (f"Outside trained domain ({ood.reason}); deferred to human. "
                                 f"No action proposed.")
            self._last["manager"] = {"verdict": "OOD", "ts": t0}
            result["agents"] = self.agents()
            result["latency_ms"] = int((time.time() - t0) * 1000)
            return result

        mres = predictive_simulation(
            bres["fused"], self.value_post, self.objective_keys, self._action_of,
            n_samples=opt.n_samples, method=opt.mcmc_method, seed=opt.seed)
        result["mcmc"] = mres
        result["predicted_action"] = mres.get("predicted_action")
        result["action_confidence"] = mres.get("action_confidence")
        result["verdict"] = self._verdict(bres, mres, confidence, opt.narrate)
        self._last["manager"] = {"verdict": result["predicted_action"],
                                 "p": result.get("action_confidence"), "ts": t0}
        result["agents"] = self.agents()
        result["latency_ms"] = int((time.time() - t0) * 1000)
        return result

    # -- verdict ---------------------------------------------------------
    def _verdict(self, bres: dict, mres: dict, confidence: float, narrate: bool) -> str:
        top = bres.get("top")
        top_p = bres.get("top_p", 0.0)
        action = mres.get("predicted_action")
        ap = mres.get("action_confidence", 0.0)
        vs = mres.get("value_summary", {})
        vbits = ", ".join(f"{k}={v['mean']:.2f} [{v['ci_low']:.2f},{v['ci_high']:.2f}]"
                          for k, v in vs.items())
        deterministic = (
            f"Predicted class '{top}' (p={top_p:.2f}); recommended action '{action}' "
            f"(p={ap:.2f}). Objective forecast: {vbits}. Match confidence {confidence:.2f}.")
        if not narrate or self.llm is None:
            return deterministic
        try:
            modality_lines = "; ".join(f"{m}: {max(p, key=p.get)}={max(p.values()):.2f}"
                                       for m, p in bres["per_modality"].items())
            prompt = (
                "You are the Inference Management agent. Summarise this probabilistic "
                "prediction for an operator in 2-3 sentences. Be precise about uncertainty; "
                "do not invent numbers.\n"
                f"Fused top class: {top} (p={top_p:.2f}). Per-modality: {modality_lines}. "
                f"Predicted action: {action} (p={ap:.2f}). Value forecast: {vbits}. "
                f"Entropy: {bres.get('entropy', 0):.2f}.")
            return self.llm.reason(prompt, source="inference_manager", max_tokens=220).strip()
        except Exception:
            return deterministic
