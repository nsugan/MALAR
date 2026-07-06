"""Learning campaign (Planner) — training mode.

Explore frontiers + active learning toward under-covered classes -> identify -> anchor
-> learn {v_{k,c}} and F_j from data + HITL -> Curator stores/merges/decays + propagates
corrections -> track coverage per objective/class -> STOP on coverage targets or budget.

Coverage signals (rolling window):
  * novelty_rate            — fraction of recent ticks that INSERTED a new memory.
  * per_class_confidence    — mean identification/recall confidence per class.
  * ood_rate                — fraction of recent ticks flagged out-of-distribution.
  * value_uncertainty       — spread of learned values (proxy: 1 - mean confidence).

Stops when all targets are met, or the tick / HITL-label budget is exhausted.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from malar.core.loop import MalarEngine
from malar.training.domainspec import DomainSpec, load_domainspec


@dataclass
class CoverageReport:
    tick: int
    novelty_rate: float
    ood_rate: float
    per_class_confidence: dict
    value_uncertainty: float
    targets_met: bool
    memory_size: int
    stop_reason: str = ""
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tick": self.tick, "novelty_rate": round(self.novelty_rate, 3),
                "ood_rate": round(self.ood_rate, 3),
                "per_class_confidence": {k: round(v, 3) for k, v in self.per_class_confidence.items()},
                "value_uncertainty": round(self.value_uncertainty, 3),
                "targets_met": self.targets_met, "memory_size": self.memory_size,
                "stop_reason": self.stop_reason, "history": self.history}


class Campaign:
    def __init__(self, spec: DomainSpec, window: int = 6):
        self.spec = spec
        self.window = window
        self.adapter = spec.make_adapter()
        self.engine = MalarEngine(self.adapter, world_id=spec.domain,
                                  objectives=spec.objectives, functionals=spec.functionals,
                                  budget=128)
        self.insert_window: deque = deque(maxlen=window)
        self.ood_window: deque = deque(maxlen=window)
        self.class_conf: dict[str, list] = {}
        self.history: list[dict] = []

    def _update_coverage(self, st) -> None:
        action = st.curator_decision.action
        self.insert_window.append(1 if action in ("insert", "evict+insert") else 0)
        conf_ok = st.curator_decision.novelty.get("U", 0.0) < 0.5
        self.ood_window.append(0 if conf_ok else 1)
        cls = st.objects[0]["class"]
        score = st.stage_outputs.get("identify", {}).get("score", 0.0)
        # confidence proxy: 1 - novelty(phi) once a class is recognised
        conf = max(0.0, 1.0 - st.curator_decision.novelty.get("d_phi", 1.0))
        self.class_conf.setdefault(cls, []).append(max(conf, score))

    def _coverage(self, tick: int) -> CoverageReport:
        nov = float(np.mean(self.insert_window)) if self.insert_window else 1.0
        ood = float(np.mean(self.ood_window)) if self.ood_window else 1.0
        per_class = {c: float(np.mean(v[-self.window:])) for c, v in self.class_conf.items()}
        mean_conf = float(np.mean(list(per_class.values()))) if per_class else 0.0
        value_unc = 1.0 - mean_conf
        ct = self.spec.coverage
        met = (nov <= ct.novelty_rate_below and ood <= ct.ood_rate_below
               and value_unc <= ct.value_uncertainty_below
               and (per_class and min(per_class.values()) >= ct.per_class_confidence_above))
        return CoverageReport(tick=tick, novelty_rate=nov, ood_rate=ood,
                              per_class_confidence=per_class, value_uncertainty=value_unc,
                              targets_met=bool(met), memory_size=self.engine.memory_size())

    def run(self, verbose: bool = True) -> CoverageReport:
        max_ticks = self.spec.budget.max_ticks
        tick = 0
        report = self._coverage(0)
        for batch in self.adapter.stream():
            if tick >= max_ticks:
                report.stop_reason = "budget: max_ticks reached"
                break
            st = self.engine.step(batch, mode="train")
            self._update_coverage(st)
            report = self._coverage(tick)
            self.history.append(report.to_dict())
            if verbose:
                print(f"t={tick} nov={report.novelty_rate:.2f} ood={report.ood_rate:.2f} "
                      f"val_unc={report.value_uncertainty:.2f} mem={report.memory_size} "
                      f"met={report.targets_met}")
            tick += 1
            if report.targets_met and tick >= self.window:
                report.stop_reason = "coverage targets met"
                break
        report.history = self.history
        if not report.stop_reason:
            report.stop_reason = "stream exhausted"
        return report


def run_campaign(domain: str, verbose: bool = True) -> CoverageReport:
    spec = load_domainspec(domain if domain.endswith(".yaml") else f"domains/{domain}.yaml")
    return Campaign(spec).run(verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="raman_virus")
    ap.add_argument("--out", default="data/coverage_report.json")
    args = ap.parse_args(argv)
    report = run_campaign(args.domain)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nstop_reason: {report.stop_reason}")
    print(f"coverage report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
