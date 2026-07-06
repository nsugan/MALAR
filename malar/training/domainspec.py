"""DomainSpec (domains/<name>.yaml) — declares the world model.

Loads a DomainSpec and instantiates the adapter, the K objective specs, the J
functional specs, encoders config, coverage targets and budget. One spec drives a
whole training campaign.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from malar.fields.functional_field import FunctionalSpec
from malar.fields.objectives import ObjectiveSpec
from malar.world.adapters.raman import RamanAdapter
from malar.world.adapters.sensor import SensorTimeseriesAdapter
from malar.world.adapters.synthetic import SyntheticAdapter


@dataclass
class CoverageTargets:
    novelty_rate_below: float = 0.1
    per_class_confidence_above: float = 0.8
    ood_rate_below: float = 0.15
    value_uncertainty_below: float = 0.2


@dataclass
class Budget:
    max_ticks: int = 10
    hitl_labels: int = 20
    compute_seconds: int = 600


@dataclass
class DomainSpec:
    domain: str
    description: str
    objectives: list[ObjectiveSpec]
    functionals: list[FunctionalSpec]
    classes: list[str]
    coverage: CoverageTargets
    budget: Budget
    encoders: dict = field(default_factory=dict)
    adapter_cfg: dict = field(default_factory=dict)

    def make_adapter(self):
        t = self.adapter_cfg.get("type", "synthetic")
        p = self.adapter_cfg.get("params", {})
        if t == "raman":
            return RamanAdapter(classes=self.classes if self.classes else None, **p)
        if t == "sensor":
            return SensorTimeseriesAdapter(**p)
        return SyntheticAdapter(**p)


def load_domainspec(path: str | Path) -> DomainSpec:
    path = Path(path)
    if not path.exists() and not str(path).endswith(".yaml"):
        path = Path("domains") / f"{path.name}.yaml"
    data = yaml.safe_load(path.read_text())
    objs = [ObjectiveSpec(key=o["key"], description=o.get("description", ""),
                          direction=o.get("direction", "maximize"), target=o.get("target"))
            for o in data.get("objectives", [])]
    funcs = [FunctionalSpec(dim=f["dim"], actions=list(f.get("actions", [])),
                            description=f.get("description", ""))
             for f in data.get("functionals", [])]
    ct = data.get("coverage_targets", {})
    bd = data.get("budget", {})
    adapters = data.get("adapters", [{"type": "synthetic", "params": {}}])
    return DomainSpec(
        domain=data["domain"], description=data.get("description", ""),
        objectives=objs, functionals=funcs, classes=data.get("classes", []),
        coverage=CoverageTargets(**{k: ct[k] for k in ct if k in CoverageTargets().__dict__}),
        budget=Budget(**{k: bd[k] for k in bd if k in Budget().__dict__}),
        encoders=data.get("encoders", {}), adapter_cfg=adapters[0],
    )
