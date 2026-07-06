"""DomainManager — per-domain isolation (UI plan §1.3).

Owns the domain_id <-> (Qdrant collection `mem__{id}`, data dir, engine) mapping and
guarantees every store call is scoped to the active domain. Domains are created,
selected, reset, deleted; switching never bleeds data across domains because each
domain owns its own MalarEngine instance (isolated registry / memory graph / Qdrant
collection / artifact dir).

Persisted: data/domains/registry.json (metadata) + data/domains/{id}/ (spec, dirs,
results). Engines are built lazily and cached.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from malar.core.config import get_settings
from malar.core.loop import MalarEngine
from malar.fields.functional_field import FunctionalSpec
from malar.fields.objectives import ObjectiveSpec
from malar.world.adapters.raman import RamanAdapter
from malar.world.adapters.sensor import SensorTimeseriesAdapter
from malar.world.adapters.synthetic import SyntheticAdapter

_DEFAULT_OBJECTIVES = {
    "raman": [("sensitivity", 0.9), ("specificity", 0.9)],
    "sensor": [("yield", 0.9), ("defect_rate", 0.05)],
    "synthetic": [("quality", 0.8)],
}
_DEFAULT_ACTIONS = {
    "raman": ["confirm_rtpcr", "escalate", "watchlist"],
    "sensor": ["adjust_feed_rate", "re_tool", "hold_lot", "flag"],
    "synthetic": ["flag", "watchlist"],
}


@dataclass
class DomainMeta:
    id: str
    name: str
    description: str = ""
    adapter_type: str = "raman"            # raman | sensor | synthetic
    data_folders: list[str] = field(default_factory=list)
    data_description: str = ""
    objectives: list[dict] = field(default_factory=list)   # [{key,target}]
    functionals: list[dict] = field(default_factory=list)  # [{dim,actions}]
    classes: list[str] = field(default_factory=list)
    coverage_targets: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)            # LLM-derived learning plan
    extra_agents: list = field(default_factory=list)    # LLM-proposed process agents
    llm_assist: bool = False                            # call LLM per item during training
    created: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or f"domain_{int(time.time())}"


class DomainManager:
    def __init__(self):
        self.s = get_settings()
        self.root = self.s.data_dir / "domains"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.metas: dict[str, DomainMeta] = {}
        self._engines: dict[str, MalarEngine] = {}
        self.active_id: str | None = None
        self._load()

    # -- persistence ---------------------------------------------------
    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
            except Exception as e:  # noqa: BLE001
                # tolerate a corrupt/truncated registry (e.g. interrupted sync):
                # back it up and start clean rather than crashing the whole app.
                backup = self.registry_path.with_suffix(".json.corrupt")
                try:
                    self.registry_path.replace(backup)
                    print(f"[domains] corrupt registry.json ({e}); backed up to {backup.name}")
                except Exception:
                    pass
                data = {}
            for m in data.get("domains", []):
                try:
                    known = {k: m[k] for k in DomainMeta.__dataclass_fields__ if k in m}
                    self.metas[m["id"]] = DomainMeta(**known)
                except Exception:
                    continue
            self.active_id = data.get("active_id")
        if self.active_id not in self.metas:
            self.active_id = next(iter(self.metas), None)

    def _save(self) -> None:
        self.registry_path.write_text(json.dumps(
            {"active_id": self.active_id,
             "domains": [asdict(m) for m in self.metas.values()]}, indent=2))

    # -- CRUD ----------------------------------------------------------
    def create(self, name: str, description: str = "", adapter_type: str = "raman",
               classes: list[str] | None = None) -> DomainMeta:
        did = _slug(name)
        base = did
        i = 1
        while did in self.metas:
            did = f"{base}_{i}"
            i += 1
        objs = [{"key": k, "target": t} for k, t in _DEFAULT_OBJECTIVES.get(adapter_type, [("quality", 0.8)])]
        funcs = [{"dim": "response", "actions": _DEFAULT_ACTIONS.get(adapter_type, ["flag"])}]
        if adapter_type == "raman" and not classes:
            classes = ["sars_cov_2", "influenza_a", "rsv", "negative"]
        meta = DomainMeta(id=did, name=name, description=description, adapter_type=adapter_type,
                          objectives=objs, functionals=funcs, classes=classes or [],
                          coverage_targets={"novelty_rate_below": 0.1,
                                            "per_class_confidence_above": 0.7,
                                            "ood_rate_below": 0.15,
                                            "value_uncertainty_below": 0.2})
        self.metas[did] = meta
        (self.root / did).mkdir(parents=True, exist_ok=True)
        for sub in ("artifacts", "snapshots", "results", "raw"):
            (self.root / did / sub).mkdir(exist_ok=True)
        if self.active_id is None:
            self.active_id = did
        self._save()
        self._write_spec(meta)
        return meta

    def list(self) -> list[DomainMeta]:
        return list(self.metas.values())

    def get(self, did: str) -> DomainMeta | None:
        return self.metas.get(did)

    def active(self) -> str | None:
        return self.active_id

    def select(self, did: str) -> DomainMeta:
        if did not in self.metas:
            raise KeyError(did)
        self.active_id = did
        self.metas[did].last_activity = time.time()
        self._save()
        return self.metas[did]

    def reset(self, did: str) -> None:
        """Wipe a domain's memory + artifacts, keep its config."""
        if did not in self.metas:
            raise KeyError(did)
        eng = self._engines.pop(did, None)
        if eng is not None:
            try:
                eng.c.store.qdrant.client.delete_collection(f"mem__{did}")
            except Exception:
                pass
        for sub in ("artifacts", "snapshots", "results"):
            d = self.root / did / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        self._save()

    def delete(self, did: str) -> None:
        self._engines.pop(did, None)
        self.metas.pop(did, None)
        shutil.rmtree(self.root / did, ignore_errors=True)
        if self.active_id == did:
            self.active_id = next(iter(self.metas), None)
        self._save()

    # -- config --------------------------------------------------------
    def update_config(self, did: str, **kw) -> DomainMeta:
        meta = self.metas[did]
        for k in ("data_folders", "data_description", "objectives", "functionals",
                  "coverage_targets", "classes", "adapter_type", "description",
                  "plan", "extra_agents", "llm_assist"):
            if k in kw and kw[k] is not None:
                setattr(meta, k, kw[k])
        meta.last_activity = time.time()
        self._save()
        self._write_spec(meta)
        # rebuild engine so new objectives/functionals take effect
        self._engines.pop(did, None)
        return meta

    def set_flag(self, did: str, **flags) -> "DomainMeta":
        """Set lightweight meta flags WITHOUT rebuilding the engine (keeps training)."""
        meta = self.metas[did]
        for k, v in flags.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
        meta.last_activity = time.time()
        self._save()
        return meta

    def _write_spec(self, meta: DomainMeta) -> None:
        import yaml

        spec = {
            "domain": meta.id, "description": meta.description,
            "adapters": [{"type": meta.adapter_type, "params": {}}],
            "encoders": {"topology": {"n_pixels": 12}, "spectral": {"latent_dim": 16},
                         "graph": {"emb_dim": 48}},
            "objectives": [{"key": o["key"], "target": o.get("target")} for o in meta.objectives],
            "functionals": meta.functionals,
            "classes": meta.classes,
            "coverage_targets": meta.coverage_targets,
            "data_folders": meta.data_folders,
            "data_description": meta.data_description,
            "budget": {"max_ticks": 24, "hitl_labels": 30},
        }
        (Path("domains")).mkdir(exist_ok=True)
        (self.root / meta.id / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        Path("domains", f"{meta.id}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    # -- engine --------------------------------------------------------
    def _make_adapter(self, meta: DomainMeta):
        if meta.adapter_type == "raman":
            return RamanAdapter(n_ticks=24, n_per_class=12, n_bands=128, k=6,
                                classes=meta.classes or None)
        if meta.adapter_type == "sensor":
            return SensorTimeseriesAdapter(n_sensors=24, n_ticks=18, k=4)
        return SyntheticAdapter(n_ticks=18, points_per_tick=40, k=6)

    def engine(self, did: str | None = None, llm_client=None) -> MalarEngine:
        did = did or self.active_id
        if did is None:
            raise RuntimeError("no active domain; create one first")
        if did in self._engines:
            return self._engines[did]
        meta = self.metas[did]
        objectives = [ObjectiveSpec(key=o["key"], target=o.get("target")) for o in meta.objectives]
        functionals = [FunctionalSpec(dim=f["dim"], actions=list(f.get("actions", [])))
                       for f in meta.functionals]
        eng = MalarEngine(self._make_adapter(meta), world_id=did, objectives=objectives,
                          functionals=functionals, budget=128, llm_client=llm_client,
                          domain_id=did)
        # Restore learned state (prototypes + values + affordances) from a prior session.
        try:
            from malar.domains.persistence import load_engine_state, state_path
            load_engine_state(eng, state_path(self.root, did))
        except Exception:
            pass
        self._engines[did] = eng
        return eng

    def save_engine_state(self, did: str) -> dict:
        """Snapshot a domain's learned state so predictions survive a restart."""
        eng = self._engines.get(did)
        if eng is None:
            return {"saved": False, "reason": "engine not loaded"}
        from malar.domains.persistence import save_engine_state, state_path
        return {"saved": True, **save_engine_state(eng, state_path(self.root, did))}

    def status(self, did: str) -> dict:
        meta = self.metas[did]
        eng = self._engines.get(did)
        classes = eng.c.registry.classes() if eng else []
        mem = eng.memory_size() if eng else 0
        return {"id": did, "name": meta.name, "objects_learned": len(classes),
                "classes": classes, "memory_size": mem,
                "adapter_type": meta.adapter_type, "last_activity": meta.last_activity}


_MANAGER: DomainManager | None = None


def get_manager() -> DomainManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DomainManager()
    return _MANAGER
