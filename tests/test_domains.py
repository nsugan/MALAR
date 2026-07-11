"""Per-domain isolation + UI backend flow tests (U0-U8)."""
import numpy as np
from malar.domains.manager import DomainManager
from malar.training.folder_analyzer import analyze_folder, select_subset, FolderReport, FileEntry


def test_domain_isolation(tmp_path, monkeypatch):
    from malar.core import config
    monkeypatch.setenv("MALAR_DATA_DIR", str(tmp_path))
    config._SETTINGS = None
    dm = DomainManager()
    a = dm.create("A", adapter_type="raman")
    b = dm.create("B", adapter_type="raman")
    ea, eb = dm.engine(a.id), dm.engine(b.id)
    for i, batch in enumerate(ea.adapter.stream()):
        if i >= 4:
            break
        ea.step(batch)
    assert ea.memory_size() > 0
    assert eb.memory_size() == 0                     # no leak into B
    assert ea.c.store.qdrant.alias == f"mem__{a.id}"
    dm.reset(a.id)
    assert dm.engine(a.id).memory_size() == 0        # reset wipes A only
    config._SETTINGS = None


def test_generic_synthetic_domain_trains(tmp_path, monkeypatch):
    """A default (synthetic) domain builds a generic feature-cloud queue and trains
    end-to-end with no domain-specific (raman) coupling. (V4 genericization)"""
    from malar.core import config
    monkeypatch.setenv("MALAR_DATA_DIR", str(tmp_path))
    config._SETTINGS = None
    from malar.api.domain_service import DomainService
    dm = DomainManager()
    d = dm.create("Generic")                      # no adapter_type -> synthetic default
    assert d.adapter_type == "synthetic"
    svc = DomainService(manager=dm)
    q = svc.build_queue(d.id, n_per_class=1, replicates=8)
    assert q and all(it["kind"] == "features" for it in q)
    assert {it["label"] for it in q} == {"class_a", "class_b", "class_c"}
    r = svc.confirm(d.id, "confirm")              # commit the first item
    assert r.get("committed") and dm.engine(d.id).memory_size() >= 1
    config._SETTINGS = None


def test_consolidated_context_has_both_descriptions(tmp_path, monkeypatch):
    from malar.core import config
    monkeypatch.setenv("MALAR_DATA_DIR", str(tmp_path))
    config._SETTINGS = None
    dm = DomainManager()
    d = dm.create("Widgets", description="inspect widgets for cracks")
    dm.update_config(d.id, data_description="128-band signals per sample, folder-per-class")
    ctx = dm.get(d.id).consolidated_context()
    assert "inspect widgets for cracks" in ctx           # domain description
    assert "128-band signals per sample" in ctx          # data description
    config._SETTINGS = None


def test_planner_embeds_domain_and_data_context_in_every_query():
    """The planner/orchestrator must always send BOTH the domain and data description."""
    from malar.agents.planner import DomainPlanner

    ctx = ("Domain: Widgets\nDomain description: inspect widgets for cracks\n"
           "Data description: 128-band signals per sample")
    captured = {}

    class CapLLM:
        def reason(self, prompt, **kw):
            captured["prompt"] = prompt
            # valid-enough JSON for both plan_domain and suggest_object
            return ('{"objectives":[{"key":"quality","target":0.8}],"functionals":[],'
                    '"class":"a","affordances":[],"confidence":0.5}')

    p = DomainPlanner(CapLLM(), context=ctx)
    p.plan_domain(classes=["a", "b"], adapter_type="synthetic")
    assert "inspect widgets for cracks" in captured["prompt"]
    assert "128-band signals per sample" in captured["prompt"]

    captured.clear()
    p.suggest_object(feature_summary={"h0": 1}, candidates=["a"], functional_dims=["response"])
    assert "inspect widgets for cracks" in captured["prompt"]
    assert "128-band signals per sample" in captured["prompt"]


def test_folder_analyzer_empty():
    rep = analyze_folder("")
    assert rep.n_files == 0 and "no folder" in rep.summary


def test_select_subset_synthetic(tmp_path):
    # build a tiny fake folder with class subdirs
    for cls in ("a", "b"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            np.save(d / f"s{i}.npy", np.random.rand(8))
    rep = analyze_folder(str(tmp_path))
    assert rep.n_files == 6
    assert set(rep.detected_labels) == {"a", "b"}
    sub = select_subset(rep)
    assert len(sub["subset"]) >= 1 and sub["rationale"]
