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
