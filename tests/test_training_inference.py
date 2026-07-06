import numpy as np
from malar.training.domainspec import load_domainspec
from malar.training.campaign import Campaign
from malar.training.checkpoint import save_checkpoint, restore_checkpoint
from malar.core.loop import MalarEngine
from malar.inference.service import InferenceService
from malar.world.adapters.raman import _CLASS_PEAKS, _spectrum


def _trained():
    spec = load_domainspec("tests/fixtures/raman_test.yaml")
    camp = Campaign(spec); report = camp.run(verbose=False)
    return spec, camp, report


def test_campaign_stops_on_coverage():
    spec, camp, report = _trained()
    assert report.targets_met and report.stop_reason == "coverage targets met"
    assert report.per_class_confidence  # per-class confidence present


def test_checkpoint_roundtrip(tmp_path):
    spec, camp, report = _trained()
    save_checkpoint(camp.engine, "t1", domain="raman_virus", coverage=report.to_dict(),
                    root=str(tmp_path))
    fresh = MalarEngine(spec.make_adapter(), world_id="raman_virus",
                        objectives=spec.objectives, functionals=spec.functionals)
    restore_checkpoint(fresh, "t1", root=str(tmp_path))
    assert fresh.memory_size() == camp.engine.memory_size()
    assert fresh.c.spectral.fitted


def test_inference_trained_and_ood():
    spec, camp, report = _trained()
    svc = InferenceService(camp.engine)
    rng = np.random.default_rng(7)
    img = np.array([_spectrum(_CLASS_PEAKS["sars_cov_2"], 128, rng) for _ in range(20)])
    r = svc.infer(image=img)
    assert r["action"] is not None and not r["ood"]
    ood_img = np.clip(rng.normal(0.5, 0.3, (20, 128)), 0, None)
    r2 = svc.infer(image=ood_img)
    assert r2["ood"] and r2["action"] is None


def test_inference_text_resolves_class():
    spec, camp, report = _trained()
    svc = InferenceService(camp.engine)
    r = svc.infer(text="elevated rsv signature suspected")
    assert not r["ood"] and r["matched_objects"][0]["class"] == "rsv"
