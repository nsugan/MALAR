"""Tests for the V3 probabilistic inference & prediction layer (I0-I2, I6)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from malar.core.loop import MalarEngine
from malar.fields.objectives import ObjectiveSpec
from malar.inference.probabilistic.bayes import BayesEnsemble, fuse_posteriors
from malar.inference.probabilistic.likelihoods import GaussianLikelihood
from malar.inference.probabilistic.manager import InferenceManager, PredictOptions
from malar.inference.probabilistic.mcmc import ValuePosterior, predictive_simulation
from malar.world.adapters.synthetic import SyntheticAdapter


@pytest.fixture(scope="module")
def trained_engine():
    adapter = SyntheticAdapter(n_ticks=20, points_per_tick=40, k=6)
    eng = MalarEngine(adapter, world_id="synthetic", budget=64,
                      objectives=[ObjectiveSpec(key="sensitivity")], domain_id="pytest_inf")
    eng.run(mode="train")
    yield eng
    shutil.rmtree("data/domains/pytest_inf", ignore_errors=True)


def test_gaussian_likelihood_separates_classes():
    rng = np.random.default_rng(0)
    A = rng.normal(0.0, 0.1, size=(20, 8))
    B = rng.normal(3.0, 0.1, size=(20, 8))
    lik = GaussianLikelihood(shrinkage=0.2).fit({"A": A, "B": B})
    ll = lik.loglik(np.zeros(8))            # close to class A
    assert ll["A"] > ll["B"]
    assert set(lik.classes) == {"A", "B"}


def test_posteriors_are_normalised(trained_engine):
    be = BayesEnsemble().fit(trained_engine)
    assert be.is_fitted()
    res = be.posteriors(np.zeros(432), np.zeros(2), np.zeros(48))
    assert abs(sum(res["fused"].values()) - 1.0) < 1e-6
    for post in res["per_modality"].values():
        assert abs(sum(post.values()) - 1.0) < 1e-6


def test_fusion_disagreement_widens_posterior():
    prior = {"a": 0.5, "b": 0.5}
    agree = fuse_posteriors({"topo": {"a": 0.9, "b": 0.1}, "graph": {"a": 0.9, "b": 0.1}}, prior)
    disagree = fuse_posteriors({"topo": {"a": 0.9, "b": 0.1}, "graph": {"a": 0.1, "b": 0.9}}, prior)
    # agreement concentrates; disagreement returns to ~uniform
    assert max(agree.values()) > max(disagree.values())
    assert abs(disagree["a"] - 0.5) < 1e-6


def test_mcmc_predictive_has_credible_intervals(trained_engine):
    be = BayesEnsemble().fit(trained_engine)
    vp = ValuePosterior(trained_engine.value_store, ["sensitivity"], counts=be.counts)
    fused = {c: 1.0 / len(be.classes) for c in be.classes}
    out = predictive_simulation(fused, vp, ["sensitivity"],
                                action_of=lambda c, r: "flag", n_samples=500, seed=1)
    vs = out["value_summary"]["sensitivity"]
    assert vs["ci_low"] <= vs["mean"] <= vs["ci_high"]
    assert abs(sum(out["action_predictive"].values()) - 1.0) < 1e-6
    assert len(out["trace"]) > 0


def test_manager_predict_and_ood(trained_engine):
    mgr = InferenceManager(trained_engine)
    # known class name resolves -> in-distribution prediction with an action
    res = mgr.predict(text="circle in sample", options=PredictOptions(n_samples=400))
    assert not res["ood"]
    assert res["predicted_action"] is not None
    assert res["mcmc"] is not None
    # gibberish -> OOD, no fabricated action
    ood = mgr.predict(text="zzzz qqqq unknown xyzzy")
    assert ood["ood"]
    assert ood["predicted_action"] is None


def test_agents_roster_all_enabled(trained_engine):
    mgr = InferenceManager(trained_engine)
    ids = {a["id"]: a["enabled"] for a in mgr.agents()}
    # second pass shipped -> the full roster is active
    for aid in ("bayes_topo", "bayes_spectral", "bayes_graph",
                "crossdomain", "diffusion", "rl", "manager"):
        assert ids[aid], aid


# -- second pass: cross-domain (I3), diffusion (I4), RL (I5) ------------------

def test_crossdomain_respects_isolation(trained_engine):
    from malar.inference.probabilistic.crossdomain import CrossDomainReferencer

    class FakeDM:
        def __init__(self, eng):
            self._engines = {"other": eng}     # a different domain id, same engine
    ref = CrossDomainReferencer(FakeDM(trained_engine), current_domain_id="other")
    # current domain is excluded -> no self matches
    out = ref.search(np.zeros(432), np.zeros(2), np.zeros(48))
    assert out["enabled"] and out["searched_domains"] == []
    # a genuinely-other domain is searched and matches are down-weighted
    ref2 = CrossDomainReferencer(FakeDM(trained_engine), current_domain_id="self")
    out2 = ref2.search(np.zeros(432), np.zeros(2), np.zeros(48))
    assert out2["searched_domains"] == ["other"]
    for m in out2["matches"]:
        assert m["weight"] < 1.0


def test_diffusion_stage1_field(trained_engine):
    from malar.inference.probabilistic.diffusion import DiffusionAgent
    means = {"circle": np.ones(64), "blob": np.zeros(64)}
    agent = DiffusionAgent(trained_engine, class_means=means)
    img = np.random.default_rng(0).normal(0, 1, size=(12, 64))
    out = agent.diffuse_image(img, steps=8)
    assert out["stage"] == 1 and out["n_regions"] == 12
    assert abs(sum(out["diffusion_vote"].values()) - 1.0) < 1e-6
    assert len(out["heatmap"]) == 12


def test_diffusion_stage2_needs_gpu(trained_engine):
    from malar.inference.probabilistic.diffusion import DiffusionAgent
    out = DiffusionAgent(trained_engine).escalate_generative(np.zeros((4, 8)))
    # CI has no CUDA -> must report unavailable, never fabricate
    assert out["stage"] == 2 and out["available"] is False


def test_rl_bandit_learns_from_feedback(trained_engine):
    from malar.inference.probabilistic.rl import ContextualBandit
    rng = np.random.default_rng(0)
    ctx = rng.normal(0, 1, 16)
    bandit = ContextualBandit(["flag", "watchlist"], dim=16)
    for _ in range(40):
        bandit.update(ctx, "flag", 1.0)
        bandit.update(ctx, "watchlist", 0.0)
    means = bandit.act(ctx)["mean_reward"]
    assert means["flag"] > means["watchlist"]


def test_manager_runs_all_second_pass_agents(trained_engine):
    class FakeDM:
        def __init__(self, eng):
            self._engines = {"pytest_inf": eng}
    mgr = InferenceManager(trained_engine, domain_manager=FakeDM(trained_engine),
                           domain_id="pytest_inf")
    res = mgr.predict(text="circle in sample",
                      options=PredictOptions(n_samples=300, cross_domain=True))
    assert "cross_domain" in res and "rl" in res
    assert res["rl"]["action"] in ("flag", "watchlist")
    fb = mgr.rl_feedback(res["rl"]["action"], 1.0)
    assert fb["ok"]


def test_engine_state_roundtrip(trained_engine, tmp_path):
    from malar.core.loop import MalarEngine
    from malar.domains.persistence import load_engine_state, save_engine_state
    from malar.fields.objectives import ObjectiveSpec
    from malar.world.adapters.synthetic import SyntheticAdapter

    p = tmp_path / "state.json"
    info = save_engine_state(trained_engine, p)
    assert p.exists() and info["objects"] >= 1
    # fresh engine, no training -> load -> same classes available
    fresh = MalarEngine(SyntheticAdapter(n_ticks=1, points_per_tick=10, k=4),
                        world_id="s", objectives=[ObjectiveSpec(key="sensitivity")],
                        domain_id="pytest_persist")
    out = load_engine_state(fresh, p)
    assert out["loaded"] and set(fresh.c.registry.classes()) == set(trained_engine.c.registry.classes())
    import shutil
    shutil.rmtree("data/domains/pytest_persist", ignore_errors=True)
