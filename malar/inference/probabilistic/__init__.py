"""Probabilistic inference & prediction layer (V3).

Per-modality Bayesian classifiers (topo/spectral/graph) fit over the WHOLE training
corpus, product-of-experts fusion, an MCMC posterior-predictive simulation, and the
Inference Management agent that orchestrates them — plus the second-pass agents:
cross-domain referencing, image/video diffusion, and contextual-bandit RL.

All math is deterministic Python (numpy default); the LLM only orchestrates/narrates.
"""
from __future__ import annotations

from malar.inference.probabilistic.bayes import BayesEnsemble, fuse_posteriors
from malar.inference.probabilistic.crossdomain import CrossDomainReferencer
from malar.inference.probabilistic.diffusion import DiffusionAgent
from malar.inference.probabilistic.likelihoods import GaussianLikelihood, collect_corpus
from malar.inference.probabilistic.manager import InferenceManager, PredictOptions
from malar.inference.probabilistic.mcmc import ValuePosterior, predictive_simulation
from malar.inference.probabilistic.rl import ContextualBandit, build_from_engine

__all__ = [
    "GaussianLikelihood", "collect_corpus",
    "BayesEnsemble", "fuse_posteriors",
    "ValuePosterior", "predictive_simulation",
    "CrossDomainReferencer", "DiffusionAgent",
    "ContextualBandit", "build_from_engine",
    "InferenceManager", "PredictOptions",
]
