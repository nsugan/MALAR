"""RL agent for text / contextual information (I5) — contextual bandit (locked Q2).

A **contextual bandit with Thompson sampling** over the F-afforded action set. Each action
keeps a Bayesian linear-reward model (Bayesian linear regression: posterior over weights
given context features). To act, we Thompson-sample a weight vector per action and pick the
highest predicted reward; the per-action mean and posterior variance give the policy and
its uncertainty.

Reward signal (locked): **human feedback** and/or **training labels** (when the data is
labelled, the action that matches the labelled-correct affordance gets reward 1, others 0).
`warm_start` fits the bandit over the whole training corpus before any live use; `update`
folds in HITL feedback online. No policy-gradient network is used.
"""
from __future__ import annotations

import numpy as np


class _LinearTS:
    """Bayesian linear regression reward model for one action (Thompson sampling)."""

    def __init__(self, dim: int, sigma2: float = 1.0, prior_var: float = 1.0):
        self.dim = dim
        self.sigma2 = sigma2
        self.A = np.eye(dim) / prior_var       # precision
        self.b = np.zeros(dim)
        self.n = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float).ravel()[:self.dim]
        if x.shape[0] < self.dim:
            x = np.pad(x, (0, self.dim - x.shape[0]))
        self.A += np.outer(x, x) / self.sigma2
        self.b += x * float(reward) / self.sigma2
        self.n += 1

    def _mean_cov(self):
        cov = np.linalg.inv(self.A)
        mu = cov @ self.b
        return mu, cov

    def sample_reward(self, x: np.ndarray, rng: np.random.Generator) -> float:
        mu, cov = self._mean_cov()
        w = rng.multivariate_normal(mu, cov)
        return float(w @ _fit(x, self.dim))

    def mean_reward(self, x: np.ndarray) -> tuple[float, float]:
        mu, cov = self._mean_cov()
        xf = _fit(x, self.dim)
        return float(mu @ xf), float(xf @ cov @ xf)


class ContextualBandit:
    def __init__(self, actions: list[str], dim: int = 16, seed: int = 0):
        self.actions = list(actions)
        self.dim = dim
        self.models = {a: _LinearTS(dim) for a in self.actions}
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def act(self, context: np.ndarray) -> dict:
        if not self.actions:
            return {"action": None, "scores": {}, "uncertainty": {}}
        sampled = {a: self.models[a].sample_reward(context, self.rng) for a in self.actions}
        means = {a: self.models[a].mean_reward(context) for a in self.actions}
        best = max(sampled, key=sampled.get)
        return {
            "action": best,
            "thompson_scores": {a: round(v, 4) for a, v in sampled.items()},
            "mean_reward": {a: round(m[0], 4) for a, m in means.items()},
            "uncertainty": {a: round(float(np.sqrt(max(m[1], 0.0))), 4) for a, m in means.items()},
            "updates": self.updates,
        }

    def update(self, context: np.ndarray, action: str, reward: float) -> bool:
        if action not in self.models:
            return False
        self.models[action].update(context, reward)
        self.updates += 1
        return True

    def warm_start(self, samples: list[tuple[np.ndarray, str, float]]) -> int:
        """Fit over the whole corpus: list of (context, correct_action, reward)."""
        k = 0
        for ctx, action, reward in samples:
            if self.update(ctx, action, reward):
                k += 1
        return k


def _fit(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] >= dim:
        return x[:dim]
    return np.pad(x, (0, dim - x.shape[0]))


def build_from_engine(engine, dim: int = 16, seed: int = 0) -> ContextualBandit:
    """Construct + warm-start a bandit from the engine's afforded actions and labelled
    prototypes (context = graph embedding g; correct action = first affordance per class)."""
    actions = []
    for ff in engine.c.functional_fields:
        for a in ff.spec.actions:
            if a not in actions:
                actions.append(a)
    bandit = ContextualBandit(actions or ["flag"], dim=dim, seed=seed)
    if not actions:
        return bandit
    # warm-start over labelled prototypes (whole-corpus signal available in-session)
    samples = []
    for o in engine.c.registry.objects.values():
        if getattr(o, "candidate", False):
            continue
        ctx = np.asarray(o.g, dtype=float)
        correct = actions[0]
        for a in actions:
            samples.append((ctx, a, 1.0 if a == correct else 0.0))
    bandit.warm_start(samples)
    return bandit
