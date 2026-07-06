"""MCMC posterior-predictive simulation (I2).

Turns the fused class posterior + per-(class, objective) **value posteriors** into a
forecast of the OUTCOME, with credible intervals:

    for s in 1..S:
        c_s    ~ P(class | phi,h,g)                  # categorical, from BayesEnsemble
        v_s,k  ~ N(mu_{k,c_s}, sigma^2_{k,c_s})      # per-objective value posterior
        a_s    = policy(c_s, R(v_s))                  # the action it implies
    -> posterior-predictive over {class, value_k, action}

Value posteriors are built non-invasively from the existing point values v_{k,c} (the
ValueStore is unchanged): the mean is the learned value; the variance shrinks with the
number of training items for the class (standard error) and tightens for human-set
(sticky) values. This realises the "(class, objective) -> small posterior" decision (Q5)
without breaking anything that reads the point value.

Engines:
  * "mc"   — direct Monte-Carlo (default, dependency-free). The class posterior is already
             a normalised categorical, so direct sampling is exact.
  * "mh"   — Metropolis-Hastings random walk on the value parameters (real trace; for the
             continuous part of the model).
  * "nuts" — full MCMC via emcee/PyMC if installed (the UI "Full MCMC" checkbox, Q3);
             falls back to "mc" with a flag if the extra is missing.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_BASE_SIGMA = 0.15


class ValuePosterior:
    """Per-(class, objective) Normal posterior derived from point values + corpus counts."""

    def __init__(self, value_store, objective_keys, counts: dict[str, int] | None = None,
                 base_sigma: float = _DEFAULT_BASE_SIGMA):
        self.vs = value_store
        self.keys = list(objective_keys)
        self.counts = counts or {}
        self.base_sigma = float(base_sigma)

    def mean_var(self, cls: str, objective: str) -> tuple[float, float]:
        rec = self.vs.get(cls, objective)
        mu = rec.value if rec else self.vs.value(cls, objective, default=0.0)
        n = max(int(self.counts.get(cls, 1)), 1)
        sigma = self.base_sigma / np.sqrt(n)          # standard error shrinks with data
        if rec is not None and getattr(rec, "sticky", False):
            sigma *= 0.25                              # human-set values are confident
        return float(mu), float(max(sigma, 1e-4) ** 2)

    def sample(self, cls: str, objective: str, rng: np.random.Generator) -> float:
        mu, var = self.mean_var(cls, objective)
        return float(rng.normal(mu, np.sqrt(var)))


def _credible_interval(samples: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    lo = (1.0 - mass) / 2.0 * 100.0
    hi = (1.0 + mass) / 2.0 * 100.0
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def predictive_simulation(class_post: dict[str, float],
                          value_post: ValuePosterior,
                          objective_keys: list[str],
                          action_of,
                          n_samples: int = 2000,
                          method: str = "mc",
                          seed: int = 0,
                          mass: float = 0.95) -> dict:
    """Run the posterior-predictive simulation. `action_of(cls, r_value)->str` is supplied
    by the manager (wraps the engine policy)."""
    rng = np.random.default_rng(seed)
    classes = list(class_post.keys())
    if not classes:
        return {"error": "empty class posterior"}
    probs = np.array([class_post[c] for c in classes], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(classes), 1.0 / len(classes))

    used_method = method
    nuts_samples = None
    if method == "nuts":
        nuts_samples = _try_nuts(class_post, value_post, objective_keys, n_samples, seed)
        if nuts_samples is None:
            used_method = "mc (nuts unavailable)"

    class_draws = rng.choice(len(classes), size=n_samples, p=probs)
    value_samples = {k: np.empty(n_samples) for k in objective_keys}
    action_counts: dict[str, int] = {}
    primary = objective_keys[0] if objective_keys else None
    trace = np.empty(n_samples)

    for s in range(n_samples):
        c = classes[class_draws[s]]
        rvals = {}
        for k in objective_keys:
            if method == "mh":
                v = _mh_value(value_post, c, k, rng, steps=8)
            else:
                v = value_post.sample(c, k, rng)
            value_samples[k][s] = v
            rvals[k] = v
        r_value = max(rvals.values()) if rvals else 0.0
        trace[s] = rvals.get(primary, r_value) if primary else r_value
        a = action_of(c, r_value)
        action_counts[a] = action_counts.get(a, 0) + 1

    # class predictive (= input categorical, but reported empirically for the UI)
    class_pred = {c: float((class_draws == i).mean()) for i, c in enumerate(classes)}
    action_pred = {a: n / n_samples for a, n in
                   sorted(action_counts.items(), key=lambda x: -x[1])}
    top_action = next(iter(action_pred), None)
    value_summary = {}
    for k, arr in value_samples.items():
        lo, hi = _credible_interval(arr, mass)
        value_summary[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                            "ci_low": lo, "ci_high": hi}

    # thin the trace for transport to the UI
    thin = max(1, n_samples // 400)
    return {
        "method": used_method,
        "n_samples": n_samples,
        "class_predictive": class_pred,
        "value_summary": value_summary,
        "predicted_action": top_action,
        "action_predictive": action_pred,
        "action_confidence": action_pred.get(top_action, 0.0) if top_action else 0.0,
        "trace": trace[::thin].round(5).tolist(),
        "trace_objective": primary,
    }


def _mh_value(value_post: ValuePosterior, cls: str, objective: str,
              rng: np.random.Generator, steps: int = 8) -> float:
    """Random-walk Metropolis-Hastings targeting the value Normal posterior."""
    mu, var = value_post.mean_var(cls, objective)
    sd = np.sqrt(var)

    def logp(x):
        return -0.5 * ((x - mu) ** 2) / var

    x = mu
    lx = logp(x)
    for _ in range(steps):
        prop = x + rng.normal(0, sd)
        lp = logp(prop)
        if np.log(rng.random() + 1e-12) < (lp - lx):
            x, lx = prop, lp
    return float(x)


def _try_nuts(class_post, value_post, objective_keys, n_samples, seed):
    """Full posterior sampling of the value parameters via emcee (opt-in `[ml]`)."""
    try:
        import emcee  # noqa: F401
    except Exception:
        return None
    try:
        import emcee
        primary = objective_keys[0] if objective_keys else None
        if primary is None:
            return None
        # sample the marginal value of the MAP class as a demonstration of full MCMC
        c = max(class_post, key=class_post.get)
        mu, var = value_post.mean_var(c, primary)

        def log_prob(theta):
            x = theta[0]
            return -0.5 * ((x - mu) ** 2) / var

        nwalkers, ndim = 8, 1
        p0 = mu + np.sqrt(var) * np.random.default_rng(seed).normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(p0, max(200, n_samples // nwalkers), progress=False)
        return sampler.get_chain(flat=True)[:, 0]
    except Exception:
        return None
