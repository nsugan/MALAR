# MALAR V3 — Probabilistic Inference & Prediction Layer (PLAN, for review)

**Status: REVIEWED & APPROVED (Q1–Q7 answered 2026-06-29). Decisions locked in §8.**

> **Cross-cutting rule from review:** every inference method (likelihoods, bandit,
> diffusion, MCMC) is **fit/run over the WHOLE training corpus of the domain**, not a single
> sample. A prediction for a new input is then scored against models built from all train data.

A new layer and Web-UI tab that turns MALAR's deterministic identification into a
**probabilistic prediction engine**: a team of inference agents each runs a principled
method on one modality, optionally pulls cross-references from other domains, and reports
to an **Inference Management agent** that fuses everything with **Bayesian inference** and
forecasts the outcome with an **MCMC (Monte-Carlo) simulation** — uncertainty included.

It **extends** the existing `inference/` (identify → action, OOD gate); it does not
replace it. The current path becomes the fast "point estimate"; this layer adds the
"distribution + prediction" on top.

---

## 0. Design stance (unchanged MALAR laws)
- **Agents decide; tools compute.** Bayesian updates, MCMC, diffusion and RL run as
  deterministic Python tools. The LLM (Inference Management agent) decides which agents to
  run, sets priors/weights from the goal, narrates, and recommends — it never computes the
  posteriors itself and never sees raw vectors.
- **Local-first.** Everything runs on CPU/numpy by default (lightweight MCMC, conjugate
  updates, graph diffusion, contextual bandits). Heavy options (score-based diffusion,
  PyMC/NUTS) are opt-in `[ml]` extras.
- **Grounded & isolated.** Every prediction is anchored to a WorldContext; cross-domain
  references are explicit and opt-in (they cross the per-domain isolation boundary).

---

## 1. The agents (one task each → Inference Management)

| Agent | Input modality | Method (default) | Output |
|---|---|---|---|
| **Bayesian-Topology agent** | φ (persistence features) | Bayesian classifier: per-class likelihood of φ-summary → posterior P(class \| φ) | posterior over classes + evidence |
| **Bayesian-Graph agent** | g (graph embedding) | Bayesian classifier on g (Gaussian/Dirichlet likelihood per class) → P(class \| g) | posterior + evidence |
| **Bayesian-Spectral agent** | h (hyperspectral code) | Bayesian classifier on h → P(class \| h) | posterior + evidence |
| **Cross-Domain Reference agent** | φ/h/g | searches OTHER domains' Qdrant collections + object registries for similar objects | ranked cross-domain matches + their classes/values |
| **Diffusion agent (image/video)** | image / video region graphs | label/heat diffusion over a region-or-frame graph (default); optional score-based diffusion | per-region/frame class field + temporal aggregate |
| **RL agent (text/context)** | text + contextual features | contextual bandit (Thompson sampling) over the F-afforded action set | action policy + per-action value/uncertainty |
| **Inference Management agent** | all of the above | **Bayesian fusion** (product-of-experts across modalities) + **MCMC** predictive simulation; LLM narrates | fused posterior, predicted outcome + credible interval, recommended action, cross-refs, confidence |

Each task agent is monitored individually (same pattern as the Agents tab), and every LLM
call it makes appears in the LLM log.

---

## 2. The mathematics (what each tool actually does)

### 2.1 Per-modality Bayesian classifiers
For an input with encodings (φ, h, g) and learned classes c:
```
P(c | x_m) ∝ P(x_m | c) · P(c)            for modality m ∈ {topo, spectral, graph}
```
- **Likelihood P(x_m | c):** a per-class generative model fit from the trained memory of
  that class — default a Gaussian (diagonal/again-shrunk covariance) in the modality's
  feature space, or a kernel-density / nearest-prototype likelihood. Conjugate where
  possible so updates are closed-form (no MCMC needed at this stage).
- **Prior P(c):** class prevalence from training, or set by the goal/LLM, or uniform.
- Output: a categorical posterior per modality, plus the **marginal evidence** (model fit)
  so the manager can down-weight an unreliable modality.

### 2.2 Bayesian fusion (Inference Management)
Combine the three modality posteriors (assumed conditionally independent given class — a
"product of experts"), with learnable/`goal-set` modality weights wₘ:
```
P(c | φ,h,g) ∝ P(c) · Πₘ P(c | x_m)^{wₘ} / P(c)^{Σwₘ}
```
This yields the **fused class posterior**. Disagreement between modalities widens the
posterior (honest uncertainty), which is exactly what the OOD gate and the operator want.

All three likelihoods are fit from the **whole training corpus** of the domain (every
stored memory of every class), not the single query item.

### 2.3 MCMC predictive simulation
Given the fused posterior over classes and the per-class **value distributions** (the R
fields are learned point values today; we extend them to distributions, e.g. a
Beta/Normal per (class, objective)):
```
for s in 1..S samples:
    c_s   ~ P(c | φ,h,g)                      # sample a class
    v_s   ~ P(value | c_s, objective)          # sample its objective value
    a_s   = policy(c_s, v_s)                    # the action it implies
accumulate → posterior-predictive over {class, value, action}
report: predicted action + P(action), value mean ± 95% credible interval, entropy
```
- **Engine (locked Q3):** default a light, dependency-free sampler (direct Monte-Carlo from
  the categorical + conjugate value posteriors; Metropolis-Hastings only where needed). A
  **UI checkbox "Full MCMC (emcee/PyMC NUTS)"** switches on full posterior sampling over
  continuous parameters when the `[ml]` extra is installed.
- This is the "MCMC Monte-Carlo simulation to design and predict the outcome of the
  Bayesian analysis" — it turns the posterior into a **forecast with credible intervals**.

### 2.4 Diffusion agent (image / video) — **two-stage escalation (locked Q1)**
1. **Stage 1 — graph/heat label-diffusion (default, CPU).** Build a graph over image
   regions (or video frames) across the **whole train data**, seed it with per-region class
   evidence, and run **heat/label diffusion over the Laplacian** (same machinery as the
   V-field, guard dt < 2/λmax) to produce a smooth class field; aggregate over frames for
   video. Deterministic, fast, no GPU.
2. **Stage 2 — generative score-based diffusion (GPU), triggered after human intervention.**
   If, after Stage 1 + a HITL review, the operator judges that **not enough was learned**
   (insufficient signal / coverage), the agent escalates to a small **score-based /
   denoising diffusion** model trained on the **whole train corpus**, using the **GPU**
   (RTX 3050, torch `[ml]` extra), for generative denoising / counterfactual frames.
   The escalation is an explicit HITL action, never automatic.

### 2.5 RL agent (text / context) — **locked Q2**
- **Method:** a **contextual bandit with Thompson sampling** over the F-afforded action set,
  fit over the **whole train data**. Context = text/contextual features (LLM-parsed). Each
  action keeps a Bayesian reward posterior; Thompson sampling picks actions and updates.
- **Reward signal:** **human feedback (HITL)**, and **training-data labels when they exist**
  (correct action ⇒ reward) — both feed the same posterior. No policy-gradient net in scope.

### 2.6 Cross-domain referencing
- Per-domain isolation is a keystone, so cross-domain search is **explicit and opt-in**.
- The Cross-Domain Reference agent queries the **other domains' Qdrant collections**
  (`mem__{other_id}`) and object registries with the input's (φ,h,g), returning the nearest
  objects, their domain, class, and learned values — surfaced as "this looks 0.83 like
  `rsv` in domain X". The manager may use these as extra evidence (down-weighted, flagged
  as cross-domain) or just show them.

---

## 3. Backend design

```
malar/inference/
├── service.py              (existing — point-estimate path; unchanged interface)
├── frontend.py             (existing — image/video/text → φ/h/g)
└── probabilistic/          (NEW)
    ├── likelihoods.py      per-class generative models from memory (Gaussian/KDE/prototype)
    ├── bayes.py            per-modality posteriors + product-of-experts fusion
    ├── mcmc.py             posterior-predictive sampler (light MC default; emcee/PyMC opt-in)
    ├── diffusion.py        region/frame graph label-diffusion (+ optional score-based)
    ├── rl.py               contextual bandit (Thompson sampling) over the action set
    ├── crossdomain.py      multi-collection Qdrant/registry search (opt-in)
    └── manager.py          Inference Management agent: orchestrate → fuse → MCMC → narrate
```
- **Value distributions:** extend the value store so each (class, objective) can carry a
  small posterior (mean + variance / Beta params), updated alongside the point value. This
  is additive and backward-compatible.
- **LLM use:** the manager calls the LLM (route-aware) to choose which agents to run from
  the goal, to set priors/weights, and to write the human-readable verdict.

### 3.1 New API endpoints (per domain)
```
POST /domains/{id}/predict           {input(text|image|video|file|index), options} → full result
GET  /domains/{id}/predict/agents    live state of the inference agents
POST /domains/{id}/predict/crossref  {input, target_domains?} → cross-domain matches
GET  /domains/{id}/predict/{run_id}  fetch a stored prediction run
```
Options: which agents to enable, prior mode (uniform/prevalence/LLM), modality weights,
MCMC sample count, cross-domain on/off + target domains, RL feedback.

---

## 4. The new Web-UI tab — "Inference and Prediction" (locked Q6)

A dedicated tab, placed **last** in the tab bar (after "Test & Inference"; that tab stays),
to control all inference & prediction:

**Controls**
- Input picker: text box / image or file upload / pick a queue file / video.
- Agent toggles: enable/disable each task agent (topo / graph / spectral / cross-domain /
  diffusion / RL).
- Priors & weights: prior mode, per-modality weights, MCMC sample count, **"Full MCMC
  (emcee/PyMC)" checkbox** (Q3), cross-domain **opt-in** toggle + which domains (Q4).
- Diffusion **"Escalate to generative (GPU)"** button — enabled after a Stage-1 HITL review.
- Run / live progress.

**Results (pictorial + numeric)**
- **Per-modality posteriors** — bar/violin per class for topo, spectral, graph.
- **Fused posterior** — the combined class distribution + entropy.
- **MCMC predictive** — trace plot, posterior-predictive histogram of the value, the
  predicted action with P(action), and a 95% credible interval.
- **Cross-domain references** — ranked matches with domain/class/similarity.
- **Diffusion view** — the per-region/frame class field (heatmap / overlay) for image/video.
- **RL view** — action values + uncertainty; the Thompson-sampled choice.
- **Management verdict** — the LLM narrative: predicted outcome, confidence, recommended
  action, caveats; OOD banner if outside trained domain.

**Monitoring** — the same per-agent live cards (reusing the Agents-tab pattern) for the
inference agents, plus their LLM calls in the LLM tab.

---

## 5. Build milestones (I0–I7, PR-sized, each with a Definition of Done)

**Phasing (locked Q7): first cut = I0 → I1 → I2 → I6 → I7** (Bayes + MCMC + the
"Inference and Prediction" tab, end-to-end). **Second pass = I3 (cross-domain) → I4
(diffusion, both stages) → I5 (RL bandit).** The tab and manager are built with the agent
toggles present-but-disabled in cut 1, so cut 2 lights them up without rework.

- **I0 — Likelihoods + per-modality Bayes.** `likelihoods.py`, `bayes.py`. *DoD:* posteriors
  for φ/h/g on a trained domain; sane on held-out items; sums to 1; evidence reported.
- **I1 — Bayesian fusion.** product-of-experts with weights. *DoD:* fused posterior;
  disagreement widens it; matches the point-estimate class when confident.
- **I2 — MCMC predictive.** `mcmc.py` (light MC default). *DoD:* posterior-predictive over
  {class, value, action} with credible intervals; reproducible with a seed.
- **I3 — Cross-domain referencing.** `crossdomain.py`. *DoD:* returns nearest objects from
  other domains; isolation respected (off by default, explicit opt-in).
- **I4 — Diffusion agent.** `diffusion.py` (graph label-diffusion). *DoD:* a class field over
  image regions / video frames under the dt<2/λmax guard.
- **I5 — RL agent.** `rl.py` (contextual bandit, Thompson sampling). *DoD:* picks actions
  from the afforded set by context; updates from feedback; uncertainty shrinks with data.
- **I6 — Inference Management + API.** `manager.py` + the 4 endpoints; LLM orchestration +
  narrative. *DoD:* `/predict` returns the full fused result + MCMC forecast + cross-refs +
  verdict; OOD gate honored.
- **I7 — Web UI "Inference and Prediction" tab (last in the bar).** controls + all
  visualizations + agent monitors. *DoD:* run a prediction from the UI end-to-end;
  posteriors, MCMC predictive, cross-refs and verdict render; agent cards live.

Each milestone keeps the existing tests green and adds its own; ruff stays clean; UI builds.

---

## 6. Dependencies & local-first
- Core (already present): numpy, scipy, scikit-learn — enough for the default Bayes / MC /
  diffusion / bandit.
- Optional `[ml]` extras (opt-in): `emcee` or `pymc` (full MCMC/NUTS), `torch` (score-based
  diffusion, policy network). Default path needs **none** of these.

---

## 7. Risks
- **Calibration:** Gaussian/KDE likelihoods from few training items can be over-confident;
  mitigate with shrinkage priors and the conformal gate already in place.
- **Compute on 6 GB:** keep MCMC sample counts modest by default; heavy options opt-in.
- **Cross-domain leakage of meaning:** keep it explicit, flagged, down-weighted — never
  silently merge another domain's evidence.
- **Scope creep:** four method families (Bayes, MCMC, diffusion, RL) is a lot; the
  milestones let you stop after any one and still have a working, valuable layer.

---

## 8. Decisions (locked from review, 2026-06-29)

- **Q1 — Diffusion:** two-stage. **Stage 1 graph/heat label-diffusion (CPU)** first; **Stage
  2 generative score-based diffusion on the GPU** only after a HITL review finds Stage 1
  learned too little. Both fit over the **whole train data**. *(see §2.4)*
- **Q2 — RL:** **contextual bandit with Thompson sampling**, fit over the **whole train
  data**; reward = **human feedback and/or training labels**. No policy-gradient net. *(§2.5)*
- **Q3 — MCMC:** light Monte-Carlo + Metropolis-Hastings by default; **emcee/PyMC (NUTS)
  opt-in via a UI checkbox**. Runs over the **whole train data**. *(§2.3)*
- **Q4 — Cross-domain:** **off by default, opt-in per run.** *(§2.6)*
- **Q5 — Value distributions:** **yes** — each (class, objective) gets a small posterior
  (mean+variance), backward-compatible, so MCMC can sample outcomes. *(§3)*
- **Q6 — Tab:** named **"Inference and Prediction"**, placed **last** in the tab bar;
  "Test & Inference" stays. *(§4)*
- **Q7 — Scope:** phased — **first cut I0,I1,I2,I6,I7** (Bayes + MCMC + UI), **second pass
  I3,I4,I5** (cross-domain, diffusion, RL). *(§5)*

---

*Plan approved. Building now in the locked order, milestone by milestone, keeping existing
tests green / ruff clean / UI building at each step.*
