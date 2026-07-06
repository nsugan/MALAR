# MALAR — Changelog

Versioning note: "V2" and "V3" refer to the project iteration (the folder line).
The `v2` git tag and `MALAR_V2_backup_*.zip` freeze the V2 state; the working tree is V3.

## [Unreleased] — V3 (3.0.0.dev) — in progress
See `ROADMAP_V3.md` for the planned direction and `MALAR_INFERENCE_LAYER_PLAN.md` for the
inference layer design.

### Added — Probabilistic Inference & Prediction layer (first cut: I0–I2, I6, I7)
- `malar/inference/probabilistic/`: per-modality Bayesian classifiers (`likelihoods.py`,
  `bayes.py`) fit over the **whole training corpus**; product-of-experts fusion; MCMC
  posterior-predictive simulation (`mcmc.py`, `mc`/`mh` default + `nuts` emcee opt-in) with
  per-(class, objective) value posteriors and 95% credible intervals; Inference Management
  agent (`manager.py`) orchestrating fuse → MCMC → verdict behind the OOD gate.
- `QdrantMemory.scroll_all()` to read the whole corpus for likelihood fitting.
- API: `POST /domains/{id}/predict`, `GET /domains/{id}/predict/agents`,
  `POST /domains/{id}/predict/crossref` (`predict_service.py`, `predict_routes.py`).
- Web UI: **"Inference and Prediction"** tab (last in the bar) — per-modality posteriors,
  fused posterior, MCMC forecast (action predictive, trace, objective CIs), verdict, and
  per-agent monitors with cut-2 agents shown as present-but-disabled.
- Tests: `tests/test_inference_probabilistic.py` (6); full suite 36 passed; ruff clean on
  all new modules.

### Added — second pass agents (I3, I4, I5) — now shipped
- `crossdomain.py` (I3): cross-domain referencing — opt-in per request, searches only other
  loaded domains' registries, matches flagged + down-weighted (×0.4); never overwrites
  in-domain meaning.
- `diffusion.py` (I4): Stage-1 graph/heat label-diffusion over image-region / video-frame
  graphs (CPU, dt<2/λmax), per-class field + vote + heatmap; Stage-2 generative
  score-based diffusion (GPU/torch), HITL-triggered, reports cleanly when no GPU.
- `rl.py` (I5): contextual bandit with Thompson sampling over the afforded action set,
  Bayesian linear-reward models, warm-started over the corpus; reward = human feedback
  and/or training labels.
- Manager now runs all seven agents; new endpoints `predict/feedback` (RL reward) and
  `predict/escalate` (Stage-2 diffusion). UI tab gained cross-domain matches, diffusion
  field, and an RL policy card with ✓/✗ feedback buttons.
- `emcee` added to the `[ml]` extra (Full-MCMC checkbox). Tests: 41 passed; ruff clean.

## [v2] — 3.0.0.dev cut point — as-built, frozen
The complete, verified MALAR build. Tagged `v2`; snapshot in `MALAR_V2_backup_*.zip`.

### Engine (M0–M12)
- World graph + Hodge Laplacians + region sampler + snapshots + world context.
- Three encoders: topology (ripser→persistence image), spectral (PCA autoencoder),
  graph (spectral/WL). Per-modality change stats.
- Objects + LLM-assisted identification (conformal-gated); objective/value/functional
  fields with sticky human values; V diffusion under the dt<2/λmax guard.
- Combined memory: Neo4j graph + Qdrant named vectors + artifacts; fused retrieval;
  async Curator (novelty/merge/insert/evict/defer, 1−1/e compaction, blue-green
  re-index, bounded rollback-able correction propagation).
- Core loop (Algorithm 1); LangGraph agent layer; generation + critic; FastAPI + eval;
  training campaign (stops on coverage) + checkpoints; inference with hard OOD gate.

### Web UI (U0–U8) + post-plan additions
- Per-domain isolation (mem__{id} collections, scoped graph, per-domain dirs).
- Studio tabs: Domains, Configure, Train (supervised + unsupervised), World Graph,
  Knowledge, Test & Inference, Agents, LLM, Debug.
- Debug input inspector (file → spectra → matrix heatmap → graph → persistence diagram →
  spectral encode/reconstruct → graph embedding → process), pictorial + numeric.
- Agents tab: per-agent monitors with individual checkboxes (11 agents + LLM extras).
- Learning-parameter editor in the Correct flow (all math/weights, per-domain).
- Light, professional theme.

### LLM layer
- LiteLLM gateway (pinned `main-stable`); local Gemma 3n (`gemma3n:e4b/e2b`).
- Frontier providers: DeepSeek, OpenAI/ChatGPT, Anthropic Claude — each by alias + key.
- Runtime model routing per agent role (LLM tab); call log + manual console.
- LLM-driven Orchestrator/Planner (derives R/F/extra-agents from the data description);
  opt-in per-item LLM identification (`llm_assist`).
- Configurable timeout (`LLM_TIMEOUT`, default 120 s); removed the confusing keyless
  frontier fallback so each alias surfaces its own error.

### Data & ops
- Real Raman ingestion (14 viruses × 10 spectra); Windows→container path mapping;
  multi-column .txt parsing; corpus-wide spectral fit.
- Verified: 30 unit tests pass · ruff clean · UI builds · no cross-domain leakage.

### Fixes made during development
- gemma4→gemma3n model id; litellm v1.80.0→main-stable image; node:20-slim UI build
  (esbuild/alpine); _decode_image base64; curator merge on fused distance; topology
  persistence-image fixed grid; spectral width alignment; resilient domain registry
  loader; isolated test data dir.
