# MALAR — Changelog

Versioning note: "V2"/"V3"/"V4" refer to the project iteration (the folder line).
The `v2` git tag and `MALAR_V2_backup_*.zip` freeze the V2 state; V4 lives in `C:\Dev\MALAR_V4`
(a clean copy of the final V3 tree, no `.git`/`node_modules`/caches).

## [Unreleased] — V4 (4.0.0.dev) — in progress

Iteration bumped to V4 (folder `MALAR_V4`). First V4 work: a data-flow audit and fixes —
see `docs/V4_DATAFLOW_FINDINGS.md` for the full findings, evidence, and deferred items.

### Changed — training acceleration (honest) + agent outputs in Train tab
- **The real bottleneck is LLM calls, not the math.** Per-item deterministic work (ripser
  PH + small numpy) is ms-scale — GPU/NPU offload of it does not help. So acceleration is
  applied where it's real: the independent **in-loop agent calls now run concurrently**
  (`malar/core/accel.py::parallel_map`) — `run_validated_agents` (sandboxed code) and
  `run_extra_agents` (LLM proposals) overlap, then apply feedback serially. The local LLM
  (Ollama) already runs on the **RTX 3050** via `gpus: all`; route agent roles to
  malar-reasoner/fast/vision (LLM tab) to use it instead of the cloud.
- **Honest device report** at `GET /accel` + a **Compute** status line in the Train tab:
  shows CUDA device (if any), CPU cores, parallel workers, and states plainly that the
  **AMD NPU (XDNA) and Radeon iGPU are not wired** (they need an ONNX / Ryzen-AI / DirectML
  rewrite of the encoders and wouldn't help this small per-item data).
- **Agent Factory outputs in the Train tab.** A new panel (supervised + unsupervised) shows
  what each validated factory agent produced on the training data and what it fed back into
  its parent field, so you can judge the data flow. Populated when "use in training" is on.
- Tests: `tests/test_accel.py` (parallel_map order/edges; device report is honest about NPU).

### Changed — Agent Factory
- **Generated agents can now use any library / do any task.** Removed the `math`/`numpy`-only
  import allowlist and the call/dunder blocklist from `validate.py`, and switched
  `sandbox.py` to run generated agents with full builtins and real imports. Validation now
  only checks that the code parses and implements `GeneratedAgent.run(ctx)`, and additionally
  reports the imports the code uses (for review). The algorithm/code prompts were updated to
  tell the model it may import and use anything it needs. **Security note:** validated agents
  run with full capabilities (file/network/etc.) and auto-run in the training loop when "use
  in training" is on — only validate agents you have reviewed.
- **Per-agent input/output is now visible for verification.** Every agent run (auto-test on
  generation, manual "Test on domain sample", "Run now", and in-loop training runs) records
  its last input (ctx) and output on the agent; the Agent Factory shows a "Last run — input &
  output" panel when you click an agent, so you can confirm it does its task. Backed by new
  `last_input`/`last_output`/`last_run` columns in the agent store (`agent_memory.py`,
  migrated in place) and `AgentFactory.record_io`.
- Tests: `test_validate_allows_any_imports`, `test_sandbox_runs_with_full_capabilities`,
  `test_records_last_input_and_output` (existing restriction tests updated to the new behavior).

### Fixed — Web UI (build-verified)
- **Blank screen after visiting the LLM or Debug tab (needed a reload).** Root cause: an
  uncaught render error unmounted the whole React tree (no error boundary). Added a per-tab
  `ErrorBoundary` (`ui/src/components/ErrorBoundary.jsx`), keyed by tab in `App.jsx`, so a
  view error is shown inline and the header/nav stay live — switching tabs recovers with no
  reload. Also hardened `DebugConsole` (optional chaining on partial `/debug` snapshots so it
  never throws) and fixed a `ProviderRouting` effect that returned a Promise as its cleanup.
- **DeepSeek is now the default provider for the `reasoner` and `fast` roles** (`config.py`
  `alias_reasoner`/`alias_fast` default to `malar-deepseek`; shown as the default in the LLM
  tab's Model-routing card). Requires `DEEPSEEK_API_KEY` in `.env` and sends data off-box;
  override per role via `LLM_*_ALIAS` env or the LLM tab. `vision` stays local.

### Changed — domain-agnostic / generic data model
The engine is now generic by default; nothing is hardcoded to Raman spectra, virus classes,
or sensor timeseries. What the data actually is gets described in the **Domain description**
and the **Configure** tab (free text → LLM context), not baked into code. Configure, Train,
Agents, and their actions are otherwise unchanged. (verified: full suite green; UI builds)
- **Synthetic is the default.** New domains use the domain-agnostic `SyntheticAdapter`; the
  built-in no-folder demo uses generic, class-separable feature clouds
  (`synthetic_class_cloud`) with generic labels (`class_a/b/c`) instead of virus spectra.
- **Format-agnostic folder ingestion.** `_load_points` loads any numeric table/array
  (`.csv/.txt/.tsv/.asc/.spc/.npy/.npz/.dat`) and, best-effort, images/video (optional
  `PIL`/`imageio`), reducing each to a fixed-length feature vector. `folder_analyzer` kinds
  are now generic (`features/array/image/video`).
- **Decoupled, not deleted.** `raman.py` / `sensor.py` remain on disk and still work for any
  pre-existing domain that declares them (lazy-imported); `domain_service`, `frontend`, and
  `diffusion` no longer import from `raman.py` — the generic `cosine_knn_graph` moved to
  `world/graph.py`. Defaults in `manager.py`, `planner.py`, `session.py`, `domains.py`, and
  the create form are now `synthetic`.
- **UI wording genericized.** Removed Raman/spectra/hyperspectral/timeseries/virus wording
  from the Domains create form (adapter dropdown dropped), Configure, Train, Input Inspector,
  Inference & Prediction, and the legacy console; the core `topology/spectral/graph` encoder
  names are kept (they are not domain-specific).
- Test: `tests/test_domains.py::test_generic_synthetic_domain_trains`.

### Fixed — data-flow inconsistencies (backend, pytest-verified)
- **❶ OOD validity gate consistency.** Both inference entry points now gate on `fe.g` (the
  same space as the registered `o.g` contexts); the probabilistic manager previously compared
  `fe.world_emb` against `o.g` across embedding spaces. `validation/ood.py` param renamed
  `world_emb`→`ctx_emb` with the same-space contract documented.
- **❷ Value/affordance graph sync.** `HAS_VALUE` / `AFFORDS` edges (with `:Object` /
  `:Objective` / `:Action` nodes) are now actually written by both graph backends and mirrored
  from the training write path; surfaced in the World-Graph endpoint. In-process stores remain
  the source of truth (checkpointed to `state.json`); false "synced to Neo4j" docstrings fixed.
- **❸ Qdrant read-back hydration.** Added `QdrantMemory.fetch()` and `MemoryStore._hydrate()`;
  `get_item`/`reinforce`/`merge` rehydrate real vectors from Qdrant on a cache miss instead of
  no-op'ing (merges) or losing vectors (reinforce) after a restart.
- **Routing bypass.** `frontend.py` / `worldview.py` / `health.py` now call the reasoner/fast
  ROLE via the route table instead of hardcoded alias literals.
- Tests: two new cases in `tests/test_memory.py` (cold-cache hydration; value/affordance edges).

### Deferred (documented in `docs/V4_DATAFLOW_FINDINGS.md`)
- ❹ unify the dual auto-train pipeline; ❺/❻ vestigial legacy `/ws`+`/review` control plane
  (superseded by the per-domain supervised train flow); S auto-requeue of sticky-value
  rejections to the HITL queue.

## [Unreleased] — V3 (3.0.0.dev)
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
