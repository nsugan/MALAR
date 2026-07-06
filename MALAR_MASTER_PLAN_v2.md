# MALAR — Comprehensive Master Plan & Build Record (v2)

**A fully-local, agentic implementation of MALAR (Multi-scale Adaptive Learning with
Abstraction and Recall): a combined graph + persistence-homology + hyperspectral memory,
with grounded objects carrying objective / value / functional fields anchored to the
world model they were learned in; a training layer that learns a domain to coverage; an
inference layer that acts on unknown image/video/text; and a multi-domain Web UI to
configure, train (supervised & unsupervised), monitor every agent, and run inference.**

> This v2 document supersedes and extends the original `MALAR_MASTER_PLAN.md` (engine
> milestones M0–M12) and `MALAR_WEBUI_PLAN.md` (web UI augmentation U0–U8). It records the
> **as-built** system including everything added in development: the multi-domain web UI,
> the LLM gateway + frontier-provider layer (DeepSeek / OpenAI / Claude), the LLM-driven
> Orchestrator/Planner, per-item LLM identification, the Debug / Agents / LLM monitoring
> tabs, and real-data Raman ingestion.

---

## 0. What MALAR is, and the two architectural laws

MALAR runs one repeating loop:

```
observe → topological abstraction → objective / value / functional fields
→ policy / action → combined memory (graph + persistence-homology + hyperspectral,
anchored to world context) → recall / reverse-generation → observe
```

It **trains a domain to coverage**, then **infers** on unknown inputs behind a hard
out-of-distribution (OOD) gate.

**Two laws, enforced throughout:**

1. **Agents decide; tools compute.** Persistence homology, spectral/graph encoding,
   Laplacian diffusion, and distances run in deterministic Python. The LLM only
   orchestrates, sets weights from goals, hypothesises object labels/values from
   *summaries* (never raw vectors), plans the domain, and explains.
2. **Local-first.** Default routing is local Ollama (Gemma 3n); nothing leaves the box.
   Frontier providers (DeepSeek / OpenAI / Claude) are an explicit, per-alias, runtime
   opt-in that sends data off-box.

---

## 1. Implementation status (as-built)

| Area | Status |
|---|---|
| Engine M0–M12 (world, encoders, fields, memory, curator, loop, agents, generation, critic, API, training, inference) | **Built & verified** |
| Web UI U0–U8 (multi-domain studio: Domains / Configure / Train / World Graph / Knowledge / Test & Inference) | **Built & verified** |
| LLM layer (gateway + routing + frontier providers + observability + planner + per-item assist) | **Built & verified** |
| Monitoring (Debug input inspector, Agents tab, LLM tab) | **Built & verified** |
| Real-data Raman ingestion (14 viruses × 10 spectra) | **Built & verified** |
| Verification | **30 unit tests pass · ruff clean · UI builds (≈ 850 modules) · full backend flow + no cross-domain leakage** |

Counts: **98 Python modules** across 20 packages; **8 React tabs**; **17 React
components**; **45+ REST endpoints + WebSocket**; **7 LLM aliases** (3 local + 4 frontier).

Honest caveat (unchanged from the source paper): MALAR is a framework with no published
empirical results; its proofs/algorithms were AI-drafted and author-verified. This build
implements the designed algorithm faithfully and wires its theorems in as runtime guards,
but the synthetic eval scores 1.0 only because the synthetic classes are cleanly
separable. Real metrics require a real labelled dataset (the Raman set is wired in).

---

## 2. Locked stack (as-built, with corrections)

| Concern | Decision (as-built) |
|---|---|
| Agent orchestration | **LangGraph** (cyclic typed-state graph; sequential fallback when unavailable) |
| Local LLM | **Gemma 3n** via **Ollama** — aliases `gemma3n:e4b` (reasoner/vision) and `gemma3n:e2b` (fast). *(The original plan's "gemma4:e4b" was a misnomer; the real model is Gemma 3n e4b.)* |
| LLM gateway | **LiteLLM** (image pinned to `ghcr.io/berriai/litellm:main-stable` — the originally-pinned `v1.80.0` was never published). Single OpenAI-compatible endpoint; the one place providers are chosen. |
| Frontier (opt-in) | **DeepSeek** (`deepseek/deepseek-chat`), **OpenAI** (`openai/gpt-4o-mini`), **Anthropic Claude** (`anthropic/claude-sonnet-4-6`) — each via its own alias + API key. |
| Memory graph | **Neo4j 5 Community** (+ APOC + GDS) with an in-process graph fallback used automatically when Neo4j is unreachable. |
| Vector store | **Qdrant** with named vectors `phi` / `hyper` / `graph_emb`; embedded mode (`:memory:`) used when the server is unreachable. **One collection per domain** (`mem__{domain_id}`). |
| Topology | **ripser** + **persim** (persistence images) + **POT** (Wasserstein). *(Moved into core deps so the image builds without the heavy ML extras.)* |
| Encoders | Deterministic spectral PCA autoencoder + deterministic spectral/WL graph encoder (drop-in for a PyTorch-Geometric GNN; `torch`/`torch-geometric`/`giotto-tda` are optional extras). |
| Web UI | **React 18 + Vite + Tailwind** (light, professional theme), **Cytoscape.js** graphs, **recharts** charts. Built on Debian-based `node:20-slim` (esbuild is unreliable on alpine/musl). |
| API | **Python 3.11 + FastAPI + pydantic**, REST + WebSocket. |
| Containers | **Docker Compose**: neo4j, qdrant, ollama, litellm, malar (API), webapp (nginx). |

---

## 3. Engine architecture (the deterministic core)

One timestep (`malar/core/loop.py`, Algorithm 1) wires the modules:

```
adapter → W(t)=(I,E,C,L)          world graph + Hodge Laplacians (malar/world)
  → F_rep                          smoothed node representation (importance/sampling)
  → RegionSampler(budget B)        a region S
  → WorldSnapshot + WorldContext   provenance anchor (everything is grounded to this)
  → encode:
       φ = topology(S)             ripser → persistence image (raw diagram stored)
       h = spectral(S)             PCA autoencoder over spectra (raw spectra stored)
       g = graph(S), world_emb     spectral/WL embedding
  → identify object                retrieval top-k + conformal gate (+ optional LLM)
  → fields:  R^(k)(S)              objective value fields {v_{k,c}}
             V(S,t)                diffused over L (guard dt < 2/λmax(L))
             A(S) = ∪ F_j(o)       afforded action set
  → policy π(a|S,t)                ranks A(S) by R,V
  → Curator on_candidate(...)      novelty → reinforce / merge / insert / evict / defer
```

**Theorem guards (runtime asserts):** diffusion `dt < 2/λmax(L)`; importance EMA
`ω ← ρω + (1−ρ)·gain`, `0<ρ<1`; memory insert only when novelty `≥ θ` (`θ > ε`); merges
contractive (`c < 1`).

---

## 4. The 11 agents (monitored individually in the Agents tab)

| # | Agent | Role | LLM use |
|---|---|---|---|
| 1 | **Orchestrator / Planner** | drives the loop; allocates budget B; **LLM-plans the domain** (objectives R, functional dims F, processing steps, extra agents) from the data description; tracks coverage | **Yes** (planner) |
| 2 | **Perception / Ingestion** | pulls a batch, builds W(t), runs F_rep; surfaces the LLM-derived processing steps | indirect |
| 3 | **Topology** | computes φ_topo(S,t) via persistence homology | no |
| 4 | **Objective-Field Agents Aₖ** | one per objective: identify objects, compute R^(k), learn {v_{k,c}} from data + humans (sticky); active learning + HITL | optional |
| 5 | **Functional-Field Agents Bⱼ** | one per functional dim: learn affordances F_j(o,t) from data + humans; **LLM proposes functionalities** | **Yes** (assist) |
| 6 | **Fields Coordinator** | aggregate R, diffuse V, assemble the F-afforded action set | no |
| 7 | **Memory (inline)** | per-tick retrieval + per-modality novelty read | no |
| 8 | **Policy / Action** | choose actions within the F-afforded set, ranked by R,V; justify | no |
| 9 | **Generation** | reverse world-state generation, counterfactuals | no |
| 10 | **Critic / Validation** | conformal + OOD gate before any memory write or action | no |
| 11 | **Memory Curator (async)** | novelty + budget gating, store/merge/decay, budgeted (1−1/e) compaction, encoder-drift blue-green re-index, bounded rollback-able correction propagation, maintains the WorldView | no |

**Extra (LLM-proposed) agents.** When the Orchestrator plans a domain, the LLM may
propose domain-specific helper agents (e.g. a `PeakDetector` for Raman). These are stored
on the domain and shown as their own cards in the Agents tab.

---

## 5. Combined memory subsystem

- **Two stores keyed by `memory_id`.** Neo4j = memory graph 𝓜 (nodes = memories;
  edges `SIMILAR` / `DERIVED_FROM` / `TEMPORAL_NEXT` / `GROUNDED_IN`→`WorldContext`) plus
  the structural abstraction `g_m`. Qdrant = one point per memory, named vectors
  `phi`/`hyper`/`graph_emb`, payload `{tau, omega, class, neo4j_id, encoder_version,
  world_ctx_id, snapshot_id}`.
- **Fused retrieval.** `Score(m|S,t) = w_φ·Sim(φ) + w_h·Sim(h) + w_g·Sim(g) +
  λ·TimeGate(t,τ) + μ·ω`. Per-named-vector prefetch → weighted fusion → payload filter →
  structure hydrated from Neo4j only for the top-k.
- **Per-modality novelty.** Keep `Δ_φ` (Wasserstein/bottleneck on diagram, or cosine on
  φ), `Δ_h`, `Δ_g` separate. Novel if any modality exceeds its threshold, or goal-gain G
  or uncertainty U exceeds theirs.
- **Encoder drift — two clocks.** Fast: insert/merge/decay individual memories. Slow:
  retrain an encoder → re-embed from raw artifacts into a fresh Qdrant collection → swap
  behind a collection alias (blue-green); never query across encoder versions.
- **Curator algorithm.** `reinforce / merge (contractive) / insert / evict+insert /
  defer` + budgeted `CompressAndDecay` + Critic-gated, rollback-able human-correction
  propagation (scope by lineage / similarity / shared WorldContext, capped by a budget).

---

## 6. Grounding: objects, R / V / F fields, world anchoring

- **Objects = labeled memory** `o = (class c, g_o, φ_o, h_o, ω_o, τ_o)`. Identify by
  retrieval + a conformal gate; uncertain cases become labeling candidates.
- **Objective fields** `R^(k)(S) = Σ_{o∈O(S)} v_{k,class(o)}·ctx(o,S)` — learned per-class
  values, never hand-set. **Human values are STICKY** (agents only *propose* changes,
  which re-enter the HITL queue).
- **Functional fields** `F_j(o,t)` generate the action set `A(S)`; the Policy ranks `a∈A(S)`
  by R,V. Stored `(:Object)-[:AFFORDS {dim,action,source,version,world_ctx}]→(:Action)`.
- **World anchoring.** Every learned artifact is stored *in reference to where it was
  learned*: `WorldContext = (world_id, snapshot_id, region S, t, source)`, with snapshots
  persisted (periodic full + deltas). `world_emb` anchors context similarity → validity
  gating and drift detection.
- **LLM-assisted identification.** The agent sends a **feature summary** (H0/H1/H2 counts,
  persistence ranges, spectral descriptors) + retrieval context + the **domain
  description** to the LLM → label + candidate functionalities. Deterministic Score +
  conformal gate decide; human override is final. **Raw vectors are never sent to the LLM.**

---

## 7. Training layer

- **DomainSpec** (`domains/<id>.yaml`): domain, adapters/datasets, encoders, objectives R
  (K), functional dims F (J), coverage targets (novelty rate, per-class confidence, OOD
  rate, value uncertainty), budget. Now also carries `data_folders`, `data_description`,
  the **LLM plan**, **extra_agents**, and the **llm_assist** flag.
- **Supervised (one-at-a-time).** The UI walks a representative subset item-by-item,
  showing the item's graph + persistence diagram + spectrum + proposed identification +
  values + affordances. Controls: **Confirm** (commit, anchored to WorldContext) /
  **Correct** (fix label/value + edit the learning math; sticky) / **Skip**.
- **Unsupervised (auto).** Auto-ingests the rest of the data (real folder items when
  configured, else the synthetic adapter), letting the Curator update memory + vectors;
  live coverage / novelty / OOD progress.
- **Coverage-driven stop.** Tracks novelty rate, per-class confidence, OOD rate, value
  uncertainty over a rolling window; stops on targets or budget. Verified to stop on
  coverage at t=18 for the Raman fixture.
- **Checkpoints.** Versioned, restorable bundle: memory graph dump + memory vectors +
  model registry (ψ_k, η_j, encoder state) + DomainSpec + coverage report.

---

## 8. Inference layer

- **Multimodal front-end** (`inference/frontend.py`): image (Raman map → spectra; else
  Gemma vision), video (frames → temporal aggregate), text (LLM/keyword → structured
  query + pseudo-features) → φ/h/graph + world_emb.
- **Pipeline.** identify (retrieval + conformal, validity-gated via world_emb) → look up
  R, diffused V, and affordances F → Policy selects from the F-afforded set ranked by R/V
  → action + rationale + matched objects/contexts + confidence.
- **Hard OOD gate.** Low confidence / out-of-distribution → returns **"outside trained
  domain"**, no action, optionally queues the novel case back to training.
- **Folder batch inference.** Runs the whole test folder, writes a per-run results file
  (`data/domains/{id}/results/{run_id}.{csv,json}`), and a comparison view (class
  distribution, OOD rate, overlap with trained classes).

---

## 9. Web UI augmentation (U0–U8) — MALAR Studio

A single-page React app (light, professional theme) with a global header (active-domain
selector, mode badge, **Review** + **Debug** toggles) and these tabs:

1. **Domains** — add / list / select / **reset** / delete domains, each fully isolated.
2. **Configure** — data folders + free-text description; the DomainSpec (objectives R,
   functional dims F, coverage targets); **"Plan with LLM"** (Orchestrator derives the
   spec from the description); **LLM-assist** toggle.
3. **Train** — supervised one-at-a-time review (graph + topology + spectrum + proposed
   knowledge, Confirm/Correct/Skip) and unsupervised auto with live progress. **Correct**
   opens a panel to fix the label/value AND edit every learning parameter (math/weights).
   When **Debug** is on, per-agent live panels appear under each item.
4. **World Graph** — Cytoscape canvas of objects / memories / world contexts + filters +
   node detail (Neo4j structure + Qdrant vector neighbours); opens in a standalone window.
5. **Knowledge** — learned objective values {v_{k,c}} (human-set flagged orange/sticky) +
   affordances per class + coverage chart.
6. **Test & Inference** — folder batch inference → results table + distribution/comparison
   charts.
7. **Agents** — live monitor with **a checkbox per agent** (toggle any of the 11 + extra
   LLM-proposed agents); each card shows that agent's role + live state + activity,
   auto-refreshing.
8. **LLM** — gateway status, **model-routing** selector (point each agent role at any
   provider alias), a manual **console**, and the live **call log** (prompt + response +
   latency + source).
9. **Debug** — global Debug checkbox reveals: the **Input inspector** (pick any file from
   a dropdown → file display → spectra plot → value-matrix heatmap → graph → persistence
   diagram → spectral encode/reconstruct → graph embedding → process/identification, all
   pictorial + numeric) and the agent / gate / curator / memory panels.

### Per-domain isolation (U0 — the keystone)
- **Neo4j:** every node carries `domain_id`; queries scoped by it.
- **Qdrant:** one collection per domain, `mem__{domain_id}`; reset = drop + recreate.
- **Filesystem:** `data/domains/{id}/{artifacts,snapshots,results,raw}`.
- **Engine:** one `MalarEngine` per domain (isolated registry / stores / encoder state).
- Verified: two domains hold fully separate memory; resetting one never touches the other.

---

## 10. LLM layer (gateway, routing, frontier providers, observability, planner)

**Gateway.** Agents call only `malar/llm/client.py` BY ALIAS. The alias→provider/model
mapping lives in `infra/litellm_config.yaml` — never in code.

**Aliases.**
- Local (no egress): `malar-reasoner` (gemma3n:e4b), `malar-fast` (gemma3n:e2b),
  `malar-vision` (gemma3n:e4b).
- Frontier (opt-in, API key required, data leaves the box): `malar-deepseek`,
  `malar-openai`, `malar-claude`, `malar-frontier`.

**Runtime router.** A process-level routing table maps each agent role
(`reasoner`/`fast`/`vision`) to any alias; settable from the LLM tab (`PUT /llm/route`).
This is how you switch the agents between local Gemma and a frontier provider without
restarting.

**Observability.** Every call (prompt, response or error, latency, caller `source`) is
recorded in a shared ring buffer and exposed at `GET /llm/log`; the LLM tab streams it.
A manual `POST /llm/complete` console is provided.

**LLM-driven Orchestrator (`malar/agents/planner.py`).** `plan_domain(description, …)`
asks the LLM to derive objectives R, functional dims F, processing steps, and extra
agents from the data description (strict-JSON, with a deterministic preset fallback when
the gateway is down). `suggest_object(…)` proposes, per training item, the object class +
functionalities to assign.

**Per-item LLM identification (opt-in).** When `llm_assist` is enabled for a domain, each
training item calls the LLM (route-aware) to propose its class + affordances; off by
default so training stays fast (and to avoid the long first-load latency of a local model).

**Timeout.** Configurable via `LLM_TIMEOUT` (default **120 s**) — long enough for a local
model's first load; short calls remain fast.

**Operational notes (learned in practice).**
- `docker compose restart litellm` does **not** reload `.env`; use
  `docker compose up -d --force-recreate litellm` after changing API keys.
- A frontier alias with no key returns `401 — x-api-key header is required`. The previous
  automatic fallback to a keyless frontier provider was removed so each alias surfaces its
  own real error.

---

## 11. Debug & monitoring

- **Input inspector (Debug tab).** For any chosen file: file metadata + value stats; the
  spectra plotted; the raw value matrix as a heatmap; per-row norms; the world graph; the
  weighted adjacency matrix + degrees; the H0/H1/H2 persistence diagrams; the spectral
  encode→reconstruct overlay + latent-code heatmap; the graph embedding; and the full
  process/identification + novelty-vs-nearest. Both **pictorial and numeric**.
- **Agents tab.** Live, per-agent cards with a checkbox to toggle each of the 11 agents +
  any LLM-proposed extra agents; sub-agents (A_k / B_j) expand to their learned values /
  affordances; the Curator shows its decision audit.
- **LLM tab.** Gateway status + models, model routing, manual console, live call log.

---

## 12. Real-data Raman ingestion

- The folder loader resolves Windows paths to the container mount (`C:\…\MALAR_V2\X` →
  `/app/X`), parses messy multi-column `.txt` exports (keeps the trailing numeric
  shift/intensity columns), resamples each spectrum onto a fixed band grid, L2-normalises,
  and builds a per-item replicate cloud.
- The spectral encoder is fit on the **whole corpus** (not one item) for a meaningful PCA
  basis.
- Verified on the real `Raman_Virus/` set: **14 virus folders × 10 spectra = 140 items**
  (CVB1, CVB3, EV70, EV71, H1N1, H3N2, H5N2, H7N2, PV2, REO, RSV, Rhino, coronavirus,
  influenza_B). When a folder is configured but unreadable, the UI surfaces it instead of
  silently using synthetic data.

---

## 13. API endpoint reference (as-built)

**Engine / legacy session:** `/health` · `/configure` · `/status` · `/run` · `/step` ·
`/query` · `/generate` · `/curate` · `/labels` · `/label` · `/coverage` · `/infer` ·
`/control` · `/review` · `/ws`.

**Domains & config:** `GET/POST /domains` · `POST /domains/{id}/select|reset` ·
`DELETE /domains/{id}` · `GET/PUT /domains/{id}/config`.

**Training & planning:** `/{id}/analyze-folder` · `/{id}/select-subset` ·
`/{id}/train/start|next|preview|confirm|auto` · `GET/PUT /{id}/params` · `/{id}/plan` ·
`PUT /{id}/llm-assist`.

**Knowledge & graph:** `/{id}/learned/objectives|values|affordances` · `/{id}/graph` ·
`/{id}/node/{nid}` · `/{id}/debug` · `/{id}/agents` · `/{id}/inputs` ·
`/{id}/inspect/{index}`.

**Inference & results:** `/{id}/infer/folder` · `/{id}/results/{run_id}` ·
`/{id}/results/{run_id}/compare`.

**LLM:** `/llm/status` · `/llm/log` · `/llm/complete` · `/llm/clear` ·
`GET/PUT /llm/route`.

---

## 14. Repository layout (as-built)

```
MALAR_V2/
├── docker-compose.yml  Dockerfile  Makefile  pyproject.toml  .env.example
├── CLAUDE.md  README.md  RUNBOOK.md
├── MALAR_MASTER_PLAN.md        (original engine plan, M0–M12)
├── MALAR_WEBUI_PLAN.md         (original web UI plan, U0–U8)
├── MALAR_MASTER_PLAN_v2.md     (this comprehensive as-built document)
├── infra/litellm_config.yaml   (the one place providers are chosen; 7 aliases)
├── malar/                      (98 Python modules / 20 packages)
│   ├── core/   world/   encoders/   objects/   fields/   worldview/   hitl/
│   ├── memory/  policy/  generation/  validation/  agents/  llm/  mcp/
│   ├── training/  inference/  domains/   api/
│   └── agents/planner.py        (LLM Orchestrator/Planner)
│   └── domains/manager.py       (per-domain isolation)
│   └── api/{domains,train_ui,graph,results,llm_routes,domain_service}.py
├── ui/                         (React 18 + Vite + Tailwind, 8 tabs, 17 components)
│   └── src/tabs/{Domains,Configure,Train,WorldGraph,Knowledge,TestInference,Agents,LLM}Tab.jsx
│   └── src/components/{DebugConsole,AgentPanels,InputInspector,ParamEditor,Heatmap,…}.jsx
├── domains/                    (DomainSpec yamls; raman_virus.yaml is the real one)
├── Raman_Virus/                (real data: 14 viruses × 10 spectra)
├── data/domains/{id}/…         (per-domain artifacts/snapshots/results + registry.json)
└── tests/                      (10 test files, 30 tests)
```

---

## 15. Build milestones (as-built, with DoD)

**Engine M0–M12** — all complete & verified: scaffold/infra; world + Laplacians + context;
3 encoders; objects + fields + identification; combined memory; async curator; core loop;
LangGraph agent layer; generation + critic; API + eval; training (stops on coverage);
inference (OOD gate); web control plane.

**Web UI U0–U8** — all complete & verified: domain isolation; UI shell + Domains;
Configure; folder analyzer + subset; supervised one-at-a-time; Knowledge; World Graph;
unsupervised auto; Test & Inference.

**Post-plan additions** — Debug input inspector; Agents monitoring tab; LLM tab (status,
console, log); learning-parameter editor in Correct; frontier-provider layer (DeepSeek /
OpenAI / Claude) + runtime routing; LLM-driven Orchestrator/Planner; per-item LLM-assist;
real-data Raman ingestion; light professional theme.

---

## 16. Operations

```bash
cp .env.example .env            # set NEO4J_PASSWORD, LITELLM_*; optionally provider keys
docker compose up -d --build    # 6 services
docker compose exec ollama ollama pull gemma3n:e4b
# open http://localhost:3000
```

- **Frontier keys.** Put `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in
  `.env`, then `docker compose up -d --force-recreate litellm` (recreate, not restart),
  then route an agent role to that alias in the LLM tab.
- **GPU.** `gemma3n:e4b` fits a 6 GB GPU (tight); `gemma3n:e2b` is lighter. CPU works but
  is slow. Remove the `deploy:` block under `ollama` for CPU-only.

---

## 17. Risks & caveats

- **Unvalidated science** — earns its claims only at eval on a real labelled dataset.
- **Encoder drift** — every retrain invalidates stored vectors; blue-green re-index is
  mandatory.
- **Frontier egress** — frontier aliases send data off-box; local Gemma is the default.
- **6 GB VRAM** — keep agent prompts compact (summaries + ids, never raw graphs/spectra);
  per-item LLM-assist is bounded/opt-in.
- **LLM is advisory** — the deterministic Score + conformal gate make the final
  identification decision; the LLM proposes labels/values/plan from summaries only.
- **OneDrive** — pause sync while building if file writes act oddly.

---

## 18. Probabilistic Inference & Prediction layer (V3)

A new layer and a final Web-UI tab, **"Inference and Prediction"**, that turns the
deterministic identification into a probabilistic forecast. Seven agents report to an
**Inference Management agent**:

- **Bayesian-Topology / -Spectral / -Graph** — per-modality Bayesian classifiers
  (diagonal-Gaussian, shrinkage) fit over the **whole training corpus**; each emits a
  categorical posterior + evidence.
- **Inference Management** — fuses the three posteriors by a weighted **product of
  experts**, then runs an **MCMC posterior-predictive** simulation over
  {class, value, action} (light Monte-Carlo / Metropolis by default; **emcee/PyMC NUTS**
  via the "Full MCMC" checkbox) producing the predicted action with P(action) and each
  objective's 95% credible interval. Per-(class, objective) **value posteriors**
  (mean+variance, shrinking with corpus count) feed the simulation. The OOD gate still
  refuses to fabricate an action outside the trained domain.
- **Cross-Domain Reference** (opt-in per request) — searches other *loaded* domains'
  registries; matches are flagged and down-weighted (×0.4), never overwriting in-domain
  meaning (isolation preserved).
- **Diffusion (image/video)** — Stage-1 graph/heat label-diffusion over region/frame
  graphs (CPU, dt<2/λmax) → per-class field + vote + heatmap; Stage-2 generative
  score-based diffusion (GPU/torch), HITL-triggered, reports cleanly when no GPU.
- **RL (text/context)** — contextual bandit with Thompson sampling over the afforded
  action set; reward = human feedback and/or training labels; warm-started over the corpus.

"Agents decide; tools compute" holds: all posteriors, samples, diffusion fields and bandit
draws are deterministic Python; the LLM only sets options and writes the verdict.

**Modules:** `malar/inference/probabilistic/{likelihoods,bayes,mcmc,manager,crossdomain,
diffusion,rl}.py`; `QdrantMemory.scroll_all()` reads the corpus.

**API:** `POST /domains/{id}/predict`, `GET /domains/{id}/predict/agents`,
`POST /domains/{id}/predict/crossref`, `POST /domains/{id}/predict/feedback` (RL reward),
`POST /domains/{id}/predict/escalate` (Stage-2 diffusion).

**Persistence:** `malar/domains/persistence.py` snapshots each domain's learned state
(per-class prototypes + value/affordance records) to `data/domains/{id}/state.json` on
training; the engine restores it on build, so **predictions survive a server restart**
without a live training session (in Docker the real Qdrant volume persists the full memory
corpus as well).

**Optional extras (`pip install -e ".[ml]"`):** `torch` (Stage-2 GPU diffusion),
`emcee` (Full MCMC). The default path runs entirely on CPU/numpy.
