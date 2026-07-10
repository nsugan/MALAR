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
| Probabilistic Inference layer I0–I7 (Bayes fusion, MCMC, diffusion, RL bandit, cross-domain) | **Built & verified** (§18) |
| Web UI (multi-domain studio, 11 tabs incl. Agent Factory + Inference and Prediction) | **Built & verified**, review-mode wiring incomplete (§19.4) |
| LLM layer (gateway + routing + frontier providers + observability + planner + per-item assist) | **Built & verified**, 4 hardcoded-alias call sites remain (§19.3, §20) |
| Monitoring (Debug input inspector, Agents tab, LLM tab) | **Built & verified** |
| Real-data Raman ingestion (14 viruses × 10 spectra) | **Built & verified** |
| Verification | **47/47 unit tests pass (re-run 2026-07-06 via `python:3.11-slim` in Docker — the bare-metal host only has Python 3.14, for which `ripser`/`POT` have no wheels yet) · ruff clean · UI builds · full backend flow + no cross-domain leakage** |

Counts (re-audited 2026-07-06 — see §19/§20 for the full method): **117 Python modules**
across **19 packages**; **11 React tabs** (10 dedicated tab files + Debug, which reuses
`DebugConsole.jsx` directly) and **17 React components**; **~68 REST routes + 1 WebSocket
route** across 12 API router files; **7 LLM aliases** (3 local + 4 frontier), of which the
runtime **route table only covers 5 roles** (`reasoner/fast/vision/coder/algo`).

> **This document was audited against the current code on 2026-07-06** (module-by-module
> read-through of `malar/*` and `ui/src/*`, plus a fresh `pytest` run). Sections below carry
> inline **[as-audited]** notes where the code has drifted from what was previously written
> here. §19 is a from-scratch **Algorithmic Framework** (what the system actually does,
> file-and-line accurate) and §20 is a **Master-Plan correction/action list** (what to do
> about the drift) — both new, and deliberately kept separate from the narrative sections
> above, which are left intact as the original as-built record.

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

> **[as-audited]** Only `:Memory`, `:WorldContext`, `:WorldSnapshot` (+ an undocumented
> `:AgentOutput`/`:Domain` pair for the Agent Factory) are ever written to Neo4j, with
> edges `GROUNDED_IN`, `OF`, `IN_DOMAIN`, and generic `SIMILAR`/`DERIVED_FROM`/
> `TEMPORAL_NEXT`. **`:Object`, `:Objective`, `:Action`, `:FieldSample` and the
> `HAS_VALUE`/`AFFORDS` edges described in §6 exist only as module docstrings** in
> `fields/value_store.py` / `fields/functional_store.py` — those stores are in-process
> Python dicts, checkpointed to `data/domains/{id}/state.json` by `domains/persistence.py`,
> never synced to the graph. No APOC/GDS procedure is called anywhere in the codebase. The
> retrieval formula also **normalizes the fused similarity by the sum of weights** before
> adding the time/omega terms — not a straight unweighted sum as written above. See §19.2
> for the exact as-implemented algorithm and §20 for the fix options.

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

> **[as-audited]** Confirmed true — `objects/identify.py`'s LLM prompt interpolates only
> the summary dict and candidate class/score pairs, never `phi`/`h`/`g` arrays. However
> there is no `malar/llm/prompts/` module; prompts are built inline in `identify.py`,
> `agents/planner.py`, and `inference/frontend.py`. Four call sites bypass the alias/route
> contract with a hardcoded literal instead of the route table: `inference/frontend.py:119`,
> `worldview/worldview.py:70`, `health.py:60`, and the `CompleteCmd` default in
> `api/llm_routes.py:20`; `core/config.py` also carries alias-string fallback defaults.
> See §19.3 for the full identify() pipeline and §20 for the fix.

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

> **[as-audited]** Two tabs shipped after this section was written are missing from it:
> **"Inference and Prediction"** (§18) and **"Agent Factory"** (LLM-generated domain-specific
> agents — `AgentFactoryTab.jsx`, backed by `api/agent_routes.py`), bringing the real count
> to **11 tabs**. More importantly: `api/ws.py` + `api/control.py` + `EngineSession.set_review`
> are fully implemented on the backend (a real `POST /review` rebuilds the LangGraph with
> `interrupt_before` on the selected stages), but **the shipped React frontend never opens
> `/ws` and never calls `/control` or `/review`** — the header's Review-mode checkbox only
> flips local React state. Every tab instead polls its REST endpoint on an interval (e.g.
> `DebugConsole` polls `/debug` every 2s). §9.2's "REVIEW_MODE drives interrupt-before" is
> accurate for the backend in isolation but not for the running system end-to-end. See
> §19.4 and §20.

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

> **[as-audited]** The runtime router (`client.py`'s `_ROUTE` dict, `PUT/GET /llm/route`)
> genuinely works, but only covers **5 roles** (`reasoner/fast/vision/coder/algo`); the
> frontier aliases (`malar-deepseek/openai/claude/frontier`) are reachable only by passing
> the raw alias string to `complete()`, not as a routable role. See §19.3.

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

**Probabilistic inference** *(added, missing from earlier revisions of this table —
see §18, §19.5)*: `POST /{id}/predict` · `GET /{id}/predict/agents` ·
`POST /{id}/predict/crossref` · `POST /{id}/predict/feedback` ·
`POST /{id}/predict/escalate`. (`GET /{id}/predict/{run_id}`, promised in
`MALAR_INFERENCE_LAYER_PLAN.md`, was never implemented — no run persistence.)

**Agent Factory** *(added, undocumented until this audit — §19.4, §20)*:
`POST /agents/generate` · `GET /agents` · `GET /agents/{aid}` ·
`PUT /agents/{aid}/code` · `POST /agents/{aid}/test` · `POST /agents/{aid}/validate` ·
`DELETE /agents/{aid}` · `POST /agents/{aid}/run` ·
`GET /domains/{id}/agents/proposals` · `POST /domains/{id}/agents/generate-selected` ·
`POST /domains/{id}/agents/activate` · `GET /domains/{id}/agents/outputs` ·
`POST /domains/{id}/agents/use` · `GET /domains/{id}/agents/status`.

Total as of the 2026-07-06 audit: **~68 REST routes + 1 WebSocket route** across 12
router files (up from the "45+" figure in earlier revisions).

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
├── malar/                      (117 Python modules / 19 packages — audited 2026-07-06)
│   ├── core/   world/   encoders/   objects/   fields/   worldview/   hitl/
│   ├── memory/  policy/  generation/  validation/  agents/  llm/  mcp/
│   ├── training/  inference/(+probabilistic/)  domains/   api/
│   └── agents/planner.py        (LLM Orchestrator/Planner — the real "brain"; agents/
│                                  orchestrator.py etc. are unused stub classes, §19.1)
│   └── agents/factory.py|sandbox.py|validate.py|agent_memory.py|runner.py  (Agent Factory)
│   └── domains/manager.py|persistence.py  (per-domain isolation + state snapshot)
│   └── api/{domains,train_ui,graph,results,llm_routes,predict_routes,agent_routes,
│            domain_service,agent_service,session}.py
├── ui/                         (React 18 + Vite + Tailwind, 11 tabs, 17 components)
│   └── src/tabs/{Domains,Configure,Train,WorldGraph,Knowledge,TestInference,Agents,LLM,
│                 InferencePrediction,AgentFactory}Tab.jsx  (+ Debug tab reuses DebugConsole.jsx)
│   └── src/components/{DebugConsole,AgentPanels,InputInspector,ParamEditor,Heatmap,…}.jsx
├── domains/                    (DomainSpec yamls; raman_virus.yaml is the real one)
├── Raman_Virus/                (real data: 14 viruses × 10 spectra)
├── data/domains/{id}/…         (per-domain artifacts/snapshots/results + registry.json)
└── tests/                      (10 test files, 47 tests — all pass under Python 3.11)
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

> **[as-audited]** All seven agents (I0–I7, including the second-pass cross-domain/
> diffusion/RL agents) are shipped and match their locked design almost exactly — see
> §19.5 for the full confirm/deny against `MALAR_INFERENCE_LAYER_PLAN.md`'s Q1–Q7. Two
> real gaps: (1) **value posteriors are not persisted** — `fields/value_store.py` was never
> extended with a posterior; `mcmc.py`'s `ValuePosterior` computes an ephemeral Normal
> on-the-fly from the point value + corpus count, not the "Beta/Normal, additive,
> persisted" design the plan describes. (2) The plan's `GET /predict/{run_id}` (replay a
> stored run) was **never implemented** — predictions are computed live and are not
> persisted for later retrieval.

---

## 19. Algorithmic Framework (as-implemented, audited 2026-07-06)

This section documents what the code **actually does**, independent of what earlier
sections planned. It is organized by the five subsystems requested for this audit.
File:line references are accurate as of the audit date; re-check before relying on them
after further edits.

### 19.1 Agentic workflow

**Two independent tick implementations exist**, not one:

- **Path A — `MalarEngine.step()`** (`malar/core/loop.py:113-191`), used by
  `Campaign.run()` (`training/campaign.py:96`) and the CLI (`loop.py:248`):
  ```
  perception -> F_rep(117-120) -> RegionSampler(126-128) -> snapshot+WorldContext(130-136)
    -> encode phi,h,g (137-144) -> Identifier.identify (146-151)
    -> per-objective learn_from_data + evaluate -> R (153-156)
    -> per-functional learn_affordance -> region_action_set = F (157-159)
    -> diffuse_values -> V (161-163) -> Policy.decide (165-169)
    -> MemoryItem -> Curator.on_candidate (171-180)
    -> register_object prototype EMA-blend (184-185)
    -> every compress_every ticks: curator.compress (187-189)
  ```
- **Path B — `MalarAgentGraph.run_tick()`** (`malar/agents/graph.py:196-207`), used by
  the API/`EngineSession` (`api/session.py:92-105`): the same nine stages as discrete
  node functions (`ORDER`, `graph.py:165-166`: perception → topology → identification →
  value_learning → functional_field → fields_coordinator → policy → critic →
  memory_write). If `langgraph` imports successfully **and** `review_stages` is empty, it
  runs the compiled `StateGraph` with a `MemorySaver` checkpointer and
  `interrupt_before`; **otherwise (including whenever any stage is under review — the
  default) it falls through to a plain Python `for` loop over `ORDER`** (`graph.py:204-207`).
  `Campaign` never touches this path at all — LangGraph governs only the
  API/UI-driven loop, and even then only when review mode is fully off.

**The "11 agents" are mostly empty classes.** `orchestrator.py`, `perception.py`,
`topology_agent.py`, `objective_field_agent.py`, `functional_field_agent.py`,
`fields_coordinator.py`, `memory_agent.py`, `policy_agent.py`, `critic.py`,
`curator_agent.py`, `generation_agent.py` are all `role`-docstring + `__init__` +
`__repr__` stubs, never instantiated anywhere. The real per-stage logic lives in
module-level functions in `agents/graph.py` (`_node_perception`, etc.), in
`MalarEngine.step()`, and in the underlying deterministic components (`Identifier`,
`ObjectiveField`, `FunctionalField`, `Policy`, `MemoryCurator`). The actual "orchestrator
brain" is `agents/planner.py`'s `DomainPlanner`, not `orchestrator.py`.

**Agents beyond the documented 11** (all real, all live): `agents/factory.py`
(`AgentFactory` — LLM-generates/validates/runs domain-specific agent code),
`agents/sandbox.py` (restricted exec), `agents/validate.py` (AST safety check),
`agents/agent_memory.py` (SQLite reuse index), `agents/runner.py`
(`run_validated_agents`), `training/extra_agents.py` (`run_extra_agents`,
`run_modality_agent` — routes image/video → diffusion, text → RL bandit, spectra →
standard pipeline).

**HITL / sticky values.** `hitl/intervene.py`'s `HITLQueue` picks the
highest-uncertainty pending item (active learning) and tracks resolve/reject. The
sticky-value guarantee is enforced independently in `fields/value_store.py:44-56` and
`fields/functional_store.py:35-41`: any proposal with `source != "human"` is rejected
(audited as `reject_sticky`) once a record has `sticky=True`; only `source=="human"`
writes set it. The rejection is audit-logged, not automatically re-queued into
`HITLQueue` — the two mechanisms aren't wired together.

**Theorem guards (real, with locations):** `fields/values.py:32`
(`assert dt < 2.0/lambda_max + eps`); `world/laplacian.py:37-46` (`lambda_max`,
`safe_diffusion_dt`); `memory/store.py:40` (`assert 0.0 < rho < 1.0`);
`memory/store.py:59` (`assert c < 1.0`, contractive merge);
`memory/curator.py:31-35,72-74` (novelty thresholds/insert condition);
`core/config.py:39-44` (defaults: `rho=0.9`, `theta_novelty=0.15`, `eps=1e-3`,
`merge_contraction=0.7`, `dt_safety=0.9`).

### 19.2 Databases

**Neo4j — real schema** (`memory/neo4j_io.py`, `memory/schema.py`): node labels
actually created are `:Memory {id,tau,omega,class,cost,encoder_version,domain_id,
world_ctx_id,snapshot_id}` (L129-139), `:WorldContext {id,snapshot_id,region_id,t,
source}` (L177-185), `:WorldSnapshot {id,t,hash}` (L187-190), and an undocumented
`:AgentOutput`/`:Domain` pair (L214-224) for the Agent Factory. Edges written:
`GROUNDED_IN` (Memory→WorldContext), `OF` (WorldContext→WorldSnapshot), `IN_DOMAIN`
(AgentOutput→Domain), plus a generic `link()` used for `SIMILAR`/`DERIVED_FROM`/
`TEMPORAL_NEXT`. **`:Object`, `:Objective`, `:Action`, `:FieldSample` and
`HAS_VALUE`/`AFFORDS` are never written** — `ValueStore`/`FunctionalStore` are
in-process Python dicts, checkpointed to `data/domains/{id}/state.json` by
`domains/persistence.py`, not synced to the graph. No APOC/GDS procedure is called
anywhere. Fallback: `make_memory_graph()` (`neo4j_io.py:247-254`) wraps
`GraphDatabase.driver(...)` in a bare `try/except Exception` and falls back to
`InMemoryGraph` on **any** construction error, not specifically an unreachability
check.

**Qdrant** (`memory/qdrant_io.py`, `memory/schema.py:36-45`): one point per memory,
named vectors `phi`/`hyper`/`graph_emb`, payload exactly
`{tau,omega,class,neo4j_id,encoder_version,world_ctx_id,snapshot_id}`; one collection
per domain `mem__{domain_id}` (`core/loop.py:87`), addressed through a **collection
alias** for blue-green swaps. The `:memory:` embedded fallback is decided by the
*caller* (`loop.py:85`, `_qdrant_reachable(...)` check), not internally by
`qdrant_io.py`.

**Retrieval scoring** — real formula (`memory/retrieval.py:65-67`):
```python
fused_sim = sum(name_to_w[n] * modal.get(n, 0.0) for n in name_to_w) / wsum   # NORMALIZED
score = fused_sim + self.s.lam_time * time_gate(t, tau) + self.s.mu_omega * omega
```
Defaults (`core/config.py:39-51`): `w_phi=w_h=w_g=1.0`, `lam_time=0.1`, `mu_omega=0.2`.

**Curator algorithm** (`memory/curator.py`), real thresholds
(`theta_phi=0.25, theta_h=0.20, theta_g=0.20, theta_G=0.5, theta_U=0.6,
merge_radius=0.12, budget=200`):
```
novel = dphi>theta_phi or dh>theta_h or dg>theta_g or goal_gain>theta_G or uncertainty>theta_U
if not novel: reinforce nearest (EMA omega)
elif nearest within merge_radius (fused dist) and dphi<=theta_phi: contractive merge (c=0.7)
elif under budget: insert
else: evict weakest by utility() if new region's utility beats it, else defer
```
`compress_and_decay` (`memory/maintenance.py:28-50`) decays all items ×0.97 then
greedily keeps ≤budget by `utility(t)*(0.5+0.5*marginal_coverage)` — a submodular-style
greedy, not a proven (1−1/e) bound (that's a comment, not a proof). `reindex.py` does
real blue-green: builds `{alias}__{new_version}`, re-embeds from raw artifacts, swaps
the alias; `rollback()` points it back. `propagate.py` scopes via
`DERIVED_FROM`/`SIMILAR`/`TEMPORAL_NEXT` neighbors + same `world_ctx_id` + cosine
radius (`sim_radius=0.85`), capped at `budget=25`; relabels only where
`conf>=0.9 and gate.accept(conf)`, else flags; `rollback()` only reverses `"relabel"`
steps, silently no-ops on `"flag"` steps.

**Persistence across restarts** (`domains/persistence.py:30-91`): `save_engine_state`/
`load_engine_state` snapshot non-candidate object prototypes (phi/h/g/omega/tau/
world_ctx_id/provenance) + `ValueRecord`s + `AffordanceRecord`s to `state.json`
(atomic tmp-write+replace). This survives a restart **without** Neo4j/Qdrant running;
the full memory corpus additionally needs the Docker volumes to persist.

**Undocumented:** `wipe_domain()`/`wipe_all()` on both graph backends (used by
`DomainManager` resets); `MemoryStore`'s in-process `_cache` (`store.py:26`) is the
*only* full-fidelity read path back out of memory — Qdrant itself is write/query-only —
so a cold-started process's curator "weakest" scan only sees whatever is in that
process's cache.

### 19.3 RAG — retrieval-augmented identification

Real `identify()` pipeline (`objects/identify.py`):
```
hits = registry.topk(phi, h, g, k)                       # FusedRetriever, see §19.2
if hits[0].score >= theta_match and conformal.accept(hits[0].score):
    return metric match (method="metric")                 # no LLM call
else:
    summary = {H0,H1,H2 counts, persistence ranges, spectral peaks}   # caller-built, no raw arrays
    prompt = (
      "You identify an object from a TOPOLOGICAL/SPECTRAL feature SUMMARY and "
      "retrieval context. Choose the single best class label from the candidates, "
      "or reply UNKNOWN.\n"
      f"Feature summary: {summary}\n"
      f"Retrieval candidates:\n- {cls}: Score={score:.3f}  ...\n"
      "Reply with: LABEL: <class or UNKNOWN> ; REASON: <one line>"
    )
    label = llm.complete(alias, prompt)
    if label in known_classes and conformal.accept(hits[0].score):   # same gate result, not re-run
        return LLM-arbitrated match (method="llm")
    else:
        store as labeling candidate (method="candidate", matched=False)  # human is final arbiter
```
Confirmed: **no raw `phi`/`h`/`g` array is ever interpolated into an LLM prompt**, only
the summary dict and `(class, score)` pairs. `agents/planner.py`'s `suggest_object`
builds an analogous summary-only JSON prompt. There is no `malar/llm/prompts/` module —
prompts are inline-built in `identify.py`, `planner.py`, and `inference/frontend.py`.

**Gate math.** `ConformalGate` (`validation/conformal.py`) is split-conformal:
`threshold = max(floor, alpha_quantile_of_calibration_scores)`; `accept(score) = score
>= threshold`. `OODGate` (`validation/ood.py`) combines a retrieval-distance check
(`best_score < threshold`) with world-context drift (cosine similarity to registered
context embeddings). `Critic.review()` (`validation/critic.py:30`) requires **both**
conformal-accept **and** not-OOD — but `identify()` itself only calls the conformal
gate; OOD is not wired into the identification decision (it's applied later, in
`inference/service.py`, on the whole result).

**Reverse generation is fully implemented, not a stub.** `generation/generate.py`'s
`ReverseGenerator.generate()` deterministically synthesizes a point cloud (circle+noise
if `H1_count>=1 and max_persistence>0.1`, else a Gaussian blob), optionally decodes a
representative spectrum via `SpectralEncoder.decode_matrix`, and `counterfactual()`
perturbs `max_persistence` and regenerates — seeded and deterministic.

**MCP is dormant scaffolding.** `mcp/mcp_client.py` / `mcp/mcp_server.py` implement
`MCPToolRegistry.call()` and `Critic.review_tool_output()` (`validation/critic.py:38`)
correctly (tool outputs are gated as data, matching the design rule) — but nothing in
the codebase ever constructs an `MCPToolRegistry`, registers a tool, or invokes
`gate_tool_result`. It is implemented but unwired.

### 19.4 Web UI

**Real tab list** (`ui/src/App.jsx` `TABS`, in order) — 11, not 8: Domains,
Configure, Train, World Graph, Knowledge, Test & Inference, Agents, LLM, Debug
(reuses `DebugConsole.jsx` directly, no dedicated tab file), **Inference and
Prediction** (`InferencePredictionTab.jsx`), **Agent Factory**
(`AgentFactoryTab.jsx`, undocumented anywhere — LLM-generated domain-specific agent
authoring UI backed by `api/agent_routes.py`). 17 components under `ui/src/components/`.

**Real endpoint inventory** (~68 REST routes + 1 WebSocket route across 12 router
files: `app.py`, `ws.py`, `control.py`, `domains.py`, `train_ui.py`, `graph.py`,
`results.py`, `llm_routes.py`, `predict_routes.py`, `agent_routes.py`). The
`/domains/{id}/predict*` family (5 routes) and the entire Agent Factory family
(~14 routes: `/agents*` + `/domains/{id}/agents/*`) exist in code but are absent from
§13's endpoint reference. No documented endpoint has been removed — drift is
additive.

**REVIEW_MODE mechanism — backend real, frontend disconnected.** The backend is
correctly wired: `POST /review` (`api/control.py`, `ReviewCmd{review_mode,stages}`) →
`EngineSession.set_review()` rebuilds `MalarAgentGraph` with `review_stages =
set(STAGES)` when on, empty when off; `/ws` broadcasts `{"type":"stage",...}` frames
via `WSManager.broadcast`. **But grep across `ui/src` finds zero `new WebSocket` calls
and zero references to `/control` or `/review` in `lib/api.js`.** The header's
Review-mode checkbox only flips local React state (`DomainContext`); every tab instead
polls its REST endpoint on a timer (e.g. `DebugConsole` polls `/debug` every 2s). The
shipped system today behaves as a **polling UI with a decorative review toggle**, not
the interrupt-driven review console §9.2 describes.

Per-domain isolation in the UI (`DomainContext.jsx` → `activeId` threaded into every
tab's API calls) is confirmed to work exactly as documented.

### 19.5 Inference engine

**Point-estimate path** (`inference/service.py:55-93`):
```
fe = frontend.process(text|image|video)                 # -> phi,h,g,world_emb,confidence_scale
hits = registry.topk(fe.phi, fe.h, fe.g, k=3)
if not hits: return "outside trained domain"
confidence = hits[0].score * fe.confidence_scale
ood = OODGate.combined(hits[0].score, fe.g)              # [drift] uses fe.g, not fe.world_emb
if ood.ood or confidence < threshold: queue novel case; return "outside trained domain"
r = max(objective_fields.evaluate([{class: hits[0].obj.cls}]))     # R
v = r                                                     # no real diffusion at this entry point
actions = region_action_set(objects, functional_fields)   # F
decision = policy.decide(actions, [cls], r, v)
return {action, rationale, matched_objects, world_contexts, confidence, ood}
```
The probabilistic path (`inference/probabilistic/manager.py:129`) correctly uses
`fe.world_emb` for the same OOD check — the two entry points are inconsistent.

**Probabilistic path** (`manager.py:117-194`):
```
fe = frontend.process(...); hits = registry.topk(...); ood = OODGate.combined(best, fe.world_emb)
bres = BayesEnsemble.posteriors(fe.phi, fe.h, fe.g, prior_mode, weights)   # per-modality + fused
if opt.cross_domain: crossdomain.search(...)                               # I3, x0.4 down-weight
if image/video: diffusion.diffuse_image/video(...)                        # I4, Stage 1 only
rl_result = bandit.act(fe.g)                                               # I5, ALWAYS runs
if ood: return verdict "outside trained domain" (mcmc=None, predicted_action=None)
mres = predictive_simulation(bres.fused, value_posteriors, objective_keys, action_of, ...)  # I2
verdict = llm_narrate(...) or deterministic fallback
return {input, ood, bayes, cross_domain, diffusion, rl, mcmc, predicted_action, verdict, agents}
```

**Q1–Q7 vs. `MALAR_INFERENCE_LAYER_PLAN.md` (confirm/deny):**

| # | Locked decision | Code reality |
|---|---|---|
| Q1 | Two-stage diffusion, Stage 2 HITL-gated | **Confirmed** exactly (`probabilistic/diffusion.py:31-92`); degrades cleanly with no torch |
| Q2 | Contextual bandit, Thompson sampling | **Confirmed** — real Bayesian linear regression (`rl.py:19-51`) + Thompson pick (61-73); warm-start target is "first affordance wins" per class, a proxy, not true labelled actions |
| Q3 | Light MC/MH default, emcee NUTS opt-in | **Confirmed the methods exist**, but the actual default is `"mc"` alone, not a blended "MC/MH default" |
| Q4 | Cross-domain off-by-default, ×0.4 weight | **Confirmed exactly** (`crossdomain.py:18`, `manager.py:51`) |
| Q5 | Value posteriors, Beta/Normal, persisted | **Partially wrong** — Normal-only, computed ephemeral in `mcmc.py`, never persisted to `value_store.py` |
| Q6 | "Inference and Prediction" tab, last in bar | **Confirmed** (§19.4) |
| Q7 | Phased I0→I7, cut-2 present-but-disabled | **Confirmed shipped**, but `manager.agents()` now marks every agent `enabled: True` unconditionally — the "present-but-disabled" framing is moot |

**Undocumented capabilities:** `InferenceManager.refit()` (`manager.py:88-92`) and
`predict_service.py:49-58` auto-refit the Bayes ensemble/value posteriors/bandit when
corpus size changes; `_class_means` (`predict_service.py:32-47`) builds per-class
reference spectra specifically to seed the diffusion agent; the RL agent runs on
**every** predict call (even OOD ones) — only the final verdict/MCMC are suppressed
by the OOD gate, not RL itself.

**Missing promised capability:** the plan's `GET /domains/{id}/predict/{run_id}` does
not exist — predictions are not persisted for replay.

---

## 20. Master-Plan corrections & action list (audited 2026-07-06)

Prioritized follow-ups to reconcile this document with the running system. Nothing here
blocks current functionality; these are documentation-debt and design-decision items
surfaced by the audit in §19.

**P0 — decide and fix, these change user-visible behavior:**
1. **Review-mode UI is inert.** Either wire `ui/src/lib/api.js` to open `/ws` and call
   `POST /control` / `POST /review` (making the header toggle real), or remove the
   toggle and document today's polling-only behavior as the intended design. Currently
   the UI silently does neither what the docs nor the toggle itself imply.
2. **Neo4j schema vs. reality.** Either (a) actually write `:Object`/`:Objective`/
   `:Action`/`:FieldSample` + `HAS_VALUE`/`AFFORDS` to Neo4j as §6.5/§6.7 describe, or
   (b) rewrite those sections to describe the real design: `ValueStore`/
   `FunctionalStore` as in-process stores checkpointed to per-domain JSON. Pick one;
   right now the docs assert (a) and the code does (b).

**P1 — contract violations worth closing:**
3. Fix the 4 hardcoded LLM alias literals (`inference/frontend.py:119`,
   `worldview/worldview.py:70`, `health.py:60`, `api/llm_routes.py:20` default) to go
   through `client.py`'s route table, per the "provider chosen in
   infra/litellm_config.yaml, never in code" rule in `CLAUDE.md`.
4. Either add real APOC/GDS usage (the stack table promises it) or drop the claim.
5. Extend the runtime route table (`client.py`) to cover the frontier roles, or
   document that frontier aliases are call-by-literal-alias only, not routable roles.

**P2 — inference-layer design debt:**
6. Reconcile the value-posterior design: persist a real posterior on `ValueRecord`
   (Beta/Normal, as planned) instead of computing an ephemeral Normal at predict time,
   or update `MALAR_INFERENCE_LAYER_PLAN.md` §2.3/§8-Q5 to describe the ephemeral
   design actually shipped.
7. Implement `GET /predict/{run_id}` (persist prediction runs) or remove the promise
   from the inference plan.
8. `service.py`'s point-estimate OOD check should use `fe.world_emb` (as
   `manager.py`'s probabilistic path already does), not `fe.g`, for consistency.

**P3 — bookkeeping (low risk, do opportunistically):**
9. Refresh stats wherever quoted: **117 Python modules / 19 packages**, **11 React
   tabs / 17 components**, **~68 REST routes + 1 WS route**, **47 tests**.
10. Add the Agent Factory tab/endpoint family and the `/predict*` family to §13's
    endpoint reference table.
11. Document the Agent Factory subsystem (currently has zero coverage in any plan
    document despite being a full LLM-code-generation-and-sandboxing feature).
