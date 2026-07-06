# MALAR — Web UI Master Plan (augmentation)

**A companion plan to `MALAR_MASTER_PLAN.md`. It specifies the web application — multi-domain management, configuration, supervised one-at-a-time training, world-graph visualization, learned-knowledge review, unsupervised auto-training, and folder-level test/inference with results + comparison.**

This plan is written to be **grafted onto the existing working codebase** by Claude Code. It reuses the locked stack (FastAPI + LangGraph + Neo4j + Qdrant + Ollama `gemma4:e4b` via LiteLLM + React/Vite). It **adds** a per-domain isolation layer, new backend endpoints, and new UI views — it does not change the engine, memory, field, training, or inference semantics already defined.

> Where this plan says "existing", it means a module already specified in `MALAR_MASTER_PLAN.md` (e.g. `api/app.py`, `api/ws.py`, `api/control.py`, `training/campaign.py`, `inference/service.py`, `memory/neo4j_io.py`, `memory/qdrant_io.py`, `objects/`, `encoders/`, `fields/`).

---

## 0. Locked UI stack (reuse + additions)

| Concern | Decision |
|---|---|
| Framework | **React 18 + Vite + Tailwind** (as in main plan §9.4) |
| Backend transport | **FastAPI REST + WebSocket** (`api/ws.py` extended) |
| Graph rendering | **Cytoscape.js** (world/memory graph) |
| Charts | **recharts** (coverage, comparison, distributions) |
| App state | **React Context** (`DomainContext` for the active domain) + TanStack Query for server cache |
| Per-domain memory isolation | **`domain_id` scoping in Neo4j** (Community = single DB) + **one Qdrant collection per domain** (`mem__{domain_id}`) |
| Per-domain artifacts | `data/domains/{domain_id}/` (raw artifacts, snapshots, checkpoints, results) |

No substitutions — same rule as the main plan.

---

## 1. The central change: per-domain isolation (requirements i, ii)

Everything below depends on this. Today the stores are global; the UI needs **N isolated domains** that can be created, selected, reset, and continued.

### 1.1 Isolation model
- **Neo4j** — every node/edge carries a `domain_id` property; **all** Cypher in `neo4j_io.py` is scoped by the active `domain_id`. (Neo4j Community is single-database, so isolation is by property scoping, not separate DBs.)
- **Qdrant** — **one collection per domain**, named `mem__{domain_id}`, holding that domain's `phi/hyper/graph_emb` points. Reset = drop + recreate the collection.
- **Filesystem** — `data/domains/{domain_id}/{raw,bronze,silver,gold,artifacts,snapshots,checkpoints,results}`.
- **Model registry** — per-domain heads (`ψ_k, η_j`) and encoder versions under the domain dir.
- **DomainSpec** — `domains/{domain_id}.yaml` (extends the existing DomainSpec with `data_folders` + `data_description`, req iii).

### 1.2 Select / create / reset / continue semantics
- **Create domain** → new `domain_id`, empty namespace (fresh Neo4j scope + new Qdrant collection + dirs + blank DomainSpec).
- **Select existing domain** → set it active; **load** its accumulated state (graph, memory, learned values). Training / analysis / inference **continue** from where they were.
- **Switch domains** → the active namespace swaps; the UI shows the selected domain's data only. (This is the "reset graphs/memory for each domain" behavior — each domain's view is its own; switching never bleeds data across domains.)
- **Reset domain** (explicit button) → wipe that domain's Neo4j scope + Qdrant collection + dirs, keep the DomainSpec.
- **Delete domain** → remove everything for that `domain_id`.

### 1.3 New backend module: `DomainManager` (`malar/domains/manager.py`)
`create(name, desc) · list() · select(id) · reset(id) · delete(id) · active()`. Owns the `domain_id` ↔ collection/dir mapping and guarantees every store call is scoped. **All existing memory/field/training/inference calls gain a `domain_id` argument** sourced from the active domain.

---

## 2. UI structure — tabs

A single-page app with a global header (active-domain selector, mode badge: *supervised / unsupervised*, review toggle) and these tabs:

1. **Domains** — add / list / select / reset / delete (req i, ii)
2. **Configure** — data folders + description per domain (req iii)
3. **Train** — folder analysis → subset → supervised one-at-a-time confirm → unsupervised auto (req iv, vi, vii)
4. **World Graph** — full graph + values, opens in a separate window (req v)
5. **Knowledge** — objectives & values & affordances learned, confirm (req vi)
6. **Test & Inference** — folder → batch inference → results file + comparison (req viii, ix)

---

## 3. Domains tab (req i, ii)

- **List** of domains with status chips (objects learned, coverage %, last activity).
- **Add domain** — name + short description → creates an isolated namespace.
- **Select** — sets active; the whole app rescopes to that domain.
- **Reset / Delete** — with confirm dialogs (reset wipes data, keeps config; delete removes all).

Backend: `GET /domains`, `POST /domains {name, description}`, `POST /domains/{id}/select`, `POST /domains/{id}/reset`, `DELETE /domains/{id}`.

---

## 4. Configure tab (req iii)

For the active domain, **Configure** defines what to learn from:
- **Training-data folders** — add one or more folder paths (server-visible; mounted into the container — see §10).
- **Data description** — free-text describing the training data (what it is, labels available, instrument, etc.). Fed to the LLM as domain context for identification.
- Shows the DomainSpec essentials (objectives `R`, functional dims `F`, encoders, coverage targets) read from `domains/{id}.yaml`; editable.

Backend: `GET /domains/{id}/config`, `PUT /domains/{id}/config {data_folders[], data_description, objectives, functional_dims, coverage_targets}` → writes `domains/{id}.yaml`.

---

## 5. Train tab (req iv, vi, vii)

The core workflow. Three phases in one tab.

### 5.1 Folder analysis (req iv, part 1)
On choosing a configured folder, the **FolderAnalyzer** (`malar/training/folder_analyzer.py`):
- walks the tree, classifies files (spectra `.csv/.txt/.spc`, hyperspectral cubes `.hdr/.dat/.npy`, images, manifests/`README`/label files), records counts/sizes/structure,
- extracts any **existing information** (manifests, label columns, metadata sidecars),
- produces a **folder report** shown as a tree + summary cards.

Backend: `POST /domains/{id}/analyze-folder {path}` → `{tree, file_types, detected_labels, manifests, summary}`.

### 5.2 Similar-subset selection (req iv, part 2)
- Quick-feature + cluster the files; propose a **representative similar subset** for first training, with the rationale shown (clusters, why these items).
- User can accept/adjust the subset.

Backend: `POST /domains/{id}/select-subset {report}` → `{subset[], clusters, rationale}`.

### 5.3 Supervised one-at-a-time confirm (req iv part 3, vi)
**Mode = supervised.** The UI walks the subset **one item at a time**, and for each shows the generated representations side by side:
- **Graph** (the item's world-graph region, Cytoscape),
- **Topology** (persistence diagram / barcode from `encoders/topology.py`),
- **Hyperspectral** (the spectral embedding + raw spectrum plot from `encoders/spectral.py`),
- the **proposed identification** (LLM-assisted, §6.6 of main plan) + **objective values** `{v_{k,c}}` + **affordances** `F`.

Controls: **Confirm** (commit to memory, anchored to its `WorldContext`) · **Correct** (fix label/value → HITL, sticky) · **Skip**. "Did it learn this correctly?" is answered per item before advancing.

Backend (streamed over `/ws`):
- `GET /domains/{id}/train/next` → next item id,
- server streams `{graph, topology, hyper, identification, values, affordances}` over `/ws`,
- `POST /domains/{id}/train/confirm {item_id, decision: confirm|correct|skip, corrections?}` → writes via the existing objects/fields/memory pipeline.

### 5.4 Unsupervised auto-mode (req vii)
After the user has confirmed several items / data types, switch the mode toggle to **unsupervised**: the Train tab calls the existing **training campaign** (`training/campaign.py`) which auto-ingests the remaining data, identifies, learns values/affordances, and lets the **Curator** update the memory graph + Qdrant vectors — no per-item confirm. Live progress: coverage bars, items/sec, novel-objects count, OOD flags.

Backend: `POST /domains/{id}/train/auto {supervision:"unsupervised"}` → runs the campaign; progress over `/ws`.

> Supervised vs unsupervised maps onto the main plan's `REVIEW_MODE`: supervised = interrupt-and-confirm per item; unsupervised = stream-only, no interrupts.

---

## 6. World Graph tab (req v)

A full visualization of the active domain's world + memory graph with all associated values across the DBs, openable in a **separate window** (`/graph?domain={id}` route, `window.open`).
- **Cytoscape** canvas: objects, memories, world contexts, edges (`SIMILAR/DERIVED_FROM/TEMPORAL_NEXT/GROUNDED_IN/IDENTIFIED_IN/HAS_VALUE/AFFORDS`); filter by node/edge type; large graphs are sampled/paginated.
- **Node detail panel**: pulls from Neo4j (structure, class, provenance, world context) **and** Qdrant (vector neighbors), showing values `R`, diffused `V`, affordances `F`, and the `WorldContext` it was learned in.

Backend: `GET /domains/{id}/graph?filter=...&limit=...` → `{nodes, edges}`; `GET /domains/{id}/node/{nid}` → full detail incl. vector neighbors.

---

## 7. Knowledge tab (req vi)

Shows **what was learned**, for confirmation:
- **Objectives `R`** and, per objective, the learned **per-object values** `{v_{k,c}}` with provenance (data/human), version, and confidence.
- **Functional affordances `F`** per object class.
- Coverage per objective/class (recharts).
- **Confirm / Correct** controls (corrections go through HITL, sticky).

Backend: `GET /domains/{id}/learned/objectives`, `GET /domains/{id}/learned/values`, `GET /domains/{id}/learned/affordances`; corrections via existing `/label`.

---

## 8. Test & Inference tab (req viii, ix)

### 8.1 Folder batch inference (req viii)
- Choose a **test folder**; the same FolderAnalyzer (§5.1) reports its structure/info.
- **Run inference on the whole dataset**: each item → `inference/frontend.py` (image/video/text → `φ/h/graph`) → identify → values/`F` → action, OOD-gated (existing `inference/service.py`).
- **Store a results file**: `data/domains/{id}/results/{run_id}.{csv,json}` with per-item `{input_id, identified_object, action, confidence, ood, matched_contexts}`.

Backend: `POST /domains/{id}/infer/folder {path}` → `{run_id}` (progress over `/ws`); `GET /domains/{id}/results/{run_id}` → rows + file link.

### 8.2 Results + comparison with training (req ix)
- **Results table** (sortable/filterable): action, confidence, OOD per item; drill-down to the matched objects/contexts.
- **Comparison with existing training data**: distribution overlap (test vs trained object classes), OOD rate, per-item nearest training examples (from Qdrant), and an agreement view (how test items map onto known regimes). recharts + a side-by-side item view (test item vs nearest trained item: graph/topology/hyper).

Backend: `GET /domains/{id}/results/{run_id}/compare` → `{class_distribution, ood_rate, per_item_neighbors, overlap}`.

---

## 9. React component structure

```
ui/src/
├── App.tsx                      # router + DomainContext provider + header
├── context/DomainContext.tsx    # active domain, mode (supervised/unsupervised), review toggle
├── api/client.ts  ws.ts         # REST (TanStack Query) + WebSocket hook
├── components/
│   ├── DomainSelector.tsx  ModeBadge.tsx  ReviewToggle.tsx
│   ├── GraphView.tsx            # Cytoscape wrapper (world/memory graph)
│   ├── TopologyView.tsx         # persistence diagram / barcode
│   ├── HyperspectralView.tsx    # spectrum + embedding plot
│   ├── ValueCards.tsx  AffordanceCards.tsx  CoverageChart.tsx
│   ├── FolderTree.tsx  ItemReviewPanel.tsx  ResultsTable.tsx  CompareView.tsx
└── tabs/
    ├── DomainsTab.tsx  ConfigureTab.tsx  TrainTab.tsx
    ├── WorldGraphTab.tsx (also a standalone /graph window)  KnowledgeTab.tsx  TestInferenceTab.tsx
```

---

## 10. Backend additions (what Claude Code grafts on)

New modules:
- `malar/domains/manager.py` — DomainManager (§1.3).
- `malar/training/folder_analyzer.py` — folder analysis + subset selection (§5.1–5.2).
- `malar/api/domains.py`, `malar/api/train_ui.py`, `malar/api/graph.py`, `malar/api/results.py` — the routers above, mounted in `api/app.py`.
- `api/ws.py` — extend with channels: `train_item` (streamed reps), `campaign_progress`, `infer_progress`.

Changed (scoping):
- `memory/neo4j_io.py`, `memory/qdrant_io.py` — add `domain_id` to every read/write; Qdrant collection = `mem__{domain_id}`.
- `training/campaign.py`, `inference/service.py`, `objects/*`, `fields/*` — accept `domain_id` from the active domain.

Compose / mounts:
- Mount host training/test data into the container (e.g. `./data:/app/data` and an additional read-only mount for source folders the user configures), so configured folder paths resolve inside `malar`.
- No new services required — the existing `malar` (API) and `webapp` containers cover this.

---

## 11. Requirements traceability

| # | Requirement | Satisfied by | Backend |
|---|---|---|---|
| i | Add any number of domains | Domains tab (§3) | `DomainManager`, `POST /domains` |
| ii | Per-domain reset; same domain continues | Isolation model (§1) | `domain_id` scoping + `mem__{id}` collection; `select/reset` |
| iii | Configure: data folders + description | Configure tab (§4) | `PUT /domains/{id}/config` |
| iv | Analyze folder + choose similar subset; supervised one-at-a-time graph/topology/hyper + confirm | Train §5.1–5.3 | `analyze-folder`, `select-subset`, `train/next` + `/ws` + `train/confirm` |
| v | Visualize whole world graph + values, separate window | World Graph tab (§6) | `GET /graph`, `/node/{id}` |
| vi | Show objectives & values learned, confirm | Knowledge tab (§7) + Train §5.3 | `learned/objectives|values|affordances`, `/label` |
| vii | Same for more data types; then unsupervised auto-updates memory + vectors | Train §5.4 | `train/auto` → `campaign.py` + Curator |
| viii | Test/inference tab: folder → analyze → infer whole data → results file | Test tab §8.1 | `infer/folder` → results file |
| ix | Show results + comparison with training in UI | Test tab §8.2 | `results/{run_id}`, `/compare` |

---

## 12. Build milestones (augmentation order, U0–U8)

PR-sized; build in order; each has a Definition of Done. These graft onto the existing repo without breaking the engine.

**U0 — Domain isolation layer.** `DomainManager`; `domain_id` scoping in `neo4j_io.py`; per-domain Qdrant collections in `qdrant_io.py`; per-domain dirs; thread `domain_id` through training/inference/objects/fields; `/domains` CRUD + `select/reset`. *DoD:* two domains hold fully separate graphs/memory; selecting one shows only its data; reset wipes one without touching the other.

**U1 — UI shell + Domains tab.** Vite app, `DomainContext`, header (selector + mode badge + review toggle), tab router; Domains tab (add/list/select/reset/delete). *DoD:* create two domains, switch between them, app rescopes; status chips populate.

**U2 — Configure tab.** Folder paths + description form → `domains/{id}.yaml`; show/edit objectives & functional dims & coverage. *DoD:* config persists and reloads; folders validate as server-visible.

**U3 — Folder analyzer + subset.** `folder_analyzer.py` + `analyze-folder`/`select-subset`; UI folder tree + report + proposed subset with rationale. *DoD:* a sample folder yields a correct type/label report and an adjustable similar subset.

**U4 — Supervised one-at-a-time training.** `train/next` + `/ws` streaming of `{graph, topology, hyper, identification, values, affordances}`; `ItemReviewPanel` with Confirm/Correct/Skip → existing pipeline. *DoD:* walking the subset shows each item's three representations + proposed identity; Confirm writes an anchored memory; Correct goes to HITL and sticks.

**U5 — Knowledge tab.** `learned/objectives|values|affordances`; value/affordance cards + coverage; Confirm/Correct via `/label`. *DoD:* learned objectives and per-object values display with provenance; a correction updates the value and persists.

**U6 — World Graph tab.** `GET /graph` + `/node/{id}`; Cytoscape canvas + filters + node detail (Neo4j + Qdrant neighbors); standalone `/graph` window. *DoD:* the active domain's graph renders with values; node detail shows R/V/F + WorldContext + vector neighbors; opens in a separate window.

**U7 — Unsupervised auto-mode.** Mode toggle → `train/auto` → `campaign.py`; live coverage/novelty/OOD over `/ws`. *DoD:* switching to unsupervised auto-ingests the rest, updates the graph + Qdrant, and stops on coverage; the World Graph reflects new memories live.

**U8 — Test & Inference tab.** `infer/folder` (batch over `inference/service.py`) → results file; `ResultsTable` + `CompareView` (`results/{run_id}`, `/compare`). *DoD:* a test folder runs full-dataset inference, writes `results/{run_id}.csv|json`, and the UI shows per-item actions/OOD plus comparison (class distribution, OOD rate, nearest training examples) side by side.

---

## 13. Notes & risks
- **Per-domain isolation is the keystone** — land U0 first and test cross-domain leakage explicitly; everything else assumes it.
- **Large-graph rendering** — sample/paginate in `/graph`; never ship the full graph to Cytoscape at once. Provide class/type filters and a node cap.
- **Folder access** — configured folders must be mounted into the `malar` container; validate paths server-side and treat folder contents as untrusted data (no executing manifest contents).
- **Streaming reps** — generating graph/topology/hyper per item is compute-heavy; cache per item id and stream progressively so the panel fills in rather than blocking.
- **Supervised → unsupervised continuity** — both write the same per-domain memory; a domain can be trained partly supervised then finished unsupervised, then extended later (continue-on-reselect).
- **Results files** — keep them per-run under the domain dir; the compare view reads the run file + Qdrant neighbors, it does not recompute training.
- **6 GB VRAM** — batch inference and per-item encoding compete with the agents for the GPU; process test folders sequentially and keep `gemma4:e4b` resident.
