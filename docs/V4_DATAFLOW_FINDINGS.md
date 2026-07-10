# MALAR V4 — Data-flow inconsistencies: findings, fixes & status

**Audited 2026-07-07 against the V4 tree (`C:\Dev\MALAR_V4`), a byte-for-byte copy of the
final V3 code.** Method: module-by-module read of the actual seams (not the plan docs) plus
a full `pytest` run in a `python:3.11-slim` container (the host runs Python 3.14, for which
`ripser`/`POT` have no wheels).

This document records eight data-flow inconsistencies — places where the code's real flow
diverges from what the docstrings/plan describe, where a producer has no consumer, or where
the durable stores and the operational state have quietly drifted apart. Each item states
the evidence, the proposed fix, and whether it is **patched now** (backend, pytest-verified)
or **deferred** (with rationale and a concrete plan).

The through-line: **the durable stores (Qdrant / Neo4j) and the operational state (an
in-process RAM cache + `state.json`) diverged.** Most breaks are a documented store→store or
producer→consumer edge that was never wired, with an in-process shortcut standing in for it.

| # | Inconsistency | Severity | Status |
|---|---|---|---|
| ❶ | `world_emb` computed then dropped; two inference paths gate differently | High | **Patched** |
| ❷ | Value/affordance "Neo4j sync" has no writer (values absent from both DBs) | High | **Patched** |
| ❸ | Qdrant is write-only for the curator; merges/reinforce lost after restart | High | **Patched** |
| ❹ | "Unsupervised auto-train" runs two different pipelines by data source | Medium | **Deferred** (design) |
| ❺ | Backend `/ws` stage stream has no UI consumer | Medium | **Deferred** (UI) |
| ❻ | Review-mode toggle never reaches the backend | Medium | **Deferred** (UI) |
| R | Hardcoded LLM alias literals bypass the route table | Low | **Patched** |
| S | Sticky-value rejection is dropped, never re-queued to HITL | Low | **Deferred** (needs HITL wiring) |

---

## ❶ `world_emb` is computed, then the two inference paths gate on different vectors — PATCHED

**Evidence.** The front-end produces a distinct `world_emb` for validity-gating —
`inference/frontend.py:59-60` sets `world_emb = graph.world_embed(world)`, a *different*
projection from `g = gr.graph_emb`. But:
- `InferenceService.infer` gated on `fe.g` (`inference/service.py:65`), and registered
  trained contexts from each object's `o.g` (`service.py:51-53`). Self-consistent (query
  space == registered space), but the docstring claimed "validity-gated via world_emb".
- `InferenceManager.predict` gated on `fe.world_emb` (`inference/probabilistic/manager.py`)
  while *also* registering contexts from `o.g` (`manager.py:67-69`). So its cosine drift
  check compared **world_emb against graph_emb contexts** — across embedding spaces — making
  that OOD signal meaningless.

**Root cause.** No per-object `world_emb` is ever stored; only `o.g` is. So "validity gating
via world_emb" was never wired end-to-end, and the manager silently compared incomparable
vectors.

**Fix (shipped).** Unify both entry points on `fe.g`, matching the registered `o.g`
contexts:
- `manager.py`: `self.ood.combined(best_score, fe.world_emb)` → `... fe.g` with a comment.
- `validation/ood.py`: renamed the parameter `world_emb` → `ctx_emb` and documented the
  same-space contract (callers register `o.g`, so they must query `fe.g`).
- `service.py` docstring corrected to describe the real gate.

**Deferred enhancement.** If a richer whole-world validity gate is wanted, store `world_emb`
per object at registration (registry + `register_object` + `persistence.py`) and register/
query `world_emb` on *both* paths. Bigger change; tracked for later.

---

## ❷ Value/affordance stores claim a Neo4j sync that had no writer — PATCHED

**Evidence.** `fields/value_store.py` and `fields/functional_store.py` docstrings said
"synced to Neo4j by the memory layer", and `memory/schema.py` / the plan describe
`(:Object)-[:HAS_VALUE]->(:Objective)` and `(:Object)-[:AFFORDS]->(:Action)`. Nothing wrote
them — `neo4j_io.py` only created `:Memory`, `:WorldContext`, `:WorldSnapshot`. The API read
values straight from the in-process dicts (`api/domain_service.py:346-355`,
`eng.value_store.all()` / `eng.func_store.all()`), so values/affordances lived only in RAM +
`state.json` and were absent from both databases; the World-Graph could not show R/V/F.

**Fix (shipped).**
- Added `upsert_value` / `upsert_affordance` to **both** graph backends (`InMemoryGraph`
  and `Neo4jGraph`) in `memory/neo4j_io.py`, plus the `MemoryGraph` Protocol. They create
  `:Object` / `:Objective` / `:Action` nodes and `HAS_VALUE` / `AFFORDS` edges.
- Wired a best-effort mirror into the training write path (`DomainService.confirm`, after
  values/affordances are learned) — the in-process stores stay authoritative; the graph is a
  projection. Never raises (wrapped) so it cannot break training.
- Surfaced the new nodes/edges in `DomainService.graph()` and added `HAS_VALUE` / `AFFORDS`
  to `rel_types`.
- Corrected both docstrings to describe the real durability (`state.json` source of truth +
  graph projection).
- Test: `tests/test_memory.py::test_value_and_affordance_edges_land_in_graph`.

---

## ❸ Qdrant is write-only for the curator; the real read path was a volatile RAM cache — PATCHED

**Evidence.** `memory/store.py:26`: `self._cache = {}  # vector cache (Qdrant is opaque for
reads)`. Every maintenance op read that cache:
- `reinforce` rebuilt a **zero-vector stub** on a cache miss and then skipped the Qdrant
  re-upsert (`item.phi.size > 1` was false), so omega drifted in Neo4j only and vectors were
  lost.
- `merge` returned `None` on a cache miss — so **merges silently no-op'd after a restart.**
- `curator._weakest` / `compress` / `node_detail` / `debug` all read via the same cache, so
  they went blind cold.

Meanwhile `QdrantMemory.scroll_all()` *could* read vectors back and the probabilistic layer
used it — two subsystems read the same store two different ways; the curator picked the one
that doesn't survive a restart.

**Fix (shipped).**
- Added `QdrantMemory.fetch(mem_id)` (`client.retrieve` with vectors + payload).
- Added `MemoryStore._hydrate(mem_id)`: cache → Qdrant (real vectors + payload) → graph
  payload-only, caching the result. Routed `get_item`, `reinforce`, and `merge` through it,
  so cache misses now rehydrate real vectors instead of no-op'ing. `merge` also guards
  against a payload-only stub (`phi.size <= 1`).
- Test: `tests/test_memory.py::test_cold_cache_hydrates_from_qdrant_for_merge_and_reinforce`.

**Note.** Because `get_item` now hydrates on miss, `_weakest` / `compress` / `node_detail`
recover after a restart too. Same-process runs keep the cache warm, so behavior is unchanged
where it was already correct.

---

## ❹ "Unsupervised auto-train" runs two different pipelines by data source — DEFERRED (design decision)

**Evidence.** `api/domain_service.py:auto_train` branches: a **folder** queue commits each
item via `self.confirm(...)` (which also runs `process_training_item` extra/modality agents
and the factory agents, and writes snapshots/contexts), while the **synthetic/stream** path
calls `eng.step(batch)` (which does none of that). Identical "unsupervised training" thus
produces different memory and agent side-effects depending only on data origin.

**Proposed fix.** Extract a single `_commit_item(eng, item, supervision)` used by both
`confirm()` and `auto_train`'s synthetic branch, so the encode → identify → learn → curator
→ extra/factory-agents sequence is identical regardless of source; `eng.step` stays the
low-level primitive it wraps.

**Why deferred.** This changes training behavior and is covered by
`tests/test_training_inference.py`; it deserves its own PR with before/after coverage numbers
rather than riding along with the store fixes. No data is lost today — the paths just differ.

---

## ❺ / ❻ Legacy control plane (`/ws`, `/review`, `/control`) is vestigial in the multi-domain UI — DEFERRED (UI)

**Evidence.** The backend fully implements `/ws` (broadcasts `{"type":"stage",...}` via
`WSManager`), `/review`, and `/control` against the **singleton `EngineSession`**. But:
- No frontend file opens a WebSocket; every tab polls REST (e.g. `DebugConsole` → `/debug`
  every 2s). ❺: the stage stream has no consumer.
- `ui/src/context/DomainContext.jsx` holds a `review` boolean, but the header toggle only
  flips local React state; `ui/src/lib/api.js` never calls `/review` or `/control`. ❻: the
  toggle never reaches the backend.

**Key nuance.** The multi-domain studio does its review through the **per-domain supervised
train flow** (`/train/next` + `/train/confirm` with Confirm/Correct/Skip), which *is* fully
wired. The singleton-session `/ws`+`/review`+`/control` are leftovers from the original
single-session design, superseded by the per-domain flow — not simply "unwired features".

**Proposed fix — pick one:**
1. **Remove the vestige** (recommended): drop the dead `review` toggle from the header and
   document the supervised train flow as the review mechanism. Smallest, most honest.
2. **Wire it**: add `review()`/`control()` to `lib/api.js`, call them from the toggle, and
   add a `useWebSocket` hook that consumes `/ws` stage events into a live panel.

**Why deferred.** Both options edit React and need a UI build (`npm ci && npm run build`) to
verify; `node_modules` was intentionally left out of the V4 copy, and blind JSX edits
without a build are not safe to ship. Tracked as a UI task.

---

## R. Hardcoded LLM alias literals bypass the route table — PATCHED

**Evidence.** `inference/frontend.py:119`, `worldview/worldview.py:70`, and `health.py:60`
called `llm.complete("malar-reasoner"/"malar-fast", ...)` with a literal alias, routing
around the `_ROUTE` role table (`/llm/route`) — contrary to the "provider chosen in
`infra/litellm_config.yaml`, never in code" rule.

**Fix (shipped).** Routed all three through role methods: `llm.reason(...)` /
`llm.fast(...)`, each with a `source=` tag for the call log. Now the LLM tab's routing
actually governs these calls.

---

## S. Sticky-value rejection is dropped, never re-queued to HITL — DEFERRED

**Evidence.** `value_store.set` rejects a non-human overwrite of a sticky (human-set) value
and only appends a `reject_sticky` audit row (`fields/value_store.py:49-52`). The docstring
said the proposal "re-enters the HITL queue" — but nothing enqueues it. `ObjectiveField`
exposes `propose_change` that *returns* a proposal dict, yet `DomainService.confirm` calls
`learn_from_data` (source `data`), so during auto-train a sticky value is silently rejected.

**Proposed fix.** Give the engine a `HITLQueue` handle; on `applied is False` from a
data-sourced `set`, enqueue the `propose_change` proposal (kind `value_change`) so it
surfaces in the label/correction queue instead of vanishing. Contained but touches HITL
wiring across engine construction.

**Why deferred.** Low frequency (only when data contradicts a human value) and needs the
HITL queue threaded into the field agents; belongs with the ❹ training-path unification. The
docstring is corrected in the meantime to state the real behavior and point here.

---

## Verification

- Backend patches (❶, ❷, ❸, R) are covered by the existing suite plus two new tests in
  `tests/test_memory.py`. Target: full suite green under Python 3.11.
- Deferred items (❹, ❺, ❻, S) change behavior or require a UI build and are scoped above as
  their own follow-up PRs.
