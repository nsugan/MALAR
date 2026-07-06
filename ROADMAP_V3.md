# MALAR V3 — Roadmap & Arrangements

V2 is frozen (git tag `v2` + `MALAR_V2_backup_*.zip`). V3 develops on this working tree
(version `3.0.0.dev0`). This file is the starting backlog — refine before building.

## Guiding goals for V3
1. **Earn the science.** Move from "runs end-to-end" to "measured on real labelled data".
2. **Make the heavy maths real.** Swap the deterministic stand-ins for trained encoders
   where it improves accuracy, behind the same interfaces (no API churn).
3. **Harden for real use.** Real databases, multi-user, persistence, packaging.

## Candidate V3 work items (proposed)

### A. Modelling & accuracy
- [ ] **Constrained LLM identification** — make the identify/plan prompts choose strictly
      from the known class labels (the V2 LLM sometimes invented `NEW:VirusRamanAnalyzer`).
- [ ] **Trained graph encoder** — replace the spectral/WL stand-in with a PyTorch-Geometric
      GNN (drop-in: same `encode`/`world_embed` contract), optional `[ml]` extra.
- [ ] **giotto-tda topology option** — alongside ripser, for richer multi-parameter PH.
- [ ] **Real eval on the Raman set** — held-out split, per-virus sensitivity/specificity/
      F1/AUROC; calibrate the conformal gate on real scores.

### B. Stores & scale
- [ ] **Exercise real Neo4j + Qdrant** end-to-end (the V2 default falls back to in-process/
      embedded); integration tests against the live services.
- [ ] **Persistence across restarts** — restore a domain's engine state (registry + values +
      memory) from its checkpoint on API start, so trained domains survive a container cycle.
- [ ] **Encoder-drift re-index** wired to a real retrain trigger.

### C. Multimodal & inference
- [ ] **Video front-end** (frames → temporal aggregation) made first-class.
- [ ] **Hyperspectral cubes** (.hdr/.dat) as a real adapter, not just file detection.
- [ ] **Batch inference performance** — stream results over WS; cap GPU contention.

### D. Web UI & UX
- [ ] **Per-objective value correction** (the V2 Correct panel applies one value to all
      objectives; split it per objective key).
- [ ] **Live training over WebSocket** (replace polling) for the supervised walk.
- [ ] **Code-split the UI bundle** (currently ~1 MB single chunk).
- [ ] **Auth / multi-user** if the studio is to be shared.

### E. Engineering & ops
- [ ] **CI** (pytest + ruff + UI build) on push.
- [ ] **Packaging** — publishable wheel + versioned Docker images.
- [ ] **Move the repo out of OneDrive** (OneDrive locks corrupt `.git`; see GIT_SETUP_V3.md).
- [ ] **Secrets hygiene** — `.env` stays out of git; document a secrets workflow.

## Definition of done for V3.0
A real labelled domain trained to coverage, with a held-out evaluation report and a
restored-from-checkpoint domain serving inference — all from the web UI, on the real stack.
