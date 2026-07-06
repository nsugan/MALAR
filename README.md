# MALAR

Local, agentic implementation of **MALAR** (Multi-scale Adaptive Learning with Abstraction and
Recall): a combined graph + persistence-homology + hyperspectral memory, with grounded objects
carrying objective / value / functional fields anchored to the world model they were learned in.
A training layer learns a domain to coverage; an inference layer acts on unknown image/video/text;
a web UI lets you start, review, then run it seamlessly.

See [`docs/MALAR_MASTER_PLAN.md`](docs/MALAR_MASTER_PLAN.md) for the full design.

## Two architectural laws
1. **Agents decide; tools compute.** Persistence homology, spectral/graph encoding, diffusion, and
   distances run in deterministic Python. The LLM orchestrates, sets weights, hypothesizes labels
   from *summaries*, and explains — it never computes embeddings or runs the metric.
2. **Local-first.** Default routing is Ollama; nothing leaves the box. Frontier is opt-in.

## Quick start
```bash
cp .env.example .env          # set NEO4J_PASSWORD, LITELLM_MASTER_KEY, LITELLM_SALT_KEY
make up                       # all 6 services
make pull-model               # ollama pull gemma4:e4b
curl http://localhost:4000/v1/models
open http://localhost:3000    # web UI
```

## Layout
- `malar/` — engine: world, encoders, fields, objects, memory, curator, agents, training, inference, api.
- `ui/` — React 18 + Vite + Tailwind control plane.
- `domains/` — DomainSpecs (e.g. `raman_virus.yaml`).
- `infra/litellm_config.yaml` — the single place providers are chosen.
- `tests/` — one file per module + loop + curator + train/infer integration.

## Running locally without Docker
The deterministic engine (world, Laplacians, encoders, memory logic, loop, training/inference math)
runs in-process. The memory **stores** (Neo4j, Qdrant) and **LLM** are reached through thin clients;
when those services are unreachable the clients raise clear errors. Bring the stack up with
`make up` to exercise the full integration. Unit tests target the deterministic engine and do not
require the external services.

## Make targets
`make up | down | logs | ps | pull-model | train DOMAIN=.. | infer FILE=..|TEXT=.. | checkpoint NAME=.. | test | lint`
