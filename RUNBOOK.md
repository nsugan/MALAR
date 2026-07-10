# MALAR — how to run it on your laptop (Windows)

There are two ways to run MALAR:

- **Route A — Full system** (Docker: Neo4j + Qdrant + Ollama + LiteLLM + API + Web UI). This is the
  real, complete experience with the web control plane at `http://localhost:3000`.
- **Route B — Local engine only** (just Python, no Docker). Fastest way to see it work: runs the
  loop, training campaign, eval, inference and the test suite using in-process/embedded stores.

Your folder: `C:\Dev\MALAR`

---

## Route A — Full system with Docker (recommended)

### A1. Install prerequisites
1. **Docker Desktop for Windows** (uses the WSL2 backend) — install from docker.com, launch it, wait
   until it says *Engine running*.
2. *(Optional, for GPU acceleration)* NVIDIA driver + **NVIDIA Container Toolkit**. Your RTX 3050 (6 GB)
   fits a small Gemma model. **CPU-only also works**, just slower — see A5 if you have no GPU.

### A2. Open a terminal in the project
Open **PowerShell**, then:
```powershell
cd C:\Dev\MALAR
```

### A3. Create the .env file
```powershell
copy .env.example .env
notepad .env
```
Set at least these (any non-empty values are fine for local use):
```
NEO4J_PASSWORD=malar_dev_password
LITELLM_MASTER_KEY=sk-malar-local
LITELLM_SALT_KEY=some_random_string
```
Leave `ANTHROPIC_API_KEY` blank to stay fully local. Save and close.

### A4. Start everything
```powershell
docker compose up -d --build
```
First build takes a few minutes (it downloads images and Python/ML packages). Check status:
```powershell
docker compose ps
```
You want `neo4j`, `qdrant`, `ollama`, `litellm`, `malar`, `webapp` all *running*.

### A5. Pull the local LLM model
```powershell
docker compose exec ollama ollama pull gemma3n:e4b
```
**`gemma3n:e4b`** is Gemma 3n (≈ effective 4B params, multimodal, fits 6 GB). If you prefer something
lighter/faster, pull a smaller model and point the config at it. For example:
```powershell
docker compose exec ollama ollama pull gemma3n:e2b   # lighter, or: gemma2:2b
```
Then edit `infra\litellm_config.yaml` and replace the `ollama/gemma3n:...` model lines with the tag
you pulled, and restart the gateway:
```powershell
docker compose restart litellm
```
> The LLM is only used for control / identification hypotheses / explanation. The math (topology,
> spectral, graph, memory, fields) runs without it, so MALAR still works even if you skip the model —
> identification just falls back to the deterministic metric.

### A6. Sanity checks
```powershell
curl http://localhost:4000/v1/models      # the LiteLLM gateway
curl http://localhost:8000/health         # the MALAR API
```
Optional full health check (Neo4j + Qdrant + gateway completion):
```powershell
docker compose exec malar python -m malar.health
```

### A7. Open the web UI
Open a browser at **http://localhost:3000**. Then:
1. Pick `raman_virus` in the domain dropdown → **Configure**.
2. Keep **Review mode ON** → **Start training** (or click **Step** to advance one tick at a time).
   Each pipeline stage shows its output as an Approve / Edit / Reject card.
3. Flip **Review mode OFF** to run seamlessly (no blocking).
4. Use the **Inference console**: type e.g. `possible sars_cov_2 signature in sample 7` → you get an
   action + confidence, or *"outside trained domain"*.

### A8. Stop / restart
```powershell
docker compose logs -f          # watch logs
docker compose down             # stop (keeps data volumes)
docker compose up -d            # start again
```

---

## Route B — Local engine only (no Docker)

Use this if you just want to see the engine, training, eval and tests run. No web UI database services;
the memory layer uses an in-process graph + Qdrant's embedded mode.

### B1. Install Python 3.11
Install from python.org (tick **"Add Python to PATH"**).

### B2. Create a virtual environment + install
```powershell
cd C:\Dev\MALAR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```
This installs the core deps (numpy, scipy, fastapi, qdrant-client, neo4j, langgraph, ripser, persim,
POT, scikit-learn). The heavy ML extras (`torch`, `torch-geometric`, `giotto-tda`) are **optional** —
the engine runs without them. If you want them too: `pip install -e ".[ml,dev]"`.

> If `pip` complains about a package, the deterministic engine only really needs:
> `pip install numpy scipy networkx pydantic pyyaml scikit-learn ripser persim POT qdrant-client fastapi "uvicorn[standard]" httpx`

### B3. Run the things
```powershell
# unit tests (should report: 27 passed)
pytest -q

# the main loop over synthetic data (memory grows then stabilises)
python -m malar.core.loop --ticks 12 --points 40

# a training campaign that stops on coverage + writes data\coverage_report.json
python -m malar.training.campaign --domain raman_virus

# the eval harness -> data\eval_report.json
python -m malar.eval

# a checkpoint of the trained model
python -m malar.training.checkpoint --name my_first_ckpt --domain raman_virus

# inference from the command line
python -m malar.inference.service --domain raman_virus --text "suspected influenza_a in sample"
```

### B4. Run the API + UI locally (optional, no Docker)
Terminal 1 (API):
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn malar.api.app:app --reload --port 8000
```
Terminal 2 (UI dev server — needs Node 18+ from nodejs.org):
```powershell
cd C:\Dev\MALAR\ui
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:3000`). The UI proxies `/api` and `/ws` to the API
on port 8000. (Without Neo4j/Qdrant running, the API uses its in-process fallbacks — fine for a demo.)

---

## Troubleshooting
- **`docker` not recognised** → Docker Desktop isn't installed/running. Start it first.
- **`make` not recognised** → Windows has no `make`; use the raw `docker compose ...` commands above.
- **GPU errors on `docker compose up`** → you have no NVIDIA runtime. Open `docker-compose.yml` and
  delete the `deploy:` block under the `ollama` service, then `docker compose up -d` again (CPU mode).
- **`ollama pull` model-not-found** → use an available tag and update `infra\litellm_config.yaml`
  (see A5).
- **Ports already in use** (3000/8000/7474/6333/11434/4000) → stop whatever is using them, or change
  the left-hand port numbers in `docker-compose.yml`.
- **OneDrive locking files** → if a build acts oddly, pause OneDrive sync for the folder while running.

---

## Inference & Prediction tab (V3)

After training a domain, open the **Inference and Prediction** tab (last in the bar):

1. Enter query text (e.g. `possible coronavirus signature in sample 3`) or a server-visible
   spectra file path.
2. Optionally set the prior, per-modality weights, MCMC sample count, the **Full MCMC
   (emcee/PyMC)** checkbox, and the **Cross-domain referencing** opt-in.
3. **Run prediction** → you get per-modality posteriors, the fused posterior, an MCMC
   forecast (predicted action + P(action), objective 95% credible intervals, trace), any
   cross-domain matches, a diffusion field (for image/video), and the RL policy with ✓/✗
   feedback buttons. Outside the trained domain it returns *"outside trained domain"* and
   proposes no action.

**Restart-survival:** training writes `data/domains/<id>/state.json`, so after
`docker compose restart malar` the domain's learned classes load automatically and you can
predict immediately (no need to retrain in the session). With the real Qdrant volume the
full memory corpus persists too.

**GPU / Full-MCMC:** Stage-2 generative diffusion needs a CUDA GPU and the `[ml]` extra;
the Full-MCMC checkbox needs `emcee`. Without them everything runs on the CPU defaults.

---

## Configuring LLM models (local Ollama + cloud) — all via `.env`

Models are no longer hard-coded. Every alias's model/endpoint/key is read from `.env`
(LiteLLM `os.environ/` interpolation), so you never edit code or `infra/litellm_config.yaml`.

**Change the local Ollama model** (e.g. to a different/newer Gemma):

```powershell
docker compose exec ollama ollama pull gemma3:4b        # pull the tag you want
# in .env set (keep the ollama/ prefix):
#   OLLAMA_REASONER_MODEL=ollama/gemma3:4b
#   OLLAMA_FAST_MODEL=ollama/gemma3:1b
#   OLLAMA_VISION_MODEL=ollama/gemma3:4b
docker compose up -d --force-recreate litellm malar     # reloads .env (restart won't)
```

> There is no `gemma4` tag yet — current Google open models are `gemma3` / `gemma3n`.
> When a `gemma4` lands, `ollama pull gemma4:...` then set the `OLLAMA_*_MODEL` vars to it.

**Use a cloud model for an agent role.** Fill that provider's key in `.env`, then point the
role's alias at it. Built-in aliases: `malar-claude`, `malar-openai`, `malar-deepseek`,
`malar-glm` (Zhipu), `malar-kimi` (Moonshot), `malar-nemotron`, `malar-cosmos` (NVIDIA),
plus `malar-custom` for any OpenAI-compatible endpoint.

```ini
# .env — example: run the reasoner on GLM, keep vision local
GLM_API_KEY=your-zhipu-key
GLM_MODEL=openai/glm-4.6
GLM_API_BASE=https://api.z.ai/api/paas/v4
LLM_REASONER_ALIAS=malar-glm
```

```ini
# Kimi (Moonshot)
KIMI_API_KEY=your-moonshot-key
KIMI_MODEL=openai/kimi-k2-0711-preview
KIMI_API_BASE=https://api.moonshot.ai/v1
LLM_REASONER_ALIAS=malar-kimi
```

```ini
# NVIDIA Nemotron / Cosmos (key from build.nvidia.com)
NEMOTRON_API_KEY=your-nvidia-key
NEMOTRON_MODEL=openai/nvidia/nemotron-3-super-120b-a12b
NEMOTRON_API_BASE=https://integrate.api.nvidia.com/v1
LLM_REASONER_ALIAS=malar-nemotron
```

Then `docker compose up -d --force-recreate litellm malar`. You can also switch the route
live from the **LLM tab** in the web UI. The `*_MODEL` value carries the full model string;
the `openai/` prefix routes any OpenAI-compatible provider. Confirm the exact model id /
base URL on the provider's dashboard (intl vs CN endpoints differ for GLM and Kimi).
