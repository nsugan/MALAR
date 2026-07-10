"""Active training agents: LLM-proposed extra agents + modality-routed learners.

Called once per training item (from DomainService.confirm, which auto-train also drives),
so it covers BOTH supervised and unsupervised training. Everything here is wrapped so it
can NEVER crash the training loop — any failure is caught and reported, training continues.

- Extra agents (LLM-proposed): each one takes the current item, asks the REASONER model to
  propose an action + value for its role, and stores what it learns in memory (functional
  affordances + objective values, LLM-sourced and sticky-safe).
- Modality routing:
    image / video  -> Diffusion agent (Stage-1 graph/heat label-diffusion; Stage-2
                      generative score-based diffusion when a CUDA GPU is available).
    text           -> RL contextual bandit (Thompson sampling), rewarded from the label.
    spectra/cube   -> handled by the standard topology/spectral/graph pipeline already.
"""
from __future__ import annotations

import numpy as np

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
_VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
_TXT_EXT = (".txt", ".md", ".text")


def detect_modality(item: dict) -> str:
    """Return 'image' | 'video' | 'text' | 'spectra' for a training item."""
    kind = str(item.get("kind", "")).lower()
    if kind in ("image", "img"):
        return "image"
    if kind in ("video", "vid"):
        return "video"
    if kind in ("text", "doc", "document"):
        return "text"
    if kind in ("spectra", "cube"):
        return "spectra"
    path = str(item.get("path", "")).lower()
    if path.endswith(_VID_EXT):
        return "video"
    if path.endswith(_IMG_EXT):
        return "image"
    if path.endswith(_TXT_EXT):
        return "text"
    return "spectra"


def _parse_action_value(out: str):
    action, value = None, None
    for line in (out or "").splitlines():
        u = line.upper()
        if "ACTION:" in u and action is None:
            action = line.split(":", 1)[1].split(";")[0].strip() or None
        if "VALUE:" in u and value is None:
            tok = line.upper().split("VALUE:", 1)[1].split(";")[0].strip()
            try:
                value = float(tok)
            except ValueError:
                value = None
    if action:
        action = action.replace(" ", "_")[:40]
    if value is not None:
        value = float(np.clip(value, 0.0, 1.0))
    return action, value


def run_extra_agents(engine, extra_agents, item, feature_summary, label, ctx_id, llm) -> list:
    """Each LLM-proposed extra agent proposes an action+value for this item and stores it."""
    out = []
    if not extra_agents or llm is None:
        return out
    for a in extra_agents:
        name = a.get("name") or "agent"
        role = a.get("role") or a.get("description") or ""
        prompt = (
            f"You are MALAR's '{name}' training agent. Your role: {role}.\n"
            f"Object class: '{label}'. Feature summary: {feature_summary}.\n"
            "From your role, propose ONE action verb for this object and a relevance VALUE "
            "in [0,1]. Reply STRICTLY as:\n"
            "ACTION: <verb> ; VALUE: <0..1> ; REASON: <one line>"
        )
        try:
            resp = llm.reason(prompt, source=f"extra:{name}", max_tokens=256)
        except Exception as e:  # noqa: BLE001
            out.append({"agent": name, "ok": False, "error": str(e)[:100]})
            continue
        action, value = _parse_action_value(resp)
        stored = []
        try:
            if action and engine.c.functional_fields:
                engine.c.functional_fields[0].learn_affordance(
                    label, action, True, source="llm", world_ctx=ctx_id)
                stored.append(f"affordance:{action}")
            if value is not None and engine.c.objective_fields:
                k = next(iter(engine.c.objective_fields))
                applied, _ = engine.value_store.set(label, k, float(value),
                                                    source="llm", by=name, world_ctx=ctx_id)
                if applied:
                    stored.append(f"{k}={value:.2f}")
        except Exception as e:  # noqa: BLE001
            out.append({"agent": name, "action": action, "value": value,
                        "ok": False, "error": str(e)[:100]})
            continue
        out.append({"agent": name, "action": action, "value": value,
                    "stored": stored, "ok": True})
    return out


def _train_bandit(engine):
    b = getattr(engine, "_train_bandit_obj", None)
    if b is None:
        from malar.inference.probabilistic.rl import build_from_engine
        b = build_from_engine(engine)
        engine._train_bandit_obj = b
    return b


def _class_means(engine):
    means = getattr(engine, "_train_class_means", None)
    return means if isinstance(means, dict) else {}


def run_modality_agent(engine, modality, item, g, label, ctx_id) -> dict:
    """Activate the modality-specific learner for this item and store what it learns."""
    if modality in ("image", "video"):
        try:
            from malar.inference.probabilistic.diffusion import DiffusionAgent
            pts = np.asarray(item.get("points"), dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                return {"modality": modality, "ok": False, "reason": "not a 2D region array"}
            agent = DiffusionAgent(engine, class_means=_class_means(engine))
            res = (agent.diffuse_video([pts]) if modality == "video"
                   else agent.diffuse_image(pts))
            top = res.get("top")
            # store the diffusion vote as an LLM/data-sourced affordance signal
            if top and engine.c.functional_fields:
                engine.c.functional_fields[0].learn_affordance(
                    label, f"diffusion_{modality}", True, source="data", world_ctx=ctx_id)
            return {"modality": modality, "ok": True, "stage": res.get("stage"),
                    "diffusion_top": top, "note": res.get("note")}
        except Exception as e:  # noqa: BLE001
            return {"modality": modality, "ok": False, "error": str(e)[:120]}
    if modality == "text":
        try:
            bandit = _train_bandit(engine)
            ctx = np.asarray(g, dtype=float)
            # reward: the label's first afforded action is "correct" for this context
            actions = bandit.actions
            correct = actions[0] if actions else None
            for act in actions:
                bandit.update(ctx, act, 1.0 if act == correct else 0.0)
            pick = bandit.act(ctx)
            return {"modality": "text", "ok": True, "rl_action": pick.get("action"),
                    "updates": bandit.updates}
        except Exception as e:  # noqa: BLE001
            return {"modality": "text", "ok": False, "error": str(e)[:120]}
    return {"modality": modality, "ok": True, "note": "standard spectra pipeline"}


def process_training_item(engine, meta, item, feature_summary, g, label, ctx_id, llm) -> dict:
    """Single entry point called per training item. Never raises."""
    try:
        modality = detect_modality(item)
    except Exception:
        modality = "spectra"
    extra = []
    modres = {"modality": modality, "ok": True}
    try:
        extra = run_extra_agents(engine, getattr(meta, "extra_agents", None),
                                 item, feature_summary, label, ctx_id, llm)
    except Exception as e:  # noqa: BLE001
        extra = [{"ok": False, "error": str(e)[:100]}]
    try:
        modres = run_modality_agent(engine, modality, item, g, label, ctx_id)
    except Exception as e:  # noqa: BLE001
        modres = {"modality": modality, "ok": False, "error": str(e)[:100]}
    return {"modality": modality, "extra_agents": extra, "modality_agent": modres}
