"""PredictService — the probabilistic Inference & Prediction layer for the UI (I6 + I3/I4/I5).

Holds one InferenceManager per domain (fit over that domain's whole trained corpus) and
exposes predict / agents / crossref / feedback / escalate. Reuses DomainService for the
trained engine, the spectra loader and per-class reference spectra, so everything stays
domain-isolated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from malar.api.domain_service import _json_safe, get_service
from malar.inference.probabilistic.manager import InferenceManager, PredictOptions


class PredictService:
    def __init__(self):
        self._svc = get_service()
        self._mgrs: dict[str, InferenceManager] = {}
        self._fit_sig: dict[str, int] = {}

    @property
    def dm(self):
        return self._svc.dm

    @property
    def llm(self):
        return self._svc.llm

    def _class_means(self, did: str) -> dict:
        """Per-class mean reference spectrum from the training queue (whole-corpus signal)."""
        items = self._svc._queues.get(did) or []
        means: dict[str, list] = {}
        groups: dict[str, list] = {}
        for it in items:
            lbl = it.get("label")
            pts = it.get("points")
            if lbl is None or pts is None:
                continue
            arr = np.asarray(pts, dtype=float)
            if arr.ndim == 2 and arr.size:
                groups.setdefault(lbl, []).append(arr.mean(axis=0))
        for lbl, rows in groups.items():
            means[lbl] = np.mean(np.vstack(rows), axis=0)
        return means

    def _manager(self, did: str, refit: bool = False) -> InferenceManager:
        eng = self.dm.engine(did, llm_client=self.llm)
        sig = eng.memory_size()
        mgr = self._mgrs.get(did)
        if mgr is None or refit or self._fit_sig.get(did) != sig:
            mgr = InferenceManager(eng, llm_client=self.llm, domain_manager=self.dm,
                                   domain_id=did, class_means=self._class_means(did))
            self._mgrs[did] = mgr
            self._fit_sig[did] = sig
        return mgr

    def _options(self, payload: dict) -> PredictOptions:
        p = payload or {}
        return PredictOptions(
            prior_mode=p.get("prior_mode", "prevalence"),
            prior_override=p.get("prior_override"),
            weights=p.get("weights"),
            n_samples=int(p.get("n_samples", 2000)),
            mcmc_method=p.get("mcmc_method", "mc"),
            seed=int(p.get("seed", 0)),
            cross_domain=bool(p.get("cross_domain", False)),
            target_domains=p.get("target_domains") or [],
            narrate=bool(p.get("narrate", True)),
        )

    def _load_image(self, payload: dict):
        fpath = (payload or {}).get("file")
        if not fpath:
            return None
        root = self._svc._resolve_folder(fpath)
        if not root:
            return None
        pts = self._svc._load_points(Path(root))
        return pts

    def predict(self, did: str, payload: dict) -> dict:
        try:
            mgr = self._manager(did, refit=bool((payload or {}).get("refit")))
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "ood": True}
        opt = self._options(payload)
        p = payload or {}
        image = self._load_image(p)
        text = p.get("text") or (None if image is not None else (p.get("query") or ""))
        try:
            res = mgr.predict(text=text or None, image=image, options=opt)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "ood": True}
        return _json_safe(res)

    def agents(self, did: str) -> dict:
        try:
            mgr = self._manager(did)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "agents": []}
        return {"domain": did, "agents": _json_safe(mgr.agents()),
                "classes": mgr.bayes.classes, "n_corpus": mgr.bayes.n_corpus}

    def crossref(self, did: str, payload: dict) -> dict:
        try:
            mgr = self._manager(did)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "enabled": False, "matches": []}
        if mgr.crossref is None:
            return {"enabled": False, "matches": [], "note": "no domain manager bound"}
        p = payload or {}
        image = self._load_image(p)
        fe = mgr.frontend.process(text=(p.get("text") or None) if image is None else None,
                                  image=image)
        return _json_safe(mgr.crossref.search(fe.phi, fe.h, fe.g, p.get("target_domains")))

    def feedback(self, did: str, payload: dict) -> dict:
        try:
            mgr = self._manager(did)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": str(e)}
        p = payload or {}
        return _json_safe(mgr.rl_feedback(p.get("action"), p.get("reward", 0.0)))

    def escalate(self, did: str, payload: dict) -> dict:
        try:
            mgr = self._manager(did)
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": str(e)}
        image = self._load_image(payload)
        if image is None:
            return {"available": False, "reason": "escalation needs a server-visible spectra file"}
        return _json_safe(mgr.escalate_diffusion(image))


_PREDICT: PredictService | None = None


def get_predict_service() -> PredictService:
    global _PREDICT
    if _PREDICT is None:
        _PREDICT = PredictService()
    return _PREDICT
