"""Inference & Prediction routers (V3, probabilistic layer + second pass)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from malar.api.predict_service import get_predict_service


class PredictCmd(BaseModel):
    text: str | None = None
    file: str | None = None
    query: str | None = None
    prior_mode: str = "prevalence"
    prior_override: dict | None = None
    weights: dict | None = None
    n_samples: int = 2000
    mcmc_method: str = "mc"
    seed: int = 0
    cross_domain: bool = False
    target_domains: list | None = None
    narrate: bool = True
    refit: bool = False


class CrossRefCmd(BaseModel):
    text: str | None = None
    file: str | None = None
    target_domains: list | None = None


class FeedbackCmd(BaseModel):
    action: str
    reward: float = 1.0


class EscalateCmd(BaseModel):
    file: str | None = None


router = APIRouter(prefix="/domains", tags=["predict"])


@router.post("/{did}/predict")
def predict(did: str, cmd: PredictCmd):
    return get_predict_service().predict(did, cmd.model_dump())


@router.get("/{did}/predict/agents")
def predict_agents(did: str):
    return get_predict_service().agents(did)


@router.post("/{did}/predict/crossref")
def predict_crossref(did: str, cmd: CrossRefCmd):
    return get_predict_service().crossref(did, cmd.model_dump())


@router.post("/{did}/predict/feedback")
def predict_feedback(did: str, cmd: FeedbackCmd):
    return get_predict_service().feedback(did, cmd.model_dump())


@router.post("/{did}/predict/escalate")
def predict_escalate(did: str, cmd: EscalateCmd):
    return get_predict_service().escalate(did, cmd.model_dump())
