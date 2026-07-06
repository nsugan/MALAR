"""Test & Inference routers (UI plan §8)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from malar.api.domain_service import get_service


class InferFolderCmd(BaseModel):
    path: str = ""


router = APIRouter(prefix="/domains", tags=["results"])


@router.post("/{did}/infer/folder")
def infer_folder(did: str, cmd: InferFolderCmd):
    return get_service().infer_folder(did, cmd.path)


@router.get("/{did}/results/{run_id}")
def get_results(did: str, run_id: str):
    return get_service().results(did, run_id)


@router.get("/{did}/results/{run_id}/compare")
def compare(did: str, run_id: str):
    return get_service().compare(did, run_id)
