"""Domains + Configure routers (UI plan §3-4)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from malar.domains.manager import get_manager


class CreateDomain(BaseModel):
    name: str
    description: str = ""
    adapter_type: str = "synthetic"


class DomainConfig(BaseModel):
    data_folders: list[str] | None = None
    data_description: str | None = None
    objectives: list[dict] | None = None
    functionals: list[dict] | None = None
    coverage_targets: dict | None = None
    classes: list[str] | None = None
    adapter_type: str | None = None
    description: str | None = None


router = APIRouter(prefix="/domains", tags=["domains"])


@router.get("")
def list_domains():
    dm = get_manager()
    return {"active": dm.active(),
            "domains": [dm.status(m.id) | {"description": m.description} for m in dm.list()]}


@router.post("")
def create_domain(cmd: CreateDomain):
    m = get_manager().create(cmd.name, cmd.description, cmd.adapter_type)
    return {"id": m.id, "name": m.name, "adapter_type": m.adapter_type}


@router.post("/{did}/select")
def select_domain(did: str):
    return {"active": get_manager().select(did).id}


@router.post("/wipe-memory")
def wipe_memory():
    """Full fresh start: drop every domain's Qdrant collection and clear all of Neo4j."""
    return get_manager().wipe_all_memory()


@router.post("/{did}/reset")
def reset_domain(did: str, purge_memory: bool = True):
    return get_manager().reset(did, purge_memory=purge_memory)


@router.delete("/{did}")
def delete_domain(did: str, purge_memory: bool = True):
    return get_manager().delete(did, purge_memory=purge_memory)


@router.get("/{did}/config")
def get_config(did: str):
    m = get_manager().get(did)
    if not m:
        return {"error": "not found"}
    return {"id": m.id, "name": m.name, "description": m.description,
            "adapter_type": m.adapter_type, "data_folders": m.data_folders,
            "data_description": m.data_description, "objectives": m.objectives,
            "functionals": m.functionals, "coverage_targets": m.coverage_targets,
            "classes": m.classes, "llm_assist": getattr(m, "llm_assist", False)}


@router.put("/{did}/config")
def put_config(did: str, cfg: DomainConfig):
    m = get_manager().update_config(did, **cfg.model_dump(exclude_none=True))
    return {"id": m.id, "data_folders": m.data_folders, "objectives": m.objectives}
