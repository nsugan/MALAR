"""World-graph + Knowledge routers (UI plan §6-7)."""
from __future__ import annotations

from fastapi import APIRouter

from malar.api.domain_service import get_service

router = APIRouter(prefix="/domains", tags=["graph"])


@router.get("/{did}/graph")
def get_graph(did: str, filter: str | None = None, limit: int = 200):
    return get_service().graph(did, filter_type=filter, limit=limit)


@router.get("/{did}/node/{nid}")
def get_node(did: str, nid: str):
    return get_service().node_detail(did, nid)


@router.get("/{did}/debug")
def debug_snapshot(did: str):
    return get_service().debug_snapshot(did)


@router.get("/{did}/inputs")
def inputs(did: str):
    return get_service().queue_list(did)


@router.get("/{did}/inspect/{index}")
def inspect(did: str, index: int):
    return get_service().inspect_input(did, index)


@router.get("/{did}/agents")
def agents(did: str):
    return get_service().agents_snapshot(did)


@router.get("/{did}/learned/objectives")
def learned_objectives(did: str):
    return {"objectives": get_service().learned(did)["objectives"]}


@router.get("/{did}/learned/values")
def learned_values(did: str):
    return {"values": get_service().learned(did)["values"]}


@router.get("/{did}/learned/affordances")
def learned_affordances(did: str):
    return {"affordances": get_service().learned(did)["affordances"]}
