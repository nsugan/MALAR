"""Train routers (UI plan §5): folder analysis, subset, supervised one-at-a-time, auto."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from malar.api.domain_service import get_service
from malar.training.folder_analyzer import analyze_folder, select_subset


class FolderCmd(BaseModel):
    path: str


class SubsetCmd(BaseModel):
    report: dict | None = None


class ConfirmCmd(BaseModel):
    decision: str = "confirm"          # confirm | correct | skip
    corrections: dict | None = None


class AutoCmd(BaseModel):
    n_ticks: int = 18


router = APIRouter(prefix="/domains", tags=["train"])


@router.post("/{did}/analyze-folder")
def analyze(did: str, cmd: FolderCmd):
    return analyze_folder(cmd.path).to_dict()


@router.post("/{did}/select-subset")
def subset(did: str, cmd: SubsetCmd):
    from malar.training.folder_analyzer import FileEntry, FolderReport

    rep = cmd.report or {}
    items = [FileEntry(**{k: v for k, v in it.items() if k in
             ("path", "name", "kind", "size", "label")}) for it in rep.get("items", [])]
    report = FolderReport(root=rep.get("root", ""), n_files=rep.get("n_files", 0),
                          file_types=rep.get("file_types", {}),
                          detected_labels=rep.get("detected_labels", []),
                          manifests=rep.get("manifests", []), items=items, tree=rep.get("tree", {}))
    res = select_subset(report)
    get_service().build_queue(did, from_subset=res["subset"])
    return res


@router.get("/{did}/train/next")
def train_next(did: str):
    nxt = get_service().next_item(did)
    return nxt or {"done": True}


@router.get("/{did}/train/preview")
def train_preview(did: str):
    return get_service().preview(did)


@router.post("/{did}/train/confirm")
def train_confirm(did: str, cmd: ConfirmCmd):
    return get_service().confirm(did, cmd.decision, cmd.corrections)


@router.post("/{did}/train/auto")
def train_auto(did: str, cmd: AutoCmd):
    return get_service().auto_train(did, n_ticks=cmd.n_ticks)


@router.post("/{did}/train/start")
def train_start(did: str):
    q = get_service().build_queue(did)
    return {"queued": len(q)}


@router.get("/{did}/params")
def get_params(did: str):
    return get_service().get_params(did)


class ParamsCmd(BaseModel):
    params: dict


@router.put("/{did}/params")
def set_params(did: str, cmd: ParamsCmd):
    return get_service().set_params(did, cmd.params)


class PlanCmd(BaseModel):
    apply: bool = False
    folder_summary: str | None = None


@router.post("/{did}/plan")
def plan_domain(did: str, cmd: PlanCmd):
    return get_service().plan_domain(did, apply=cmd.apply, folder_summary=cmd.folder_summary)


class LLMAssistCmd(BaseModel):
    enabled: bool


@router.put("/{did}/llm-assist")
def set_llm_assist(did: str, cmd: LLMAssistCmd):
    from malar.domains.manager import get_manager
    m = get_manager().set_flag(did, llm_assist=cmd.enabled)
    return {"llm_assist": m.llm_assist}
