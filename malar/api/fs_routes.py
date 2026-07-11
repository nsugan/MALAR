"""Filesystem browse — pick a server-visible training-data folder in the Configure tab.

The server can only read what is mounted into it (the project folder is mounted at /app),
so a typed host path that isn't under a mount silently resolves to nothing. This endpoint
lets the UI browse exactly what the server CAN see and pick a folder that is guaranteed to
resolve at training time. Only directory names + a trainable-file count are returned —
never file contents.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from malar.core.config import get_settings
from malar.training.folder_analyzer import ARRAY_EXT, IMAGE_EXT, NUMERIC_EXT, VIDEO_EXT

router = APIRouter(prefix="/fs", tags=["fs"])

_DATA_EXT = NUMERIC_EXT | ARRAY_EXT | IMAGE_EXT | VIDEO_EXT
_SKIP = {".git", "node_modules", "__pycache__", ".venv", ".pytest_cache",
         ".ruff_cache", ".mypy_cache", "dist"}


def _roots() -> list[dict]:
    """Suggested starting points the server can see (deduped, existing only)."""
    s = get_settings()
    data = Path(s.data_dir).resolve()
    cands = [("app root", Path("/app")), ("working dir", Path.cwd()),
             ("data dir", data), ("data parent", data.parent)]
    seen, out = set(), []
    for label, p in cands:
        try:
            rp = str(p)
            if p.exists() and p.is_dir() and rp not in seen:
                seen.add(rp)
                out.append({"name": f"{label}", "path": rp})
        except Exception:
            continue
    return out


def _count_data_files(root: Path, cap: int = 500) -> int:
    """Bounded recursive count of trainable data files (prunes heavy dirs)."""
    n = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP and not d.startswith(".")]
            for f in filenames:
                if os.path.splitext(f)[1].lower() in _DATA_EXT:
                    n += 1
                    if n >= cap:
                        return n
    except Exception:
        pass
    return n


@router.get("/browse")
def browse(path: str = ""):
    """List the sub-folders (and their data-file counts) of a server-visible directory."""
    roots = _roots()
    target = (path or "").strip() or (roots[0]["path"] if roots else str(Path.cwd()))
    try:
        cur = Path(target).resolve()
        if not cur.exists() or not cur.is_dir():
            return {"error": f"not a folder on the server: {target}", "path": target,
                    "parent": None, "roots": roots, "dirs": [], "n_data_files_here": 0}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "path": target, "parent": None, "roots": roots,
                "dirs": [], "n_data_files_here": 0}

    dirs = []
    try:
        for child in sorted(cur.iterdir(), key=lambda x: x.name.lower()):
            try:
                if (not child.is_dir() or child.name in _SKIP
                        or child.name.startswith(".")):
                    continue
                dirs.append({"name": child.name, "path": str(child),
                             "n_data_files": _count_data_files(child)})
            except Exception:
                continue
    except PermissionError:
        return {"error": f"permission denied: {cur}", "path": str(cur),
                "parent": str(cur.parent), "roots": roots, "dirs": [],
                "n_data_files_here": 0}

    parent = str(cur.parent) if cur.parent != cur else None
    return {"error": None, "path": str(cur), "parent": parent, "roots": roots,
            "dirs": dirs, "n_data_files_here": _count_data_files(cur, cap=5000)}
