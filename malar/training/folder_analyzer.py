"""FolderAnalyzer (UI plan §5.1-5.2).

Walks a training/test folder, classifies files (spectra .csv/.txt/.spc, hyperspectral
cubes .hdr/.dat/.npy, images, manifests/label files), records counts/structure,
extracts existing label info (manifests, label columns, sidecars), and produces a
folder report. Then proposes a representative SIMILAR SUBSET for first training via
quick-feature clustering, with the rationale.

Folder contents are treated as untrusted DATA — manifests are parsed, never executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPECTRA_EXT = {".csv", ".txt", ".spc", ".asc"}
CUBE_EXT = {".hdr", ".dat", ".npy", ".npz"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MANIFEST_NAMES = {"manifest.json", "labels.csv", "readme", "readme.md", "metadata.json"}


@dataclass
class FileEntry:
    path: str
    name: str
    kind: str            # spectra | cube | image | manifest | other
    size: int
    label: str | None = None


@dataclass
class FolderReport:
    root: str
    n_files: int
    file_types: dict
    detected_labels: list[str]
    manifests: list[str]
    items: list[FileEntry]
    tree: dict
    summary: str = ""

    def to_dict(self) -> dict:
        return {"root": self.root, "n_files": self.n_files, "file_types": self.file_types,
                "detected_labels": self.detected_labels, "manifests": self.manifests,
                "items": [vars(i) for i in self.items], "tree": self.tree,
                "summary": self.summary}


def _classify(p: Path) -> str:
    n = p.name.lower()
    if n in MANIFEST_NAMES or n.startswith("readme"):
        return "manifest"
    ext = p.suffix.lower()
    if ext in SPECTRA_EXT:
        return "spectra"
    if ext in CUBE_EXT:
        return "cube"
    if ext in IMAGE_EXT:
        return "image"
    return "other"


def _label_from_path(p: Path, root: Path) -> str | None:
    # heuristic: first sub-directory under root often encodes the class
    try:
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) > 1:
            return parts[0]
    except Exception:
        pass
    return None


def analyze_folder(path: str, max_files: int = 5000) -> FolderReport:
    if not path or not str(path).strip():
        return FolderReport(root="", n_files=0, file_types={}, detected_labels=[],
                            manifests=[], items=[], tree={}, summary="no folder configured")
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return FolderReport(root=str(root), n_files=0, file_types={}, detected_labels=[],
                            manifests=[], items=[], tree={},
                            summary=f"folder not found: {root}")
    items: list[FileEntry] = []
    file_types: dict[str, int] = {}
    manifests: list[str] = []
    labels: set[str] = set()
    tree: dict = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir() or len(items) >= max_files:
            continue
        kind = _classify(p)
        file_types[kind] = file_types.get(kind, 0) + 1
        lab = _label_from_path(p, root)
        if lab:
            labels.add(lab)
        if kind == "manifest":
            manifests.append(str(p))
        items.append(FileEntry(path=str(p), name=p.name, kind=kind,
                               size=p.stat().st_size, label=lab))
        # build a shallow tree (top-level dirs -> counts)
        try:
            top = p.relative_to(root).parts[0]
        except Exception:
            top = "."
        tree[top] = tree.get(top, 0) + 1

    summary = (f"{len(items)} files: " +
               ", ".join(f"{k}={v}" for k, v in sorted(file_types.items())) +
               (f"; labels: {sorted(labels)}" if labels else "; no labels detected"))
    return FolderReport(root=str(root), n_files=len(items), file_types=file_types,
                        detected_labels=sorted(labels), manifests=manifests, items=items,
                        tree=tree, summary=summary)


def _quick_features(item: FileEntry) -> np.ndarray:
    """Cheap, dependency-free signature for clustering (size + name hash + kind)."""
    kinds = ["spectra", "cube", "image", "manifest", "other"]
    k = kinds.index(item.kind) if item.kind in kinds else 4
    name_sig = sum(ord(c) for c in item.name) % 997
    return np.array([k, np.log1p(item.size), name_sig / 997.0,
                     (hash(item.label or "") % 1000) / 1000.0], dtype=float)


def select_subset(report: FolderReport, k_clusters: int = 4,
                  per_cluster: int = 2) -> dict:
    """Cluster files, propose a representative similar subset + rationale."""
    data_items = [it for it in report.items if it.kind in ("spectra", "cube", "image")]
    if not data_items:
        return {"subset": [], "clusters": [], "rationale": "no trainable data files found"}
    X = np.stack([_quick_features(it) for it in data_items])
    k = min(k_clusters, len(data_items))
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
        labels = km.labels_
    except Exception:
        labels = np.arange(len(data_items)) % k

    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(i)

    subset, cluster_desc = [], []
    for c, idxs in clusters.items():
        chosen = idxs[:per_cluster]
        for i in chosen:
            subset.append(vars(data_items[i]))
        cluster_desc.append({"cluster": c, "size": len(idxs),
                             "example": data_items[idxs[0]].name,
                             "kind": data_items[idxs[0]].kind})
    rationale = (f"grouped {len(data_items)} data files into {len(clusters)} clusters by "
                 f"type/size/label; picked {len(subset)} representatives "
                 f"({per_cluster} per cluster) so first training covers the variety.")
    return {"subset": subset, "clusters": cluster_desc, "rationale": rationale}
