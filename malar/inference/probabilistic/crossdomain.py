"""Cross-domain referencing (I3) — opt-in, isolation-respecting.

Per-domain isolation is a MALAR keystone, so this never runs by default. When the
operator opts in (per request), the Cross-Domain Reference agent queries OTHER domains'
object registries with the input's (phi, h, g) and returns the nearest objects, their
domain, class and similarity. Matches are always FLAGGED as cross-domain and DOWN-WEIGHTED
(`weight` < 1) so the manager never silently fuses another domain's meaning into this one.

Only domains whose engines are already loaded in this session are searched (we do not
build/instantiate other domains' engines as a side effect of a query); an explicit
`target_domains` list may further restrict the search.
"""
from __future__ import annotations

from dataclasses import dataclass

# Cross-domain evidence is trusted less than in-domain evidence.
CROSS_DOMAIN_WEIGHT = 0.4


@dataclass
class CrossMatch:
    domain: str
    cls: str
    score: float
    object_id: str

    def to_dict(self) -> dict:
        return {"domain": self.domain, "class": self.cls,
                "score": round(float(self.score), 4), "object_id": self.object_id,
                "weight": CROSS_DOMAIN_WEIGHT}


class CrossDomainReferencer:
    def __init__(self, manager, current_domain_id: str):
        self.dm = manager
        self.current = current_domain_id

    def _candidate_domains(self, target_domains: list[str] | None) -> list[str]:
        loaded = [d for d in getattr(self.dm, "_engines", {}) if d != self.current]
        if target_domains:
            loaded = [d for d in loaded if d in set(target_domains)]
        return loaded

    def search(self, phi, h, g, target_domains: list[str] | None = None,
               k: int = 5) -> dict:
        domains = self._candidate_domains(target_domains)
        matches: list[CrossMatch] = []
        for did in domains:
            eng = self.dm._engines.get(did)
            if eng is None:
                continue
            try:
                hits = eng.c.registry.topk(phi, h, g, k=k)
            except Exception:
                continue
            for hit in hits:
                if getattr(hit.obj, "candidate", False):
                    continue
                matches.append(CrossMatch(domain=did, cls=hit.obj.cls,
                                          score=hit.score, object_id=hit.obj.id))
        matches.sort(key=lambda m: m.score, reverse=True)
        matches = matches[:k]
        return {
            "enabled": True,
            "searched_domains": domains,
            "weight": CROSS_DOMAIN_WEIGHT,
            "matches": [m.to_dict() for m in matches],
            "note": ("cross-domain matches are flagged and down-weighted; "
                     "they never overwrite in-domain meaning"),
        }
