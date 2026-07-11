"""Agent Memory — a persistent store of LLM-generated agents + a reuse index.

Backed by SQLite: each generated agent keeps its name, role, the DeepSeek-designed
algorithm, the Claude-written code, a text embedding (for reuse), a `validated` flag, a
usage count, a `note` (last validation/test result), and a `trace` (the exact algo/code
LLM queries + responses so you can see them in the Agent Factory).

Reuse: `find_similar(role)` returns the closest VALIDATED agent above a cosine threshold.
Embeddings are a deterministic hashed n-gram vector (no model). DB path is overridable with
AGENT_DB_PATH (a dedicated Docker volume), falling back to temp if SQLite locking is unsupported.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

EMB_DIM = 256


def _trim_io(obj, max_rows: int = 8, max_cols: int = 64, max_items: int = 64):
    """Shrink a run's input/output for storage/display: cap 2D feature lists to a small
    corner and long lists, so a big ctx doesn't bloat the row. Shape-preserving otherwise."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "features" and isinstance(v, list):
                rows = v[:max_rows]
                out[k] = [r[:max_cols] if isinstance(r, list) else r for r in rows]
                out["_features_shape"] = [len(v), len(v[0]) if v and isinstance(v[0], list) else 0]
            else:
                out[k] = _trim_io(v, max_rows, max_cols, max_items)
        return out
    if isinstance(obj, list):
        return [_trim_io(x, max_rows, max_cols, max_items) for x in obj[:max_items]]
    return obj


def embed_text(text: str) -> np.ndarray:
    """Deterministic hashed word + char-trigram embedding, L2-normalised."""
    v = np.zeros(EMB_DIM, dtype=float)
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", text)
    grams = list(words)
    for w in words:
        grams += [w[i:i + 3] for i in range(max(0, len(w) - 2))]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        v[h % EMB_DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class AgentMemory:
    def __init__(self, db_path: str | Path, qdrant=None):
        self.db_path = os.environ.get("AGENT_DB_PATH") or str(db_path)
        self.qdrant = qdrant
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init()
        except sqlite3.OperationalError:
            self.db_path = str(Path(tempfile.gettempdir()) / "malar_agent_memory.sqlite")
            self._init()

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        return c

    def _init(self):
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS agents(
                    id TEXT PRIMARY KEY, name TEXT, role TEXT, algorithm TEXT, code TEXT,
                    embedding TEXT, validated INTEGER DEFAULT 0, usage_count INTEGER DEFAULT 0,
                    provider_algo TEXT, provider_code TEXT, created REAL, note TEXT, trace TEXT,
                    last_input TEXT, last_output TEXT, last_run REAL, domain_id TEXT)""")
            # migrate older DBs that predate newer columns
            for col, decl in (("trace", "TEXT"), ("last_input", "TEXT"),
                              ("last_output", "TEXT"), ("last_run", "REAL"),
                              ("domain_id", "TEXT")):
                try:
                    c.execute(f"ALTER TABLE agents ADD COLUMN {col} {decl}")
                except sqlite3.OperationalError:
                    pass

    def add(self, name: str, role: str, algorithm: str, code: str,
            provider_algo: str = "", provider_code: str = "", note: str = "",
            trace: str = "", domain_id: str | None = None) -> str:
        aid = "agent_" + uuid.uuid4().hex[:12]
        emb = embed_text(f"{name} {role}")
        with self._conn() as c:
            c.execute(
                """INSERT INTO agents(id,name,role,algorithm,code,embedding,validated,
                   usage_count,provider_algo,provider_code,created,note,trace,domain_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (aid, name, role, algorithm, code, json.dumps(emb.tolist()), 0, 0,
                 provider_algo, provider_code, time.time(), note, trace, domain_id))
        return aid

    def get(self, aid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
        return dict(r) if r else None

    def all(self, domain_id: str | None = None, with_code: bool = True,
            include_cross_domain: bool = False) -> list[dict]:
        """Agents for `domain_id` (each domain owns its agents — isolation).

        * domain_id=None → every agent (unscoped / admin view).
        * domain_id set → only that domain's agents. If include_cross_domain, ALSO
          return other domains' VALIDATED agents, each flagged `cross_domain=True` — the
          opt-in path the orchestrator uses for complex cross-domain inference.
        """
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM agents ORDER BY created DESC")]
        out = []
        for r in rows:
            r["embedding"] = None
            own = (domain_id is None) or (r.get("domain_id") == domain_id)
            if not own:
                # foreign-domain agent: only surfaced when cross-domain is opted in, and
                # only if it is validated (never run another domain's unreviewed code).
                if not (include_cross_domain and r.get("validated") and r.get("domain_id")):
                    continue
                r["cross_domain"] = True
            else:
                r["cross_domain"] = False
            if not with_code:
                r.pop("code", None)
                r.pop("algorithm", None)
                r.pop("trace", None)
                r.pop("last_input", None)     # kept light for the list view
                r.pop("last_output", None)
            out.append(r)
        return out

    def record_io(self, aid: str, input_obj, output_obj) -> None:
        """Persist the LAST input this agent ran on and the output it produced, so the
        Agent Factory can show it for verification. Trimmed + best-effort (never raises)."""
        try:
            with self._conn() as c:
                c.execute(
                    "UPDATE agents SET last_input=?, last_output=?, last_run=? WHERE id=?",
                    (json.dumps(_trim_io(input_obj))[:12000],
                     json.dumps(_trim_io(output_obj))[:12000], time.time(), aid))
        except Exception:
            pass

    def update_code(self, aid: str, code: str) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE agents SET code=?, validated=0 WHERE id=?", (code, aid))
            return cur.rowcount > 0

    def set_note(self, aid: str, note: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE agents SET note=? WHERE id=?", (note, aid))

    def set_trace(self, aid: str, trace: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE agents SET trace=? WHERE id=?", (trace, aid))

    def mark_validated(self, aid: str, validated: bool = True) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE agents SET validated=? WHERE id=?",
                            (1 if validated else 0, aid))
            return cur.rowcount > 0

    def bump_usage(self, aid: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE agents SET usage_count=usage_count+1 WHERE id=?", (aid,))

    def delete(self, aid: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM agents WHERE id=?", (aid,))
            return cur.rowcount > 0

    def find_similar(self, role: str, name: str = "", threshold: float = 0.82,
                     only_validated: bool = True, domain_id: str | None = None) -> dict | None:
        """Closest reusable agent. Scoped to `domain_id` when given, so a domain reuses
        its OWN validated agents and never silently adopts another domain's."""
        q = embed_text(f"{name} {role}")
        best, best_s = None, threshold
        with self._conn() as c:
            rows = c.execute("SELECT * FROM agents").fetchall()
        for r in rows:
            if only_validated and not r["validated"]:
                continue
            if domain_id is not None and r["domain_id"] != domain_id:
                continue
            try:
                emb = np.array(json.loads(r["embedding"]), dtype=float)
            except Exception:
                continue
            s = float(q @ emb) if emb.shape == q.shape else 0.0
            if s >= best_s:
                best, best_s = dict(r), s
        if best is not None:
            best["similarity"] = round(best_s, 4)
            best["embedding"] = None
        return best
