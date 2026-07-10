"""Neo4j I/O: the memory graph M (nodes = memories; structural abstraction g_m;
edges SIMILAR / DERIVED_FROM / TEMPORAL_NEXT / GROUNDED_IN -> WorldContext) plus
WorldSnapshot / WorldContext / Object / Objective / Action grounding nodes.

Two backends with the SAME interface:
  * Neo4jGraph    — production, real bolt driver (NEO4J_URI).
  * InMemoryGraph — offline/test fallback used automatically when the server is
                    unreachable; pure-Python, same method surface.

`make_memory_graph()` returns the real driver when it can connect, else the
in-memory one (logging which). This keeps the production path on real Neo4j while
the deterministic logic stays verifiable without Docker.
"""
from __future__ import annotations

from typing import Protocol

from malar.core.config import get_settings
from malar.memory.schema import MemoryItem

REL_TYPES = ("SIMILAR", "DERIVED_FROM", "TEMPORAL_NEXT", "GROUNDED_IN")


class MemoryGraph(Protocol):
    def upsert_memory(self, item: MemoryItem) -> None: ...
    def get_memory(self, mem_id: str) -> dict | None: ...
    def delete_memory(self, mem_id: str) -> None: ...
    def all_memories(self) -> list[dict]: ...
    def link(self, rel: str, src: str, dst: str, props: dict | None = None) -> None: ...
    def neighbors(self, mem_id: str, rel: str) -> list[str]: ...
    def upsert_world_context(self, ctx: dict) -> None: ...
    def upsert_snapshot(self, snap: dict) -> None: ...
    def memories_in_snapshot(self, snapshot_id: str) -> list[str]: ...
    # Grounded knowledge edges (V4 fix ❷): mirror the in-process value/functional
    # stores into the graph so (:Object)-[:HAS_VALUE]->(:Objective) and
    # (:Object)-[:AFFORDS]->(:Action) actually exist and can be visualised.
    def upsert_value(self, object_class: str, field_k: str, value: float, source: str,
                     version: int, world_ctx: str | None = None) -> None: ...
    def upsert_affordance(self, object_class: str, dim: str, action: str, source: str,
                          version: int, world_ctx: str | None = None) -> None: ...


class InMemoryGraph:
    backend = "in_memory"

    def __init__(self, domain_id: str = "default") -> None:
        self.domain_id = domain_id
        self.memories: dict[str, dict] = {}
        self.contexts: dict[str, dict] = {}
        self.snapshots: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str, dict]] = []  # (rel, src, dst, props)

    def upsert_memory(self, item: MemoryItem) -> None:
        self.memories[item.id] = {
            "id": item.id, "tau": item.tau, "omega": item.omega, "class": item.cls,
            "cost": item.cost, "encoder_version": item.encoder_version,
            "world_ctx_id": item.world_ctx_id, "snapshot_id": item.snapshot_id,
            "g_subgraph": item.g_subgraph,
        }
        if item.world_ctx_id:
            self.link("GROUNDED_IN", item.id, item.world_ctx_id)

    def get_memory(self, mem_id: str) -> dict | None:
        return self.memories.get(mem_id)

    def delete_memory(self, mem_id: str) -> None:
        self.memories.pop(mem_id, None)
        self.edges = [e for e in self.edges if e[1] != mem_id and e[2] != mem_id]

    def all_memories(self) -> list[dict]:
        return list(self.memories.values())

    def link(self, rel: str, src: str, dst: str, props: dict | None = None) -> None:
        self.edges.append((rel, src, dst, props or {}))

    def neighbors(self, mem_id: str, rel: str) -> list[str]:
        out = []
        for r, s, d, _ in self.edges:
            if r != rel:
                continue
            if s == mem_id:
                out.append(d)
            elif d == mem_id:
                out.append(s)
        return out

    def upsert_world_context(self, ctx: dict) -> None:
        self.contexts[ctx["id"]] = ctx
        if ctx.get("snapshot_id"):
            self.link("OF", ctx["id"], ctx["snapshot_id"])

    def upsert_snapshot(self, snap: dict) -> None:
        self.snapshots[snap["id"]] = snap

    def memories_in_snapshot(self, snapshot_id: str) -> list[str]:
        ctx_ids = {c["id"] for c in self.contexts.values() if c.get("snapshot_id") == snapshot_id}
        return [m["id"] for m in self.memories.values() if m.get("world_ctx_id") in ctx_ids]

    def wipe_domain(self) -> int:
        n = len(self.memories)
        self.memories.clear()
        self.contexts.clear()
        self.snapshots.clear()
        self.edges.clear()
        return n

    def wipe_all(self) -> int:
        return self.wipe_domain()

    # -- grounded knowledge (values / affordances) — V4 fix ❷ ----------
    def upsert_value(self, object_class: str, field_k: str, value: float, source: str,
                     version: int, world_ctx: str | None = None) -> None:
        self.objects = getattr(self, "objects", {})
        self.objectives = getattr(self, "objectives", {})
        oid, kid = f"obj::{object_class}", f"objective::{field_k}"
        self.objects.setdefault(oid, {"id": oid, "class": object_class})
        self.objectives.setdefault(kid, {"id": kid, "key": field_k})
        self.edges = [e for e in self.edges
                      if not (e[0] == "HAS_VALUE" and e[1] == oid and e[2] == kid)]
        self.link("HAS_VALUE", oid, kid, {"field": field_k, "value": float(value),
                  "source": source, "version": version, "world_ctx": world_ctx})

    def upsert_affordance(self, object_class: str, dim: str, action: str, source: str,
                          version: int, world_ctx: str | None = None) -> None:
        self.objects = getattr(self, "objects", {})
        self.actions = getattr(self, "actions", {})
        oid, aid = f"obj::{object_class}", f"action::{dim}::{action}"
        self.objects.setdefault(oid, {"id": oid, "class": object_class})
        self.actions.setdefault(aid, {"id": aid, "dim": dim, "action": action})
        self.edges = [e for e in self.edges
                      if not (e[0] == "AFFORDS" and e[1] == oid and e[2] == aid)]
        self.link("AFFORDS", oid, aid, {"dim": dim, "action": action, "source": source,
                  "version": version, "world_ctx": world_ctx})

    def add_agent_output(self, rec: dict) -> None:
        if not hasattr(self, "agent_outs"):
            self.agent_outs = []
        self.agent_outs.append(dict(rec))

    def agent_outputs(self, domain_id: str | None = None) -> list[dict]:
        outs = getattr(self, "agent_outs", [])
        return [o for o in outs if domain_id is None or o.get("domain_id") == domain_id]


class Neo4jGraph:
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, domain_id: str = "default"):
        from neo4j import GraphDatabase

        self.domain_id = domain_id
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()

    def _init_schema(self) -> None:
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT mem_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE")
            s.run("CREATE CONSTRAINT ctx_id IF NOT EXISTS FOR (c:WorldContext) REQUIRE c.id IS UNIQUE")
            s.run("CREATE CONSTRAINT snap_id IF NOT EXISTS FOR (w:WorldSnapshot) REQUIRE w.id IS UNIQUE")

    def upsert_memory(self, item: MemoryItem) -> None:
        with self.driver.session() as s:
            s.run(
                """MERGE (m:Memory {id:$id})
                   SET m.tau=$tau, m.omega=$omega, m.class=$cls, m.cost=$cost,
                       m.encoder_version=$ev, m.world_ctx_id=$wc, m.snapshot_id=$snap,
                       m.domain_id=$dom""",
                id=item.id, tau=item.tau, omega=item.omega, cls=item.cls, cost=item.cost,
                ev=item.encoder_version, wc=item.world_ctx_id, snap=item.snapshot_id,
                dom=self.domain_id,
            )
            if item.world_ctx_id:
                s.run(
                    """MATCH (m:Memory {id:$id}) MERGE (c:WorldContext {id:$wc})
                       MERGE (m)-[:GROUNDED_IN]->(c)""",
                    id=item.id, wc=item.world_ctx_id,
                )

    def get_memory(self, mem_id: str) -> dict | None:
        with self.driver.session() as s:
            rec = s.run("MATCH (m:Memory {id:$id}) RETURN m", id=mem_id).single()
            return dict(rec["m"]) if rec else None

    def delete_memory(self, mem_id: str) -> None:
        with self.driver.session() as s:
            s.run("MATCH (m:Memory {id:$id}) DETACH DELETE m", id=mem_id)

    def all_memories(self) -> list[dict]:
        with self.driver.session() as s:
            return [dict(r["m"]) for r in s.run(
                "MATCH (m:Memory {domain_id:$dom}) RETURN m", dom=self.domain_id)]

    def link(self, rel: str, src: str, dst: str, props: dict | None = None) -> None:
        rel = rel if rel.isidentifier() else "SIMILAR"
        with self.driver.session() as s:
            s.run(
                f"""MATCH (a {{id:$src}}), (b {{id:$dst}})
                    MERGE (a)-[r:{rel}]->(b) SET r += $props""",
                src=src, dst=dst, props=props or {},
            )

    def neighbors(self, mem_id: str, rel: str) -> list[str]:
        rel = rel if rel.isidentifier() else "SIMILAR"
        with self.driver.session() as s:
            res = s.run(
                f"MATCH (m {{id:$id}})-[:{rel}]-(n) RETURN n.id AS nid", id=mem_id)
            return [r["nid"] for r in res if r["nid"]]

    def upsert_world_context(self, ctx: dict) -> None:
        with self.driver.session() as s:
            s.run(
                """MERGE (c:WorldContext {id:$id})
                   SET c.snapshot_id=$snap, c.region_id=$region, c.t=$t, c.source=$source
                   WITH c MATCH (w:WorldSnapshot {id:$snap}) MERGE (c)-[:OF]->(w)""",
                id=ctx["id"], snap=ctx.get("snapshot_id"), region=ctx.get("region_id"),
                t=ctx.get("t"), source=ctx.get("source"),
            )

    def upsert_snapshot(self, snap: dict) -> None:
        with self.driver.session() as s:
            s.run("MERGE (w:WorldSnapshot {id:$id}) SET w.t=$t, w.hash=$hash",
                  id=snap["id"], t=snap.get("t"), hash=snap.get("hash"))

    def memories_in_snapshot(self, snapshot_id: str) -> list[str]:
        with self.driver.session() as s:
            res = s.run(
                """MATCH (m:Memory)-[:GROUNDED_IN]->(c:WorldContext {snapshot_id:$snap})
                   RETURN m.id AS id""", snap=snapshot_id)
            return [r["id"] for r in res]

    def wipe_domain(self) -> int:
        """Delete every Memory (and orphaned grounding) for THIS domain only."""
        with self.driver.session() as s:
            rec = s.run(
                """MATCH (m:Memory {domain_id:$d}) DETACH DELETE m RETURN count(m) AS n""",
                d=self.domain_id).single()
            return int(rec["n"]) if rec and rec["n"] is not None else 0

    def wipe_all(self) -> int:
        """Delete EVERYTHING in the graph (full fresh start across all domains)."""
        with self.driver.session() as s:
            rec = s.run("MATCH (n) DETACH DELETE n RETURN count(n) AS n").single()
            return int(rec["n"]) if rec and rec["n"] is not None else 0


    # -- grounded knowledge (values / affordances) — V4 fix ❷ ----------
    def upsert_value(self, object_class: str, field_k: str, value: float, source: str,
                     version: int, world_ctx: str | None = None) -> None:
        with self.driver.session() as s:
            s.run(
                """MERGE (o:Object {class:$cls, domain_id:$dom})
                   MERGE (k:Objective {key:$field, domain_id:$dom})
                   MERGE (o)-[r:HAS_VALUE]->(k)
                   SET r.value=$value, r.source=$source, r.version=$version,
                       r.world_ctx=$wc""",
                cls=object_class, field=field_k, value=float(value), source=source,
                version=version, wc=world_ctx, dom=self.domain_id)

    def upsert_affordance(self, object_class: str, dim: str, action: str, source: str,
                          version: int, world_ctx: str | None = None) -> None:
        with self.driver.session() as s:
            s.run(
                """MERGE (o:Object {class:$cls, domain_id:$dom})
                   MERGE (a:Action {dim:$dim, action:$action, domain_id:$dom})
                   MERGE (o)-[r:AFFORDS]->(a)
                   SET r.source=$source, r.version=$version, r.world_ctx=$wc""",
                cls=object_class, dim=dim, action=action, source=source,
                version=version, wc=world_ctx, dom=self.domain_id)

    def add_agent_output(self, rec: dict) -> None:
        import json as _json
        with self.driver.session() as s:
            s.run(
                """MERGE (a:AgentOutput {id:$id})
                   SET a.agent_id=$agent_id, a.agent_name=$agent_name, a.parent=$parent,
                       a.domain_id=$domain_id, a.cls=$cls, a.output=$output, a.ts=$ts
                   WITH a MERGE (d:Domain {id:$domain_id}) MERGE (a)-[:IN_DOMAIN]->(d)""",
                id=rec["id"], agent_id=rec.get("agent_id"), agent_name=rec.get("agent_name"),
                parent=rec.get("parent"), domain_id=rec.get("domain_id"), cls=rec.get("cls"),
                output=_json.dumps(rec.get("output")), ts=rec.get("ts"))

    def agent_outputs(self, domain_id: str | None = None) -> list[dict]:
        import json as _json
        with self.driver.session() as s:
            q = "MATCH (a:AgentOutput) "
            if domain_id:
                q += "WHERE a.domain_id=$d "
            res = s.run(q + "RETURN a ORDER BY a.ts DESC", d=domain_id)
            out = []
            for r in res:
                a = dict(r["a"])
                try:
                    a["output"] = _json.loads(a.get("output") or "null")
                except Exception:
                    pass
                out.append(a)
            return out

    def close(self) -> None:
        self.driver.close()


def make_memory_graph(prefer_real: bool = True, domain_id: str = "default") -> MemoryGraph:
    s = get_settings()
    if prefer_real:
        try:
            return Neo4jGraph(s.neo4j_uri, s.neo4j_user, s.neo4j_password, domain_id=domain_id)
        except Exception as e:  # noqa: BLE001
            print(f"[memory] Neo4j unreachable ({e}); in-memory graph fallback (domain={domain_id}).")
    return InMemoryGraph(domain_id=domain_id)
