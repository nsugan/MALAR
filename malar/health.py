"""Health check: pings Neo4j + Qdrant + LiteLLM gateway (incl. a completion).

Usage: python -m malar.health
Exit code 0 if all green, 1 otherwise.
"""
from __future__ import annotations

import sys

from malar.core.config import get_settings


def _ok(label: str) -> None:
    print(f"  [ OK ] {label}")


def _fail(label: str, err: Exception) -> None:
    print(f"  [FAIL] {label}: {err}")


def check_neo4j() -> bool:
    s = get_settings()
    try:
        from neo4j import GraphDatabase

        drv = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        with drv.session() as sess:
            sess.run("RETURN 1 AS ok").single()
        drv.close()
        _ok(f"Neo4j {s.neo4j_uri}")
        return True
    except Exception as e:  # noqa: BLE001
        _fail("Neo4j", e)
        return False


def check_qdrant() -> bool:
    s = get_settings()
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(url=s.qdrant_url)
        c.get_collections()
        _ok(f"Qdrant {s.qdrant_url}")
        return True
    except Exception as e:  # noqa: BLE001
        _fail("Qdrant", e)
        return False


def check_gateway(do_completion: bool = True) -> bool:
    s = get_settings()
    try:
        from malar.llm.client import LLMClient

        client = LLMClient()
        models = client.list_models()
        _ok(f"Gateway {s.llm_base_url} ({len(models)} models)")
        if do_completion:
            # 'fast' ROLE via the route table, not a hardcoded alias literal. (V4 fix)
            out = client.fast("Reply with the single word: ok", source="health")
            _ok(f"Gateway completion -> {out[:40]!r}")
        return True
    except Exception as e:  # noqa: BLE001
        _fail("Gateway", e)
        return False


def main() -> int:
    print("MALAR health check")
    results = [check_neo4j(), check_qdrant(), check_gateway()]
    green = all(results)
    print("ALL GREEN" if green else "SOME CHECKS FAILED")
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
