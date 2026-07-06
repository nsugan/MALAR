from fastapi.testclient import TestClient
from malar.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["ok"]


def test_configure_run_query_label_flow():
    assert client.post("/configure", json={"domain": "raman_virus", "review_mode": False}).status_code == 200
    res = client.post("/run", params={"mode": "train", "n": 4}).json()
    assert len(res["result"]) == 4
    mems = client.get("/query", params={"k": 5}).json()["memories"]
    assert isinstance(mems, list)
    # review toggle (UI contract)
    assert client.post("/review", json={"review_mode": True}).json()["review_mode"] is True
    # infer endpoint returns a structured result
    out = client.post("/infer", json={"text": "possible sars_cov_2 signature"}).json()
    assert "ood" in out
