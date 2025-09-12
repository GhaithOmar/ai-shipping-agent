import os
os.environ["AGENT_OFFLINE"] = "1"  # set before importing app

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_ok():
    r = client.post("/chat?agent=0", json={"message": "Track order 12345678 with Shipping_A", "top_k": 1})
    assert r.status_code == 200
    js = r.json()
    assert "answer" in js and isinstance(js["answer"], str)

def test_search_ok():
    r = client.post("/search", json={"query": "label pricing", "k": 1})
    assert r.status_code == 200
    js = r.json()
    assert "results" in js
