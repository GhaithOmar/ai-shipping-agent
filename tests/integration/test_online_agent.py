# tests/integration/test_online_agent.py
import os
import time
import pytest

# Gate: only run when you *explicitly* enable it locally.
RUN_ONLINE = os.getenv("RUN_ONLINE_TESTS") == "1"

# Skip in CI/offline unless you set RUN_ONLINE_TESTS=1 yourself
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_ONLINE, reason="Enable with RUN_ONLINE_TESTS=1"),
]

def _unset_offline_env():
    """Ensure the app takes the online path (model/Qdrant if available)."""
    for k in ["AGENT_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        if k in os.environ:
            del os.environ[k]

def test_health_and_agent_chat_online_import():
    """
    Imports the FastAPI app in ONLINE mode, hits /health,
    and makes a tiny /chat call via TestClient (agent path).
    Skips unless RUN_ONLINE_TESTS=1 to avoid CI cost.
    """
    _unset_offline_env()

    # Optional: make base selection predictable for local CPU-only testing
    # You can point to a small open model you've already downloaded, or leave your defaults.
    # os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    # os.environ.setdefault("ADAPTER_ID", "")  # ensure no LoRA is required

    # Import after flipping env
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # /health should succeed
    r = client.get("/health", timeout=30)
    assert r.status_code == 200
    js = r.json()
    assert js.get("status") == "ok"
    assert "base_model" in js and "agent_enabled" in js

    # Tiny agent call (short message, expect a structured answer & citations array)
    payload = {"message": "Track order 12345678 with Shipping_A", "top_k": 1}
    r2 = client.post("/chat?agent=1", json=payload, timeout=60)
    assert r2.status_code == 200
    out = r2.json()
    assert isinstance(out.get("answer"), str) and out["answer"].strip()
    assert isinstance(out.get("citations"), list)
