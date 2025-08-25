import pytest
from fastapi.testclient import TestClient

import backend.main as appmod

@pytest.fixture(scope="module")
def client():
    # Build/ensure agent graph at import time (done in backend.main), then create client
    return TestClient(appmod.app)

def test_chat_agent_branch_monkeypatched(client, monkeypatch):
    # offline safe reply (no model use)
    def offline_guarded(user_msg, top_k_context, tok=None, model=None, provided_tracking=None):
        parts = []
        parts.append("Thanks for reaching out.")
        if provided_tracking:
            parts.append(f"Tracking: {provided_tracking}")
        if top_k_context:
            parts.append(f"Using {len(top_k_context)} context chunk(s).")
        parts.append("This is a handbook-based response (no live tracking).")
        return "\n".join(f"- {p}" for p in parts)

    # Patch the generator used by the agent graph at runtime
    monkeypatch.setattr(appmod, "infer_guarded", offline_guarded, raising=True)

    # Rebuild agent to capture patched infer_guarded
    from backend.agent.graph import build_graph
    def gen_fn(msg, ctx, tid=None):  # will call patched appmod.infer_guarded
        return appmod.infer_guarded(msg, ctx, tok=None, model=None, provided_tracking=tid)
    appmod.agent_app = build_graph(gen_fn)

    payload = {"message": "Track order 12345 with Shipping_A and give last two scan events."}
    r = client.post("/chat?agent=1", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "handbook-based" in data["answer"]
