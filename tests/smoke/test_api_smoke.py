import json
from typing import List, Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agent.graph import build_graph, run_agent
from backend.tools.parse_tracking import parse_tracking

def offline_generate(user_msg: str, top_k_context: List[str], provided_tracking: Optional[str] = None) -> str:
    """Deterministic generator for CI (no model downloads)."""
    parsed = parse_tracking(user_msg)
    ids = parsed.get("ids") or ([provided_tracking] if provided_tracking else [])
    bullets = []
    if not ids:
        bullets.append("Please share a valid tracking ID (and carrier if known).")
    else:
        bullets.append(f"Parsed tracking ID: {ids[0]} (carrier: {parsed.get('carrier') or 'unknown'}).")
        bullets.append("I’ll summarize the latest two scan events from our knowledge base.")
    if top_k_context:
        bullets.append(f"Using {len(top_k_context)} retrieved context chunks.")
    bullets.append("No live tracking is used; info is handbook-based.")
    return "\n".join(f"- {b}" for b in bullets)

# Build a minimal FastAPI app purely for smoke tests
_graph = build_graph(offline_generate)
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message", "")
    top_k = int(payload.get("top_k", 2) or 2)
    # In smoke we just pass empty context; graph still runs the tooling router.
    out = run_agent(_graph, message)
    return {"answer": out.get("answer", ""), "citations": out.get("citations", [])}

@app.post("/chat/stream")
def chat_stream(payload: dict):
    # For smoke we return a synthetic SSE-like body without real streaming.
    message = payload.get("message", "")
    out = run_agent(_graph, message)
    chunks = [
        {"event": "start", "data": {"status": "start", "agent": True}},
        {"event": "token", "data": {"token": out.get("answer", "")[:32]}},
        {"event": "end", "data": {"citations": out.get("citations", [])}},
    ]
    # Join into a single string that looks like SSE
    text = "".join(f"event: {c['event']}\ndata: {json.dumps(c['data'])}\n\n" for c in chunks)
    return {"sse": text}

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"

def test_chat_ok():
    r = client.post("/chat", json={"message": "Track order 12345678 with Shipping_A", "top_k": 2})
    assert r.status_code == 200
    js = r.json()
    assert "answer" in js and isinstance(js["answer"], str) and len(js["answer"]) > 0

def test_chat_stream_minimal():
    r = client.post("/chat/stream", json={"message": "Return policy for fragile items", "top_k": 2})
    assert r.status_code == 200
    js = r.json()
    assert "sse" in js and "event: start" in js["sse"] and "event: end" in js["sse"]
