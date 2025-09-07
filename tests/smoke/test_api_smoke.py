# tests/smoke/test_api_smoke.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.tools.parse_tracking import parse_tracking  # safe import
import json

def offline_generate(user_msg: str, top_k_context=None, provided_tracking=None):
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
    answer = "\n".join(f"- {b}" for b in bullets)
    citations = [{"source": "kb/offline_stub.md", "chunk_id": 1}]
    return {"answer": answer, "citations": citations}

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message", "")
    out = offline_generate(message, [])
    return out

@app.post("/chat/stream")
def chat_stream(payload: dict):
    message = payload.get("message", "")
    out = offline_generate(message, [])
    events = [
        {"event": "start", "data": {"status": "start", "agent": True}},
        {"event": "token", "data": {"token": out["answer"][:32]}},
        {"event": "end", "data": {"citations": out["citations"]}},
    ]
    text = "".join(f"event: {e['event']}\ndata: {json.dumps(e['data'])}\n\n" for e in events)
    return {"sse": text}

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_chat_ok():
    r = client.post("/chat", json={"message": "Track order 12345678 with Shipping_A", "top_k": 2})
    assert r.status_code == 200
    js = r.json()
    assert "answer" in js and js["answer"]

def test_chat_stream_minimal():
    r = client.post("/chat/stream", json={"message": "Return policy for fragile items"})
    assert r.status_code == 200
    js = r.json()
    assert "sse" in js and "event: start" in js["sse"] and "event: end" in js["sse"]
