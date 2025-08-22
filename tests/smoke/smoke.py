# tests/smoke/smoke.py
import os, re, sys, json, time
import requests

BASE = os.getenv("SHIPPING_API_BASE", "http://127.0.0.1:8000")

def assert_no_links(txt: str):
    assert not re.search(r"https?://|www\.|\.(com|net|org|io|co|ly)\b|@\w+", txt, re.I), f"Link/handle leaked: {txt}"

def health():
    r = requests.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    assert data.get("status") == "ok", data
    print("✓ /health OK:", json.dumps(data))

def chat(msg, top_k=2):
    r = requests.post(f"{BASE}/chat", json={"message": msg, "top_k": top_k}, timeout=30)
    r.raise_for_status()
    out = r.json()
    ans = out["answer"]
    cits = out["citations"]
    print(f"\n{msg}\n---\n{ans}\nCitations: {cits}")
    assert_no_links(ans)
    return ans

def main():
    print(f"Target API: {BASE}")
    health()

    # 1) Missing ID -> must ask for it
    a1 = chat("Track my order")
    assert re.search(r"(tracking|waybill|order)\s*(number|id)", a1, re.I), "Should ask for tracking/waybill ID"

    # 2) Link request -> explicit refusal + ID ask
    a2 = chat("Give me the DHL tracking link now")
    assert re.search(r"can.?t share tracking links|cannot share links", a2, re.I), "Should refuse to share links"
    assert re.search(r"(tracking|waybill|order)\s*(number|id)", a2, re.I), "Should still ask for ID"

    # 3) Customs info -> should NOT ask for ID (intent-aware)
    a3 = chat("Customs hold message received. What documents are needed?")
    assert not re.search(r"(tracking|waybill|order)\s*(number|id)", a3, re.I), "Customs advice shouldn't force ID"

    # 4) Delivery issue -> should ask for ID
    a4 = chat("Package marked delivered but I didn't get it.")
    assert re.search(r"(tracking|waybill|order)\s*(number|id)", a4, re.I), "Delivery issue should ask for ID"

    print("\n✓ Smoke tests passed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n✗ Smoke tests failed:", e)
        sys.exit(1)
