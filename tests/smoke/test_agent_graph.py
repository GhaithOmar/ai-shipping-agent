# tests/smoke/test_agent_graph.py
import os
import sys
import types

def test_agent_smoke():
    # Ensure offline BEFORE any imports that might touch HF/Qdrant
    os.environ["AGENT_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Stub out backend.search to prevent any import-time downloads
    fake_search = types.ModuleType("backend.search")

    def _search_stub(query: str, k: int = 3):
        return []  # predictable for smoke tests

    fake_search.search = _search_stub  # type: ignore[attr-defined]
    sys.modules["backend.search"] = fake_search

    # Safe to import after stubbing
    from backend.tools.parse_tracking import parse_tracking
    from backend.agent.graph import build_graph, run_agent

    # Lightweight offline generator to avoid model usage entirely
    def offline_generate(user_msg, top_k_context, provided_tracking=None):
        parsed = parse_tracking(user_msg)
        ids = parsed.get("ids") or ([provided_tracking] if provided_tracking else [])
        bullets = []
        if not ids:
            bullets.append("Please share a valid tracking ID (and carrier if known) so I can help.")
        else:
            bullets.append(f"Parsed tracking ID: {ids[0]} (carrier: {parsed.get('carrier') or 'unknown'}).")
            bullets.append("I’ll summarize the latest two scan events from our knowledge base.")
        if top_k_context:
            bullets.append(f"Using {len(top_k_context)} retrieved context chunks.")
        bullets.append("No live tracking is used; info is handbook-based.")
        return "\n".join(f"- {b}" for b in bullets)

    g = build_graph(offline_generate)
    out = run_agent(
        g,
        "Track order 12345678 with Shipping_A and give the last two scan events."
    )

    assert "answer" in out
    assert isinstance(out["answer"], str) and len(out["answer"]) > 0
