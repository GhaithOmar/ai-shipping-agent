from backend.agent.graph import build_graph, run_agent
from backend.tools.parse_tracking import parse_tracking

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

def test_agent_smoke():
    app = build_graph(offline_generate)
    out = run_agent(app, "Track order 12345 with Shipping_A and give the last two scan events.")
    assert "answer" in out
    assert isinstance(out["answer"], str) and len(out["answer"]) > 0
