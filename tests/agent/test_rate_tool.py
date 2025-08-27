from backend.agent.graph import build_graph, run_agent

def test_rate_quote_smoke():
    def offline_gen(user_msg, top_k_context, provided_tracking=None):
        return "- ok"
    app = build_graph(offline_gen)
    out = run_agent(app, "How much to ship 1kg from JO to AE express?")
    assert "answer" in out and isinstance(out["answer"], str)
