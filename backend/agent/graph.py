from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Literal
from typing_extensions import TypedDict
from dataclasses import asdict
from backend.search import search as vector_search  # legacy fallback search


from langgraph.graph import StateGraph, START, END

# Tools
from backend.tools.search_kb import search_kb, format_citations, KBHit
from backend.tools.parse_tracking import parse_tracking
from backend.tools.estimate_eta import estimate_eta
from backend.tools.rate_quote import rate_quote

# Memory 
from backend.agent.memory import ShortMemory
_memory = ShortMemory(max_turns=6)

# Helper function
def _ascii_safe(s: str) -> str:
    if not s:
        return s
    table = {
        "–": "-", "—": "-", "−": "-",
        "→": "->", "←": "<-",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "…": "...", "\u00A0": " ",
    }
    for k, v in table.items():
        s = s.replace(k, v)
    return s

# -------- Agent State --------
class AgentState(TypedDict, total=False):
    user_msg: str
    # tool outputs
    kb_hits: List[Dict[str, Any]]
    citations: List[Dict[str, str]]
    parsed: Dict[str, Any]            # parse_tracking result
    eta: Dict[str, Any]               # estimate_eta result
    # control
    needed: List[str]                 # list of needs: ["tracking_id", "retrieval", "eta"]
    done: bool                        # terminal flag
    # final
    answer: str
    history: List[str]
    rate: Dict[str, Any]              # rate_quote result

# Type of the generation function we’ll receive from the backend
GenerateFn = Callable[[str, List[str], Optional[str]], str]
# (user_msg, top_k_context, provided_tracking) -> answer_text

# -------- Nodes --------
def node_understand(state: AgentState) -> AgentState:
    """Decide what we need: parse IDs, retrieve KB, estimate ETA."""
    msg = state["user_msg"]

    # Always try to parse possible carrier + tracking IDs
    parsed = parse_tracking(msg)
    needs: List[str] = []

    # If user asked about tracking/scan/quote/where is my package: want retrieval too
    lower = msg.lower()
    intents = {
        "needs_eta": any(k in lower for k in ["eta", "how long", "when arrive", "delivery time"]),
        "needs_tracking": any(k in lower for k in ["track", "tracking", "scan", "status"]),
        "needs_quote": any(k in lower for k in ["quote", "rate", "price", "how much", "shipping cost", "label price"])  # NEW
    }

    if intents["needs_quote"]:
        needs.append("quote")


    if intents["needs_tracking"]:
        needs.append("retrieval")
        if not parsed.get("ids"):
            needs.append("tracking_id")

    if intents["needs_eta"]:
        # ask for countries if not present later; for now we try to infer or let model ask user
        needs.append("eta")

    state["parsed"] = parsed
    state["needed"] = needs
    state["done"] = False
    return state

def node_tool_router(state: AgentState) -> AgentState:
    """Run the tools we decided we need."""
    needs = state.get("needed", [])
    msg = state["user_msg"]

    # 1) KB retrieval
    kb_hits: List[KBHit] = []
    if "retrieval" in needs:
        carrier = state.get("parsed", {}).get("carrier")
        kb_hits = search_kb(msg, k=4, carrier=carrier)

        # NEW: fallback — if Qdrant returns nothing, try legacy vector_search
        if not kb_hits:
            legacy = vector_search(msg, k=4)
            kb_hits = []
            for h in legacy:
                txt = h.get("text") or h.get("chunk") or ""
                src = h.get("source") or ""
                chk = str(h.get("chunk_id") or h.get("id") or "")
                kb_hits.append(KBHit(text=txt, score=float(h.get("score", 0.0) or 0.0),
                                     source=src, chunk_id=chk, meta=h))

        state["kb_hits"] = [asdict(h) for h in kb_hits]
        state["citations"] = format_citations(kb_hits)


    # 2) ETA estimate (try only if user asked)
    if "eta" in needs:
        # naive detection of country codes in message (e.g., "JO to AE express")
        import re
        cc = re.findall(r"\b([A-Z]{2})\b", msg.upper())
        origin_cc = cc[0] if len(cc) >= 1 else "JO"       # default to JO (your locale) if missing
        dest_cc   = cc[1] if len(cc) >= 2 else origin_cc
        # detect service keyword
        lvl = "standard"
        for key in ["express", "standard", "economy"]:
            if key in msg.lower():
                lvl = key
                break
        state["eta"] = estimate_eta(origin_cc, dest_cc, lvl)
    # 3) Rate quote (only if asked)
    if "quote" in needs:
        # naive parse for weight and dims; in real app we would add a tiny parser.
        # for now, assume we extract something later; here we demo with safe defaults and let the model ask for missing info.
        rate = rate_quote(
            origin_cc=state.get("origin_cc", "JO"),
            dest_cc=state.get("dest_cc", "AE"),
            weight_kg=state.get("weight_kg", 1.0),
            dims_cm=state.get("dims_cm"),   # (L,W,H) if detected, else None
            service_level="standard",
        )
        state["rate"] = rate

    state["citations"] = format_citations(kb_hits)  # list[dict] with 'ref'


    return state

from typing import List  # ensure this import exists at top of file

def node_respond(state: AgentState, generate_fn: GenerateFn) -> AgentState:
    """Call the guarded generator, then append ETA/Rate/Parsed-ID bullets."""
    # 1) Build context for the generator
    top_k_context: List[str] = []
    for h in state.get("kb_hits", []):
        txt = h.get("text") or ""
        if txt:
            top_k_context.append(txt)

    # add a compact ETA note into context (helps the model phrase it correctly)
    eta = state.get("eta")
    if eta:
        top_k_context.append(
            f"ETA handbook estimate: {eta['eta_business_days']['min']}–{eta['eta_business_days']['max']} business days "
            f"for {eta['service_level']} from {eta['origin_cc']} to {eta['dest_cc']}. "
            f"Clarify that this is not live tracking."
        )

    history_lines = state.get("history", [])
    if history_lines:
        top_k_context = ["Previous turns:\n" + "\n".join(history_lines)] + top_k_context

    # keep context lean
    top_k_context = top_k_context[:4]

    # 2) Provide first parsed tracking id (if any)
    tracking_id = None
    parsed = state.get("parsed", {}) or {}
    ids = parsed.get("ids") or []
    if ids:
        tracking_id = ids[0]

    # 3) Generate the main answer
    model_ans = generate_fn(state["user_msg"], top_k_context, tracking_id).strip()
    model_ans = _ascii_safe(model_ans) 

    # 4) Append structured bullets for parsed ID, ETA, and RATE
    parts: List[str] = []

    if ids:
        parts.append(f"- Parsed tracking ID: {ids[0]} (carrier: {parsed.get('carrier') or 'unknown'}).")

    if eta:
        parts.append(
            f"- ETA (handbook): {eta['eta_business_days']['min']}–{eta['eta_business_days']['max']} business days "
            f"({eta['service_level']}, {eta['origin_cc']}→{eta['dest_cc']})."
        )

    rate = state.get("rate")
    if rate:
        parts.append(
            f"- Estimated label price: ~${rate['price_usd_est']} USD "
            f"({rate['service_level']}, zone: {rate['zone']}, billable {rate['billable_weight_kg']} kg)."
        )
        parts.append("- Note: This is a non-binding estimate; surcharges and contracts vary.")

    # 5) Combine
    if parts and model_ans:
        answer = "\n".join(parts + [model_ans])
    elif parts:
        answer = "\n".join(parts)
    else:
        answer = model_ans

    state["answer"] = _ascii_safe(answer)
    state["done"] = True
    return state


def node_history_read(state: AgentState) -> AgentState:
    state["history"] = _memory.as_lines()
    return state

def node_history_write(state: AgentState) -> AgentState:
    # persist last user message and model answer
    _memory.add("user", state.get("user_msg","")[:500])
    _memory.add("assistant", (state.get("answer") or "")[:500])
    return state
# -------- Builder --------
def build_graph(generate_fn: GenerateFn):
    """
    Build and return a compiled LangGraph app. The generate_fn is injected
    so we reuse the model already loaded in backend.main.
    """
    g = StateGraph(AgentState)

    # Wrap respond to inject generate_fn
    def _respond(state: AgentState) -> AgentState:
        return node_respond(state, generate_fn)

    g.add_node("understand", node_understand)
    g.add_node("tool_router", node_tool_router)
    g.add_node("respond", _respond)
    g.add_node("history_read", node_history_read)
    g.add_node("history_write", node_history_write)

    g.add_edge(START, "history_read")
    g.add_edge("history_read", "understand")

    g.add_edge("understand", "tool_router")
    g.add_edge("tool_router", "respond")

    g.add_edge("respond", "history_write")
    g.add_edge("history_write", END)


    return g.compile()

# -------- Runner convenience --------
def run_agent(app, message: str) -> Dict[str, Any]:
    """Utility to run the compiled graph and return the final state."""
    init: AgentState = {"user_msg": message}
    final_state = app.invoke(init)
    return final_state
