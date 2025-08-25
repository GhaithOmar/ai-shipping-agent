# backend/main.py
from typing import List, Optional
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi import Query
from backend.agent.graph import build_graph, run_agent

from fastapi.responses import StreamingResponse
import asyncio


# RAG search (kept as-is)
from backend.search import search as vector_search

# Guarded generation + loader
from backend.generation import load_model_and_tokenizer, infer_guarded

from dotenv import load_dotenv

load_dotenv()  # loads variables from .env at repo root



log = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

# ========= Config via env =========
# NOTE: If you don't have access to the Llama base (gated), either:
#  1) set ADAPTER_ID="" to run a plain open model, or
#  2) set BASE_MODEL to an open base you have (e.g., "Qwen/Qwen2.5-3B-Instruct")
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER_ID = os.getenv("ADAPTER_ID", "GhaithOmar/ai-shipping-agent-llama3.1-8b-lora-day4")
HF_TOKEN   = os.getenv("HUGGINGFACE_TOKEN", None)

# Optional non-gated fallback if loading the above fails and ADAPTER_ID is empty.
FALLBACK_BASE = os.getenv("FALLBACK_BASE", "Qwen/Qwen2.5-3B-Instruct")

# ========= Load model once =========
tok_backend = None
model_backend = None

def _boot():
    global tok_backend, model_backend
    try:
        tok_backend, model_backend = load_model_and_tokenizer(
            base_model_id=BASE_MODEL,
            adapter_id=ADAPTER_ID if ADAPTER_ID else None,
            hf_token=HF_TOKEN,
        )
        log.info(f"Loaded base={BASE_MODEL} adapter={ADAPTER_ID or 'None'}")
    except Exception as e:
        log.error(f"Primary model load failed: {e}")
        # Only try fallback if user didn’t request a specific adapter
        if not ADAPTER_ID:
            try:
                tok_backend, model_backend = load_model_and_tokenizer(
                    base_model_id=FALLBACK_BASE,
                    adapter_id=None,
                    hf_token=HF_TOKEN,
                )
                log.info(f"Loaded fallback base={FALLBACK_BASE}")
            except Exception as ee:
                log.exception(f"Fallback load failed: {ee}")
                raise

_boot()
# ========= Agent (LangGraph) bootstrap =========
agent_app = None

def _build_agent():
    global agent_app

    def offline_generate(user_msg, top_k_context, provided_tracking=None):
        # safe, model-free reply
        from backend.tools.parse_tracking import parse_tracking
        parsed = parse_tracking(user_msg)
        ids = parsed.get("ids") or ([provided_tracking] if provided_tracking else [])
        bullets = []
        if not ids:
            bullets.append("Please share a valid tracking ID (and carrier if known).")
        else:
            bullets.append(f"Parsed tracking ID: {ids[0]} (carrier: {parsed.get('carrier') or 'unknown'}).")
        if top_k_context:
            bullets.append(f"Using {len(top_k_context)} retrieved context chunk(s).")
        bullets.append("No live tracking is used; info is handbook-based.")
        return "\n".join(f"- {b}" for b in bullets)

    def generate_fn(user_msg, top_k_context, provided_tracking=None):
        # use offline fallback if model not loaded
        if tok_backend is None or model_backend is None:
            return offline_generate(user_msg, top_k_context, provided_tracking)
        return infer_guarded(
            user_msg=user_msg,
            top_k_context=top_k_context,
            tok=tok_backend,
            model=model_backend,
            provided_tracking=provided_tracking,
        )

    agent_app = build_graph(generate_fn)


# Build if enabled (via env var from settings or default True)
try:
    _build_agent()
    log.info("Agent graph compiled")
except Exception as e:
    log.error(f"Agent graph build failed (will fall back to legacy path): {e}")
    agent_app = None


async def _stream_answer_chunks(text: str, delay_s: float = 0.02):
    # naive chunker: one line at a time; adjust as you like
    for line in text.splitlines(True):
        yield line
        await asyncio.sleep(delay_s)

# ========= FastAPI app =========
app = FastAPI(title="AI Shipping Agent", version="0.4")

# ========= Schemas =========
class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    tracking: Optional[str] = None
    carrier: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchHit(BaseModel):
    text: str
    source: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchHit]

# ========= Endpoints =========
@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_ID or "",
    }

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    agent: Optional[int] = Query(default=0, description="Set 1 to use LangGraph agent"),
):
    if agent and agent_app is None:
        # Agent requested but not available → soft fallback to legacy RAG
        log.warning("Agent requested but not available; using legacy path.")

    # ===== Agent path =====
    if agent and agent_app is not None:
        # Run the graph
        final_state = run_agent(agent_app, req.message)

        # citations in agent path (if any) come as list of dicts -> stringify
        citations = []
        for c in final_state.get("citations", []):
            ref = c.get("ref") if isinstance(c, dict) else str(c)
            if ref:
                citations.append(ref)

        answer = final_state.get("answer", "")
        return ChatResponse(answer=answer, citations=citations)

    # ===== Legacy RAG path (existing behavior) =====
    if tok_backend is None or model_backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    hits = vector_search(req.message, req.top_k)

    contexts: List[str] = []
    citations: List[str] = []
    for h in hits:
        if isinstance(h, str):
            contexts.append(h); citations.append("")
        elif isinstance(h, dict):
            contexts.append(h.get("text") or h.get("chunk") or "")
            citations.append(h.get("source") or h.get("id") or "")
        else:
            txt = getattr(h, "text", "") or getattr(h, "chunk", "") or str(h)
            src = getattr(h, "source", "") or getattr(h, "id", "")
            contexts.append(txt); citations.append(src)

    answer = infer_guarded(
        user_msg=req.message,
        top_k_context=[c for c in contexts if c][:req.top_k],
        tok=tok_backend,
        model=model_backend,
        provided_tracking=req.tracking,
    )
    return ChatResponse(answer=answer, citations=citations[:req.top_k])


@app.post("/search", response_model=SearchResponse)
def search_api(req: SearchRequest):
    hits = vector_search(req.query, req.k)
    return {"results": hits}

@app.post("/chat/stream")
def chat_stream(req: ChatRequest, agent: Optional[int] = Query(1)):
    if not agent or agent_app is None:
        raise HTTPException(status_code=400, detail="Streaming is only supported for agent=1 right now.")

    # Run the graph to get final state
    final_state = run_agent(agent_app, req.message)
    answer = final_state.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)

    return StreamingResponse(_stream_answer_chunks(answer), media_type="text/plain")
