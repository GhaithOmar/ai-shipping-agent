# backend/main.py
from typing import List, Optional
import os
import logging
import json, time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi import Query
from backend.agent.graph import build_graph, run_agent

from fastapi.responses import StreamingResponse
import asyncio

# RAG search (kept as-is)
from backend.search import search as vector_search

# Guarded generation + loader
from backend.generation import load_model_and_tokenizer, infer_guarded, stream_guarded 
from backend.settings import settings
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

def _ascii_safe(s: str) -> str:
    if not s:
        return s
    table = {"–":"-","—":"-","−":"-","→":"->","←":"<-","’":"'","‘":"'","“":'"',"”":'"',"…":"...","\u00A0":" "}
    for k,v in table.items():
        s = s.replace(k,v)
    return s

def _sse(data: dict, event: str = "message") -> bytes:
    return (f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n").encode("utf-8")


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

def _offline_legacy_reply(user_msg: str, top_k: int = 2):
    # lightweight KB pull
    hits = vector_search(user_msg, k=top_k) if top_k else []
    try:
        from backend.tools.parse_tracking import parse_tracking
    except Exception:
        # if tools module path differs in your repo, adjust import
        from backend.parse_tracking import parse_tracking  # fallback

    parsed = parse_tracking(user_msg)
    ids = parsed.get("ids") or []
    bullets = []
    if not ids:
        bullets.append("Please share your tracking/waybill ID (and carrier if known).")
    if hits:
        bullets.append(f"Using {len(hits)} handbook chunk(s) from the knowledge base.")
    bullets.append("No live tracking is used; info is handbook-based.")

    answer = "\n".join(f"- {b}" for b in bullets)
    citations = []
    for h in hits:
        src = h.get("source") or ""
        chk = str(h.get("chunk_id") or h.get("id") or "")
        label = f"{src}#{chk}" if src and chk else (src or (chk and f"kb#{chk}") or "kb")
        citations.append(label)

    return answer, citations


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
        "agent_enabled": bool(settings.agent_enable),  
        "agent_loaded": bool(agent_app is not None),   
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    agent: Optional[int] = Query(
        default=1 if settings.agent_enable else 0,  # default from settings
        description="Set 1 to use LangGraph agent"
    ),
):
    if agent and agent_app is None:
        # Agent requested but not available → soft fallback to legacy RAG
        log.warning("Agent requested but not available; using legacy path.")

    # ===== Agent path =====
    if agent and agent_app is not None:
        final_state = run_agent(agent_app, req.message)

        # citations in agent path (if any) come as list of dicts -> stringify
        citations = []
        for c in final_state.get("citations", []):
            ref = c.get("ref") if isinstance(c, dict) else str(c)
            if ref:
                citations.append(ref)

        # fallback from kb_hits if citations empty
        if not citations:
            for h in final_state.get("kb_hits", []):
                src = h.get("source") or ""
                chk = str(h.get("chunk_id") or h.get("id") or "")
                label = f"{src}#{chk}" if src and chk else (src or (chk and f"kb#{chk}") or "kb")
                citations.append(label)

        answer = final_state.get("answer", "")
        return ChatResponse(answer=answer, citations=citations)


    # ===== Legacy RAG path (existing behavior) =====
    if agent == 0:
        if model_backend is None or tok_backend is None:
            # NEW: offline fallback instead of raising "Model not loaded"
            ans, cits = _offline_legacy_reply(req.message, getattr(req, "top_k", 2))
            return ChatResponse(answer=ans, citations=cits)
    

    hits = vector_search(req.message, req.top_k)

    contexts: List[str] = []
    citations = []
    for h in hits:
        src = h.get("source") or ""
        chk = h.get("chunk_id") or h.get("id") or ""
        label = f"{src}#{chk}" if src and chk else (src or (chk and f"kb#{chk}") or "kb")
        citations.append(label)



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
def chat_stream(req: ChatRequest, agent: Optional[int] = Query(default=1 if settings.agent_enable else 0)):
    """
    Server-Sent Events (SSE) stream:
      - start   {meta}
      - token   {token}
      - end     {citations}
    """
    def legacy_stream():
        # 1) Build retrieval context + citations
        hits = vector_search(req.message, k=getattr(req, "top_k", 2) or 2)
        contexts, citations = [], []
        for h in hits:
            txt = h.get("text") or ""
            src = h.get("source") or ""
            chk = str(h.get("chunk_id") or h.get("id") or "")
            contexts.append(txt)
            label = f"{src}#{chk}" if src and chk else (src or (chk and f"kb#{chk}") or "kb")
            citations.append(label)

        # 2) Parse tracking ID
        try:
            from backend.tools.parse_tracking import parse_tracking
        except Exception:
            from backend.parse_tracking import parse_tracking
        parsed = parse_tracking(req.message)
        ids = parsed.get("ids") or []
        tracking_id = ids[0] if ids else None

        # 3) Token streaming with offline fallback
        def gen():
            yield _sse({"status": "start", "agent": False, "citations": []}, "start")

            try:
                # if model is missing, force offline path
                if model_backend is None or tok_backend is None:
                    raise RuntimeError("Model not loaded")

                # real token-by-token streaming
                for token in stream_guarded(req.message, contexts[:4], tracking_id):
                    yield _sse({"token": _ascii_safe(token)}, "token")

            except Exception:
                # OFFLINE FALLBACK: build a safe text and stream it in small chunks
                ans, cits_off = _offline_legacy_reply(req.message, getattr(req, "top_k", 2))
                text = _ascii_safe(ans or "")
                CHUNK = 48
                for i in range(0, len(text), CHUNK):
                    yield _sse({"token": text[i:i+CHUNK]}, "token")
                # prefer offline citations if present, else the earlier retrieval labels
                nonlocal citations
                if cits_off:
                    citations = cits_off

            yield _sse({"citations": citations}, "end")

        return StreamingResponse(gen(), media_type="text/event-stream")


    def agent_stream():
        # Keep agent path working; stream in chunks (answer already composed by the graph)
        if agent_app is None:
            # fallback to legacy if agent not built
            return legacy_stream()

        final_state = run_agent(agent_app, req.message)

        # citations from state (dicts with 'ref') → strings
        citations = []
        for c in final_state.get("citations", []):
            ref = c.get("ref") if isinstance(c, dict) else str(c)
            if ref:
                citations.append(ref)
        if not citations:
            for h in final_state.get("kb_hits", []):
                src = h.get("source") or ""
                chk = str(h.get("chunk_id") or h.get("id") or "")
                label = f"{src}#{chk}" if src and chk else (src or (chk and f"kb#{chk}") or "kb")
                citations.append(label)

        answer = _ascii_safe(final_state.get("answer", "").strip())

        def gen():
            yield _sse({"status": "start", "agent": True, "citations": []}, "start")
            # stream the composed text in small chunks
            CHUNK = 32
            for i in range(0, len(answer), CHUNK):
                yield _sse({"token": answer[i:i+CHUNK]}, "token")
                time.sleep(0.01)
            yield _sse({"citations": citations}, "end")

        return StreamingResponse(gen(), media_type="text/event-stream")

    # route to agent or legacy
    return agent_stream() if agent else legacy_stream()
