# backend/main.py
from typing import List, Optional
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
def chat(req: ChatRequest):
    if tok_backend is None or model_backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1) Retrieve top-k chunks (RAG)
    hits = vector_search(req.message, req.top_k)

    # Normalize to strings & collect citations
    contexts: List[str] = []
    citations: List[str] = []
    for h in hits:
        if isinstance(h, str):
            contexts.append(h)
            citations.append("")
        elif isinstance(h, dict):
            contexts.append(h.get("text") or h.get("chunk") or "")
            citations.append(h.get("source") or h.get("id") or "")
        else:
            txt = getattr(h, "text", "") or getattr(h, "chunk", "") or str(h)
            src = getattr(h, "source", "") or getattr(h, "id", "")
            contexts.append(txt)
            citations.append(src)

    # 2) Guarded generation (asks for missing IDs, blocks links)
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
