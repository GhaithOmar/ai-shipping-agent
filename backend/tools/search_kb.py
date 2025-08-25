from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# If you later add qdrant_path to Settings, you can import it there.
# For now we read env/path directly to stay decoupled.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "shipping_kb")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

_EMBEDDER = None
_QDRANT: Optional[QdrantClient] = None

def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("BAAI/bge-m3")
    return _EMBEDDER

def _get_qdrant() -> Optional[QdrantClient]:
    """
    Prefer embedded (path) if folder exists or env set; else try host:port.
    Return None if neither is available.
    """
    global _QDRANT
    if _QDRANT is not None:
        return _QDRANT

    try:
        if QDRANT_PATH and os.path.isdir(QDRANT_PATH):
            _QDRANT = QdrantClient(path=QDRANT_PATH)
            return _QDRANT
    except Exception:
        _QDRANT = None  # continue to host/port attempt

    try:
        _QDRANT = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False)
        return _QDRANT
    except Exception:
        _QDRANT = None
        return None

@dataclass
class KBHit:
    text: str
    score: float
    source: str | None = None
    chunk_id: str | None = None
    meta: Dict[str, Any] | None = None

def search_kb(query: str, k: int = 4, carrier: str | None = None) -> List[KBHit]:
    """
    Semantic search over the shipping_kb collection.
    Returns top-k hits with text, score, source (filename/url), and chunk_id if present.
    If no Qdrant backend is reachable, returns an empty list (agent can still respond).
    """
    cli = _get_qdrant()
    if cli is None:
        return []

    emb = _get_embedder().encode(query, normalize_embeddings=True).tolist()

    qfilter = None
    if carrier:
        qfilter = Filter(must=[FieldCondition(key="carrier", match=MatchValue(value=carrier))])

    try:
        # Use query_points (newer API)
        points = cli.query_points(
            collection_name=QDRANT_COLLECTION,
            query=emb,
            limit=k,
            with_payload=True,
            query_filter=qfilter,
        ).points
    except Exception:
        # If anything goes wrong (e.g., collection missing), fail soft
        return []

    hits: List[KBHit] = []
    for p in points:
        payload = p.payload or {}
        hits.append(
            KBHit(
                text=payload.get("text", ""),
                score=float(p.score),
                source=payload.get("source") or payload.get("file"),
                chunk_id=str(payload.get("chunk_id") or payload.get("id") or ""),
                meta=payload,
            )
        )
    return hits

def format_citations(hits: List[KBHit]) -> List[Dict[str, str]]:
    """Convert hits to a compact citation list for responses."""
    out = []
    for h in hits:
        label = h.source or "kb"
        if h.chunk_id:
            label = f"{label}#{h.chunk_id}"
        out.append({"ref": label, "score": f"{h.score:.3f}"})
    return out
