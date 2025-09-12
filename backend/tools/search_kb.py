# backend/tools/search_kb.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

# If you later add qdrant_path to Settings, you can import it there.
# For now we read env/path directly to stay decoupled.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "shipping_kb")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Lazy singletons (None until first online use)
_EMBEDDER: Any | None = None
_QDRANT: Any | None = None


def _is_offline() -> bool:
    """Decide offline at call-time so tests/CI can set ENV late."""
    return (
        os.environ.get("AGENT_OFFLINE") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )


def _get_embedder():
    """
    Lazily load a SentenceTransformer only when online.
    In offline/CI, return a lightweight dummy with the same encode() surface.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    if _is_offline():
        # Zero-download dummy
        import numpy as np

        class _DummyEmbedder:
            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                # fixed-size zero vectors; exact width doesn't matter for dummy path
                return np.zeros((len(texts), 384), dtype=np.float32)

        _EMBEDDER = _DummyEmbedder()
        return _EMBEDDER

    # Online path: import lazily to avoid import-time downloads in CI
    from sentence_transformers import SentenceTransformer

    _EMBEDDER = SentenceTransformer("BAAI/bge-m3")
    return _EMBEDDER


def _get_qdrant():
    """
    Prefer embedded (path) if folder exists or env set; else try host:port.
    Return None if neither is available or if construction fails.
    Lazily import QdrantClient to avoid heavy imports at module import-time.
    """
    global _QDRANT
    if _QDRANT is not None:
        return _QDRANT

    # Import here, not at module import-time
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception:
        return None

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
    If offline or no Qdrant backend is reachable, returns an empty list (agent can still respond).
    """
    # Short-circuit completely in offline/CI runs
    if _is_offline():
        return []

    cli = _get_qdrant()
    if cli is None:
        return []

    emb = _get_embedder().encode(query, normalize_embeddings=True).tolist()

    # Import filter models lazily to keep module import light
    try:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore
    except Exception:  # pragma: no cover (if qdrant models missing)
        return []

    qfilter = None
    if carrier:
        qfilter = Filter(
            must=[FieldCondition(key="carrier", match=MatchValue(value=carrier))]
        )

    try:
        # Use query_points (newer API)
        points = cli.query_points(
            collection_name=QDRANT_COLLECTION,
            query=emb,
            limit=k,
            with_payload=True,
            query_filter=qfilter,
        ).points
        if not points and qfilter is not None:
            # Retry unfiltered to avoid empty results when 'carrier' isn’t in payload
            points = cli.query_points(
                collection_name=QDRANT_COLLECTION,
                query=emb,
                limit=k,
                with_payload=True,
            ).points
    except Exception:
        # If anything goes wrong (e.g., collection missing), fail soft
        return []

    hits: List[KBHit] = []
    for p in points:
        payload = p.payload or {}
        src = payload.get("source") or payload.get("file") or "kb"
        chunk_id = str(payload.get("chunk_id") or getattr(p, "id", "") or "")
        hits.append(
            KBHit(
                text=payload.get("text", ""),
                score=float(getattr(p, "score", 0.0) or 0.0),
                source=src,
                chunk_id=chunk_id,
                meta=payload,
            )
        )

    return hits


def format_citations(hits: List[KBHit]) -> List[Dict[str, Any]]:
    """Dict citations for internal agent state: [{'ref': 'source#chunk', 'score': 0.87}, ...]."""
    out: List[Dict[str, Any]] = []
    for h in hits:
        label = h.source or "kb"
        if h.chunk_id:
            label = f"{label}#{h.chunk_id}"
        out.append({"ref": label, "score": float(h.score)})
    return out


def format_citation_strings(hits: List[KBHit]) -> List[str]:
    """String citations for API responses: ['source#chunk', ...]."""
    out: List[str] = []
    for h in hits:
        label = h.source or "kb"
        if h.chunk_id:
            label = f"{label}#{h.chunk_id}"
        out.append(label)
    return out
