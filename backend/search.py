# backend/search.py
"""
Search helper: open the embedded Qdrant DB, encode the query, and return top-k hits.

CI/offline safe:
- No heavy imports or model instantiation at module import time.
- We check "offline" at call time.
- If offline, we return [] with zero downloads.
- If online, we lazily import and cache the real deps on first use.
"""

import os
from typing import Any, Dict, List

COLLECTION = "shipping_kb"
QDRANT_PATH = "qdrant_db"
EMBED_MODEL_NAME = "BAAI/bge-m3"

# Singletons for lazy load (None until first online call)
_embedder: Any | None = None
_client: Any | None = None


def _is_offline() -> bool:
    """Decide offline at call-time so tests/CI can set ENV late."""
    return (
        os.environ.get("AGENT_OFFLINE") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )


def _ensure_online_loaded() -> None:
    """Lazy-load the real libraries and singletons exactly once."""
    global _embedder, _client
    if _embedder is not None and _client is not None:
        return

    # Import here (not at module import) to avoid triggering downloads in CI.
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    _client = QdrantClient(path=QDRANT_PATH)


def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    # Short-circuit offline runs (CI/tests) with predictable behavior.
    if _is_offline():
        return []

    # Online path: load deps once, then search.
    _ensure_online_loaded()
    assert _embedder is not None and _client is not None

    vec = _embedder.encode(query).tolist()
    hits = _client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=k,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", "") or payload.get("file", ""),
                "chunk_id": str(getattr(h, "id", "") or payload.get("chunk_id") or ""),
                "score": float(getattr(h, "score", 0.0)),
            }
        )
    return results
