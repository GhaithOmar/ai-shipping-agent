# backend/search.py
"""
Search helper: open the embedded Qdrant DB, encode the query, and return top-k hits.
Offline-safe: when AGENT_OFFLINE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set,
we fall back to light dummies that never download anything.
"""

from typing import Any, Dict, List
import os

COLLECTION = "shipping_kb"
QDRANT_PATH = "qdrant_db"
EMBED_MODEL_NAME = "BAAI/bge-m3"

# Detect offline mode early (used both at import and call time)
_OFFLINE = (
    os.environ.get("AGENT_OFFLINE") == "1"
    or os.environ.get("HF_HUB_OFFLINE") == "1"
    or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
)

# Singletons (can be real libs or dummies)
_embedder: Any
_client: Any

if not _OFFLINE:
    # Real deps only when online
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    _client = QdrantClient(path=QDRANT_PATH)
else:
    # Lightweight dummies for CI/tests — no downloads
    import numpy as np

    class _DummyEmbedder:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # fixed-size zero vectors; dimension doesn't matter for dummy search
            return np.zeros((len(texts), 384), dtype=np.float32)

    class _DummyClient:
        def search(self, *args, **kwargs):
            return []

    _embedder = _DummyEmbedder()
    _client = _DummyClient()


def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    # Belt-and-suspenders: short-circuit at call time too
    if _OFFLINE:
        return []

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
