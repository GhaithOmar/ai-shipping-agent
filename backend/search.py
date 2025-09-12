"""
Search helper: open the same embedded Qdrant DB, encode the query, and return top-k hits.
We keep the embedder and client as module-level singletons so they don't reload on every request.
"""

import os
from typing import Any, Dict, List

COLLECTION = "shipping_kb"
QDRANT_PATH = "qdrant_db"
EMBED_MODEL_NAME = "BAAI/bge-m3"

# Detect offline mode (CI/tests)
_OFFLINE = os.environ.get("AGENT_OFFLINE") == "1" or \
           os.environ.get("HF_HUB_OFFLINE") == "1" or \
           os.environ.get("TRANSFORMERS_OFFLINE") == "1"

# Singletons (can be real libs or dummies in offline mode)
_embedder: Any
_client: Any

if not _OFFLINE:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Load once at import time (API startup). This avoids per-request model loads.
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    _client = QdrantClient(path=QDRANT_PATH)
else:
    # Lightweight dummies for offline CI/tests — no downloads, return empty hits
    import numpy as np

    class _DummyEmbedder:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # fixed-size zero vectors — shape doesn't matter here since we short-circuit search
            return np.zeros((len(texts), 384), dtype=np.float32)

    class _DummyClient:
        def search(self, *args, **kwargs):
            return []  # no hits offline

    _embedder = _DummyEmbedder()
    _client = _DummyClient()



def search(query: str, k: int = 5) -> List[Dict]:
    """
    Encode the query (normalized), search Qdrant, and return payload + score.
    Note: score in Qdrant for cosine is "similarity", higher is better.
    """
    if _OFFLINE:
        return []  # predictable offline behavior for CI/smoke tests
        
    vec = _embedder.encode(query, normalize_embeddings=True).tolist()
    hits = _client.search(collection_name=COLLECTION, query_vector=vec, limit=k)
    results = []
    for h in hits:
        payload = h.payload or {}
        # inside the loop where you build each hit dict
        results.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", "") or payload.get("file", ""),
                "chunk_id": str(getattr(h, "id", "") or payload.get("chunk_id") or ""),
                "score": float(h.score),
            }
        )

    return results
