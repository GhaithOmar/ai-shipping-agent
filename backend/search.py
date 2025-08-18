"""
Search helper: open the same embedded Qdrant DB, encode the query, and return top-k hits.
We keep the embedder and client as module-level singletons so they don't reload on every request.
"""

from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "shipping_kb"
QDRANT_PATH = "qdrant_db"
EMBED_MODEL_NAME = "BAAI/bge-m3"

# Load once at import time (API startup). This avoids per-request model loads.
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
_client = QdrantClient(path=QDRANT_PATH)


def search(query: str, k: int = 5) -> List[Dict]:
    """
    Encode the query (normalized), search Qdrant, and return payload + score.
    Note: score in Qdrant for cosine is "similarity", higher is better.
    """
    vec = _embedder.encode(query, normalize_embeddings=True).tolist()
    hits = _client.search(collection_name=COLLECTION, query_vector=vec, limit=k)
    results = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "score": float(h.score),
            }
        )
    return results
