"""
Ingestion pipeline: read markdown files -> chunk -> embed (BGE-M3) -> store in Qdrant (embedded, persisted).

Run:
    python rag/ingest.py
"""
import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ----------------------------
# config
# ----------------------------
COLLECTION = "shipping_kb"
QDRANT_PATH = "qdrant_db"      # persisted embedded db folder
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1000              # ~characters
CHUNK_OVERLAP = 200            # characters

@dataclass
class DocFile:
    text: str
    source: str                 # e.g., "company_a_faq.md"
    carrier: Optional[str]      # e.g., "Shipping_A" if we can infer it


def infer_carrier_from_source(basename: str) -> Optional[str]:
    """Heuristic mapping from filename to carrier alias."""
    name = (basename or "").lower()
    if any(k in name for k in ["shipping_a", "company_a"]):
        return "Shipping_A"
    if any(k in name for k in ["shipping_b", "company_b"]):
        return "Shipping_B"
    if any(k in name for k in ["shipping_c", "company_c"]):
        return "Shipping_C"
    return None


def read_markdown_files(pattern: str = "rag/data/*.md") -> Iterable[DocFile]:
    """Yield file contents + inferred metadata for each .md under rag/data/."""
    for fp in glob.glob(pattern):
        basename = os.path.basename(fp)
        if basename.lower() == "readme.md":   # skip README
            continue
        with open(fp, "r", encoding="utf-8") as f:
            text = (f.read() or "").strip()
        if not text:
            continue
        yield DocFile(text=text, source=basename, carrier=infer_carrier_from_source(basename))


def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Larger chunks = more context; overlap preserves continuity across boundaries.
    """
    chunks: List[str] = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap if j < n else j
        if i < 0:
            i = 0
    return chunks


def ensure_collection(client: QdrantClient, dim: int):
    """Create the collection cleanly (drop if exists to avoid schema drift)."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def main():
    # 1) Embedder
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    dim = embedder.get_sentence_embedding_dimension()

    # 2) Embedded Qdrant
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    # 3) Fresh collection
    ensure_collection(client, dim)

    # 4) Read -> chunk -> embed -> upsert
    points: List[PointStruct] = []
    pid = 0
    total_chunks = 0

    for doc in read_markdown_files():
        per_file_index = 0
        for ch in chunk_text(doc.text):
            vec = embedder.encode(ch, normalize_embeddings=True).tolist()

            payload: Dict[str, object] = {
                "text": ch,
                "source": doc.source,          # REQUIRED for citations
                "chunk_id": str(per_file_index)  # REQUIRED for citations (stable per file)
            }
            if doc.carrier:                     # add if known (enables carrier filter)
                payload["carrier"] = doc.carrier

            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload=payload,
                )
            )
            pid += 1
            per_file_index += 1
            total_chunks += 1

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Indexed {total_chunks} chunks into collection '{COLLECTION}'")
    else:
        print("No data found under rag/data/*.md")

    client.close()


if __name__ == "__main__":
    main()
