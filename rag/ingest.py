"""
Ingestion pipeline: read markdown files -> chunk -> embed (BGE-M3) -> store in Qdrant (embedded, persisted).

Run:
    uv run python rag/ingest.py
or:
    python rag/ingest.py
"""
import glob
import os 
from dataclasses import dataclass 
from typing import Iterable, List

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ----------------------------
# config
# ----------------------------
COLLECTION = "shipping_kb"
QDRANT_PATH = "qdrant_db"  # persisted embedded db folder
EMBED_MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1000               # ~characters; good start
CHUNK_OVERLAP = 200             # overlap helps continuity across chunks

@dataclass
class DocChunk:
    text: str 
    source: str 

def read_markdown_files(pattern: str = "rag/data/*.md") -> Iterable[DocChunk]:
    """Load each .md file and yield chunk candidates (one big string per file, for now)."""
    for fp in glob.glob(pattern):
        basename = os.path.basename(fp)
        if basename.lower() == "readme.md":   # <- skip README
            continue
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                yield DocChunk(text=text, source=basename)

def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Why char-based (not tokens)? It's lighter and works well enough to start.
    Options:
      - LangChain's RecursiveCharacterTextSplitter (adds a dep)
      - Token-based splitters (more precise but heavier)
    Tradeoff:
      - Larger chunks = more context but risk of irrelevant content
      - Some overlap preserves continuity across boundaries
    """
    chunks = [] 
    i = 0 
    n = len(text)
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
    """Create the collection if it doesn't exist, or recreate it cleanly."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)  # clear existing
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def main():
    # 1) load embedder
    # BGE models prefer cosine similarity with  normalized victors.
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    dim = embedder.get_sentence_embedding_dimension()

    # 2) connect to embedded Qdrant with presistent path
    client = QdrantClient(path=QDRANT_PATH)

    # 3) (re)create collection
    ensure_collection(client, dim)

    # 4) read files -> chunk -> embed -> upsert
    points: List[PointStruct] = []
    pid = 0
    for doc in read_markdown_files():
        for ch in chunk_text(doc.text):
            # Normalize embeddings for cosin distance
            vec = embedder.encode(ch, normalize_embeddings=True).tolist()
            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={"text": ch, "source": doc.source},
                )
            )
            pid +=1

    # Upsert in batches
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Indexed {len(points)} chunks into collection '{COLLECTION}'")
    else:
        print("No Data found under rag/data/*.md")

    client.close()

if __name__ == "__main__":
    main()
