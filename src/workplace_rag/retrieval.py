"""Vector index (ChromaDB) and retriever.

Why ChromaDB?
- Runs locally with zero setup and persists to disk.
- Stores metadata next to vectors, so we can filter by topic (used by the
  agent in step 5) and show title/URL/heading for citations.
- At 463 chunks, plain numpy would also work. Chroma is chosen for the
  metadata filtering and because it scales to far more documents unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class RetrievedChunk:
    rank: int
    score: float      # cosine similarity, 1.0 = identical meaning
    chunk_id: str
    title: str
    heading: str
    url: str
    topic: str
    body: str


def _client(persist_dir: str | Path | None):
    return chromadb.PersistentClient(path=str(persist_dir)) if persist_dir else chromadb.EphemeralClient()


def build_index(chunks: list[dict], embedder, persist_dir: str | Path | None,
                collection_name: str, batch_size: int = 256):
    """(Re)create the collection from scratch: rebuilds are deterministic."""
    client = _client(persist_dir)
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    col = client.create_collection(collection_name,
                                   configuration={"hnsw": {"space": "cosine"}},
                                   metadata={"embedding_model": embedder.model_name})
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        col.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors[i:i + batch_size],
            documents=[c["body"] for c in batch],
            metadatas=[{"title": c["title"], "heading": c["heading"], "url": c["url"],
                        "topic": c["topic"], "doc_id": c["doc_id"]} for c in batch],
        )
    return col


class Retriever:
    def __init__(self, embedder, persist_dir: str | Path | None, collection_name: str):
        self.embedder = embedder
        self.collection = _client(persist_dir).get_collection(collection_name)
        built_with = (self.collection.metadata or {}).get("embedding_model")
        if built_with and built_with != embedder.model_name:
            raise ValueError(f"Index was built with {built_with!r} but config uses "
                             f"{embedder.model_name!r}. Re-run scripts/build_index.py.")

    def search(self, query: str, k: int = 5, topic: str | None = None) -> list[RetrievedChunk]:
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where={"topic": topic} if topic else None,
        )
        out = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
            out.append(RetrievedChunk(rank=rank, score=round(1 - dist, 4), chunk_id=cid,
                                      title=meta["title"], heading=meta["heading"], url=meta["url"],
                                      topic=meta["topic"], body=doc))
        return out
