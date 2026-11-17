"""Wire the components together from config.yaml (one place, used by CLI and UI)."""
from __future__ import annotations

from dotenv import load_dotenv

from .config import PROJECT_ROOT, load_config
from .embeddings import Embedder
from .llm import LLMClient
from .rag import RAGPipeline
from .retrieval import Retriever


def load_pipeline(cfg: dict | None = None) -> RAGPipeline:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = cfg or load_config()
    e, ix, r = cfg["embeddings"], cfg["index"], cfg["retrieval"]
    embedder = Embedder(e["model_name"], e.get("query_prefix", ""))
    retriever = Retriever(embedder, PROJECT_ROOT / ix["persist_dir"], ix["collection"])
    return RAGPipeline(retriever, LLMClient.from_config(cfg["llm"]),
                       top_k=r["top_k"], min_score=r["min_score"],
                       log_path=PROJECT_ROOT / cfg["paths"]["query_log"])
