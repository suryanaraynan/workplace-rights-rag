"""Step 2a: embed every chunk and store it in a local ChromaDB index.

Usage:  python scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workplace_rag.config import PROJECT_ROOT, load_config  # noqa: E402
from workplace_rag.embeddings import Embedder  # noqa: E402
from workplace_rag.retrieval import build_index  # noqa: E402


def main() -> None:
    cfg = load_config()
    chunks_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.jsonl"
    if not chunks_path.exists():
        sys.exit("No chunks found. Run scripts/ingest.py first.")
    chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]

    e, ix = cfg["embeddings"], cfg["index"]
    embedder = Embedder(e["model_name"], e.get("query_prefix", ""))
    t0 = time.perf_counter()
    col = build_index(chunks, embedder, PROJECT_ROOT / ix["persist_dir"], ix["collection"])
    print(f"Indexed {col.count()} chunks with {e['model_name']} "
          f"in {time.perf_counter() - t0:.1f}s -> {ix['persist_dir']}")


if __name__ == "__main__":
    main()
