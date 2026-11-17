"""Step 1 pipeline: discover -> fetch -> parse -> chunk -> save.

Usage:
    python scripts/ingest.py              # uses cached raw files where present
    python scripts/ingest.py --refresh    # re-download everything
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workplace_rag.chunking import chunk_document  # noqa: E402
from workplace_rag.config import PROJECT_ROOT, load_config  # noqa: E402
from workplace_rag.ingest.govuk_client import GovUKClient  # noqa: E402
from workplace_rag.ingest.parse import parse_content  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest")


def discover_paths(client: GovUKClient, cfg: dict) -> list[str]:
    paths = list(cfg["seed_paths"])
    allowed = set(cfg["allowed_formats"])
    for q in cfg["search_queries"]:
        for r in client.search(q, cfg["results_per_query"]):
            link = r.get("link", "")
            if link.startswith("/") and r.get("format") in allowed:
                paths.append(link)
    return list(dict.fromkeys(paths))  # de-duplicate, keep order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    args = ap.parse_args()

    cfg = load_config()
    g, ch, p = cfg["govuk"], cfg["chunking"], cfg["paths"]
    client = GovUKClient(g["base_url"], PROJECT_ROOT / p["raw_dir"],
                         delay=g["request_delay_seconds"], timeout=g["timeout_seconds"])
    out_dir = PROJECT_ROOT / p["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_paths(client, g)
    log.info("Discovered %d candidate pages", len(paths))

    docs, chunks, skipped = [], [], Counter()
    for path in paths:
        raw = client.get_content(path, use_cache=not args.refresh)
        if raw is None:
            skipped["not_found"] += 1
            continue
        if raw.get("schema_name") not in g["allowed_formats"]:
            skipped[f"format:{raw.get('schema_name')}"] += 1
            continue
        doc = parse_content(raw, g["base_url"])
        if not doc.sections:
            skipped["empty"] += 1
            continue
        docs.append(doc)
        chunks += chunk_document(doc, ch["max_words"], ch["overlap_words"])

    with open(out_dir / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    lengths = sorted(len(c.body.split()) for c in chunks) or [0]
    topics = Counter(t for d in docs for t in d.topics)
    log.info("Documents: %d | Chunks: %d | Skipped: %s", len(docs), len(chunks), dict(skipped))
    log.info("Chunk words: min %d, median %d, max %d",
             lengths[0], lengths[len(lengths) // 2], lengths[-1])
    log.info("Top topics: %s", topics.most_common(10))


if __name__ == "__main__":
    main()
