"""Step 1 pipeline: curated corpus -> fetch -> parse -> chunk -> save.

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


def fetch_following_redirects(client: GovUKClient, path: str, use_cache: bool,
                              max_hops: int = 3) -> dict | None:
    """GOV.UK pages move. A 'redirect' item names its new destination; follow it."""
    for _ in range(max_hops):
        raw = client.get_content(path, use_cache=use_cache)
        if raw is None or raw.get("schema_name") != "redirect":
            return raw
        dest = (raw.get("redirects") or [{}])[0].get("destination", "")
        if not dest.startswith("/"):
            return None
        log.info("Redirect: %s -> %s (update config.yaml)", path, dest)
        path = dest.split("#")[0]
    return None


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

    docs, chunks, skipped, seen = [], [], Counter(), set()
    for topic, paths in cfg["corpus"].items():
        for path in paths:
            raw = fetch_following_redirects(client, path, use_cache=not args.refresh)
            if raw is None:
                skipped["not_found"] += 1
                continue
            if raw.get("schema_name") not in g["allowed_formats"]:
                skipped[f"format:{raw.get('schema_name')}"] += 1
                continue
            doc = parse_content(raw, g["base_url"], topic=topic)
            if doc.doc_id in seen or not doc.sections:
                skipped["duplicate_or_empty"] += 1
                continue
            seen.add(doc.doc_id)
            docs.append(doc)
            chunks += chunk_document(doc, ch["max_words"], ch["overlap_words"], ch["min_words"])

    with open(out_dir / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    lengths = sorted(len(c.body.split()) for c in chunks) or [0]
    small = sum(n < ch["min_words"] for n in lengths)
    log.info("Documents: %d | Chunks: %d | Skipped: %s", len(docs), len(chunks), dict(skipped))
    log.info("Chunk words: min %d, median %d, max %d | under %d words: %d (%.0f%%)",
             lengths[0], lengths[len(lengths) // 2], lengths[-1],
             ch["min_words"], small, 100 * small / max(len(lengths), 1))
    log.info("Chunks per topic: %s", Counter(c.topic for c in chunks).most_common())


if __name__ == "__main__":
    main()
