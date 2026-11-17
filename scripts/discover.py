"""Suggest GOV.UK pages that are NOT yet in the curated corpus.

Prints candidates with their GOV.UK tags for a human to review. Nothing is
added automatically: v1 auto-included search results and ~40% of chunks
ended up off-topic (visas, driving tests). Human-in-the-loop curation fixes that.

Usage:  python scripts/discover.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workplace_rag.config import PROJECT_ROOT, load_config  # noqa: E402
from workplace_rag.ingest.govuk_client import GovUKClient  # noqa: E402


def main() -> None:
    cfg = load_config()
    g, d = cfg["govuk"], cfg["discovery"]
    client = GovUKClient(g["base_url"], PROJECT_ROOT / cfg["paths"]["raw_dir"],
                         delay=g["request_delay_seconds"], timeout=g["timeout_seconds"])
    known = {p for paths in cfg["corpus"].values() for p in paths}
    found: dict[str, tuple[str, str, set]] = {}
    for q in d["search_queries"]:
        for r in client.search(q, d["results_per_query"]):
            link = r.get("link", "")
            if link.startswith("/") and r.get("format") in g["allowed_formats"] and link not in known:
                title, fmt, queries = found.get(link, (r.get("title", ""), r.get("format"), set()))
                queries.add(q)
                found[link] = (title, fmt, queries)

    print(f"{len(found)} candidate pages not in corpus (most-matched first):\n")
    for link, (title, fmt, queries) in sorted(found.items(), key=lambda kv: -len(kv[1][2])):
        print(f"{len(queries)} hits  {link}  [{fmt}]  {title}")


if __name__ == "__main__":
    main()
