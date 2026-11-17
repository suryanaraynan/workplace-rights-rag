"""Thin client for the public GOV.UK Content and Search APIs.

Why a client class instead of loose requests calls?
- One place for politeness (rate limiting) and timeouts.
- Raw responses are cached on disk, so re-running the pipeline is fast,
  reproducible and doesn't hammer GOV.UK.
- The HTTP session can be swapped for a fake in tests.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class GovUKClient:
    def __init__(self, base_url: str, cache_dir: str | Path,
                 delay: float = 0.5, timeout: float = 20, session=None):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "workplace-rights-rag (portfolio project)"})

    # -- helpers ---------------------------------------------------------
    def _cache_file(self, path: str) -> Path:
        return self.cache_dir / (path.strip("/").replace("/", "__") + ".json")

    def _get_json(self, url: str, params: dict | None = None) -> dict | None:
        time.sleep(self.delay)
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 404:
            log.warning("404 Not Found: %s", url)
            return None
        resp.raise_for_status()
        return resp.json()

    # -- public API --------------------------------------------------------
    def get_content(self, path: str, use_cache: bool = True) -> dict | None:
        """Fetch one page from the Content API, e.g. path='/statutory-sick-pay'."""
        cache = self._cache_file(path)
        if use_cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        data = self._get_json(f"{self.base_url}/api/content{path}")
        if data is not None:
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def search(self, query: str, count: int = 10) -> list[dict]:
        """Discover pages via the Search API. Returns [{link, title, format}, ...]."""
        data = self._get_json(
            f"{self.base_url}/api/search.json",
            params={"q": query, "count": count, "fields": "title,link,format"},
        )
        return (data or {}).get("results", [])
