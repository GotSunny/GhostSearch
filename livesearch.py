"""GHOST live layer: DuckDuckGo meta-search blended in when the local index
is thin. This is how GHOST answers queries about anything on the internet
without crawling it all — the same trick DuckDuckGo itself started with.

Every function is failure-safe: any DDG outage or rate limit degrades to an
empty list, and GHOST falls back to local-only results.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from ddgs import DDGS

log = logging.getLogger("ghost.live")


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except ValueError:
        return url


def _safesearch(safe: bool) -> str:
    return "on" if safe else "off"


def live_text(query: str, max_results: int = 12, safe: bool = True) -> list[dict]:
    """Web results shaped like search.search_web rows, plus source='live'."""
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results, safesearch=_safesearch(safe)))
    except Exception as e:  # rate limit / network / schema change
        log.warning("live_text failed: %s", e)
        return []
    out = []
    for r in rows:
        url = r.get("href") or r.get("url") or ""
        if not url.startswith(("http://", "https://")):
            continue
        out.append(
            {
                "title": (r.get("title") or url).strip(),
                "url": url,
                "display_url": _host(url),
                "favicon": None,
                "snippet": (r.get("body") or "").strip(),
                "score": 0.0,
                "source": "live",
            }
        )
    return out


def live_images(query: str, max_results: int = 40, safe: bool = True) -> list[dict]:
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.images(query, max_results=max_results, safesearch=_safesearch(safe)))
    except Exception as e:
        log.warning("live_images failed: %s", e)
        return []
    out = []
    for r in rows:
        src = r.get("image") or ""
        page = r.get("url") or ""
        if not src.startswith(("http://", "https://")):
            continue
        out.append(
            {
                "src": src,
                "thumbnail": r.get("thumbnail") or None,
                "alt": (r.get("title") or "").strip()[:300],
                "width": r.get("width"),
                "height": r.get("height"),
                "page_url": page,
                "page_title": _host(page) if page else "",
                "source": "live",
            }
        )
    return out


def live_videos(query: str, max_results: int = 16, safe: bool = True) -> list[dict]:
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.videos(query, max_results=max_results, safesearch=_safesearch(safe)))
    except Exception as e:
        log.warning("live_videos failed: %s", e)
        return []
    out = []
    for r in rows:
        page = r.get("content") or r.get("embed_url") or ""
        if not page.startswith(("http://", "https://")):
            continue
        # 'images' is a dict of thumbnail sizes in ddgs; be defensive.
        thumbs = r.get("images") or {}
        thumb = None
        if isinstance(thumbs, dict):
            for key in ("large", "medium", "small", "all"):
                cand = thumbs.get(key)
                if isinstance(cand, str) and cand.startswith(("http://", "https://")):
                    thumb = cand
                    break
        elif isinstance(thumbs, str) and thumbs.startswith(("http://", "https://")):
            thumb = thumbs
        duration = r.get("duration")
        out.append(
            {
                "src": page,
                "kind": (r.get("publisher") or "web").lower()[:20],
                "thumbnail": thumb,
                "title": (r.get("title") or page).strip()[:200],
                "duration": int(duration) if isinstance(duration, (int, float)) and duration > 0 else None,
                "page_url": page,
                "source": "live",
            }
        )
    return out
