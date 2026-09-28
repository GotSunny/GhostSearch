"""GHOST Search server: Universal Search API + static frontend.

Run:
    python app.py            # http://127.0.0.1:8000

Endpoints:
    GET  /                            search UI
    GET  /api/search?q=&type=&page=&safe=   web|images|videos results
    GET  /api/suggest?q=              autocomplete
    GET  /api/stats                   index statistics
    GET  /api/health                  liveness + db size + last crawl
    POST /api/click                   {"query", "url", "position"} click log
    POST /api/crawl                   {"seeds", "max_pages", "allow_external", "refresh"}
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import livesearch
import search
from crawler import Crawler, DB_PATH, init_schema, normalize_url, get_db

RESULTS_PER_PAGE = 10
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW = 10.0  # seconds
CACHE_MAXSIZE = 512
LIVE_THRESHOLD = 5        # local results below this trigger the live layer
LIVE_WEB_RESULTS = 12

app = FastAPI(title="GHOST Search")
app.mount("/static", StaticFiles(directory="static"), name="static")
_crawl_lock = threading.Lock()

_search_cache: dict[tuple, dict] = {}
_rate_log: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=RATE_LIMIT_REQUESTS))


def _rate_limit(request: Request) -> None:
    """Sliding-window limiter per client IP."""
    ip = request.client.host if request.client else "?"
    now = time.monotonic()
    bucket = _rate_log[ip]
    if bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
        bucket.clear()
    if len(bucket) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(429, "too many requests — slow down")
    bucket.append(now)


@app.on_event("startup")
def startup() -> None:
    init_schema(get_db())


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/search")
def api_search(request: Request, q: str, type: str = "web", page: int = 1,
               safe: str = "on", source: str = "auto") -> dict:
    _rate_limit(request)
    q = q.strip()
    if not q:
        raise HTTPException(422, "empty query")
    if type not in ("web", "images", "videos"):
        raise HTTPException(422, "type must be web, images or videos")
    if source not in ("auto", "local", "live"):
        raise HTTPException(422, "source must be auto, local or live")
    if page < 1 or page > 100:
        raise HTTPException(422, "page out of range")
    safe_on = safe != "off"

    cache_key = (q.lower(), type, page, safe_on, source)
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    if type == "images":
        res = search.search_images(q, safe=safe_on)
        results, live_count = res["results"], 0
        if source == "live" or (source == "auto" and len(results) < LIVE_THRESHOLD):
            live = livesearch.live_images(q, max_results=40, safe=safe_on)
            results = (results + live) if source == "auto" else live
            live_count = len(live)
        payload = {
            "query": q, "type": type, "page": 1,
            "per_page": len(results), "total": len(results), "results": results,
            "did_you_mean": None, "took_ms": res["took_ms"], "live_count": live_count,
        }
    elif type == "videos":
        res = search.search_videos(q, safe=safe_on)
        results, live_count = res["results"], 0
        if source == "live" or (source == "auto" and len(results) < LIVE_THRESHOLD):
            live = livesearch.live_videos(q, max_results=16, safe=safe_on)
            results = (results + live) if source == "auto" else live
            live_count = len(live)
        payload = {
            "query": q, "type": type, "page": 1,
            "per_page": len(results), "total": len(results), "results": results,
            "did_you_mean": None, "took_ms": res["took_ms"], "live_count": live_count,
        }
    else:
        res = search.search_web(q, limit=RESULTS_PER_PAGE, offset=(page - 1) * RESULTS_PER_PAGE, safe=safe_on)
        results, live_count = res["results"], 0
        if source == "live":
            results = livesearch.live_text(q, max_results=LIVE_WEB_RESULTS, safe=safe_on)
            live_count = len(results)
        elif source == "auto" and res["total"] < LIVE_THRESHOLD:
            live = livesearch.live_text(q, max_results=LIVE_WEB_RESULTS, safe=safe_on)
            results = results + live
            live_count = len(live)
        else:
            search.log_query(q)
        payload = {
            "query": q, "type": type, "page": page, "per_page": RESULTS_PER_PAGE,
            "total": live_count if source == "live"
                     else res["total"] + (live_count if source == "auto" else 0),
            "results": results,
            "did_you_mean": res["did_you_mean"], "took_ms": res["took_ms"],
            "live_count": live_count,
        }

    if len(_search_cache) >= CACHE_MAXSIZE:
        _search_cache.clear()  # simple reset; keys are cheap to recompute
    _search_cache[cache_key] = payload
    return payload


@app.get("/api/suggest")
def api_suggest(request: Request, q: str) -> dict:
    _rate_limit(request)
    return {"suggestions": search.suggest(q)}


@app.post("/api/click")
def api_click(request: Request, body: dict) -> dict:
    _rate_limit(request)
    query = str(body.get("query", "")).strip()
    url = str(body.get("url", "")).strip()
    position = int(body.get("position", 0))
    if not query or not url.startswith(("http://", "https://")):
        raise HTTPException(422, "query and http(s) url required")
    search.log_click(query, url, position)
    return {"ok": True}


def _db_size_mb() -> float:
    try:
        return round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
    except OSError:
        return 0.0


def _index_stats() -> dict:
    # Dedicated connection: closing get_db()'s thread-local would break the
    # next request reusing this thread.
    conn = sqlite3.connect(DB_PATH)
    try:
        pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        queries = conn.execute("SELECT COUNT(DISTINCT query) FROM query_log").fetchone()[0]
        try:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            vocab = conn.execute("SELECT COUNT(*) FROM terms_df").fetchone()[0]
        except Exception:
            docs, vocab = 0, 0
        row = conn.execute("SELECT MAX(last_crawled_at) FROM pages").fetchone()
        last_crawl = row[0] if row and row[0] else None
    finally:
        conn.close()
    return {
        "pages_crawled": pages,
        "documents_indexed": docs,
        "vocabulary_size": vocab,
        "images": images,
        "videos": videos,
        "links": links,
        "queries_logged": queries,
        "db_size_mb": _db_size_mb(),
        "last_crawl_at": last_crawl,
    }


@app.get("/api/stats")
def api_stats() -> dict:
    return _index_stats()


@app.get("/api/health")
def api_health() -> dict:
    stats = _index_stats()
    return {"ok": True, **stats}


class CrawlRequest(BaseModel):
    seeds: list[str] = Field(min_length=1)
    max_pages: int = Field(default=100, ge=1, le=1000)
    allow_external: bool = False
    refresh: bool = False


@app.post("/api/crawl")
def api_crawl(req: CrawlRequest) -> dict:
    seeds = [u for u in (normalize_url(s, s) for s in req.seeds) if u]
    if not seeds:
        raise HTTPException(422, "no valid http(s) seed URLs given")
    if not _crawl_lock.acquire(blocking=False):
        raise HTTPException(409, "a crawl is already running")
    try:
        crawler = Crawler(
            seeds, max_pages=req.max_pages,
            same_domain=not req.allow_external, refresh=req.refresh,
        )
        stats = crawler.run()
        import indexer

        indexed = indexer.build()
        _search_cache.clear()
        return {
            "crawled": stats["crawled"],
            "errors": stats["errors"],
            "duplicates": stats["duplicates"],
            "not_modified": stats["not_modified"],
            "skipped_fresh": stats["skipped_fresh"],
            "indexed": indexed,
        }
    finally:
        _crawl_lock.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
