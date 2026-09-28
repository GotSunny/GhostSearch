"""GHOST Search crawler: fetches pages and everything Universal Search needs
(images, videos, favicons, headings/lists for snippets), builds the link
graph, dedupes near-identical pages, and re-crawls incrementally.

Usage:
    python crawler.py https://example.com --max 100 [--allow-external] [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

DB_PATH = "search.db"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)
FETCH_TIMEOUT = 10
POLITENESS_DELAY = 0.5       # seconds between requests to the same domain
MAX_CONCURRENT_PER_DOMAIN = 2
GLOBAL_MIN_INTERVAL = 0.1    # configurable global rate limit (seconds)
MAX_WORKERS = 8
MAX_DEPTH = 4
STALE_AFTER_DAYS = 30
SIMHASH_HAMMING_MAX = 3      # closer than this counts as a duplicate page
MAX_IMAGES_PER_PAGE = 50
MAX_LISTS_PER_PAGE = 10
TRACKING_PARAMS = frozenset(
    {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
     "ref", "ref_src", "fbclid", "gclid"}
)

YOUTUBE_RE = re.compile(r"(?:youtube\.com/(?:embed|shorts|live)/|youtu\.be/)([\w-]{6,})")
VIMEO_RE = re.compile(r"player\.vimeo\.com/video/(\d+)")
ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
WORD_RE = re.compile(r"\w+", re.UNICODE)

# Rough language markers for a dependency-free heuristic detector.
LANG_MARKERS = {
    "en": ("the", "and", "of", "to", "in", "is", "that", "for", "with"),
    "es": ("el", "la", "de", "que", "y", "en", "los", "del", "una"),
    "fr": ("le", "la", "les", "des", "est", "une", "pour", "dans", "sur"),
    "de": ("der", "die", "und", "den", "von", "ist", "mit", "auf", "das"),
    "it": ("il", "di", "che", "per", "con", "una", "sono", "gli"),
    "pt": ("de", "que", "não", "uma", "para", "com", "como", "mais", "dos"),
}

_local = threading.local()


def get_db() -> sqlite3.Connection:
    """One connection per thread (sqlite3 objects are not thread-safe)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        init_schema(conn)
        _local.conn = conn
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    # Upgrade path: a pre-GHOST pages table lacks the new columns — drop it.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(pages)")}
    if cols and "simhash" not in cols:
        conn.executescript("DROP TABLE IF EXISTS pages;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pages (
            url              TEXT PRIMARY KEY,
            title            TEXT NOT NULL DEFAULT '',
            text             TEXT NOT NULL DEFAULT '',
            lang             TEXT NOT NULL DEFAULT 'en',
            favicon          TEXT,
            meta_description TEXT,
            og_title         TEXT,
            og_image         TEXT,
            canonical_url    TEXT,
            headings         TEXT NOT NULL DEFAULT '[]',
            lists            TEXT NOT NULL DEFAULT '[]',
            content_hash     TEXT,
            simhash          INTEGER,
            etag             TEXT,
            last_modified    TEXT,
            last_crawled_at  TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS images (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            page_url  TEXT NOT NULL,
            src       TEXT NOT NULL,
            alt       TEXT NOT NULL DEFAULT '',
            width     INTEGER,
            height    INTEGER
        );
        CREATE TABLE IF NOT EXISTS videos (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            page_url   TEXT NOT NULL,
            src        TEXT NOT NULL,
            kind       TEXT NOT NULL,
            video_id   TEXT,
            thumbnail  TEXT,
            title      TEXT NOT NULL DEFAULT '',
            duration   INTEGER
        );
        CREATE TABLE IF NOT EXISTS links (
            src_url TEXT NOT NULL,
            dst_url TEXT NOT NULL,
            PRIMARY KEY (src_url, dst_url)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS domains (
            domain      TEXT PRIMARY KEY,
            favicon     TEXT,
            crawl_delay REAL
        );
        CREATE TABLE IF NOT EXISTS query_log (
            query TEXT NOT NULL,
            ts    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS click_log (
            query    TEXT NOT NULL,
            url      TEXT NOT NULL,
            position INTEGER NOT NULL,
            ts       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_images_page ON images(page_url);
        CREATE INDEX IF NOT EXISTS idx_videos_page ON videos(page_url);
        """
    )
    conn.commit()


# --------------------------------------------------------------------------
# URL / content helpers
# --------------------------------------------------------------------------

def normalize_url(base: str, href: str) -> str | None:
    """Resolve href against base; drop fragments and tracking params."""
    href = (href or "").strip()
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
        return None
    absolute = urldefrag(urljoin(base, href))[0]
    parts = urlparse(absolute)
    if parts.scheme not in ("http", "https"):
        return None
    if parts.query:
        kept = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        absolute = urlunparse(parts._replace(query=urlencode(kept)))
    return absolute


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def simhash(tokens: list[str]) -> int:
    """64-bit simhash over token frequencies; near-dupes differ in few bits."""
    weights = [0] * 64
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    for token, tf in counts.items():
        h = int.from_bytes(hashlib.md5(token.encode("utf-8", "ignore")).digest()[:8], "big")
        for i in range(64):
            weights[i] += tf if (h >> i) & 1 else -tf
    bits = 0
    for i, w in enumerate(weights):
        if w > 0:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def detect_lang(html_lang: str | None, text: str) -> str:
    if html_lang:
        return html_lang.split("-")[0].lower()[:5]
    sample = text[:4000].lower()
    letters = [c for c in sample if c.isalpha()]
    if letters:
        cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff") / len(letters)
        cjk = sum(1 for c in letters if "\u4e00" <= c <= "\u9fff") / len(letters)
        if cyrillic > 0.3:
            return "ru"
        if cjk > 0.3:
            return "zh"
    words = re.findall(r"[a-zà-ÿ]+", sample)
    if not words:
        return "en"
    counts = {lang: sum(1 for w in words if w in markers) for lang, markers in LANG_MARKERS.items()}
    best = max(counts, key=lambda k: counts[k])  # type: ignore[arg-type]
    return best if counts[best] >= 3 else "en"


def parse_iso_duration(value: str) -> int | None:
    m = ISO_DURATION_RE.match((value or "").strip())
    if not m:
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s if (h or mi or s) else None


def _pixel(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(str(value).strip().removesuffix("px"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract(page_html: str, base_url: str) -> dict:
    """Pull every useful signal out of an HTML page."""
    soup = BeautifulSoup(page_html, "html.parser")

    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = (tag.get("name") or tag.get("property") or "").lower()
        if key and tag.get("content"):
            meta[key] = tag["content"].strip()

    canonical = None
    for link in soup.find_all("link", rel=lambda v: v and "canonical" in v):
        href = link.get("href")
        if href:
            resolved = urljoin(base_url, href.strip())
            if urlparse(resolved).scheme in ("http", "https"):
                canonical = urldefrag(resolved)[0]
                break

    favicon = None
    for link in soup.find_all("link", rel=lambda v: v and "icon" in v.lower()):
        href = link.get("href")
        if href and not href.startswith("data:"):
            resolved = urljoin(base_url, href.strip())
            if urlparse(resolved).scheme in ("http", "https"):
                favicon = resolved
                break

    title = (soup.title.string or "").strip() if soup.title else ""
    if not title:
        title = meta.get("og:title", "")

    html_lang = soup.html.get("lang") if soup.html else None

    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = h.get_text(" ", strip=True)
        if t:
            headings.append(t)
    headings = headings[:30]

    lists_content = []
    for lst in soup.find_all(["ol", "ul"]):
        items = [li.get_text(" ", strip=True) for li in lst.find_all("li")]
        items = [i for i in items if 15 < len(i) < 300]
        if len(items) >= 2:  # nav bars are mostly short labels — skip them
            lists_content.append(items)
        if len(lists_content) >= MAX_LISTS_PER_PAGE:
            break

    # Media must be collected before stripping tags below.
    images: list[dict] = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src or src.startswith("data:"):
            continue
        abs_src = urljoin(base_url, src.strip())
        if urlparse(abs_src).scheme not in ("http", "https"):
            continue
        w, hgt = _pixel(img.get("width")), _pixel(img.get("height"))
        if (w is not None and w <= 16) or (hgt is not None and hgt <= 16):
            continue  # tracking pixels / spacer gifs
        images.append(
            {
                "src": abs_src,
                "alt": (img.get("alt") or "").strip()[:300],
                "width": w,
                "height": hgt,
            }
        )
        if len(images) >= MAX_IMAGES_PER_PAGE:
            break

    videos: dict[str, dict] = {}
    for v in soup.find_all("video"):
        src = v.get("src")
        if not src:
            inner = v.find("source", src=True)
            src = inner["src"] if inner else None
        if not src or src.startswith("data:"):
            continue
        poster = v.get("poster")
        videos[urljoin(base_url, src.strip())] = {
            "src": urljoin(base_url, src.strip()),
            "kind": "video",
            "video_id": None,
            "thumbnail": urljoin(base_url, poster.strip()) if poster else None,
            "title": (v.get("title") or meta.get("og:title") or "")[:200],
            "duration": parse_iso_duration(v.get("data-duration") or ""),
        }
    for iframe in soup.find_all("iframe", src=True):
        s = iframe["src"] or ""
        yt = YOUTUBE_RE.search(s)
        vm = VIMEO_RE.search(s)
        if yt:
            vid = yt.group(1)
            videos[f"https://www.youtube.com/watch?v={vid}"] = {
                "src": f"https://www.youtube.com/watch?v={vid}",
                "kind": "youtube",
                "video_id": vid,
                "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
                "title": (iframe.get("title") or iframe.get("aria-label") or "")[:200],
                "duration": None,
            }
        elif vm:
            vid = vm.group(1)
            videos[f"https://vimeo.com/{vid}"] = {
                "src": f"https://vimeo.com/{vid}",
                "kind": "vimeo",
                "video_id": vid,
                "thumbnail": None,
                "title": (iframe.get("title") or "")[:200],
                "duration": None,
            }
    for a in soup.find_all("a", href=True):
        path = urlparse(urljoin(base_url, a["href"])).path.lower()
        if path.endswith((".mp4", ".webm")):
            abs_href = urljoin(base_url, a["href"].strip())
            videos[abs_href] = {
                "src": abs_href,
                "kind": "file",
                "video_id": None,
                "thumbnail": None,
                "title": a.get_text(" ", strip=True)[:200],
                "duration": None,
            }

    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)

    links = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(base_url, a["href"])
        if u:
            links.append(u)

    return {
        "title": title or base_url,
        "text": text,
        "lang": detect_lang(html_lang, text),
        "meta_description": meta.get("description", ""),
        "og_title": meta.get("og:title", ""),
        "og_image": meta.get("og:image", ""),
        "canonical_url": canonical,
        "favicon": favicon,
        "headings": headings,
        "lists": lists_content,
        "images": images,
        "videos": list(videos.values()),
        "links": links,
    }


# --------------------------------------------------------------------------
# Fetching infrastructure
# --------------------------------------------------------------------------

class RobotsCache:
    """Caches robots.txt parsers and crawl-delays per domain, thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _get(self, host: str) -> urllib.robotparser.RobotFileParser | None:
        with self._lock:
            if host in self._cache:
                return self._cache[host]
        parser = urllib.robotparser.RobotFileParser()
        try:
            resp = requests.get(
                f"https://{host}/robots.txt", timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )
            if resp.status_code == 404:
                parser = None  # type: ignore[assignment]  # no robots.txt -> everything allowed
            else:
                parser.parse(resp.text.splitlines())
        except requests.RequestException:
            parser = None  # type: ignore[assignment]  # unreachable -> be permissive
        with self._lock:
            self._cache[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        parser = self._get(urlparse(url).netloc)
        return parser is None or parser.can_fetch(USER_AGENT, url)

    def crawl_delay(self, url: str) -> float:
        """Per-domain delay in seconds (robots crawl-delay, capped at 10s)."""
        parser = self._get(urlparse(url).netloc)
        try:
            delay = parser.crawl_delay(USER_AGENT) if parser else None
        except Exception:
            delay = None
        return min(float(delay), 10.0) if delay else 0.0


class RateLimiter:
    """Per-domain delay + per-domain concurrency cap + global min interval."""

    def __init__(self, domain_delay: float, per_domain: int, global_interval: float) -> None:
        self.domain_delay = domain_delay
        self.global_interval = global_interval
        self._lock = threading.Lock()
        self._next_slot: dict[str, float] = {}
        self._sems: dict[str, threading.Semaphore] = {}
        self._per_domain = per_domain
        self._global_next = 0.0

    def _sem(self, host: str) -> threading.Semaphore:
        with self._lock:
            if host not in self._sems:
                self._sems[host] = threading.Semaphore(self._per_domain)
            return self._sems[host]

    def wait(self, host: str) -> threading.Semaphore:
        sem = self._sem(host)
        sem.acquire()
        try:
            while True:
                with self._lock:
                    now = time.monotonic()
                    domain_wait = self._next_slot.get(host, 0) - now
                    global_wait = self._global_next - now
                    delay = max(domain_wait, global_wait, 0)
                    if delay <= 0:
                        slot = now + self.domain_delay
                        self._next_slot[host] = slot
                        self._global_next = now + self.global_interval
                        return sem
                time.sleep(min(delay, 0.25))
        except BaseException:
            sem.release()
            raise

    def release(self, host: str, sem: threading.Semaphore) -> None:
        sem.release()


def fetch(url: str, etag: str | None = None, last_modified: str | None = None) -> tuple[str, tuple | None]:
    """Download a page.

    Returns ("ok", (html, etag, last_modified)), ("not_modified", None) on a
    304 conditional-GET hit, or ("error", None) on any failure.
    """
    headers = {"User-Agent": USER_AGENT}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=headers, allow_redirects=True)
        if resp.status_code == 304:
            return "not_modified", None
        if resp.status_code >= 400:
            return "error", None
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            return "error", None
        # requests falls back to ISO-8859-1 when no charset is sent; that
        # guess mangles UTF-8 pages, so re-detect when it happens.
        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding
        return "ok", (resp.text, resp.headers.get("ETag"), resp.headers.get("Last-Modified"))
    except requests.RequestException:
        return "error", None


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def store_page(data: dict, url: str, etag: str | None, last_modified: str | None) -> None:
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(
        (data["title"] + "\n" + data["text"]).encode("utf-8", "ignore")
    ).hexdigest()
    tokens = _words(data["title"]) + _words(data["text"])
    sh = simhash(tokens)
    # SQLite INTEGER is signed 64-bit; fold the unsigned hash into range.
    if sh >= 1 << 63:
        sh -= 1 << 64
    conn.execute(
        """INSERT OR REPLACE INTO pages
           (url, title, text, lang, favicon, meta_description, og_title, og_image,
            canonical_url, headings, lists, content_hash, simhash, etag,
            last_modified, last_crawled_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            url, data["title"], data["text"], data["lang"], data["favicon"],
            data["meta_description"], data["og_title"], data["og_image"],
            data["canonical_url"], json.dumps(data["headings"]), json.dumps(data["lists"]),
            content_hash, sh, etag, last_modified, now,
        ),
    )
    conn.execute("DELETE FROM images WHERE page_url = ?", (url,))
    conn.executemany(
        "INSERT INTO images (page_url, src, alt, width, height) VALUES (?, ?, ?, ?, ?)",
        [(url, i["src"], i["alt"], i["width"], i["height"]) for i in data["images"]],
    )
    conn.execute("DELETE FROM videos WHERE page_url = ?", (url,))
    conn.executemany(
        "INSERT INTO videos (page_url, src, kind, video_id, thumbnail, title, duration)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (url, v["src"], v["kind"], v["video_id"], v["thumbnail"], v["title"], v["duration"])
            for v in data["videos"]
        ],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO links (src_url, dst_url) VALUES (?, ?)",
        [(url, dst) for dst in data["links"]],
    )
    conn.commit()


def cache_domain_favicon(domain: str, favicon: str | None) -> None:
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO domains (domain, favicon) VALUES (?, ?)", (domain, favicon)
    )
    conn.commit()


def get_cached_favicon(domain: str) -> str | None:
    row = get_db().execute("SELECT favicon FROM domains WHERE domain = ?", (domain,)).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------------

class Crawler:
    def __init__(self, seeds: list[str], max_pages: int = 100, same_domain: bool = True,
                 refresh: bool = False) -> None:
        self.max_pages = max_pages
        self.allowed_domains = {urlparse(s).netloc for s in seeds} if same_domain else None
        self.refresh = refresh
        self.frontier: Queue[tuple[str, int]] = Queue()
        for s in seeds:
            self.frontier.put((s, 0))
        self.visited: set[str] = set()
        self.seen_simhashes: list[tuple[int, str]] = []
        self.robots = RobotsCache()
        self.lock = threading.Lock()
        self.count = 0
        self.errors = 0
        self.duplicates = 0
        self.skipped_fresh = 0
        self.not_modified = 0
        # Termination accounting: a worker may exit only when the queue is
        # empty AND every URL ever enqueued has been fully processed, because
        # processing a page is what adds new URLs to the queue.
        self.enqueued = len(seeds)
        self.finished = 0

    def _should_visit(self, url: str, depth: int) -> bool:
        if url in self.visited or depth > MAX_DEPTH:
            return False
        if self.allowed_domains and urlparse(url).netloc not in self.allowed_domains:
            return False
        return True

    def _is_duplicate(self, sh: int, url: str) -> bool:
        for other, other_url in self.seen_simhashes:
            if hamming(sh, other) <= SIMHASH_HAMMING_MAX and other_url != url:
                return True
        return False

    def _resolve_favicon(self, url: str, page_favicon: str | None) -> str | None:
        """Page <link> first, then the per-domain cache, then /favicon.ico."""
        if page_favicon:
            return page_favicon
        domain = urlparse(url).netloc
        cached = get_cached_favicon(domain)
        if cached is not None:
            return cached
        probed = None
        try:
            resp = requests.head(
                f"https://{domain}/favicon.ico", timeout=5, headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            )
            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
                probed = f"https://{domain}/favicon.ico"
        except requests.RequestException:
            pass
        cache_domain_favicon(domain, probed)
        return probed

    def _enqueue(self, url: str, depth: int) -> None:
        with self.lock:
            self.enqueued += 1
        self.frontier.put((url, depth))

    def _worker(self) -> None:
        while True:
            with self.lock:
                if self.count >= self.max_pages:
                    return
            try:
                url, depth = self.frontier.get(timeout=0.25)
            except Empty:
                with self.lock:
                    if self.finished >= self.enqueued:
                        return  # frontier permanently exhausted
                continue
            try:
                with self.lock:
                    if not self._should_visit(url, depth):
                        continue
                    self.visited.add(url)

                # Incremental: pages crawled recently are left alone unless
                # --refresh; older ones get a conditional GET.
                if not self.refresh:
                    row = get_db().execute(
                        "SELECT last_crawled_at FROM pages WHERE url = ?", (url,)
                    ).fetchone()
                    if row and row[0]:
                        age = datetime.now(timezone.utc) - datetime.fromisoformat(row[0])
                        if age < timedelta(days=STALE_AFTER_DAYS):
                            with self.lock:
                                self.skipped_fresh += 1
                            continue
                    etag_lm = get_db().execute(
                        "SELECT etag, last_modified FROM pages WHERE url = ?", (url,)
                    ).fetchone()
                else:
                    etag_lm = None

                if not self.robots.allowed(url):
                    continue
                delay = max(POLITENESS_DELAY, self.robots.crawl_delay(url))
                limiter = self.limiter
                sem = limiter.wait(urlparse(url).netloc)
                try:
                    status, payload = fetch(
                        url,
                        etag=etag_lm[0] if etag_lm else None,
                        last_modified=etag_lm[1] if etag_lm else None,
                    )
                finally:
                    limiter.release(urlparse(url).netloc, sem)

                if status == "error":
                    with self.lock:
                        self.errors += 1
                    continue
                if status == "not_modified":
                    get_db().execute(
                        "UPDATE pages SET last_crawled_at = ? WHERE url = ?",
                        (datetime.now(timezone.utc).isoformat(), url),
                    )
                    get_db().commit()
                    with self.lock:
                        self.not_modified += 1
                    continue

                html, etag, last_modified = payload
                data = extract(html, url)

                # Canonical URLs collapse ?utm_* / mirror variants into one.
                effective = data["canonical_url"] or url
                with self.lock:
                    if effective != url and effective in self.visited:
                        self.duplicates += 1
                        continue
                    self.visited.add(effective)

                sh = simhash(_words(data["title"]) + _words(data["text"]))
                if data["text"] and self._is_duplicate(sh, effective):
                    with self.lock:
                        self.duplicates += 1
                    continue
                with self.lock:
                    self.seen_simhashes.append((sh, effective))

                with self.lock:
                    if self.count >= self.max_pages:
                        continue
                    self.count += 1
                    print(f"[{self.count}/{self.max_pages}] {data['title'][:70]}")
                data["favicon"] = self._resolve_favicon(url, data["favicon"])
                store_page(data, effective, etag, last_modified)
                if url != effective:
                    get_db().execute(
                        "INSERT OR IGNORE INTO links (src_url, dst_url) VALUES (?, ?)", (url, effective)
                    )
                    get_db().commit()

                for link in data["links"]:
                    with self.lock:
                        if self.count >= self.max_pages:
                            break
                    self._enqueue(link, depth + 1)
            finally:
                with self.lock:
                    self.finished += 1

    def run(self) -> dict:
        init_schema(get_db())
        self.limiter = RateLimiter(POLITENESS_DELAY, MAX_CONCURRENT_PER_DOMAIN, GLOBAL_MIN_INTERVAL)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(self._worker) for _ in range(MAX_WORKERS)]
            for f in futures:
                f.result()
        return {
            "crawled": self.count,
            "errors": self.errors,
            "duplicates": self.duplicates,
            "skipped_fresh": self.skipped_fresh,
            "not_modified": self.not_modified,
        }


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Crawl seed URLs into search.db")
    parser.add_argument("seeds", nargs="+", help="seed URLs to start from")
    parser.add_argument("--max", type=int, default=100, help="max pages to fetch")
    parser.add_argument("--allow-external", action="store_true", help="follow links to other domains")
    parser.add_argument("--refresh", action="store_true", help="force re-crawl, ignore freshness/304s")
    args = parser.parse_args()
    seeds = [normalize_url(s, s) or s for s in args.seeds]
    crawler = Crawler(seeds, max_pages=args.max, same_domain=not args.allow_external, refresh=args.refresh)
    stats = crawler.run()
    print(
        f"\nDone: {stats['crawled']} stored, {stats['duplicates']} duplicates skipped, "
        f"{stats['not_modified']} unchanged (304), {stats['skipped_fresh']} fresh-skipped, "
        f"{stats['errors']} failed."
    )


if __name__ == "__main__":
    main()
