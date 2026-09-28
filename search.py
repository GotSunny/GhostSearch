"""GHOST query engine: operator parsing (site:, "phrases", -exclude,
filetype:), TF-IDF x PageRank x freshness ranking, "did you mean",
list-style snippets for "how to" queries, and image/video search.

Usage:
    python search.py "your query"
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from crawler import DB_PATH
from indexer import porter_stem, tokenize

SNIPPET_RADIUS = 130
SNIPPET_MAX = 320
DID_YOU_MEAN_MIN_RESULTS = 3
EDIT_DISTANCE_MAX = 2

# Query-time synonym expansion (raw tokens; multi-word allowed).
SYNONYMS: dict[str, list[str]] = {
    "recipe": ["how to make"],
    "pics": ["images", "pictures"],
    "photo": ["image", "picture"],
    "picture": ["image", "photo"],
    "tutorial": ["guide", "how to"],
    "guide": ["tutorial"],
    "buy": ["purchase"],
    "purchase": ["buy"],
    "make": ["create", "build"],
    "create": ["make"],
    "video": ["clip"],
    "docs": ["documentation"],
    "documentation": ["docs"],
}
SYNONYM_WEIGHT = 0.5

# Rough SafeSearch blocklist — checked as substrings against page text /
# image alt text when safe=on.
SAFESEARCH_BLOCKLIST = (
    "porn", "xxx", "nsfw", "nude", "nudes", "erotic", "escort service",
    "camgirl", "onlyfans", "hookup sex", "sex video",
)

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------
# Query parsing
# --------------------------------------------------------------------------

def parse_query(query: str) -> dict:
    """Split a query into terms, phrases, exclusions, site:, filetype:."""
    exclusions: list[str] = []
    site = None
    filetype = None
    phrases = re.findall(r'"([^"]*)"', query)
    query = re.sub(r'"[^"]*"', " ", query)

    tokens = []
    for tok in query.split():
        if tok.startswith("site:") and len(tok) > 5:
            site = tok[5:].lower().lstrip("*.")
        elif tok.startswith("-") and len(tok) > 1:
            exclusions.append(tok[1:].lower())
        elif tok.startswith("filetype:") and len(tok) > 9:
            filetype = tok[9:].lstrip(".").lower()
        else:
            tokens.append(tok)

    terms = [t for t in WORD_RE.findall(" ".join(tokens).lower()) if len(t) > 1]
    return {
        "terms": terms,
        "phrases": [p.strip() for p in phrases if p.strip()],
        "exclusions": exclusions,
        "site": site,
        "filetype": filetype,
    }


def expand_terms(terms: list[str]) -> list[tuple[str, float]]:
    """(stemmed_term, weight) pairs including synonym expansions."""
    out: dict[str, float] = {}
    for t in terms:
        out.setdefault(porter_stem(t), 1.0)
        for syn in SYNONYMS.get(t, []):
            for st in tokenize(syn):
                out.setdefault(st, SYNONYM_WEIGHT)
    return list(out.items())


# --------------------------------------------------------------------------
# Snippets
# --------------------------------------------------------------------------

def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _best_window(text: str, terms: list[str]) -> str:
    lowered = text.lower()
    anchors: list[int] = []
    for term in terms:
        start = 0
        while len(anchors) < 200:
            idx = lowered.find(term, start)
            if idx < 0:
                break
            anchors.append(idx)
            start = idx + len(term)
    if not anchors:
        return text[:SNIPPET_MAX].rstrip() + "…"
    best_start, best_score = 0, -1.0
    for idx in anchors[:200]:
        start = max(0, idx - SNIPPET_RADIUS)
        window = lowered[start : start + 2 * SNIPPET_RADIUS]
        score = sum(1.0 for t in set(terms) if t in window)
        if score > best_score:
            best_start, best_score = start, score
    end = min(len(text), best_start + 2 * SNIPPET_RADIUS)
    prefix = "…" if best_start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[best_start:end].strip()[:SNIPPET_MAX] + suffix


def _list_snippet(lists: list[list[str]], raw_terms: list[str]) -> str | None:
    """Numbered steps for 'how to' queries, from stored <ol>/<ul> content.

    Nav bars and breadcrumbs are filtered by requiring several substantial
    items free of '»' separators.
    """
    if not raw_terms:
        return None
    best, best_score = None, 0
    for items in lists:
        items = [
            i for i in items
            if len(i) > 25 and "»" not in i and "|" not in i
        ]
        if len(items) < 3:
            continue
        joined = " ".join(items).lower()
        score = sum(1 for t in raw_terms if t in joined)
        if score > best_score:
            best, best_score = items, score
    if not best or best_score < 2:
        return None
    steps = [f"{i}. {item}" for i, item in enumerate(best[:4], 1)]
    snippet = " ".join(steps)
    return snippet[:SNIPPET_MAX] + ("…" if len(snippet) > SNIPPET_MAX else "")


def _is_list_query(raw_terms: list[str]) -> bool:
    return "how" in raw_terms and "to" in raw_terms


# --------------------------------------------------------------------------
# Did you mean
# --------------------------------------------------------------------------

def _levenshtein(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def did_you_mean(conn: sqlite3.Connection, raw_terms: list[str]) -> str | None:
    """Closest vocabulary term (Levenshtein <= 2) per low-df query token."""
    if not raw_terms:
        return None
    vocab = conn.execute("SELECT term, df, display FROM terms_df").fetchall()
    vocab = [(r["term"], r["df"], r["display"] or r["term"]) for r in vocab if r["term"].isascii()]
    suggestion = list(raw_terms)
    for i, token in enumerate(suggestion):
        stem = porter_stem(token)
        row = conn.execute("SELECT df FROM terms_df WHERE term = ?", (stem,)).fetchone()
        if row and row["df"] >= 2:
            continue  # common enough as-is
        best, best_score = None, 0
        for term, df, display in vocab:
            if _levenshtein(token, term, EDIT_DISTANCE_MAX) <= EDIT_DISTANCE_MAX:
                score = df * (2 if term[0] == token[0] else 1)
                if score > best_score:
                    best, best_score = (display or term), score
        if best and porter_stem(best) != stem:
            suggestion[i] = best
    return " ".join(suggestion) if suggestion != raw_terms else None


# --------------------------------------------------------------------------
# Web search
# --------------------------------------------------------------------------

def _raw_words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if len(w) > 1]


def _freshness_boost(last_crawled_at: str) -> float:
    if not last_crawled_at:
        return 1.0
    try:
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(last_crawled_at)).days
    except ValueError:
        return 1.0
    return 1.0 + 0.15 * math.exp(-max(age_days, 0) / 14.0)


def search_web(query: str, limit: int = 10, offset: int = 0, safe: bool = True) -> dict:
    t0 = time.perf_counter()
    parsed = parse_query(query)
    raw_terms = parsed["terms"] + [w for p in parsed["phrases"] for w in _raw_words(p)]
    conn = _connect()
    n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    result = {"total": 0, "results": [], "did_you_mean": None}
    if n_docs == 0 or not raw_terms:
        conn.close()
        result["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # 1. Score candidates from postings (stemmed terms + synonym expansions).
    scored: dict[int, float] = {}
    for stem, weight in expand_terms(raw_terms):
        rows = conn.execute(
            "SELECT p.doc_id, p.tf_title, tf_head, tf_body, d.norm"
            " FROM postings p JOIN documents d ON d.doc_id = p.doc_id WHERE p.term = ?",
            (stem,),
        ).fetchall()
        if not rows:
            continue
        df = len(rows)
        idf = math.log(n_docs / df) + 1.0
        for r in rows:
            w = 0.0
            if r["tf_title"]:
                w += (1 + math.log(r["tf_title"])) * idf * 3.0
            if r["tf_head"]:
                w += (1 + math.log(r["tf_head"])) * idf * 2.0
            if r["tf_body"]:
                w += (1 + math.log(r["tf_body"])) * idf
            scored[r["doc_id"]] = scored.get(r["doc_id"], 0.0) + weight * idf * w / r["norm"]

    if not scored:
        result["did_you_mean"] = did_you_mean(conn, parsed["terms"])
        conn.close()
        result["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # 2. Authority + freshness multipliers.
    docs = {
        r["doc_id"]: r
        for r in conn.execute(
            "SELECT doc_id, url, title, pagerank, last_crawled_at FROM documents"
        ).fetchall()
    }
    final: dict[int, float] = {}
    for doc_id, tfidf in scored.items():
        d = docs[doc_id]
        final[doc_id] = tfidf * (0.3 + 0.7 * d["pagerank"]) * _freshness_boost(d["last_crawled_at"])

    # 3. Operator filters.
    exclude_stems = {porter_stem(e) for e in parsed["exclusions"]}
    exclude_stems.discard("")
    for doc_id in list(final):
        d = docs[doc_id]
        url = d["url"]
        if parsed["site"] and urlparse(url).netloc.lower().endswith(parsed["site"]) is False:
            del final[doc_id]
            continue
        if parsed["filetype"] and not urlparse(url).path.lower().endswith("." + parsed["filetype"]):
            del final[doc_id]
            continue
        if parsed["site"] is None and parsed["filetype"] is None and not parsed["phrases"] and not exclude_stems:
            continue
        page = conn.execute("SELECT text FROM pages WHERE url = ?", (url,)).fetchone()
        text = (page["text"].lower() if page else "")
        if any(phrase.lower() not in text for phrase in parsed["phrases"]):
            del final[doc_id]
            continue
        if exclude_stems:
            stems = {porter_stem(w) for w in _raw_words(text)}
            if stems & exclude_stems:
                del final[doc_id]
                continue
        if safe and any(word in text for word in SAFESEARCH_BLOCKLIST):
            del final[doc_id]

    ranked_ids = sorted(final, key=final.get, reverse=True)  # type: ignore[arg-type]
    result["total"] = len(ranked_ids)
    if not ranked_ids:
        result["did_you_mean"] = did_you_mean(conn, parsed["terms"])
        conn.close()
        result["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    snippet_terms = list(dict.fromkeys(_raw_words(" ".join(raw_terms))))
    for doc_id in ranked_ids[offset : offset + limit]:
        d = docs[doc_id]
        page = conn.execute(
            "SELECT text, favicon, headings, lists FROM pages WHERE url = ?", (d["url"],)
        ).fetchone()
        snippet = _best_window(page["text"], snippet_terms) if page else ""
        if _is_list_query(parsed["terms"]) and page:
            try:
                lists = json.loads(page["lists"])
            except (ValueError, TypeError):
                lists = []
            list_snip = _list_snippet(lists, parsed["terms"])
            if list_snip:
                snippet = list_snip
        host = urlparse(d["url"]).netloc.removeprefix("www.")
        path = urlparse(d["url"]).path.strip("/")
        crumbs = host + (" › " + " › ".join(path.split("/")[:3]) if path else "")
        result["results"].append(
            {
                "title": d["title"] or d["url"],
                "url": d["url"],
                "display_url": crumbs,
                "favicon": page["favicon"] if page else None,
                "snippet": snippet,
                "score": round(final[doc_id], 4),
                "source": "local",
            }
        )
    conn.close()
    result["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# --------------------------------------------------------------------------
# Image / video search
# --------------------------------------------------------------------------

def search_images(query: str, limit: int = 40, safe: bool = True) -> dict:
    t0 = time.perf_counter()
    raw_terms = [t for t in WORD_RE.findall(query.lower()) if len(t) > 1]
    stems = [porter_stem(t) for t in raw_terms]
    conn = _connect()
    result: dict = {"total": 0, "results": [], "took_ms": 0.0}
    if not stems:
        conn.close()
        return result

    stem_params = ",".join("?" * len(stems))

    # Two signals merged in Python: page-level relevance (postings + PageRank)
    # and image-level matches in alt text / page title.
    doc_rows = conn.execute(
        "SELECT d.url, d.pagerank, d.title FROM documents d WHERE d.doc_id IN ("
        " SELECT DISTINCT doc_id FROM postings WHERE term IN (" + stem_params + "))",
        stems,
    ).fetchall()
    page_scores = {r["url"]: (1.0 + r["pagerank"], r["title"]) for r in doc_rows}

    candidates: dict[int, tuple[float, sqlite3.Row]] = {}
    rows = conn.execute(
        "SELECT i.*, p.title AS page_title, p.text AS page_text, p.favicon"
        " FROM images i JOIN pages p ON p.url = i.page_url LIMIT 4000"
    ).fetchall()
    for r in rows:
        alt = r["alt"].lower()
        title = (r["page_title"] or "").lower()
        score = 0.0
        for raw in raw_terms:
            if raw in alt:
                score += 2.0
            if raw in title:
                score += 1.0
        page_score = page_scores.get(r["page_url"])
        if page_score:
            score += page_score[0]
        if score <= 0:
            continue
        if safe and any(w in alt or w in title for w in SAFESEARCH_BLOCKLIST):
            continue
        candidates[r["id"]] = (score, r)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)
    result["total"] = len(ranked)
    result["results"] = [
        {
            "src": r["src"],
            "thumbnail": None,
            "alt": r["alt"],
            "width": r["width"],
            "height": r["height"],
            "page_url": r["page_url"],
            "page_title": r["page_title"] or r["page_url"],
            "source": "local",
        }
        for _, (_, r) in ranked[:limit]
    ]
    conn.close()
    result["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def search_videos(query: str, limit: int = 30, safe: bool = True) -> dict:
    t0 = time.perf_counter()
    raw_terms = [t for t in WORD_RE.findall(query.lower()) if len(t) > 1]
    stems = [porter_stem(t) for t in raw_terms]
    conn = _connect()
    result: dict = {"total": 0, "results": [], "took_ms": 0.0}
    if not stems:
        conn.close()
        return result

    stem_params = ",".join("?" * len(stems))
    doc_rows = conn.execute(
        "SELECT url, pagerank, title FROM documents WHERE doc_id IN ("
        " SELECT DISTINCT doc_id FROM postings WHERE term IN (" + stem_params + "))",
        stems,
    ).fetchall()
    page_scores = {r["url"]: (1.0 + r["pagerank"], r["title"]) for r in doc_rows}

    candidates: dict[int, tuple[float, sqlite3.Row]] = {}
    rows = conn.execute(
        "SELECT v.*, p.title AS page_title, p.text AS page_text"
        " FROM videos v JOIN pages p ON p.url = v.page_url LIMIT 2000"
    ).fetchall()
    for r in rows:
        v_title = r["title"].lower()
        page_title = (r["page_title"] or "").lower()
        score = 0.0
        for raw in raw_terms:
            if raw in v_title:
                score += 2.0
            if raw in page_title:
                score += 1.0
        page_score = page_scores.get(r["page_url"])
        if page_score:
            score += page_score[0]
        if score <= 0:
            continue
        if safe and any(w in v_title or w in page_title for w in SAFESEARCH_BLOCKLIST):
            continue
        candidates[r["id"]] = (score, r)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)
    result["total"] = len(ranked)
    result["results"] = [
        {
            "src": r["src"],
            "kind": r["kind"],
            "thumbnail": r["thumbnail"],
            "title": r["title"] or r["page_title"] or r["src"],
            "duration": r["duration"],
            "page_url": r["page_url"],
            "source": "local",
        }
        for _, (_, r) in ranked[:limit]
    ]
    conn.close()
    result["took_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# --------------------------------------------------------------------------
# Autocomplete + logging
# --------------------------------------------------------------------------

def suggest(prefix: str, limit: int = 8) -> list[str]:
    prefix = prefix.strip().lower()
    if len(prefix) < 2:
        return []
    conn = _connect()
    like = f"%{_escape_like(prefix)}%"
    out: list[str] = []
    # past queries first (trending), then indexed headings, then vocabulary
    for row in conn.execute(
        "SELECT query, COUNT(*) c FROM query_log WHERE query LIKE ? ESCAPE '\\'"
        " GROUP BY query ORDER BY c DESC LIMIT ?", (like, limit)
    ):
        out.append(row["query"])
    if len(out) < limit:
        for row in conn.execute(
            "SELECT headings, url FROM pages WHERE headings LIKE ? ESCAPE '\\' LIMIT ?",
            (like, limit * 2),
        ):
            try:
                for h in json.loads(row["headings"]):
                    if prefix in h.lower() and h not in out and len(h) < 80:
                        out.append(h)
            except (ValueError, TypeError):
                pass
    if len(out) < limit:
        for row in conn.execute(
            "SELECT term, display, SUM(df) s FROM terms_df WHERE term LIKE ? ESCAPE '\\'"
            " GROUP BY term ORDER BY s DESC LIMIT ?", (like + "%", limit)
        ):
            word = row["display"] or row["term"]
            if row["term"].isascii() and word not in out:
                out.append(word)
    conn.close()
    return out[:limit]


def log_query(query: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO query_log (query, ts) VALUES (?, ?)",
        (query.strip(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def log_click(query: str, url: str, position: int) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO click_log (query, url, position, ts) VALUES (?, ?, ?, ?)",
        (query.strip(), url, position, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    query = " ".join(sys.argv[1:])
    if not query:
        print("usage: python search.py <query>")
        return
    res = search_web(query)
    print(f"{res['total']} result(s) for {query!r} ({res['took_ms']} ms)\n")
    if res["did_you_mean"]:
        print(f"Did you mean: {res['did_you_mean']!r}\n")
    for i, r in enumerate(res["results"], 1):
        print(f"{i}. {r['title']}  (score {r['score']})")
        print(f"   {r['url']}")
        print(f"   {r['snippet'][:160]}")
        print()


if __name__ == "__main__":
    main()
