# GHOST Search 👻

A self-hosted, no-AI, Google-style **Universal Search engine**: a crawler that
extracts everything (text, images, videos, favicons, link graph), a TF-IDF ×
PageRank index over Porter-stemmed tokens, a **DuckDuckGo live web layer**
blended in for anything your local index doesn't cover, and a dark-mode web
UI with Web / Images / Videos tabs, operators, "did you mean", lightbox, and
autocomplete.

## Quick start

```bash
pip install -r requirements.txt

# 1. Crawl (incremental: re-runs skip pages <30 days old and 304-unchanged ones)
python crawler.py https://en.wikipedia.org/wiki/Search_engine https://docs.python.org/3/ --max 100

# 2. Build the index (stems, field weights, PageRank)
python indexer.py

# 3. Run the web app — open http://127.0.0.1:8000
python app.py
```

The UI's **⚙︎ Crawl & index** panel (bottom-right) crawls and re-indexes
without leaving the browser, including a "refresh all" force flag.

## Architecture

| File | Role |
|---|---|
| `crawler.py` | Multi-threaded BFS. robots.txt (incl. crawl-delay), per-domain concurrency cap + global rate limit, canonical-URL & tracking-param stripping, **simhash near-duplicate rejection**, **incremental re-crawl** (ETag/If-Modified-Since + 30-day freshness skip, `--refresh` to force). Extracts per page: title, text, lang, meta/OG tags, favicon (with per-domain cache), h1–h3 headings, list content, `<img>`s, videos (`<video>`, YouTube/Vimeo iframes, .mp4/.webm links), and all outbound links (link graph). |
| `indexer.py` | Full **Porter stemmer** (no deps). Inverted index with per-field term counts — title 3×, headings 2×, body 1× — a `terms_df` table (df + most common surface form per stem), and **PageRank** (power iteration, damping 0.85) computed from the `links` table. |
| `search.py` | Query parser (**`site:` `"exact phrase"` `-exclude` `filetype:`**), synonym expansion, scoring `TF-IDF cosine × (0.3 + 0.7·PageRank) × freshness`, **"did you mean"** (Levenshtein ≤ 2 over the vocabulary), densest-window snippets plus **numbered step snippets for "how to" queries** from stored lists, image & video search, autocomplete (query log → headings → vocabulary), SafeSearch blocklist, query/click logging. |
| `app.py` | FastAPI: `/api/search?q=&type=web|images|videos&page=&safe=&source=auto|local|live`, `/api/suggest`, `/api/click`, `/api/stats`, `/api/health`, `/api/crawl`. In-memory result cache, per-IP rate limiting (20 req / 10 s), `took_ms` in every response. |
| `livesearch.py` | DuckDuckGo meta-search layer (`ddgs` package): live web/images/videos results shaped exactly like local ones (`source: "live"`). Used automatically when the local index has fewer than 5 hits, or on demand via `source=live` / the 🌐 toggle in the UI. Failure-safe: any DDG outage degrades to local-only. |
| `static/index.html` | GHOST UI: dark theme by default with persisted toggle, animated All/Images/Videos tabs, favicon + breadcrumb results, `<mark>` highlighting, lazy image grid + lightbox (Esc to close), video cards, did-you-mean banner, skeleton loaders, numbered pagination, `/` and `Esc` shortcuts, debounced autocomplete with arrow keys, aria labels, responsive grids. |

## Search operators

```
web search engine site:wikipedia.org     # restrict to a host
"information retrieval" -audio           # exact phrase, exclude term
filetype:pdf machine learning            # by file extension in the URL
how to search the web                    # triggers numbered-step snippets
```

## Response shape

```json
{
  "query": "web search engine", "type": "web", "page": 1, "per_page": 10,
  "total": 49, "took_ms": 14.5, "cached": false,
  "did_you_mean": null,
  "results": [
    { "title": "…", "url": "…", "display_url": "en.wikipedia.org › wiki › Search_engine",
      "favicon": "https://…/favicon.ico", "snippet": "…", "score": 0.42 }
  ]
}
```

## Notes & limits

- Everything lives in one SQLite file (`search.db`). Delete it to reset; the
  crawler auto-migrates a pre-GHOST schema.
- Favicons come from the crawl (`<link rel=icon>` → `/favicon.ico` probe,
  cached per domain); the UI falls back to Google's s2 favicon service and
  then a letter tile when offline.
- PageRank is only as good as the crawled link graph — small crawls make
  heavily-linked hub pages (e.g. Wikipedia disambiguation pages) outrank
  content pages; grow the index or tune the `0.3 + 0.7·pagerank` blend.
- Educational project: no web-scale engineering, no spam fighting, no AI.
