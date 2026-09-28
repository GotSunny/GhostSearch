"""GHOST indexer: inverted index over stemmed tokens with per-field term
frequencies (title / headings / body), a terms_df table, and a PageRank
authority score computed from the crawled link graph.

Usage:
    python indexer.py
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import Counter

from crawler import DB_PATH, get_db, init_schema

WORD_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in is it its
    of on or she that the their there they this to was were will with you your""".split()
)

# Field boosts: a term hit in the title counts 3x, in an h1-h3 2x, body 1x.
FIELD_WEIGHTS = {"title": 3.0, "heading": 2.0, "body": 1.0}


# --------------------------------------------------------------------------
# Stemming — compact implementation of the classic Porter stemmer (1980).
# --------------------------------------------------------------------------

_VOWELS = "aeiou"


def _is_cons(word: str, i: int) -> bool:
    c = word[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return i == 0 or not _is_cons(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Number of VC sequences: [C](VC){m}[V]."""
    m = 0
    prev_v = False
    for i in range(len(stem)):
        v = not _is_cons(stem, i)
        if prev_v and not v:
            m += 1
        prev_v = v
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _ends_double_cons(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and _is_cons(word, len(word) - 1)
        and word[-1] not in ("l", "s", "z")
    )


def _cvc(word: str) -> bool:
    """stem ends cvc where second c is not w,x,y (for step 1b)."""
    if len(word) < 3:
        return False
    return (
        _is_cons(word, len(word) - 3)
        and not _is_cons(word, len(word) - 2)
        and _is_cons(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


_STEP2 = {
    "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
    "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
    "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
    "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
    "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
}
_STEP3 = {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": ""}
_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def porter_stem(word: str) -> str:
    if len(word) <= 2:
        return word
    w = word

    # Step 1a
    if w.endswith("sses") or w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_cons(w):
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_cons(w):
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"

    # Step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2
    for suffix, repl in _STEP2.items():
        if w.endswith(suffix):
            if _measure(w[: -len(suffix)]) > 0:
                w = w[: -len(suffix)] + repl
            break

    # Step 3
    for suffix, repl in _STEP3.items():
        if w.endswith(suffix):
            if _measure(w[: -len(suffix)]) > 0:
                w = w[: -len(suffix)] + repl
            break

    # Step 4
    for suffix in _STEP4:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if suffix == "ion" and stem and stem[-1] in "st":
                if _measure(stem) > 1:
                    w = stem
            elif _measure(stem) > 1:
                w = stem
            break

    # Step 5
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    if w.endswith("l") and _measure(w[:-1]) > 1 and w.endswith("ll"):
        w = w[:-1]

    return w


def tokenize(text: str) -> list[str]:
    """Lowercase, stopword-filtered, Porter-stemmed word tokens."""
    return [
        porter_stem(t)
        for t in WORD_RE.findall(text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


# --------------------------------------------------------------------------
# PageRank
# --------------------------------------------------------------------------

def compute_pagerank(damping: float = 0.85, iterations: int = 20) -> dict[str, float]:
    """Power iteration over the links table; returns normalized scores [0,1]."""
    conn = get_db()
    out: dict[str, list[str]] = {}
    indeg: dict[str, int] = {}
    nodes: set[str] = set()
    for src, dst in conn.execute("SELECT src_url, dst_url FROM links"):
        nodes.add(src)
        nodes.add(dst)
        out.setdefault(src, []).append(dst)
        indeg[dst] = indeg.get(dst, 0) + 1
    if not nodes:
        return {}
    rank = {u: 1.0 / len(nodes) for u in nodes}
    for _ in range(iterations):
        # Dangling pages (no outbound links) redistribute uniformly.
        base = (1.0 - damping) / len(nodes)
        incoming_sum: dict[str, float] = {u: 0.0 for u in nodes}
        dangling_mass = 0.0
        for u in nodes:
            targets = out.get(u)
            if not targets:
                dangling_mass += rank[u]
                continue
            share = damping * rank[u] / len(targets)
            for t in targets:
                incoming_sum[t] += share
        rank = {
            u: base + incoming_sum[u] + damping * dangling_mass / len(nodes)
            for u in nodes
        }
    max_r = max(rank.values()) or 1.0
    return {u: r / max_r for u, r in rank.items()}


# --------------------------------------------------------------------------
# Index build
# --------------------------------------------------------------------------

def init_index_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS postings;
        DROP TABLE IF EXISTS terms_df;
        CREATE TABLE documents (
            doc_id          INTEGER PRIMARY KEY,
            url             TEXT NOT NULL UNIQUE,
            title           TEXT NOT NULL,
            norm            REAL NOT NULL,
            pagerank        REAL NOT NULL DEFAULT 0.0,
            lang            TEXT NOT NULL DEFAULT 'en',
            last_crawled_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE postings (
            term     TEXT NOT NULL,
            doc_id   INTEGER NOT NULL REFERENCES documents(doc_id),
            tf_title INTEGER NOT NULL DEFAULT 0,
            tf_head  INTEGER NOT NULL DEFAULT 0,
            tf_body  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (term, doc_id)
        ) WITHOUT ROWID;
        CREATE TABLE terms_df (
            term    TEXT PRIMARY KEY,
            df      INTEGER NOT NULL,
            display TEXT NOT NULL DEFAULT ''
        );
        """
    )


def build() -> int:
    """Rebuild the whole index from the pages table. Returns number of docs."""
    conn = get_db()
    init_index_schema(conn)

    rows = conn.execute(
        "SELECT url, title, text, lang, headings, last_crawled_at FROM pages"
    ).fetchall()

    docs = []
    df: Counter[str] = Counter()
    # Most frequent surface form per stem, so suggestions can show real
    # words ("engine") instead of stems ("engin").
    surface: dict[str, Counter[str]] = {}
    for url, title, text, lang, headings_json, last_crawled_at in rows:
        title_terms = tokenize(title)
        try:
            heading_terms = tokenize(" ".join(json.loads(headings_json)))
        except (ValueError, TypeError):
            heading_terms = []
        body_terms = tokenize(text)
        docs.append((url, title, lang, title_terms, heading_terms, body_terms, last_crawled_at))
        for term in set(title_terms) | set(heading_terms) | set(body_terms):
            df[term] += 1
        raw = [
            t for t in WORD_RE.findall((title + " " + text).lower())
            if t not in STOPWORDS and len(t) > 1
        ]
        for tok in raw:
            stem = porter_stem(tok)
            surface.setdefault(stem, Counter())[tok] += 1

    n_docs = len(docs)
    if n_docs == 0:
        conn.commit()
        return 0
    idf = {t: math.log(n_docs / d) + 1.0 for t, d in df.items()}
    pagerank = compute_pagerank()

    postings_rows: list[tuple[str, int, int, int, int]] = []
    doc_norms: list[float] = []
    for doc_id, (url, title, lang, t_terms, h_terms, b_terms, crawled) in enumerate(docs):
        tf_t, tf_h, tf_b = Counter(t_terms), Counter(h_terms), Counter(b_terms)
        sq_sum = 0.0
        for term in set(tf_t) | set(tf_h) | set(tf_b):
            w = 0.0
            if tf_t.get(term):
                w += (1 + math.log(tf_t[term])) * idf[term] * FIELD_WEIGHTS["title"]
            if tf_h.get(term):
                w += (1 + math.log(tf_h[term])) * idf[term] * FIELD_WEIGHTS["heading"]
            if tf_b.get(term):
                w += (1 + math.log(tf_b[term])) * idf[term] * FIELD_WEIGHTS["body"]
            sq_sum += w * w
            postings_rows.append((term, doc_id, tf_t.get(term, 0), tf_h.get(term, 0), tf_b.get(term, 0)))
        doc_norms.append(math.sqrt(sq_sum) or 1.0)

    conn.executemany(
        "INSERT INTO documents (doc_id, url, title, norm, pagerank, lang, last_crawled_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (i, docs[i][0], docs[i][1], doc_norms[i], pagerank.get(docs[i][0], 0.0),
             docs[i][2], docs[i][6])
            for i in range(n_docs)
        ],
    )
    conn.executemany("INSERT INTO postings VALUES (?, ?, ?, ?, ?)", postings_rows)
    conn.executemany(
        "INSERT INTO terms_df VALUES (?, ?, ?)",
        [(t, d, surface.get(t, Counter()).most_common(1)[0][0] if surface.get(t) else t)
         for t, d in df.items()],
    )
    conn.commit()
    return n_docs


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    init_schema(get_db())
    n = build()
    vocab = get_db().execute("SELECT COUNT(*) FROM terms_df").fetchone()[0]
    print(f"Indexed {n} documents, vocabulary size {vocab}.")


if __name__ == "__main__":
    main()
