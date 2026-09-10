#!/usr/bin/env python3
"""Fetch a Wikipedia article's plain-text lead for the lookup loop.

SKETCH — companion to ``fetch_current_events.py``. The current-events fetcher
deposits a day's news digest; an ``[ask:search]`` Ava raised from that digest is
resolved to a subject by the agentic extractor (``inference/core/agentic.py``),
and THIS module turns a subject into a clean article snippet the next learning
pass can read. Closing the loop, that learning pass can ``[resolved]`` the
originating ask and distil a ``[fact]``.

Pipeline position::

    digest → [ask:search] → agentic.extract_subjects → fetch_article → snippet
           → next learning pass → [resolved] + [fact]

Design choices (see the thread in AVA_DESIGN_LEGACY.md):
  * Use the MediaWiki **TextExtracts** API (``prop=extracts&explaintext=1``) for a
    clean plaintext lead — NOT ``clean_wikitext`` from the sibling module, which is
    tuned for the ``{{Current events}}`` template, not full articles.
  * ``redirects=1`` absorbs spelling/title slop ("Abelardo de la Espriella" →
    whatever the canonical title is); a ``missing`` page is the existence check.
  * An **opensearch fallback** rescues a subject whose exact title doesn't exist —
    now viable because the extractor already produced a clean query string (the
    thing a dumb script could never do from the raw ask prose).
  * Length is capped — articles are long and this text feeds the next cycle's
    context window.

Stdlib only — no third-party dependencies (matches fetch_current_events.py).

Usage::

    python fetch_article.py "Abelardo de la Espriella"
    python fetch_article.py "Strait of Hormuz" --full --print
    python fetch_article.py "some obscure thing" --no-search   # exact title only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ProjectAva-TIL/0.1 (https://github.com/; lookup fetcher)"
# One subdir per source kind (`lookups/` beside `news/` and `wander/`), so the three
# stay distinguishable and no kind can clobber another's filenames.
# Provenance snippets live under the ordered state home (server/data/til), not next to the
# fetch code. __file__ is server/til/fetch_article.py.
LOOKUPS_DIR = Path(__file__).resolve().parent.parent / "data" / "til" / "snippets" / "lookups"

# Default cap on extracted text fed into the next learning cycle (chars).
DEFAULT_MAX_CHARS = 4000


# ── low-level API access ────────────────────────────────────────────────────── #

def _get_json(params: dict, *, timeout: float = 20.0, retries: int = 3):
    """GET the MediaWiki API and return parsed JSON (dict or list).

    Retries transient throttling (HTTP 429 / 503) with short exponential backoff,
    mirroring ``fetch_current_events.fetch_wikitext``. Raises ``RuntimeError`` on
    transport failure. API-level ``error`` payloads are left for the dict-shaped
    callers to interpret (opensearch returns a list, so a generic check here would
    misfire)."""
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 3)   # 3s, 6s, …
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        except Exception as e:  # network, decode, etc.
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"request failed: {e}") from e
    raise RuntimeError("request failed after retries (throttled)")


def article_url(title: str) -> str:
    """Browser URL for an article title (spaces → underscores), for provenance."""
    slug = title.replace(" ", "_")
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(slug, safe=":/_()")


# ── fetch + resolve ─────────────────────────────────────────────────────────── #

def fetch_extract(title: str, *, intro_only: bool = True,
                  max_chars: int = DEFAULT_MAX_CHARS) -> dict | None:
    """Fetch a title's plaintext extract. Returns a record, or None if missing.

    Record: ``{title (canonical), extract, url, redirected_from?}``. ``redirects=1``
    means *title* need not be canonical; the returned ``title`` is whatever the
    article actually is. ``None`` means Wikipedia has no such page (try search).
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    if intro_only:
        params["exintro"] = "1"

    data = _get_json(params)
    if not isinstance(data, dict) or "error" in data:
        info = (data.get("error", {}) or {}).get("info", "unknown error") \
            if isinstance(data, dict) else "non-dict response"
        raise RuntimeError(f"extracts API error: {info}")

    pages = (data.get("query", {}) or {}).get("pages") or []
    if not pages:
        return None
    page = pages[0]
    if page.get("missing"):
        return None

    extract = (page.get("extract") or "").strip()
    if not extract:
        return None
    extract = _truncate(extract, max_chars)

    record = {
        "title": page.get("title") or title,
        "extract": extract,
        "url": article_url(page.get("title") or title),
    }
    redirects = (data.get("query", {}) or {}).get("redirects") or []
    if redirects:
        # The first hop's source is what we asked for if it differed.
        frm = redirects[0].get("from")
        if frm and frm != record["title"]:
            record["redirected_from"] = frm
    return record


def search_title(query: str, *, limit: int = 1) -> str | None:
    """Best-matching article title for a free-text query (opensearch). None if no hit."""
    params = {
        "action": "opensearch",
        "format": "json",
        "search": query,
        "limit": str(limit),
        "namespace": "0",
    }
    data = _get_json(params)
    # opensearch returns [query, [titles], [descriptions], [urls]].
    if not isinstance(data, list) or len(data) < 2:
        return None
    titles = data[1] or []
    return titles[0] if titles else None


def resolve_subject(subject: str, *, allow_search: bool = True,
                    intro_only: bool = True,
                    max_chars: int = DEFAULT_MAX_CHARS) -> dict | None:
    """Resolve a subject to an article record: direct title, then search fallback.

    Returns the ``fetch_extract`` record with an added ``via`` ("direct"/"search"),
    or ``None`` if nothing resolves. The two-step (exact title → search) is the
    whole point: the agentic extractor gives a clean subject, exact lookup catches
    the common case, and search catches the rest."""
    subject = (subject or "").strip()
    if not subject:
        return None

    direct = fetch_extract(subject, intro_only=intro_only, max_chars=max_chars)
    if direct is not None:
        direct["via"] = "direct"
        direct["subject"] = subject
        return direct

    if not allow_search:
        return None

    hit = search_title(subject)
    if not hit:
        return None
    found = fetch_extract(hit, intro_only=intro_only, max_chars=max_chars)
    if found is None:
        return None
    found["via"] = "search"
    found["subject"] = subject
    return found


# ── snippet building / writing ──────────────────────────────────────────────── #

def build_article_snippet(subject: str, **kwargs) -> dict | None:
    """Resolve *subject* and wrap it as a TIL snippet record (or None if unresolved).

    The shape mirrors ``fetch_current_events.build_snippet`` so the downstream
    learning pass treats a looked-up article exactly like a fetched digest."""
    resolved = resolve_subject(subject, **kwargs)
    if resolved is None:
        return None
    record = {
        "kind": "lookup",
        "source": "wikipedia:article",
        "date": date.today().isoformat(),
        "subject": subject,
        "title": resolved["title"],
        "source_url": resolved["url"],
        "via": resolved["via"],
        "fetched_at": datetime.now().astimezone().isoformat(),
        "text": resolved["extract"],
        "sources": [resolved["url"]],
    }
    if resolved.get("redirected_from"):
        record["redirected_from"] = resolved["redirected_from"]
    return record


def write_article_snippet(record: dict, out_dir: Path = LOOKUPS_DIR) -> tuple[Path, Path]:
    """Write the article snippet (.txt) and a provenance sidecar (.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_stamp(record)}_{_slug(record['title'])}"
    txt_path, json_path = _unique_pair(out_dir, stem)

    header = (
        f"# Looked up — {record['subject']} → {record['title']}\n"
        f"# {record['source_url']}\n\n"
    )
    txt_path.write_text(header + record["text"] + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return txt_path, json_path


# ── helpers ─────────────────────────────────────────────────────────────────── #

# \w is Unicode-aware, so a Cyrillic (or any non-ASCII) title survives into the slug
# instead of collapsing to "untitled" and colliding with every other such lookup.
_SLUG_RE = re.compile(r"\W+", re.UNICODE)


def _slug(text: str, *, limit: int = 60) -> str:
    s = _SLUG_RE.sub("_", text).strip("_")
    return (s[:limit].rstrip("_") or "untitled")


def _stamp(record: dict) -> str:
    """Second-granularity, filesystem-safe timestamp from the record's fetch time —
    keys the stem so same-day lookups don't overwrite each other. Falls back to
    wall-clock now when ``fetched_at`` is missing/malformed."""
    try:
        dt = datetime.fromisoformat(record.get("fetched_at") or "")
    except (TypeError, ValueError):
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def _unique_pair(out_dir: Path, stem: str) -> tuple[Path, Path]:
    """(.txt, .json) paths for *stem*, disambiguated ``-2``/``-3``/… if one exists."""
    candidate, n = stem, 2
    while (out_dir / f"{candidate}.txt").exists() or (out_dir / f"{candidate}.json").exists():
        candidate, n = f"{stem}-{n}", n + 1
    return out_dir / f"{candidate}.txt", out_dir / f"{candidate}.json"


def _truncate(text: str, max_chars: int) -> str:
    """Cut to <= max_chars at a paragraph/sentence boundary, adding an ellipsis."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Prefer a paragraph break, then a sentence end, then a hard cut.
    for sep in ("\n\n", ". ", "\n"):
        idx = window.rfind(sep)
        if idx > max_chars // 2:
            return window[:idx + (len(sep) if sep != ". " else 1)].rstrip() + " …"
    return window.rstrip() + " …"


# ── CLI ─────────────────────────────────────────────────────────────────────── #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subject", help="article subject to look up")
    parser.add_argument("--full", action="store_true",
                        help="fetch the whole article, not just the intro lead")
    parser.add_argument("--no-search", dest="search", action="store_false",
                        help="exact title only; do not fall back to opensearch")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help=f"truncate extract to N chars (default {DEFAULT_MAX_CHARS})")
    parser.add_argument("--print", dest="echo", action="store_true",
                        help="also print the extract to stdout")
    args = parser.parse_args(argv)

    try:
        record = build_article_snippet(
            args.subject, allow_search=args.search,
            intro_only=not args.full, max_chars=args.max_chars,
        )
    except RuntimeError as e:
        print(f"[lookup] fetch error: {e}", file=sys.stderr)
        return 1
    if record is None:
        print(f"[lookup] no Wikipedia article for {args.subject!r}", file=sys.stderr)
        return 2

    txt_path, _ = write_article_snippet(record)
    via = record["via"]
    redir = f" (redirected from {record['redirected_from']!r})" if record.get("redirected_from") else ""
    print(f"[lookup] {args.subject!r} → {record['title']!r} via {via}{redir}: "
          f"{len(record['text'])} chars → {txt_path}")
    if args.echo:
        print("\n" + record["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
