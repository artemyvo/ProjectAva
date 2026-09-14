#!/usr/bin/env python3
"""Fetch a random page from any MediaWiki — the 'wander' serendipity source.

Generalizes ``fetch_article.py`` (Wikipedia-only subject lookup) to *any* MediaWiki
instance, for the entropy/serendipity side of Ava's learning: instead of a day's
news, pull a random article from a user-approved wiki and let her reflect on it.
The set of approved wikis lives in ``wiki_sources.json`` (per language); only
``enabled`` sources are ever drawn from.

Everything is standard MediaWiki API, so it works across installs:
  * random page  — ``action=query&list=random&rnnamespace=0&rnfilterredir=nonredirects``
  * lead text    — ``prop=extracts`` when the wiki has the TextExtracts extension
                   (Wikimedia does); otherwise fall back to ``action=parse`` lead
                   HTML stripped to text (most third-party wikis need this).
The only per-wiki config is the ``api.php`` base URL (``/w/api.php`` on Wikimedia,
often ``/api.php`` elsewhere).

Stdlib only — no third-party dependencies (matches fetch_current_events.py).

Usage::

    python fetch_wiki.py --list                       # show configured sources
    python fetch_wiki.py --api https://lurkmore.media/api.php --print
    python fetch_wiki.py --lang ru --print            # random enabled ru source
    python fetch_wiki.py --source "Urban Culture" --print
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

USER_AGENT = "ProjectAva-TIL/0.1 (https://github.com/; wiki wander fetcher)"
SOURCES_PATH = Path(__file__).resolve().parent / "wiki_sources.json"
# Provenance snippets live under the ordered state home (server/data/til), not next to the
# fetch code — the same tree as the durable wander corpus. __file__ is server/til/fetch_wiki.py.
WANDER_DIR = Path(__file__).resolve().parent.parent / "data" / "til" / "snippets" / "wander"

DEFAULT_MAX_CHARS = 4000
DEFAULT_MIN_CHARS = 200       # skip stubs/disambig — re-roll a random page below this
DEFAULT_RANDOM_TRIES = 6      # how many random pages to try before giving up


# ── low-level API access ────────────────────────────────────────────────────── #

def _get_json(api: str, params: dict, *, timeout: float = 20.0, retries: int = 3,
              no_cache: bool = False):
    """GET a MediaWiki API endpoint and return parsed JSON (dict or list).

    Retries transient throttling (429/503) with exponential backoff. Raises
    ``RuntimeError`` on transport failure. API-level ``error`` payloads are left for
    callers to interpret. ``no_cache`` adds request headers asking proxies not to
    serve a cached response — needed for ``list=random`` on small wikis behind a
    caching CDN, which otherwise return the same 'random' page every time."""
    url = api + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": USER_AGENT}
    if no_cache:
        headers["Cache-Control"] = "no-cache"
        headers["Pragma"] = "no-cache"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"request failed: {e}") from e
    raise RuntimeError("request failed after retries (throttled)")


# ── HTML → text (parse fallback for wikis without TextExtracts) ──────────────── #

_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_TAG_RE = re.compile(r"<[^>]+>")
_EDIT_RE = re.compile(r"\[\s*edit\s*\]", re.IGNORECASE)
_REF_RE = re.compile(r"\[\d+\]")            # [1] reference superscripts
_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def _html_to_text(markup: str) -> str:
    """Flatten a MediaWiki lead-section HTML blob to readable plain text."""
    if not markup:
        return ""
    text = _SCRIPT_STYLE_RE.sub("", markup)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _EDIT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for sep in ("\n\n", ". ", "\n"):
        idx = window.rfind(sep)
        if idx > max_chars // 2:
            return window[:idx + (len(sep) if sep != ". " else 1)].rstrip() + " …"
    return window.rstrip() + " …"


# ── random page + lead fetch (generic) ──────────────────────────────────────── #

def random_title(api: str, *, namespace: int = 0) -> str | None:
    """A random non-redirect article title from *api*, or None.

    Cache-busted: ``maxage/smaxage=0`` plus a per-call nonce and no-cache headers,
    because small wikis behind a caching proxy serve the identical ``list=random``
    URL from cache — returning the same 'random' page on every call. Pulls a small
    batch and picks one client-side for extra variety even if a cache slips through."""
    data = _get_json(api, {
        "action": "query", "format": "json", "formatversion": "2",
        "list": "random", "rnnamespace": str(namespace),
        "rnlimit": "10", "rnfilterredir": "nonredirects",
        "maxage": "0", "smaxage": "0",
        "_cb": f"{random.getrandbits(48):x}",   # cache-buster for URL-keyed proxies
    }, no_cache=True)
    if not isinstance(data, dict):
        return None
    rnd = (data.get("query", {}) or {}).get("random") or []
    titles = [r.get("title") for r in rnd if r.get("title")]
    return random.choice(titles) if titles else None


def _error_mentions_extracts(data) -> bool:
    if not isinstance(data, dict):
        return False
    info = (data.get("error", {}) or {}).get("info", "") if "error" in data else ""
    return "extract" in info.lower()


def fetch_lead(api: str, title: str, *, intro_only: bool = True,
               max_chars: int = DEFAULT_MAX_CHARS) -> dict | None:
    """Fetch a page's lead text + canonical URL. Returns a record or None if missing.

    Tries ``prop=extracts`` (TextExtracts) for a clean plaintext lead; on a wiki
    without that extension, falls back to ``action=parse`` lead-section HTML stripped
    to text. ``prop=info&inprop=url`` gives the canonical ``fullurl`` regardless of
    the wiki's article-path layout."""
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts|info", "inprop": "url", "redirects": "1",
        "explaintext": "1", "titles": title,
    }
    if intro_only:
        params["exintro"] = "1"

    data = _get_json(api, params)
    if _error_mentions_extracts(data):
        # Wiki lacks TextExtracts — re-query for info only, then parse-fallback below.
        data = _get_json(api, {
            "action": "query", "format": "json", "formatversion": "2",
            "prop": "info", "inprop": "url", "redirects": "1", "titles": title,
        })
    if not isinstance(data, dict) or "error" in data:
        info = (data.get("error", {}) or {}).get("info", "unknown") if isinstance(data, dict) else "non-dict"
        raise RuntimeError(f"API error: {info}")

    pages = (data.get("query", {}) or {}).get("pages") or []
    if not pages or pages[0].get("missing"):
        return None
    page = pages[0]
    title_c = page.get("title") or title
    url = page.get("fullurl") or ""
    extract = (page.get("extract") or "").strip()

    if not extract:
        extract = _parse_lead(api, title_c, intro_only=intro_only)
    if not extract:
        return None
    return {"title": title_c, "extract": _truncate(extract, max_chars), "url": url}


def _parse_lead(api: str, title: str, *, intro_only: bool = True) -> str:
    """Fallback: render the article to HTML via action=parse, strip to text.

    With ``intro_only`` parse just the lead section (section 0); otherwise the whole
    page (more text — what the random 'wander' wants)."""
    params = {
        "action": "parse", "format": "json", "formatversion": "2",
        "page": title, "prop": "text", "redirects": "1",
    }
    if intro_only:
        params["section"] = "0"
    try:
        data = _get_json(api, params)
    except RuntimeError:
        return ""
    if not isinstance(data, dict) or "error" in data:
        return ""
    text_html = (data.get("parse", {}) or {}).get("text")
    # formatversion=2 returns a string; older returns {"*": "..."}.
    if isinstance(text_html, dict):
        text_html = text_html.get("*", "")
    return _html_to_text(text_html or "")


def _candidate_apis(api: str) -> list[str]:
    """The configured ``api.php`` plus its common-layout sibling.

    The usual MediaWiki path ambiguity is ``/w/api.php`` (Wikimedia) vs ``/api.php``
    (many third-party installs) — and a source pointed at the wrong one hard-404s. So
    a fetch tries the configured path first, then the swap, and self-heals instead of
    failing. A non-standard api path (not ending in ``api.php``) yields just itself."""
    api = (api or "").strip()
    if api.endswith("/w/api.php"):
        alt = api[: -len("/w/api.php")] + "/api.php"
    elif api.endswith("/api.php"):
        alt = api[: -len("/api.php")] + "/w/api.php"
    else:
        alt = None
    return [api, alt] if alt and alt != api else [api]


def fetch_random(api: str, *, namespace: int = 0, min_chars: int = DEFAULT_MIN_CHARS,
                 max_chars: int = DEFAULT_MAX_CHARS, intro_only: bool = False,
                 tries: int = DEFAULT_RANDOM_TRIES) -> dict | None:
    """Fetch one substantive random page from *api* (re-rolls past stubs/disambig).

    ``intro_only`` defaults False here: a random page is for reflection, so the
    whole article is wanted (truncated to *max_chars*), not just the one-sentence
    lead an ``exintro`` extract returns — that short lead would fail *min_chars* on
    many wikis and reject otherwise-substantive pages.

    Resilient two ways: it tries both ``api.php`` path variants (see
    :func:`_candidate_apis`) so a mis-pathed source self-heals, and a transient
    per-page error re-rolls another random page rather than aborting. Only when *no*
    path variant responds at all does the underlying error propagate."""
    last_err: RuntimeError | None = None
    any_path_ok = False
    for cand in _candidate_apis(api):
        path_ok = False
        for _ in range(max(1, tries)):
            try:
                title = random_title(cand, namespace=namespace)
            except RuntimeError as e:
                last_err = e            # path-level failure (e.g. 404) — try the sibling
                break
            path_ok = any_path_ok = True
            if not title:
                continue
            try:
                rec = fetch_lead(cand, title, intro_only=intro_only, max_chars=max_chars)
            except RuntimeError as e:
                last_err = e            # this page errored — re-roll another random page
                continue
            if rec and len(rec["extract"]) >= min_chars:
                return rec
        if path_ok:
            break                       # path works; the sibling won't find more pages
    if not any_path_ok and last_err is not None:
        raise last_err                  # source is genuinely unreachable on either path
    return None                         # path worked but every roll was a stub/short


# ── sources config ──────────────────────────────────────────────────────────── #

def load_sources(path: Path = SOURCES_PATH) -> dict:
    """Load the per-language wiki-sources config. Returns {} if absent/unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def enabled_sources(config: dict, *, lang: str | None = None) -> list[dict]:
    """Flatten the config to the list of enabled sources, optionally one language.

    Each returned source dict is annotated with its ``lang`` bucket. Keys beginning
    with ``_`` (e.g. ``_comment``) are skipped."""
    out: list[dict] = []
    for bucket, sources in config.items():
        if bucket.startswith("_") or (lang and bucket != lang):
            continue
        if not isinstance(sources, list):
            continue
        for s in sources:
            if isinstance(s, dict) and s.get("enabled") and s.get("api"):
                out.append({**s, "lang": bucket})
    return out


def pick_source(config: dict, *, lang: str | None = None,
                name: str | None = None) -> dict | None:
    """Pick one source: by *name* (any bucket, enabled or not) else a random enabled."""
    if name:
        for bucket, sources in config.items():
            if bucket.startswith("_") or not isinstance(sources, list):
                continue
            for s in sources:
                if isinstance(s, dict) and s.get("name") == name:
                    return {**s, "lang": bucket}
        return None
    pool = enabled_sources(config, lang=lang)
    return random.choice(pool) if pool else None


# ── snippet building / writing ──────────────────────────────────────────────── #

# \w is Unicode-aware, so a Cyrillic (or any non-ASCII) title survives into the slug
# instead of collapsing to "untitled" and colliding with every other Russian article.
_SLUG_RE = re.compile(r"\W+", re.UNICODE)


def _slug(text: str, *, limit: int = 50) -> str:
    s = _SLUG_RE.sub("_", text).strip("_")
    return (s[:limit].rstrip("_") or "untitled")


def _stamp(record: dict) -> str:
    """Second-granularity, filesystem-safe timestamp from the record's fetch time.

    Keys the stem so two fetches from the same source on the same day don't overwrite
    each other (the old day-granularity ``date`` did). Falls back to wall-clock now
    when ``fetched_at`` is missing/malformed."""
    try:
        dt = datetime.fromisoformat(record.get("fetched_at") or "")
    except (TypeError, ValueError):
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def _unique_pair(out_dir: Path, stem: str) -> tuple[Path, Path]:
    """(.txt, .json) paths for *stem*, disambiguated ``-2``/``-3``/… if one exists —
    the final guard for a true same-second, same-title collision."""
    candidate, n = stem, 2
    while (out_dir / f"{candidate}.txt").exists() or (out_dir / f"{candidate}.json").exists():
        candidate, n = f"{stem}-{n}", n + 1
    return out_dir / f"{candidate}.txt", out_dir / f"{candidate}.json"


def build_random_snippet(source: dict, **kwargs) -> dict | None:
    """Fetch a random page from *source* and wrap it as a TIL snippet record.

    Shape mirrors ``fetch_current_events.build_snippet`` so the learning pass can
    treat a random wander page like any other digest."""
    rec = fetch_random(source["api"], namespace=int(source.get("namespace", 0)), **kwargs)
    if rec is None:
        return None
    return {
        "kind": "wiki_random",
        "source": "wiki:random",
        "wiki": source.get("name", source["api"]),
        "lang": source.get("lang", ""),
        "date": date.today().isoformat(),
        "title": rec["title"],
        "source_url": rec["url"],
        "fetched_at": datetime.now().astimezone().isoformat(),
        "text": rec["extract"],
        "sources": [rec["url"]] if rec["url"] else [],
    }


def write_random_snippet(record: dict, out_dir: Path = WANDER_DIR) -> tuple[Path, Path]:
    """Write the wander snippet (.txt) and a provenance sidecar (.json).

    The stem is timestamped (second granularity) + Unicode-slugged, so two fetches from
    the same source on the same day — or any Cyrillic/non-ASCII title — no longer collapse
    to one name and overwrite each other."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_stamp(record)}_{_slug(record.get('wiki', ''))}_{_slug(record['title'])}"
    txt_path, json_path = _unique_pair(out_dir, stem)
    header = (
        f"# Wandered into — {record['title']} ({record.get('wiki', '')})\n"
        f"# {record['source_url']}\n\n"
    )
    txt_path.write_text(header + record["text"] + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return txt_path, json_path


# ── CLI ─────────────────────────────────────────────────────────────────────── #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(SOURCES_PATH), help="wiki_sources.json path")
    parser.add_argument("--api", help="explicit api.php base (overrides config selection)")
    parser.add_argument("--lang", help="restrict random source pick to this language bucket")
    parser.add_argument("--source", help="pick a configured source by exact name")
    parser.add_argument("--list", action="store_true", help="list configured sources and exit")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--no-write", action="store_true", help="don't write a snippet")
    parser.add_argument("--print", dest="echo", action="store_true", help="echo the text")
    args = parser.parse_args(argv)

    config = load_sources(Path(args.config))

    if args.list:
        if not config:
            print(f"[wiki] no config at {args.config}", file=sys.stderr)
            return 2
        for bucket, sources in config.items():
            if bucket.startswith("_") or not isinstance(sources, list):
                continue
            print(f"[{bucket}]")
            for s in sources:
                mark = "on " if s.get("enabled") else "off"
                note = f"  — {s['note']}" if s.get("note") else ""
                print(f"  [{mark}] {s.get('name','?'):20} {s.get('api','')}{note}")
        return 0

    if args.api:
        source = {"name": args.api, "api": args.api, "lang": args.lang or ""}
    else:
        source = pick_source(config, lang=args.lang, name=args.source)
    if not source:
        print("[wiki] no source selected (none enabled? bad --source/--lang?)", file=sys.stderr)
        return 2

    try:
        record = build_random_snippet(source, min_chars=args.min_chars, max_chars=args.max_chars)
    except RuntimeError as e:
        print(f"[wiki] fetch error from {source.get('name')}: {e}", file=sys.stderr)
        return 1
    if record is None:
        print(f"[wiki] no substantive random page from {source.get('name')} "
              f"(all tries were stubs/missing?)", file=sys.stderr)
        return 2

    if not args.no_write:
        txt_path, _ = write_random_snippet(record)
        where = f" → {txt_path}"
    else:
        where = ""
    print(f"[wiki] {record['wiki']}: {record['title']!r} "
          f"({len(record['text'])} chars){where}")
    if args.echo:
        print("\n" + record["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
