#!/usr/bin/env python3
"""Fetch a day's worth of world events from Wikipedia's Current events portal.

First gradual step toward the "Today I Learned" / serendipity source (see
SCRATCHPAD.md → "The Lookup Agent & the 'Today I Learned' pass"): a network-only,
GPU-free fetcher that pulls a single day's news digest and writes it as a clean
text snippet plus a JSON sidecar carrying provenance. Nothing here touches the
model, RAG, or the reflection loop yet — it only deposits text into
``server/data/til/snippets/news/`` for a later learning pass to read.

Wikipedia keeps one subpage per calendar day:

    Portal:Current events/2026 June 20      (full month name, NO leading zero)

The MediaWiki ``parse`` API returns that page's wikitext directly, so fetching a
specific date is purely a title-formatting problem. Today's page is usually not
populated until the day is over, so the default target is *yesterday*.

Usage::

    python fetch_current_events.py                 # yesterday (default)
    python fetch_current_events.py --date 2026-06-20
    python fetch_current_events.py --days-back 2
    python fetch_current_events.py --print         # also echo the digest to stdout

Stdlib only — no third-party dependencies (matches the tools/ ethos).
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
from datetime import date, datetime, timedelta
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ProjectAva-TIL/0.1 (https://github.com/; current-events fetcher)"
# Provenance snippets live under the ordered state home (server/data/til), not next to the
# fetch code — the same move the wander/lookup fetchers already made. __file__ is
# server/til/fetch_current_events.py. One subdir per source kind (`news/` beside `wander/`
# and `lookups/`), so the three stay distinguishable and no kind can clobber another's
# filenames.
SNIPPETS_DIR = Path(__file__).resolve().parent.parent / "data" / "til" / "snippets" / "news"


# ── date / title helpers ──────────────────────────────────────────────────── #

def page_title(d: date) -> str:
    """Wikipedia subpage title for *d*, e.g. ``Portal:Current events/2026 June 20``.

    The day is rendered WITHOUT a leading zero (``June 9``, not ``June 09``) —
    that is the exact form Wikipedia uses for these subpages.
    """
    return f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}"


def human_url(d: date) -> str:
    """Browser URL for the same page (spaces → underscores), for provenance."""
    slug = f"Portal:Current_events/{d.year}_{d.strftime('%B')}_{d.day}"
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(slug, safe=":/_")


# ── fetch ─────────────────────────────────────────────────────────────────── #

def fetch_wikitext(d: date, *, timeout: float = 20.0, retries: int = 3) -> str:
    """Return the raw wikitext of the given day's portal subpage.

    Retries on transient throttling (HTTP 429 / 503) with a short exponential
    backoff — a burst of requests against the Wikipedia API is rate-limited.
    Raises ``LookupError`` if the page does not exist (e.g. a future/unpopulated
    date) and ``RuntimeError`` on transport/API errors.
    """
    params = {
        "action": "parse",
        "page": page_title(d),
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    data = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 3)   # 3s, 6s, …
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        except Exception as e:  # network, decode, etc.
            raise RuntimeError(f"request failed: {e}") from e
    if data is None:
        raise RuntimeError("request failed after retries (throttled)")

    if "error" in data:
        info = data["error"].get("info", "unknown error")
        # "missingtitle" is the API's signal for a non-existent page.
        if data["error"].get("code") == "missingtitle":
            raise LookupError(f"no current-events page for {d.isoformat()} ({info})")
        raise RuntimeError(f"API error: {info}")

    text = data.get("parse", {}).get("wikitext")
    if not text:
        raise LookupError(f"empty wikitext for {d.isoformat()}")
    return text


# ── wikitext → readable digest ────────────────────────────────────────────── #

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")            # innermost template
_EXT_LINK_RE = re.compile(r"\[(https?://\S+)\s+([^\]]+)\]")  # [url display]
_BARE_LINK_RE = re.compile(r"\[https?://\S+\]")
_PIPED_WIKILINK_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")  # [[target|display]]
_WIKILINK_RE = re.compile(r"\[\[([^\]]*)\]\]")                    # [[target]]
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^'''(.+?)'''$")
_WS_RE = re.compile(r"[ \t]+")

# Wikilink namespaces that aren't readable articles (skip when collecting links).
_NONARTICLE_NS = (
    "file:", "image:", "category:", "wikipedia:", "help:", "template:",
    "portal:", "wikt:", "media:", "special:", "s:", "commons:",
)

# Cap on how many referenced titles we surface to the learning pass (token budget).
_MAX_REFERENCED_LINKS = 80


def _is_article_link(target: str) -> bool:
    t = (target or "").strip()
    return bool(t) and not t.lower().startswith(_NONARTICLE_NS)


def _clean_link_target(target: str) -> str:
    """Normalize a wikilink target to its article title: drop #anchor, underscores."""
    t = (target or "").split("#", 1)[0].strip()
    return t.replace("_", " ")


def _extract_content(wikitext: str) -> str:
    """Pull the news body out of the ``{{Current events|…|content=…}}`` wrapper."""
    idx = wikitext.find("content=")
    body = wikitext[idx + len("content="):] if idx != -1 else wikitext
    body = _COMMENT_RE.sub("", body)
    return body.rstrip().rstrip("}").rstrip()


def _strip_inline(text: str) -> tuple[str, list[str], list[str]]:
    """Resolve links/templates/markup in one line.

    Returns ``(clean_text, urls, links)`` where *links* is the article titles of the
    wikilinks in this line (targets, not display text) — the canonical names a later
    lookup can fetch directly, so Ava needn't guess them from her own knowledge.
    """
    urls: list[str] = []
    links: list[str] = []

    # Nested templates: peel innermost until none remain.
    prev = None
    while prev != text:
        prev = text
        text = _TEMPLATE_RE.sub("", text)

    # External citations: keep the display text (usually the source name), and
    # collect the URL for provenance.
    def _ext(m: "re.Match[str]") -> str:
        urls.append(m.group(1))
        return m.group(2).strip()

    text = _EXT_LINK_RE.sub(_ext, text)
    text = _BARE_LINK_RE.sub("", text)

    # Wikilinks: render the display half (piped) or the target (plain), and collect
    # the target's article title for the lookup loop.
    def _piped(m: "re.Match[str]") -> str:
        target, display = m.group(1), m.group(2)
        if _is_article_link(target):
            links.append(_clean_link_target(target))
        return display.strip()

    def _plain(m: "re.Match[str]") -> str:
        target = m.group(1)
        title = _clean_link_target(target)
        if _is_article_link(target):
            links.append(title)
        return title

    text = _PIPED_WIKILINK_RE.sub(_piped, text)   # piped first (plain would over-match)
    text = _WIKILINK_RE.sub(_plain, text)

    # Bold/italic apostrophe markup and any stray HTML tags.
    text = text.replace("'''", "").replace("''", "")
    text = _TAG_RE.sub("", text)

    text = _WS_RE.sub(" ", text).strip()
    return text, urls, links


def clean_wikitext(wikitext: str) -> tuple[str, list[str], list[str]]:
    """Flatten the day's wikitext into an indented plain-text digest.

    Returns ``(digest, sources, links)`` — *sources* is the de-duplicated citation
    URLs and *links* the de-duplicated wikilink article titles, both in first-seen
    order. *links* are the canonical Wikipedia titles the lookup loop can fetch
    directly (see the references block appended in ``build_snippet``).
    """
    body = _extract_content(wikitext)
    out_lines: list[str] = []
    sources: list[str] = []
    links: list[str] = []
    seen_urls: set[str] = set()
    seen_links: set[str] = set()

    def _record(urls: list[str]) -> None:
        for u in urls:
            if u not in seen_urls:
                seen_urls.add(u)
                sources.append(u)

    def _record_links(ls: list[str]) -> None:
        for l in ls:
            if l and l not in seen_links:
                seen_links.add(l)
                links.append(l)

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        heading = _HEADING_RE.match(line.strip())
        if heading:
            text, urls, ls = _strip_inline(heading.group(1))
            _record(urls)
            _record_links(ls)
            out_lines.append("")
            out_lines.append(f"# {text}")
            continue

        m = re.match(r"^(\*+)\s*(.*)$", line)
        if m:
            depth = len(m.group(1)) - 1
            text, urls, ls = _strip_inline(m.group(2))
            _record(urls)
            _record_links(ls)
            if text:
                out_lines.append("  " * depth + "- " + text)
            continue

        # Anything else: a stray paragraph line — keep it, flattened.
        text, urls, ls = _strip_inline(line)
        _record(urls)
        _record_links(ls)
        if text:
            out_lines.append(text)

    digest = "\n".join(out_lines).strip() + "\n"
    return digest, sources, links


# ── orchestration ─────────────────────────────────────────────────────────── #

def _references_block(links: list[str], limit: int = _MAX_REFERENCED_LINKS) -> str:
    """A compact, capped list of referenced article titles for the learning pass.

    Appended to the digest so Ava can bind an ``[ask:search]`` to an exact title
    (``(lookup: ...)``) instead of leaving the canonical name to be guessed later.
    """
    if not links:
        return ""
    shown = links[:limit]
    body = "; ".join(shown)
    more = f" (+{len(links) - limit} more)" if len(links) > limit else ""
    return f"\n\n# Referenced Wikipedia articles\n{body}{more}\n"


def build_snippet(d: date) -> dict:
    """Fetch + clean one day; return a record ready to write (and reuse later)."""
    wikitext = fetch_wikitext(d)
    digest, sources, links = clean_wikitext(wikitext)
    text = digest + _references_block(links)
    return {
        "kind": "serendipity",
        "source": "wikipedia:current_events",
        "date": d.isoformat(),
        "title": page_title(d),
        "source_url": human_url(d),
        "fetched_at": datetime.now().astimezone().isoformat(),
        "text": text,
        "sources": sources,
        "links": links,
    }


def write_snippet(record: dict, out_dir: Path = SNIPPETS_DIR) -> tuple[Path, Path]:
    """Write the digest (.txt) and a provenance sidecar (.json). Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = record["date"]
    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"

    header = (
        f"# Today I Learned — {record['date']} "
        f"(via Wikipedia Current events)\n"
        f"# {record['source_url']}\n\n"
    )
    txt_path.write_text(header + record["text"], encoding="utf-8")
    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return txt_path, json_path


def _target_date(args: argparse.Namespace) -> date:
    if args.date:
        return datetime.strptime(args.date, "%Y-%m-%d").date()
    return date.today() - timedelta(days=max(1, args.days_back))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="explicit date YYYY-MM-DD (overrides --days-back)")
    parser.add_argument("--days-back", type=int, default=1,
                        help="how many days before today to fetch (default 1 = yesterday)")
    parser.add_argument("--print", dest="echo", action="store_true",
                        help="also print the digest to stdout")
    args = parser.parse_args(argv)

    d = _target_date(args)
    try:
        record = build_snippet(d)
    except LookupError as e:
        print(f"[til] {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[til] fetch error: {e}", file=sys.stderr)
        return 1

    txt_path, json_path = write_snippet(record)
    print(f"[til] {d.isoformat()}: {len(record['text'])} chars, "
          f"{len(record['sources'])} sources → {txt_path}")
    if args.echo:
        print("\n" + record["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
