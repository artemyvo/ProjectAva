"""TIL / Wikipedia "wander" learning subsystem, extracted from server.py.

Ava's self-directed learning off her own transcripts:

  * **TIL fetch / Learn** — pull a day's Wikipedia Current-events digest and run a
    dry-run learning-reflection pass over it (``handle_til_fetch``).
  * **Lookup** — resolve Ava's own open ``[ask:search]`` questions by fetching the
    articles they point at, then reflect on the digest (``handle_til_lookup``).
  * **Wander** — reflect on a random page from a user-approved wiki, or a page the
    operator explicitly visits, manual (free) or autonomous (rationed by a token
    budget) (``handle_til_wander`` / ``run_autonomous_wander_blocking``).
  * **Apply** — persist a dry-run pass's modifiers into live memory
    (``handle_til_apply`` → ``_apply_learning_text_live``).

The reflection *ingestion* phase reuses two of these as sources
(``ingest_news`` / ``ingest_lookup``), called by ``server._run_ingestion_phase``.

Wiring: this module is import-light and never imports ``server``. The GPU/session
capabilities it needs (the WebSocket ``send``, the GPU ``executor``, the model
``backend``, the RAG/writer accessors, the reflect/agentic generate factories,
the live reflection-run flag, and the on-disk paths) are injected once at startup
via :func:`configure`. Session/model state is read directly from
``core.runtime_state``. The moved code is otherwise verbatim.
"""
from __future__ import annotations

import asyncio
import html
import json
import re
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, reflect_window as _reflect_window
from core.reflection_memory import ReflectionMemory
from core.llm_shared import ensure_chat_template
from core import activity_log
from core import reasoning_text
from core import til_facts
from core import til_gist

# ── Injected server capabilities (populated by configure(); see module docstring) ──
_send: Callable = None                       # async _send(ws, msg)
_executor: Any = None                        # ThreadPoolExecutor (single GPU worker)
_backend: Any = None                         # UnslothBackend
_get_rag: Callable = None
_get_reflection_writer: Callable = None
_make_sync_reflect_generate: Callable = None
_make_agentic_generate: Callable = None
_read_user_tokens: Callable = None           # cumulative user-token counter (budget)
_reflection_active: Callable = None          # () -> bool, live reflection-run flag
_WANDER_TOKENS_PER: int = 1000               # user tokens that earn one autonomous wander
_MEMORY_DIR: Path = None
_CONSOLIDATION_DIR: Path = None
_PROMPTS_DIR: Path = None
_TIL_DIR: Path = None                        # server/til/ — CODE only (the fetch_* scripts);
                                             # their output lands in server/data/til/snippets/
_TIL_SNIPPETS_DIR: Path = None               # …that output dir, derived in configure()

# Derived from _MEMORY_DIR in configure().
_WIKI_BUDGET_FILE: Path = None
_WANDER_LOG_FILE: Path = None

# Most-recent wander exchange, held for "Apply learning" -> SFT learning dataset
# (Ambient Enculturation). One slot: the UI previews a single wander at a time, and
# a new wander or a consuming Apply replaces/clears it.
_last_wander_exchange: Optional[dict] = None


def configure(*, send, executor, backend, get_rag, get_reflection_writer,
              make_sync_reflect_generate, make_agentic_generate, read_user_tokens,
              reflection_active, wander_tokens_per, memory_dir, consolidation_dir,
              prompts_dir, til_dir) -> None:
    """Wire in the server capabilities the moved TIL/wander code depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    """
    global _send, _executor, _backend, _get_rag, _get_reflection_writer
    global _make_sync_reflect_generate, _make_agentic_generate, _read_user_tokens
    global _reflection_active, _WANDER_TOKENS_PER
    global _MEMORY_DIR, _CONSOLIDATION_DIR, _PROMPTS_DIR, _TIL_DIR
    global _WIKI_BUDGET_FILE, _WANDER_LOG_FILE, _TIL_SNIPPETS_DIR
    _send = send
    _executor = executor
    _backend = backend
    _get_rag = get_rag
    _get_reflection_writer = get_reflection_writer
    _make_sync_reflect_generate = make_sync_reflect_generate
    _make_agentic_generate = make_agentic_generate
    _read_user_tokens = read_user_tokens
    _reflection_active = reflection_active
    _WANDER_TOKENS_PER = wander_tokens_per
    _MEMORY_DIR = memory_dir
    _CONSOLIDATION_DIR = consolidation_dir
    _PROMPTS_DIR = prompts_dir
    _TIL_DIR = til_dir
    # Where the fetchers put their output. Derived rather than passed, from the same
    # `<repo>/server/data/til/snippets/<kind>` the three `fetch_*.py` scripts each resolve
    # off their own __file__ — one place decides the layout, and a snippet's protocol
    # sidecar has to land beside the snippet that produced it.
    _TIL_SNIPPETS_DIR = til_dir.parent / "data" / "til" / "snippets"
    _WIKI_BUDGET_FILE = memory_dir / "wiki_budget.json"
    _WANDER_LOG_FILE = memory_dir / "wander_log.jsonl"


# ══════════════════════════════════════════════════════════════════════════════
# Moved verbatim from server.py (see module docstring for the four substitutions).
# ══════════════════════════════════════════════════════════════════════════════


def _read_wiki_budget() -> dict:
    """Ingestion bookkeeping: {last_news_date, wander_count}. {} if absent."""
    try:
        data = json.loads(_WIKI_BUDGET_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _write_wiki_budget(budget: dict) -> None:
    try:
        _WIKI_BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WIKI_BUDGET_FILE.write_text(json.dumps(budget, ensure_ascii=False),
                                     encoding="utf-8")
    except Exception:
        pass

def _wander_budget_available() -> int:
    """Unconsumed wanders: one per _WANDER_TOKENS_PER user tokens, minus those spent.

    Earned wanders = cumulative user tokens // _WANDER_TOKENS_PER; consumed =
    wander_count in wiki_budget.json. The difference is how many autonomous wanders
    the accrued relationship currently affords (never negative)."""
    earned = _read_user_tokens() // _WANDER_TOKENS_PER
    consumed = int(_read_wiki_budget().get("wander_count", 0) or 0)
    return max(0, earned - consumed)

def _consume_wander_budget() -> int:
    """Mark one autonomous wander as spent (advance wander_count). Returns the new count."""
    budget = _read_wiki_budget()
    budget["wander_count"] = int(budget.get("wander_count", 0) or 0) + 1
    _write_wiki_budget(budget)
    return budget["wander_count"]

def wander_stats() -> dict:
    """Snapshot of the wander token economy for status/debug readouts.

    ``accumulated`` = cumulative live user tokens counted (the earn side);
    ``wanders_consumed`` = autonomous wanders spent; ``consumed`` = the token
    equivalent of those wanders (wanders_consumed × tokens_per); ``available`` =
    unconsumed wanders currently affordable; ``tokens_per`` = the earn rate."""
    accumulated = int(_read_user_tokens() or 0)
    consumed_wanders = int(_read_wiki_budget().get("wander_count", 0) or 0)
    per = int(_WANDER_TOKENS_PER)
    return {
        "accumulated": accumulated,
        "consumed": consumed_wanders * per,
        "wanders_consumed": consumed_wanders,
        "available": max(0, accumulated // per - consumed_wanders),
        "tokens_per": per,
    }

def _append_wander_log(source: dict, rec: dict, auto: bool) -> None:
    """Append one wandered page to the wander log (best-effort; never disrupts a wander)."""
    try:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "wiki": (source or {}).get("name", ""),
            "lang": (source or {}).get("lang", ""),
            "title": (rec or {}).get("title", ""),
            "url": (rec or {}).get("source_url", ""),
            "auto": bool(auto),
        }
        _WANDER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _WANDER_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _read_wander_log(limit: int = 200) -> list[dict]:
    """Read the wander log, most-recent-first, capped to *limit* entries."""
    try:
        lines = _WANDER_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict] = []
    for line in reversed(lines):       # newest first
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out

def _apply_learning_text_live(text: str, source: str, *, truncated: bool = False) -> dict:
    """Route a learning-pass output's modifiers into LIVE memory + refresh RAG.

    The shared "Apply" path: parse the WEIGHTS/RAG/RESOLVED of *text* and write them
    to the live memory/consolidation stores (``write_consolidation`` → RAG memory +
    weights store + ledger anchors), then rebuild the reflection index so they're
    recalled at chat time. Used by the manual Apply button and the automated
    ingestion phase (which has no human to press Apply). Returns the routing counts.

    *truncated* forwards the token-cap flag of the generating pass so ``write_consolidation``
    drops the final (cut mid-string) item rather than persisting a corrupted fact/persona
    fragment. It defaults False for callers that can't supply it (a manual Apply whose
    generation flag was lost across the client round-trip).
    """
    from core.reflection_writer import register_consolidation_anchors
    text = (text or "").strip()
    if not text:
        return {}
    writer = _get_reflection_writer()
    run_id = f"{source}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # No ``source_user``: this is Ava's own reading (``wiki:``/``til:``/``lookup``), not
    # something a person told her. Attribution stays empty, so anything distilled here is
    # ``observed`` and promotes to the weights exactly as it did before attribution.
    summary = writer.write_consolidation(run_id=run_id, source_session=source, text=text,
                                         truncated=bool(truncated))
    register_consolidation_anchors(_CONSOLIDATION_DIR, summary, source)
    try:
        _get_rag().refresh_reflection_memory()
    except Exception:
        pass
    return {k: v for k, v in summary.items() if k != "facts"}

async def handle_get_wander_log(ws) -> None:
    """Return the log of pages Ava wandered into (title + url), newest first, for the Debug tab."""
    try:
        entries = _read_wander_log()
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"get_wander_log failed: {e}"})
        return
    await _send(ws, {"type": "wander_log", "entries": entries})

def _run_til_fetch(date_str: Optional[str] = None) -> dict:
    """Fetch + write one day's Current events digest into ``server/data/til/snippets/news/``.

    Synchronous (network-bound, no GPU) — dispatched via ``run_in_executor`` so it
    never blocks the asyncio loop. Loads the standalone ``server/til`` fetcher by
    path so the inference package needn't depend on it. Returns a small summary
    (no raw wikitext) plus a bounded preview for the Debug tab.
    """
    import importlib.util
    from datetime import date as _date, datetime as _dt, timedelta as _td

    til_path = _TIL_DIR / "fetch_current_events.py"
    spec = importlib.util.spec_from_file_location("til_fetch_current_events", til_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load TIL fetcher at {til_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if date_str:
        d = _dt.strptime(date_str, "%Y-%m-%d").date()
    else:
        d = _date.today() - _td(days=1)   # today's portal page isn't populated yet

    record = mod.build_snippet(d)
    txt_path, _json_path = mod.write_snippet(record)
    return {
        "date": record["date"],
        "title": record["title"],
        # Carried through for the protocol record's provenance (`til_facts.build_doc`),
        # which identifies the TEXT rather than a speaker — on this lane that identity is
        # the only thing distinguishing a sourced digest from a humour wiki.
        "source": record.get("source", ""),
        "source_url": record["source_url"],
        "chars": len(record["text"]),
        "sources": len(record["sources"]),
        "path": str(txt_path),
        "text": record["text"],
        "preview": record["text"][:6000],
    }

# What a learning pass's ANSWER must show some sign of to count as the structured
# reflection its prompt contracts for: a section header, or at least one tagged item.
# Lowercased containment — the parser is lenient about placement, so the check is too.
_LEARNING_SECTION_MARKERS = ("## weights", "## rag", "## resolved",
                             "[fact]", "[ask", "[resolved]")


def learning_contract_problem(raw: str) -> str:
    """Why *raw* is not a structured learning reflection, or ``""`` when it is one.

    The failure this names (2026-08-18, observed live): a freshly-trained adapter,
    entrained by the wander SFT corpus, answered the learning pass with a second persona
    ESSAY — fluent, on-topic, and carrying not one section or tagged line. Every guard
    is blind to it (it is healthy prose), and `parse_consolidation` correctly yields
    zero modifiers — which is exactly the problem: zero modifiers is also what an
    ordinary quiet day produces, so the collapse of a whole channel read as a series of
    uneventful passes. The check is deliberately loose (ANY marker anywhere in the
    answer passes): its job is to name total abandonment of the contract, not to grade
    partial compliance — a reflection that kept nothing but wrote its empty sections
    passes, as it should.
    """
    answer = reasoning_text.answer_after_think(str(raw or ""))
    if not answer.strip():
        return "no answer region (generation ended inside its reasoning)"
    low = answer.lower()
    if not any(marker in low for marker in _LEARNING_SECTION_MARKERS):
        return "no structured sections — the pass wrote prose instead of the reflection"
    return ""


def _run_learning_pass(digest_text: str, date: str, on_chunk: Callable) -> dict:
    """Run ONE dry-run learning reflection over a TIL digest. Writes nothing.

    Frames the fetched digest as external information Ava came across and generates
    a structured reflection under ``learning_prompt.txt`` — the same WEIGHTS / RAG /
    RESOLVED format consolidation uses, so :func:`parse_consolidation` yields the
    exact modifiers a real learning pass *would* route. Nothing is persisted: no
    memory, ledger, RAG, or staging write.

    RAG retrieval is *enabled* (read-only) so the digest is read through the prism of
    what Ava already knows and is — her recalled facts, open questions, and persona
    self-statements are injected as context, just as in a real reflection pass. It
    only reads the index; it never writes to it.

    Returns ``{text, report}`` where *text* is the cleaned reflection and *report*
    is the parsed ``{weights, rag, resolved}`` breakdown.
    """
    from core.reflection_writer import parse_consolidation

    learning_prompt = (_PROMPTS_DIR / "learning_prompt.txt").read_text(encoding="utf-8").strip()
    content = (
        f"TODAY I LEARNED — world events from {date} that you came across:\n\n"
        f"{digest_text}"
    )
    activity_log.set_ambient_label("til_learn")
    generate_fn = _make_sync_reflect_generate(_get_rag())
    text = generate_fn(
        content, learning_prompt,
        temperature=0.9, top_p=0.95,   # reflection default; recommended top_p (+ family top_k)
        max_new_tokens_setting="4096",
        disable_rag=False,         # read Ava's memory/persona as context (read-only)
        on_chunk=on_chunk,
    )
    # A pass that hit the token cap (vs. ended on EOS) is cut mid-string, so its final
    # item is a half-written [fact]/[ask]. Carry the flag through to the apply so the
    # writer drops that fragment instead of persisting a corrupted fact/persona string.
    trunc = bool(getattr(generate_fn, "last_truncated", False))
    return {"text": text, "report": parse_consolidation(text, truncated=trunc),
            "truncated": trunc,
            "contract_problem": learning_contract_problem(text)}

async def handle_til_fetch(ws, msg: dict) -> None:
    """Fetch a day's Wikipedia Current events digest into ``server/til`` (manual debug).

    The Sleep tab's "Learn" button triggers this. Always runs the network fetch off
    the event loop and replies with a ``til_fetched`` digest summary. When
    ``reflect`` is set, it then runs a **dry-run** learning reflection over the
    digest (streaming ``til_reflect_chunk`` deltas) and replies with a terminal
    ``til_reflect_done`` carrying the cleaned reflection + the parsed modifiers
    (``report``). Nothing is persisted — this only shows what Ava *would* keep.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_til_fetch, msg.get("date"))
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"til_fetch failed: {e}"})
        return
    await _send(ws, {"type": "til_fetched", **result})

    if not msg.get("reflect"):
        return

    # Dry-run learning pass over the digest just fetched. Requires a loaded model;
    # if none is loaded, terminate the sequence with a skipped marker so the client
    # stops waiting.
    if _runtime.model is None:
        await _send(ws, {"type": "til_reflect_done", "skipped": "no model loaded",
                         "text": "", "report": {}})
        return

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "til_reflect_chunk", "text": delta}), loop
        )

    try:
        learned = await loop.run_in_executor(
            _executor, _run_learning_pass,
            result.get("text", ""), result.get("date", ""), _on_chunk,
        )
    except Exception as e:
        # Without a traceback the swallowed exception leaves nothing to diagnose in
        # server.log (the client only gets the short skipped message).
        traceback.print_exc()
        await _send(ws, {"type": "til_reflect_done", "skipped": f"reflection failed: {e}",
                         "text": "", "report": {}})
        return
    await _send(ws, {"type": "til_reflect_done", **learned})

def _run_extract_subjects_sync(asks: list[str], on_log: Optional[Callable] = None) -> list:
    """Extract Wikipedia-lookup subjects from open questions, on the CLEAN base.

    Subject extraction is a mechanical tool call, so it runs against the frozen base
    with the adapter swapped out (``CleanBaseSession``) — the persona adapter is
    trained to have opinions and might refuse or answer in character instead of
    extracting. Executor-thread only (exclusive GPU access)."""
    from core import agentic

    def _prepare(tok, model_id) -> None:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if not getattr(tok, "chat_template", None):
            ensure_chat_template(tok, model_name=model_id,
                                 emit=(on_log or (lambda m: None)))

    # Dedicated agentic generate: thinking OFF (so the SUBJECT: block isn't truncated
    # away by a CoT budget) and RAG off. Reads _runtime at call time, so it runs
    # on the clean base swapped in by the session below.
    generate = _make_agentic_generate()
    with agentic.CleanBaseSession(_backend, _runtime, prepare=_prepare, on_log=on_log):
        subjects = agentic.run_task(generate, "extract_subjects", asks)
        if not subjects and on_log is not None:
            raw = (getattr(generate, "last_raw", "") or "")[:300].replace("\n", " / ")
            on_log(f"extractor produced no SUBJECT: lines — raw head: {raw}")
        return subjects

def _fetch_articles_sync(subjects: list[str]) -> list:
    """Resolve each subject to a Wikipedia article snippet (network, no GPU).

    Loads the standalone ``til/fetch_article.py`` by path (mirrors ``_run_til_fetch``)
    so the inference package needn't import it. Returns ``[(subject, record|None)]``
    in input order — ``None`` for subjects with no article."""
    import importlib.util

    fa_path = _TIL_DIR / "fetch_article.py"
    spec = importlib.util.spec_from_file_location("til_fetch_article", fa_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load article fetcher at {fa_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out: list = []
    for s in subjects:
        try:
            rec = mod.build_article_snippet(s)
            if rec is not None:
                # Persist for provenance (mirrors Learn writing its digest to til/);
                # this is disk only — nothing reaches RAG/memory until Apply.
                txt_path, _ = mod.write_article_snippet(rec)
                rec["path"] = str(txt_path)
        except Exception:
            traceback.print_exc()
            rec = None
        out.append((s, rec))
    return out

def _assemble_lookup_digest(questions: list[str], found: list) -> str:
    """Frame the fetched articles as answers to the questions Ava raised.

    Pairs each article with EVERY question it was fetched for; the verbatim question
    text is what the lookup prompt asks the model to echo into ``[resolved]`` so the
    eviction key matches the open ask.

    Every question, because several may share one subject and one article then answers
    all of them — showing the pass only the first would leave the others unresolvable by
    construction, which is the retrieval-side half of the re-fetch loop `_add` describes:
    one ask silently un-markable, the other silently un-resolvable, both from the same
    one-question-per-subject assumption.
    """
    blocks: list[str] = []
    for (subject, rec, question) in found:
        qs = ([q for q in question if q] if isinstance(question, (list, tuple))
              else ([question] if question else []))
        label = ("\n".join(f"QUESTION: {q}" for q in qs) if qs
                 else f"QUESTION: (about {subject})")
        blocks.append(
            f"{label}\n"
            f"ARTICLE — {rec['title']} ({rec['source_url']}):\n{rec['text']}\n"
        )
    return "\n\n".join(blocks)

def _run_lookup_learning_pass(digest_text: str, on_chunk: Callable) -> dict:
    """Dry-run learning pass over looked-up articles. Writes nothing.

    Like ``_run_learning_pass`` but uses ``lookup_prompt.txt`` (framed as "answers to
    questions you raised", biased toward ``[resolved]``). RAG read-only so Ava reads
    the answers through what she already knows. Returns ``{text, report}``."""
    from core.reflection_writer import parse_consolidation

    lookup_prompt = (_PROMPTS_DIR / "lookup_prompt.txt").read_text(encoding="utf-8").strip()
    activity_log.set_ambient_label("til_lookup")
    generate_fn = _make_sync_reflect_generate(_get_rag())
    text = generate_fn(
        digest_text, lookup_prompt,
        temperature=0.9, top_p=0.95,
        max_new_tokens_setting="4096",
        disable_rag=False,
        on_chunk=on_chunk,
    )
    trunc = bool(getattr(generate_fn, "last_truncated", False))
    return {"text": text, "report": parse_consolidation(text, truncated=trunc),
            "truncated": trunc,
            "contract_problem": learning_contract_problem(text)}

async def handle_til_lookup(ws, msg: dict) -> None:
    """Resolve open ``[ask:search]`` questions by looking them up on Wikipedia.

    The lookup loop, end to end: collect Ava's open search questions → extract their
    subjects on the clean base → fetch each subject's article → run a dry-run
    learning pass over the answers (streamed). The terminal message is the same
    ``til_reflect_done {text, report}`` the Learn flow emits, so "Apply Learning"
    persists it unchanged — and any ``[resolved]`` evicts the question it answered,
    closing the loop. Nothing is written until the operator applies."""
    loop = asyncio.get_event_loop()

    # The clean-base swap unloads/reloads the model, so it must never run while a
    # reflection run holds the executor and the loaded model.
    if _reflection_active():
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": "a reflection run is active — try lookup after it finishes",
                         "text": "", "report": {}})
        return

    # 1. Collect open search asks not yet looked up. `lookupable_questions` enforces
    #    fetch-once: a still-open question an article failed to resolve is not
    #    re-fetched on a later run (no endless fetch loop). Keep each question's key
    #    so we can write its fetch-once marker after the attempt.
    mem = ReflectionMemory(_MEMORY_DIR)
    recs = [{"content": (q.get("content") or "").strip(),
             "key": q.get("key") or "",
             "lookup": (q.get("lookup") or "").strip()}
            for q in mem.lookupable_questions()]
    recs = [r for r in recs if r["content"]]
    questions = [r["content"] for r in recs]
    key_by_content = {r["content"]: r["key"] for r in recs}
    await _send(ws, {"type": "til_lookup_collected",
                     "count": len(questions), "questions": questions})
    if not questions:
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": "no open [ask:search] questions to look up",
                         "text": "", "report": {}})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "til_reflect_done", "skipped": "no model loaded",
                         "text": "", "report": {}})
        return

    def _log(m: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "til_lookup_log", "message": m}), loop)

    # 2. Determine each question's lookup subject. Asks that bound a title at creation
    #    (copied from the digest's referenced articles) fetch DIRECTLY — no extraction,
    #    no model swap. Only the rest need the clean-base extractor.
    subj_to_q: dict[str, str] = {}   # subject -> the question it came from
    subjects: list[str] = []

    def _add_subject(subject: str, question: Optional[str]) -> None:
        subject = (subject or "").strip()
        if subject and subject not in subj_to_q:
            subj_to_q[subject] = question
            subjects.append(subject)

    bound = [r for r in recs if r["lookup"]]
    unbound = [r for r in recs if not r["lookup"]]
    for r in bound:
        _add_subject(r["lookup"], r["content"])
    if bound:
        _log(f"{len(bound)} question(s) carry a bound title — fetching directly "
             f"(no model swap).")

    if unbound:
        try:
            extracted = await loop.run_in_executor(
                _executor, _run_extract_subjects_sync,
                [r["content"] for r in unbound], _log)
        except Exception as e:
            traceback.print_exc()
            await _send(ws, {"type": "til_reflect_done",
                             "skipped": f"subject extraction failed: {e}",
                             "text": "", "report": {}})
            return
        # Attribute extracted subjects to the unbound questions: 1:1 by position when
        # the count matches (the extractor is asked for one per question in order),
        # else by substring match.
        unbound_qs = [r["content"] for r in unbound]
        aligned = len(extracted) == len(unbound_qs)
        for idx, s in enumerate(extracted):
            q = next((c for c in unbound_qs if s.lower() in c.lower()), None)
            if q is None and aligned:
                q = unbound_qs[idx]
            _add_subject(s, q)

    await _send(ws, {"type": "til_lookup_subjects",
                     "subjects": subjects, "bound_count": len(bound)})
    if not subjects:
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": "no lookupable subjects (none bound or extracted)",
                         "text": "", "report": {}})
        return

    # 3. Fetch each subject's article (network, no GPU).
    try:
        fetched = await loop.run_in_executor(None, _fetch_articles_sync, subjects)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": f"article fetch failed: {e}",
                         "text": "", "report": {}})
        return

    # Each fetched subject already knows the question it came from (subj_to_q).
    found: list = []
    attempted: dict[str, tuple] = {}   # question -> (subject, found_bool, title)
    for subject, rec in fetched:
        question = subj_to_q.get(subject)
        if rec is None:
            await _send(ws, {"type": "til_lookup_fetched",
                             "subject": subject, "missing": True})
        else:
            found.append((subject, rec, question))
            await _send(ws, {"type": "til_lookup_fetched", "subject": subject,
                             "title": rec["title"], "via": rec["via"],
                             "chars": len(rec["text"]), "url": rec["source_url"],
                             "redirected_from": rec.get("redirected_from")})
        if question is not None:
            attempted[question] = (subject, rec is not None,
                                   rec["title"] if rec is not None else "")

    # Fetch-once bookkeeping: mark ONLY the questions we actually looked up (a subject
    # was attributed and fetched, found or not), keyed by content_key — so an article
    # that doesn't resolve a question leaves it open but no longer re-fetched. A
    # question we couldn't extract a subject for (e.g. a future event with no article)
    # is left untouched, so a bad extraction can't wrongly retire it. Resolution stays
    # a separate evict on Apply. Written before the learning pass so it holds even if
    # the user never applies — the one durable side effect of an otherwise dry run.
    try:
        writer = _get_reflection_writer()
        run_id = f"lookup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        for content, (subject, found_bool, title) in attempted.items():
            key = key_by_content.get(content)
            if key:
                writer.write_lookup(key=key, subject=subject, found=found_bool,
                                    title=title, run_id=run_id)
    except Exception:
        traceback.print_exc()

    if not found:
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": "no Wikipedia articles found for the extracted subjects",
                         "text": "", "report": {}})
        return

    # 4. Dry-run learning pass over the answers (adapter back ON — this is Ava's
    #    judgement, not a tool call). Streamed; terminal message reuses Learn's.
    digest = _assemble_lookup_digest(questions, found)

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "til_reflect_chunk", "text": delta}), loop)

    try:
        learned = await loop.run_in_executor(
            _executor, _run_lookup_learning_pass, digest, _on_chunk)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": f"reflection failed: {e}",
                         "text": "", "report": {}})
        return
    await _send(ws, {"type": "til_reflect_done", **learned})

def _load_fetch_wiki():
    """Load the standalone ``til/fetch_wiki.py`` by path (mirrors _run_til_fetch)."""
    import importlib.util
    fw_path = _TIL_DIR / "fetch_wiki.py"
    spec = importlib.util.spec_from_file_location("til_fetch_wiki", fw_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load wiki fetcher at {fw_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_VISIT_USER_AGENT = "ProjectAva-TIL/0.1 (manual wander visit)"
_TEXT_WS_RE = re.compile(r"[ \t]+")
_TEXT_BLANKS_RE = re.compile(r"\n{3,}")


class _ReadableHTMLParser(HTMLParser):
    """Small stdlib HTML-to-text extractor for manually visited pages."""

    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "tr", "ul",
    }
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 - HTMLParser API
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.body_parts.append(data)


def _collapse_page_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_TEXT_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _TEXT_BLANKS_RE.sub("\n\n", "\n".join(line for line in lines if line)).strip()


def _truncate_page_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for sep in ("\n\n", ". ", "\n"):
        idx = window.rfind(sep)
        if idx > max_chars // 2:
            return window[:idx + (len(sep) if sep != ". " else 1)].rstrip() + " …"
    return window.rstrip() + " …"


def _normalize_visit_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("empty URL")
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be http(s)")
    return urllib.parse.urlunparse(parsed)


def _mediawiki_title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if qs.get("title") and qs["title"][0].strip():
        return qs["title"][0].strip().replace("_", " ")
    parts = [p for p in parsed.path.split("/") if p]
    if "wiki" in parts:
        idx = parts.index("wiki")
        if idx + 1 < len(parts):
            return urllib.parse.unquote("/".join(parts[idx + 1:])).replace("_", " ")
    return ""


def _match_mediawiki_source_for_url(url: str) -> Optional[dict]:
    mod = _load_fetch_wiki()
    cfg = mod.load_sources()
    visited_host = (urllib.parse.urlparse(url).hostname or "").lower()
    if not visited_host:
        return None
    for bucket, sources in cfg.items():
        if bucket.startswith("_") or not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict) or not source.get("api"):
                continue
            api_host = (urllib.parse.urlparse(source["api"]).hostname or "").lower()
            if api_host == visited_host:
                return {**source, "lang": bucket}
    return None


def _fetch_mediawiki_visit(url: str, max_chars: int) -> Optional[tuple[dict, dict]]:
    source = _match_mediawiki_source_for_url(url)
    title = _mediawiki_title_from_url(url)
    if not source or not title:
        return None
    mod = _load_fetch_wiki()
    rec = mod.fetch_lead(source["api"], title, intro_only=False, max_chars=max_chars)
    if rec is None:
        return None
    record = {
        "kind": "wiki_visit",
        "source": "wiki:visit",
        "wiki": source.get("name", source["api"]),
        "lang": source.get("lang", ""),
        "date": datetime.now().date().isoformat(),
        "title": rec["title"],
        "source_url": rec.get("url") or url,
        "fetched_at": datetime.now().astimezone().isoformat(),
        "text": rec["extract"],
        "sources": [rec.get("url") or url],
    }
    return source, record


def _fetch_generic_visit(url: str, max_chars: int) -> tuple[dict, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": _VISIT_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            raw = resp.read(max(1, max_chars) * 8)
            final_url = resp.geturl() or url
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"request failed: {e}") from e

    charset = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        charset = match.group(1).strip()
    decoded = raw.decode(charset, errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        parser = _ReadableHTMLParser()
        parser.feed(decoded)
        title = _collapse_page_text(" ".join(parser.title_parts))
        text = _collapse_page_text("\n".join(parser.body_parts))
    elif content_type.lower().startswith("text/") or not content_type:
        title = ""
        text = _collapse_page_text(decoded)
    else:
        raise RuntimeError(f"unsupported content type: {content_type or 'unknown'}")

    parsed = urllib.parse.urlparse(final_url)
    host = parsed.netloc or "visited page"
    if not title:
        title = urllib.parse.unquote(Path(parsed.path).name or host).replace("_", " ")
    text = _truncate_page_text(text, max_chars)
    if len(text) < 200:
        raise RuntimeError("page did not contain enough readable text")

    source = {"name": host, "lang": ""}
    record = {
        "kind": "web_visit",
        "source": "web:visit",
        "wiki": host,
        "lang": "",
        "date": datetime.now().date().isoformat(),
        "title": title,
        "source_url": final_url,
        "fetched_at": datetime.now().astimezone().isoformat(),
        "text": text,
        "sources": [final_url],
    }
    return source, record


def _fetch_visit_url(url: str, max_chars: int) -> tuple[dict, dict]:
    """Fetch a user-supplied page URL and wrap it in the wander record shape."""
    url = _normalize_visit_url(url)
    mediawiki = _fetch_mediawiki_visit(url, max_chars)
    if mediawiki is not None:
        return mediawiki
    return _fetch_generic_visit(url, max_chars)


def _wander_max_chars(context_length: int) -> int:
    """Article char-cap for a wander fetch, scaled to the model's context window.

    The wander truncates the whole article to this many chars and feeds it into each
    of the two reflection passes, so the cap has to leave room for the prompt + the
    generation. Budget ~30% of the window to the article and convert tokens→chars
    conservatively (~2.7 chars/token, sized for Cyrillic's denser tokenization so a
    non-English page still fits its token budget). Floored so a tiny context still
    gets a usable page, and ceilinged so a huge window doesn't pull a pathological
    100k-char featured article that would dilute attention and slow generation.

    At the usual 24k context this lands ~19.9k chars — enough to swallow a whole
    typical article (the old flat 4000-char default clipped everything past the lead,
    e.g. the Stierlitz reception section sat ~10k chars beyond the cut)."""
    chars = int(max(1, context_length) * 0.30 * 2.7)
    return max(6000, min(chars, 40000))

def _wander_article_content(record: dict) -> str:
    """The user-turn text presenting a wandered article — the VOICE pass's builder.

    Since 2026-08-18 the voice pass's alone: the learning pass builds its own
    (:func:`_wander_learning_content`, which owns the reason for the split). This one
    is bound by training parity — the prompt + content are persisted as the SFT
    example — so it must not drift casually.

    Opens with the real wall-clock 'now' so the wander is time-anchored: the reaction
    pass trains on this prompt, so each (especially autonomous) wander lands a tick of
    the current moment in the SFT stream — the training-side counterpart of the chat
    ``_temporal_anchor()`` (AVA_DESIGN_LEGACY.md → *Toward an autonomous wander*: time passing
    becomes something Ava is trained on, not only told)."""
    now = datetime.now().strftime("%A, %B %-d, %Y, %H:%M")
    if record.get("kind") in {"wiki_visit", "web_visit"}:
        lead = f"You chose to visit this page on {record.get('wiki', 'the web')}:"
    else:
        lead = (
            f"You wandered into this article on {record.get('wiki', 'a wiki')} "
            f"(picked at random):"
        )
    return (
        f"It is {now}.\n\n"
        f"{lead}\n\n"
        f"TITLE: {record['title']}\n{record.get('source_url', '')}\n\n{record['text']}"
    )

def _wander_learning_content(record: dict) -> str:
    """The learning pass's OWN presentation of a wandered page.

    Deliberately NOT :func:`_wander_article_content`, and the split is load-bearing
    (2026-08-18, observed live): the two wander passes shared that builder byte for
    byte, and every wander capture in the SFT corpus is exactly that user turn — under
    the VOICE prompt — mapped to a free-form persona essay, retrained from scratch each
    cycle. A freshly-trained adapter learned the mapping off the user turn strongly
    enough to steamroll the learning pass's system prompt: its CoT recited the voice
    pass's instructions from memory ("avoid 'As an AI'", "embrace the persona's
    voice" — text that appears nowhere in `wander_prompt.txt`) and its answer was a
    second essay, no sections at all. The voice pass must keep its builder verbatim
    (its prompt + content are persisted as the training example, so parity binds it);
    this pass has no parity constraint, so it de-anchors instead: a different lead, and
    the task restated AFTER the article — nearest to generation, the position the
    protocol/summary closings already use for the same reason
    (`til_facts._CLOSING`, `reflection_chunking.SUMMARY_CLOSING`).
    """
    now = datetime.now().strftime("%A, %B %-d, %Y, %H:%M")
    if record.get("kind") in {"wiki_visit", "web_visit"}:
        lead = f"The page below is one you chose to visit on {record.get('wiki', 'the web')}."
    else:
        lead = (f"The page below crossed your path on {record.get('wiki', 'a wiki')}, "
                f"picked at random.")
    return (
        f"It is {now}.\n\n"
        f"{lead} Read it, then file the structured reflection your instructions "
        f"describe.\n\n"
        f"TITLE: {record['title']}\n{record.get('source_url', '')}\n\n{record['text']}\n\n"
        f"— end of the page —\n\n"
        f"Now the structured reflection. Not a reaction, not an essay, not a message to "
        f"anyone: the ## WEIGHTS, ## RAG and ## RESOLVED sections exactly as your "
        f"instructions describe, each left empty where you keep nothing."
    )


def _run_wander_learning_pass(record: dict, on_chunk: Callable) -> dict:
    """Dry-run learning pass over a random wiki page (the **facts/asks → memory** pass).

    Uses ``wander_prompt.txt`` (framed as "you wandered into this at random") with
    the same WEIGHTS/RAG/RESOLVED structure as Learn, so the result feeds Apply
    unchanged. RAG read-only so Ava reads it through what she already is."""
    from core.reflection_writer import parse_consolidation

    wander_prompt = (_PROMPTS_DIR / "wander_prompt.txt").read_text(encoding="utf-8").strip()
    content = _wander_learning_content(record)
    activity_log.set_ambient_label("wander_learn")
    generate_fn = _make_sync_reflect_generate(_get_rag())
    text = generate_fn(
        content, wander_prompt,
        temperature=0.9, top_p=0.95,
        max_new_tokens_setting="4096",
        disable_rag=False,
        on_chunk=on_chunk,
    )
    trunc = bool(getattr(generate_fn, "last_truncated", False))
    return {"text": text, "report": parse_consolidation(text, truncated=trunc),
            "truncated": trunc,
            "contract_problem": learning_contract_problem(text),
            "system_prompt": wander_prompt, "prompt": content}

def _run_wander_voice_pass(record: dict, on_chunk: Callable) -> dict:
    """Free-form, same-language reaction to a random wiki page — the **language-bleed** pass.

    This is the pass whose output lands in the SFT learning dataset. Unlike the structured
    learning pass (English WEIGHTS/RAG extraction for memory), it asks Ava to react to the
    article *in the article's own language*, in her own voice, with no structured sections,
    so its phrasing/register bleed into her generation (AVA_DESIGN_LEGACY.md → Ambient
    Enculturation). ``wander_voice_prompt.txt`` deliberately does **not** name a ``<think>``
    block: gemma-4 already reasons in its native channel, and naming ``<think>`` made it
    emit a *second*, literal block — the double-think target that render then rejects. RAG
    is disabled so her past (mostly English) chat context doesn't dilute the bleed, and so
    the system prompt at generation matches what we store (clean train/inference parity)."""
    voice_prompt = (_PROMPTS_DIR / "wander_voice_prompt.txt").read_text(encoding="utf-8").strip()
    content = _wander_article_content(record)
    activity_log.set_ambient_label("wander_voice")
    generate_fn = _make_sync_reflect_generate(_get_rag())
    text = generate_fn(
        content, voice_prompt,
        temperature=0.9, top_p=0.95,
        max_new_tokens_setting="2048",
        disable_rag=True,
        on_chunk=on_chunk,
    )
    return {"text": text, "system_prompt": voice_prompt, "prompt": content}

def _pick_and_fetch_wander(lang: Optional[str], name: Optional[str], max_chars: int):
    """Pick an enabled wiki source and fetch one random substantive page.

    Returns ``(source, record)`` — ``(None, None)`` when no source is enabled,
    ``(source, None)`` when a source was chosen but yielded no substantive page, or
    raises if every candidate source was unreachable. Shared by the manual
    ``handle_til_wander`` button and the autonomous idle wander.
    """
    import random as _random
    mod = _load_fetch_wiki()
    cfg = mod.load_sources()
    # An explicit source name is an exact choice — honour it, no fallback. Otherwise
    # try the enabled sources in random order so one dead/misconfigured source (bad
    # api path, blocked host, all-stubs) doesn't make Wander look broken: we move on
    # to the next instead of failing the whole button.
    if name:
        picked = mod.pick_source(cfg, lang=lang, name=name)
        candidates = [picked] if picked else []
    else:
        candidates = mod.enabled_sources(cfg, lang=lang)
        _random.shuffle(candidates)
    if not candidates:
        return None, None
    last_exc = None
    for source in candidates:
        try:
            rec = mod.build_random_snippet(source, max_chars=max_chars)
        except Exception as e:
            last_exc = e          # this source is unreachable — try the next one
            continue
        if rec is not None:
            # The provenance snippet is written at Apply time (_write_wander_exchange), NOT
            # here at fetch: a manual wander the operator declines must not leave a snippet in
            # Ava's data tree (snippets are now a RAG-adjacent source). The landing itself is
            # logged separately (wander_log) by the caller.
            return source, rec
    # No source yielded a page: surface a real error if one occurred, else report the
    # last source tried so the "no substantive page" message has a name.
    if last_exc is not None:
        raise last_exc
    return candidates[-1], None

async def handle_til_wander(ws, msg: dict) -> None:
    """Reflect on a random page, or on a user-supplied URL.

    Without ``url``, picks a random *enabled* source from ``wiki_sources.json``
    (optionally filtered by ``lang`` or a specific ``source`` name), fetches a random
    substantive page, and runs a dry-run learning pass over it (streamed). With
    ``url``, fetches that page directly and then runs the same two-pass wander
    sequence. The terminal message is the same ``til_reflect_done`` the Learn flow
    emits, so "Apply Learning" persists it unchanged. Nothing reaches RAG/memory until
    the operator applies.

    **Manual wander never touches the token budget** — only the autonomous idle wander
    (``_run_autonomous_wander``) earns/consumes the wander economy."""
    loop = asyncio.get_event_loop()
    lang = (msg.get("lang") or "").strip() or None
    name = (msg.get("source") or "").strip() or None
    url = (msg.get("url") or "").strip() or None
    # Size the article truncation to the loaded model's context window (defaults to the
    # configured context_length when no model is loaded yet — the fetch runs before the
    # model-loaded check below).
    max_chars = _wander_max_chars(int(_runtime.context_length or 32768))

    try:
        if url:
            source, rec = await loop.run_in_executor(None, _fetch_visit_url, url, max_chars)
        else:
            source, rec = await loop.run_in_executor(
                None, _pick_and_fetch_wander, lang, name, max_chars)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": f"{'visit' if url else 'wander'} fetch failed: {e}",
                         "text": "", "report": {}})
        return
    if source is None:
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": "no enabled wiki source (approve one in wiki_sources.json)",
                         "text": "", "report": {}})
        return
    if rec is None:
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": f"couldn't fetch a substantive random page from "
                                    f"{source.get('name')}", "text": "", "report": {}})
        return
    await _send(ws, {"type": "til_wandered",
                     "wiki": source.get("name", ""), "lang": source.get("lang", ""),
                     "title": rec["title"], "url": rec["source_url"],
                     "chars": len(rec["text"]), "mode": "visit" if url else "wander"})
    _append_wander_log(source, rec, auto=False)   # record the page she landed on

    if _runtime.model is None:
        await _send(ws, {"type": "til_reflect_done", "skipped": "no model loaded",
                         "text": "", "report": {}})
        return

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "til_reflect_chunk", "text": delta}), loop)

    def _banner(text: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "til_reflect_chunk", "text": text}), loop)

    # THREE passes per wander, and all three are shown before anything is kept: (1) a
    # free-form, same-language *reaction* (the language-bleed pass → SFT dataset), streamed
    # first so the operator can see the bleed actually happening; (2) the structured
    # facts/asks reflection (→ memory); (3) the prose recap (→ `<stem>.summary.json`).
    # All read the same article.
    #
    # The recap joined them here rather than staying inside Apply because this flow is a
    # DRY RUN and Apply is the consent point: a press meaning "keep what I just read" was
    # producing a third generation nobody had seen, whose prose then reaches live chat
    # (`rag_engine._render_til_nomination`) and reach-out messages
    # (`outreach._source_material`). Generated here, it is reviewed like the other two and
    # Apply only writes it — so what is shown is what is written, and a declined wander
    # still leaves nothing on disk.
    try:
        _banner("\n— reaction (in the article's language; for the learning dataset) —\n\n")
        voice = await loop.run_in_executor(_executor, _run_wander_voice_pass, rec, _on_chunk)
        _banner("\n\n— reflection (facts / asks; for memory) —\n\n")
        learned = await loop.run_in_executor(
            _executor, _run_wander_learning_pass, rec, _on_chunk)
        _banner("\n\n— recap (what this was, for remembering it later) —\n\n")
        gist = await loop.run_in_executor(
            _executor, lambda: generate_gist_text(rec, "wander", on_chunk=_on_chunk))
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "til_reflect_done",
                         "skipped": f"reflection failed: {e}", "text": "", "report": {}})
        return
    # Stash both for "Apply learning": the same-language voice exchange → SFT dataset, the
    # structured text → memory. One most-recent slot; the next wander or a consuming Apply
    # replaces it.
    global _last_wander_exchange
    _last_wander_exchange = {
        "sft": {"system_prompt": voice.get("system_prompt", ""),
                "prompt": voice.get("prompt", ""), "target": voice.get("text", "")},
        "memory_text": learned.get("text", ""),
        "truncated": learned.get("truncated", False),
        "source": {"wiki": source.get("name", ""), "title": rec["title"],
                   "url": rec["source_url"], "lang": source.get("lang", "")},
        "rec": rec,   # raw fetched record — the provenance snippet is written on Apply
        # The reviewed recap, generated above. Apply WRITES this rather than regenerating,
        # so the persisted recap is byte-identical to the one shown. Absent (a failed or
        # skipped pass) ⇒ Apply writes none and the snippet stays in the gist backlog,
        # which is exactly what happens for a text nobody previewed.
        "gist": gist,
    }
    # til_reflect_done carries the structured text+report (drives the modifiers preview).
    # `contract_problem` names a pass that abandoned the WEIGHTS/RAG/RESOLVED contract
    # outright (wrote an essay, or never reached its answer) — without it, that collapse
    # and an uneventful page both render as "no modifiers".
    await _send(ws, {"type": "til_reflect_done", "text": learned.get("text", ""),
                     "report": learned.get("report", {}),
                     "truncated": learned.get("truncated", False),
                     "contract_problem": learned.get("contract_problem", "")})

def _write_wander_exchange(ex: dict) -> dict:
    """Route one stashed wander exchange into the SFT dataset + live memory (blocking).

    The shared commit path for both the operator's Apply button
    (``_apply_wander_to_dataset``) and the autonomous idle wander
    (``_run_autonomous_wander``): the same-language voice pass becomes a one-shot SFT
    example (register/language bleed), and the structured facts/asks pass is routed
    into live memory through the same writer Learn/Lookup use. Returns the routing
    counts (incl. ``wander_examples`` / ``wander_trainable``)."""
    from core.wander_sft import append_example, looks_trainable
    sft = (ex or {}).get("sft") or {}
    wiki = (ex.get("source") or {}).get("wiki", "")
    memory_text = ex.get("memory_text", "") or ""
    # 1. SFT learning dataset (one-shot training tokens) — the same-language voice pass.
    wrote = append_example(
        system_prompt=sft.get("system_prompt", ""), prompt=sft.get("prompt", ""),
        target=sft.get("target", ""), source=ex.get("source", {}),
    )
    # 2. Provenance snippet — written at APPLY (gated), not at fetch, so a declined manual
    # wander never lands one in the data tree. Best-effort.
    #
    # **Written BEFORE live memory now, and the order is the point.** Everything this pass
    # distils is filed under `mem_source`, and that used to be `wiki:<site>` — the site,
    # not the page. Measured on the live store: 17 of 30 self-directed asks point at
    # "Lurkmore"/"Urban Culture"/"WikiTropes" as though those named a text, so a question
    # she raises from an article can never recover the article (`til_gist.resolve_source`,
    # `outreach._source_material`). The snippet's own stem is the only identifier that
    # names the page, and it exists as soon as the snippet is written — so the write moves
    # up and the id is taken from it, rather than the id being invented from what was to
    # hand at the top of the function.
    rec = (ex or {}).get("rec")
    txt_path = None
    if wrote and rec:
        try:
            txt_path, _json_path = _load_fetch_wiki().write_random_snippet(rec)
        except Exception:
            txt_path = None
    # Falls back to the old site-level id when there is no snippet (a declined wander, or
    # a failed write): a coarse provenance string is still better than none, and this is
    # exactly the case where there is no text on disk to point at anyway.
    mem_source = (til_facts.input_id(txt_path, "wander") if txt_path
                  else (f"wiki:{wiki}" if wiki else "wiki"))
    # 3. Live memory (same routing as Learn/Lookup) — the structured facts/asks pass.
    counts = dict(_apply_learning_text_live(memory_text, mem_source,
                                            truncated=ex.get("truncated", False)) or {}) \
        if memory_text.strip() else {}
    counts["wander_examples"] = 1 if wrote else 0
    # Flag whether the captured target will actually survive render's guard, so the
    # operator gets immediate feedback (e.g. a double-think target lands here but never
    # trains). The example is written regardless — train_cycle skips an untrainable one.
    counts["wander_trainable"] = bool(wrote and looks_trainable(sft.get("target", "")))
    if txt_path is not None:
        # 4. The recap — WRITTEN, not generated. It was produced as the third preview pass
        # and the operator has read it, so Apply persists exactly that text. Nothing here
        # touches the GPU.
        gist = (ex or {}).get("gist") or {}
        if gist and not gist.get("skipped"):
            try:
                counts["gist_written"] = bool(
                    write_gist_text(gist, rec, "wander", txt_path,
                                    run_id=mem_source).get("written"))
            except Exception:
                traceback.print_exc()
        # 5. The PROTOCOL is deliberately NOT produced here, and this is the one derived
        # artifact Apply does not carry. It is the box's longest single generation (a
        # 12,288-token budget per block, observed at 21 minutes on a 26k-char page) — far
        # past the client's `til_apply` timeout, so every manual Apply reported failure
        # while the server quietly finished, and a retry inside that window wrote a SECOND
        # wander capture (`wander_sft.append_example` appends unconditionally) that then
        # trained twice.
        #
        # Deferring it costs nothing: the snippet is on disk now, so `til_facts.list_backlog`
        # picks it up and the next reflection run's ingestion phase derives it — the same
        # function, in the same place the NEWS lane has always called it from. And unlike
        # the recap it does not need reviewing: a protocol is a witness record of the text
        # ("record what the text says, not what you make of it"), closer to a photocopy
        # than to a judgement, and nobody reviews the news lane's either.
    # 6. Make the new wander retrievable this server lifetime (chat-RAG wander channel).
    if wrote:
        try:
            _get_rag().refresh_wander()
        except Exception:
            pass
    return counts

async def _apply_wander_to_dataset(ws) -> None:
    """Commit the most-recent wander exchange to BOTH the SFT learning dataset and memory.

    Wander Apply serves two purposes:
      1. **Language/register bleed** — the exchange (her own thought carrying the article's
         tongue) becomes a single one-shot training example for the next train cycle
         (``core.wander_sft``).
      2. **Belief** — the reflection's WEIGHTS/RAG/RESOLVED modifiers are routed into Ava's
         live memory through the same writer Learn/Lookup use (``_apply_learning_text_live``).

    The exchange was stashed by ``handle_til_wander``; this clears the slot once applied so
    the same wander can't be double-applied. Best-effort across both writes — a failure in
    one is reported but does not abort the other (counts reflect what landed)."""
    global _last_wander_exchange
    ex = _last_wander_exchange
    sft = (ex or {}).get("sft") or {}
    if not ex or not (sft.get("target") or "").strip():
        await _send(ws, {"type": "til_applied",
                         "skipped": "no wander exchange to apply (run Wander first)",
                         "counts": {}})
        return

    try:
        counts = await asyncio.get_event_loop().run_in_executor(
            _executor, _write_wander_exchange, ex)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "til_applied",
                         "skipped": f"wander apply failed: {e}", "counts": {}})
        return

    _last_wander_exchange = None        # one-shot: consumed
    await _send(ws, {"type": "til_applied", "source": "wander", "counts": counts})

def run_autonomous_wander_blocking() -> dict:
    """Fetch one random article, run both wander passes, auto-apply, consume budget.

    Runs on the single executor thread (so it serializes with chat/reflection
    generation — never concurrent). Returns a small summary dict; raises on a hard
    failure (logged by the caller). Budget is consumed ONLY on a successful, applied
    wander, so a fetch/generation failure costs nothing."""
    if _runtime.model is None:
        return {"skipped": "no model loaded"}

    max_chars = _wander_max_chars(int(_runtime.context_length or 32768))
    source, rec = _pick_and_fetch_wander(None, None, max_chars)
    if source is None:
        return {"skipped": "no enabled wiki source"}
    if rec is None:
        return {"skipped": f"no substantive random page from {source.get('name')}"}

    title = rec.get("title", "")
    print(f"[wander] autonomous: {source.get('name','?')} — {title!r}", flush=True)
    _append_wander_log(source, rec, auto=True)   # record the page she landed on

    # Same two passes as the manual button, but no streaming (no client to stream to).
    voice = _run_wander_voice_pass(rec, on_chunk=lambda _d: None)
    learned = _run_wander_learning_pass(rec, on_chunk=lambda _d: None)
    contract_problem = learned.get("contract_problem") or ""
    if contract_problem:
        # No operator here, so the named event goes to the journal/log — otherwise an
        # entrained adapter's essay applies as zero modifiers and the wake reads as an
        # ordinary quiet one.
        print(f"[wander] learning pass abandoned its contract: {contract_problem}",
              flush=True)
    ex = {
        "sft": {"system_prompt": voice.get("system_prompt", ""),
                "prompt": voice.get("prompt", ""), "target": voice.get("text", "")},
        "memory_text": learned.get("text", ""),
        "truncated": learned.get("truncated", False),
        "source": {"wiki": source.get("name", ""), "title": title,
                   "url": rec.get("source_url", ""), "lang": source.get("lang", "")},
        "rec": rec,   # raw fetched record — snippet written on (auto-)apply below
        # No "gist": there is no operator to review one, so nothing is gained by generating
        # it inside this wake and something is lost — the recap and the protocol both fall
        # to the backlog, and the idle job hands the GPU back in seconds instead of holding
        # it for the length of two long passes.
    }
    counts = _write_wander_exchange(ex)
    spent = _consume_wander_budget()   # success → ration one wander
    print(f"[wander] autonomous applied: {counts} (wander_count={spent}, "
          f"remaining={_wander_budget_available()})", flush=True)
    # Episodic worklog: record the page I wandered into, in my own voice — a self-directed
    # episode with no open loop (nothing is awaited). Consumed by a later deliberation task.
    try:
        from core import worklog
        worklog.record(
            "wander",
            f"On my own, I wandered into '{title}' and reflected on it.",
            refs={"title": title, "url": rec.get("source_url", ""),
                  "wiki": source.get("name", "")},
        )
    except Exception:
        traceback.print_exc()
    result = {"applied": True, "title": title, "counts": counts}
    if contract_problem:
        result["learning_contract_problem"] = contract_problem
    return result

async def handle_til_apply(ws, msg: dict) -> None:
    """Persist the modifiers from a "Learn" dry-run pass into Ava's live memory.

    The Sleep tab's "Apply learning" button sends back the cleaned reflection text
    produced by the preceding dry-run learning pass (``handle_til_fetch`` with
    ``reflect``). This routes that text through the very same writer the real
    consolidation phase uses — ``write_consolidation`` → RAG memory + weights store,
    plus ledger anchors — then refreshes the reflection RAG index so the new
    facts/asks are recalled at chat time. This is the one place a learning pass
    actually writes anything; everything up to it is preview-only.
    """
    # Wander applies have two destinations: the SFT learning dataset (one-shot Ambient
    # Enculturation example — language/register bleed) AND Ava's live memory (the same
    # WEIGHTS/RAG/RESOLVED routing Learn/Lookup use). The exchange (system/user/assistant)
    # was stashed when the wander ran; the operator's Apply press commits both.
    if (msg.get("kind") or "").strip() == "wander":
        await _apply_wander_to_dataset(ws)
        return

    text = (msg.get("text") or "").strip()
    date = msg.get("date") or ""
    if not text:
        await _send(ws, {"type": "til_applied",
                         "skipped": "no learning text to apply", "counts": {}})
        return

    loop = asyncio.get_event_loop()
    source = f"til:{date}" if date else "til"

    # If the client echoes back the dry-run's truncation flag, honor it so a manual
    # Apply of a cut pass drops the fragment too; absent ⇒ False (unchanged behavior).
    truncated = bool(msg.get("truncated", False))

    def _apply() -> dict:
        return _apply_learning_text_live(text, source, truncated=truncated)

    try:
        counts = await loop.run_in_executor(_executor, _apply)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "til_applied",
                         "skipped": f"apply failed: {e}", "counts": {}})
        return
    await _send(ws, {"type": "til_applied", "counts": counts, "source": source})

def run_facts_pass(record: dict, kind: str, snippet_path, *,
                   run_id: str = "", on_block: Optional[Callable] = None) -> dict:
    """Write the fact-extraction protocol for one fetched text. Blocking (GPU thread).

    The TIL lane's counterpart of `reflection_runner._run_chat_facts_pass_for_session`,
    and the answer to the same question the chat protocol answers: what did this *text*
    establish, as against the handful of items the learning pass judged worth keeping.
    See `core.til_facts` for why that record is not just more `[fact]` items in the live
    store, and why its subject namespace is not the chat lane's.

    Runs once per reading block (`til_facts.build_reading_blocks`) — an article is bounded
    by nothing and the live corpus holds one of 26k characters — and writes the joined
    result beside the snippet. **Best-effort by construction:** it is the last thing the
    ingest/apply path does, after the learning output has already reached live memory, so
    a failure here costs a record and never the reading it came from.
    """
    if _runtime.model is None:
        return {"skipped": "no_model"}
    prompt_path = _PROMPTS_DIR / "til_facts_prompt.txt"
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return {"skipped": "no_prompt"}          # degrade to previous behaviour, silently
    if not system_prompt:
        return {"skipped": "no_prompt"}

    blocks = til_facts.build_reading_blocks(
        record, kind,
        til_facts.block_budget_chars(int(_runtime.context_length or 32768)))
    if not blocks:
        return {"skipped": "no_text"}

    # Name this pass for the activity journal. It is the longest single generation the
    # box runs (12,288-token budget over a whole article, observed at 21 minutes on a
    # 26k-char wiki page), so it is exactly the one whose heartbeat an operator needs.
    activity_log.set_ambient_label(f"til_facts:{kind}")
    generate_fn = _make_sync_reflect_generate(_get_rag())
    facts: list = []
    truncated_any = False
    cut_parts = 0
    for i, content in enumerate(blocks, 1):
        if on_block is not None:
            try:
                on_block(i, len(blocks))
            except Exception:
                pass
        raw = generate_fn(
            content, system_prompt,
            temperature=0.7, top_p=0.95,
            max_new_tokens_setting="12288",      # matches the chat protocol's budget
            disable_rag=True,                    # a protocol is of the text, nothing else
            # BOTH guards off for the same deterministic reason the chat pass turns them
            # off: every line opens `[fact] (about: …) (class: …)`, a prefix that
            # tokenizes past the verbatim guard's 12-token window (the window sits wholly
            # inside it, identical on every line sharing an (about, class) pair — fires
            # on the 4th), and a run of such lines ALSO craters the diversity guard's
            # rolling distinct-token ratio, which was believed immune and proven not to
            # be (2026-08-18, live: one lookups pass halted at its 7th near-identical
            # line, another mid-think while drafting them — both reported 0 facts and
            # wedged the drain head). On THIS lane the repetitive shape is the normal
            # case, not the corner: a news digest is many events about one country. The
            # token cap bounds a genuine runaway.
            stop_on_repeat=False,
            degen_stop=False,
        )
        trunc = bool(getattr(generate_fn, "last_truncated", False))
        truncated_any = truncated_any or trunc
        # A generation cut inside an unclosed reasoning channel has no answer region at
        # all — and can normalize to UNTAGGED prose (the gemma-4 shape), where any
        # [fact]-looking line is deliberation, not protocol. `parse_facts` reads only the
        # answer region, but this case it cannot see from the text; the truncation flag
        # is the only signal, so the block is refused here rather than parsed. Observed
        # live 2026-08-18: a lookups pass spent its whole budget deliberating whether a
        # post-cutoff war article was fiction and its think drafts would have become the
        # protocol.
        if reasoning_text.truncated_before_answer(raw or "", trunc):
            cut_parts += 1
            continue
        facts.extend(til_facts.parse_facts(raw or "", truncated=trunc))

    written = til_facts.write_facts(snippet_path, record=record, kind=kind,
                                    facts=facts, run_id=run_id)
    return {"facts": len(facts), "parts": len(blocks), "truncated": truncated_any,
            # Blocks that never reached an answer region — reported separately, because
            # "the text established nothing" and "the pass failed before answering" must
            # not share the number 0 (the modules workbench draws the same line).
            "cut_parts": cut_parts,
            "written": str(written) if written else "",
            "counts": til_facts.class_counts(facts)}


def _record_facts_protocol(record: dict, kind: str, snippet_path, *,
                           run_id: str = "", emit: Optional[Callable] = None) -> None:
    """Best-effort :func:`run_facts_pass`, swallowing everything. Never fails a caller."""
    try:
        res = run_facts_pass(record, kind, snippet_path, run_id=run_id)
    except Exception:
        traceback.print_exc()
        return
    if emit is not None and res.get("facts"):
        try:
            emit("pass_progress", phase="ingestion",
                 message=(f"Protocol: recorded {res['facts']} fact(s) from the {kind} "
                          f"text ({res['parts']} part(s))."))
        except Exception:
            pass
    print(f"[til] facts protocol ({kind}): {res}", flush=True)


_READING_BLOCK_DEFAULT = (
    "Things you already know, looked up because this text seemed to touch them:\n\n"
    "{facts}\n\n"
    "Nobody's opinion is in here. The plain lines are things you know. Any line naming a "
    "text that reported something is exactly that and no more — what that source said at "
    "the time, which may since have turned out wrong; if you use one, keep the source "
    "attached to it the way it is written here, and never restate it as something simply "
    "known. None of this is what you are recapping, and most of what you read will have "
    "nothing to do with it: note where the text genuinely meets one and let the rest of "
    "the recap be about the text itself. Do not list them back, and do not turn a recap of "
    "the world into a recap of the people you know."
)


def _load_reading_block_template() -> str:
    """The wrapper around a fetched facts blob on the READING lane (``{facts}`` slot).

    A sibling of chat's ``facts_block_prompt.txt`` rather than a reuse of it, because that
    one is written for a pass about to answer somebody ("work into the reply") and this one
    is about to write down what it read. Default-written on first miss, like its chat
    counterpart, so it is tunable on disk without a restart.

    Its closing sentence is the anti-narrowing rule. Every candidate this box can currently
    offer is a claim about the one person Ava talks to, so a block injected into a news
    reading with no such instruction pulls the recap toward being about him rather than
    about the events — the whole risk of conditioning this lane at all.
    """
    path = _PROMPTS_DIR / "facts_reading_block_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_READING_BLOCK_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _READING_BLOCK_DEFAULT


def _reading_facts_config() -> dict:
    """Settings for the facts block on the reading lane. Read per pass, so no restart.

    Rides ``graph.enabled`` — this is the same channel as chat's, pointed at a different
    kind of material, so a box that has turned the channel off has turned this off too —
    and adds ``graph.gist_facts`` to cut it alone while leaving live chat conditioned.

    ``gist_max_claims`` is deliberately tighter than chat's 6. A chat turn answers one
    message and a fact that misses is quietly dropped; a recap is a few hundred characters
    total, so the same number of claims would be a much larger share of what the pass has
    in front of it.
    """
    try:
        import sys as _sys
        _server_dir = Path(__file__).resolve().parent.parent.parent
        if str(_server_dir) not in _sys.path:
            _sys.path.insert(0, str(_server_dir))
        from training.reflections_path import load_server_config
        cfg = ((load_server_config() or {}).get("graph") or {})
    except Exception:
        # Same reasoning as `generation._facts_channel_config`: an unreadable config is not
        # evidence the operator wanted this on, and it costs a generation per text.
        return {"enabled": False}
    return {
        "enabled": bool(cfg.get("enabled", True)) and bool(cfg.get("gist_facts", True)),
        "max_claims": int(cfg.get("gist_max_claims", 3) or 3),
        # The attributed-report channel: how many `report` claims — what a TEXT asserted,
        # as against what is simply known — the recap may carry. `0` restores the
        # knowledge-only list, i.e. the behaviour before 2026-08-15.
        "max_reports": max(0, int(cfg.get("gist_max_reports", 2) or 0)),
        "til_max_age_days": cfg.get("til_max_age_days"),
        "max_new_tokens": int(cfg.get("fetch_max_new_tokens", 512) or 512),
    }


def _describe_source(lane: str, ref: str) -> str:
    """A `(lane, ref)` rendered as the phrase a report is attributed to.

    Injected into `graph.blob.report_line` rather than imported by it, the pattern that file
    already uses for `words_match`: resolving a ref means reading the snippets tree, and
    `graph/` never imports the inference role.

    The phrase names the KIND as well as the date, because on this box that is the
    difference between evidence and a joke — the approved wiki list is chosen for tone, not
    truth, so "a Lurkmore article" and "a digest of world events" asserting one sentence are
    not the same thing, and a recap that flattens them has lost what it most needed to keep.

    An unresolvable ref falls back to the caller's own (the raw ref): `resolve_source`
    returns None for the ids that name no text and reports rather than guessing. Never
    raises — a failure here costs a hedge's precision, and the caller must not lose a recap
    over it.
    """
    try:
        if (lane or "") == "chat":
            # A chat-lane report should not exist — `(chat, stated)` folds to `position`,
            # which is offered to nobody — but say something honest rather than nothing if
            # that mapping ever changes underneath this.
            return "a conversation"
        path = til_gist.resolve_source(_TIL_SNIPPETS_DIR, str(ref or ""))
        if not path:
            return ""
        # `resolve_source` may land on the `.txt`; the record beside it carries the
        # provenance fields, and the two always share a stem.
        rec = {}
        for cand in (path.with_suffix(".json"), path):
            if cand.suffix == ".json" and cand.is_file():
                try:
                    rec = json.loads(cand.read_text(encoding="utf-8")) or {}
                except Exception:
                    rec = {}
                break
        kind = str(rec.get("kind") or path.parent.name or "").strip()
        date = str(rec.get("date") or "").strip()
        title = str(rec.get("title") or "").strip()
        if path.parent.name == "news" or kind == "serendipity":
            date = date or path.stem
            return f"a digest of world events from {date}" if date else "a news digest"
        site = str(rec.get("wiki") or rec.get("source") or "").strip()
        what = f"an article about {title}" if title else "an article"
        return f"{what} on {site}" if site else what
    except Exception:
        return ""


def _fetch_reading_facts(record: dict, kind: str) -> dict:
    """Stage 1 for a reading pass: which recorded facts this text touches. GPU, blocking.

    The reading-lane twin of ``generation._fetch_facts_block_sync``, and it goes through the
    same ``fact_fetch`` pass over the same candidate list with the same prompt and the same
    module spec — so what a live turn fetches and what a recap fetches cannot become two
    different pieces of code that merely resemble each other.

    Never raises. A recap that would otherwise have been written must not be lost because a
    retrieval channel failed, so every path returns an empty block with a named reason; the
    caller degrades to the unconditioned recap it produced before this existed.
    """
    cfg = _reading_facts_config()
    if not cfg["enabled"]:
        return {"text": "", "skipped": "disabled"}
    if _runtime.model is None:
        return {"text": "", "skipped": "no_model"}
    try:
        import datetime as _dt
        import sys as _sys
        _server_dir = Path(__file__).resolve().parent.parent.parent
        if str(_server_dir) not in _sys.path:
            _sys.path.insert(0, str(_server_dir))
        from core import fact_fetch
        from core.modules import MODULES
        from graph import store as graph_store

        doc = graph_store.read_tree()
        if doc is None:
            return {"text": "", "skipped": "no_tree"}
        spec = MODULES.get("fact_fetch")
        if spec is None:
            return {"text": "", "skipped": "no_module"}
        try:
            prompt = (_PROMPTS_DIR / spec.prompt_file).read_text(encoding="utf-8").strip()
        except Exception:
            prompt = ""
        if not prompt:
            return {"text": "", "skipped": "no_prompt"}

        reflect = _make_sync_reflect_generate(_get_rag())

        def generate(content: str, system_prompt: str, *, max_new_tokens: int) -> str:
            with activity_log.pass_context("fact_fetch:reading"):
                return reflect(
                    content, system_prompt,
                    # Selection, not expression — greedy, as on the chat path.
                    temperature=0.0, top_p=1.0,
                    max_new_tokens_setting=str(max_new_tokens),
                    before_session="", disable_rag=True,
                    disable_thinking=spec.disable_thinking,
                    stop_on_repeat=spec.stop_on_repeat)

        return fact_fetch.fetch_blob_for_text(
            doc=doc,
            framing=til_gist.framing_for(kind, record),
            title=str((record or {}).get("title") or ""),
            # The WHOLE text, where the recap below may read a truncated first part. Stage
            # 1 can afford it (the largest text on disk is 26k chars against a ~98k budget
            # at a 32,768-token window), and judging more can only find more connections.
            # The two would disagree only once `build_reading_content` starts truncating,
            # which it does not on the live corpus — its `truncated` stamp is the signal
            # that this seam needs closing.
            text=str((record or {}).get("text") or ""),
            generate=generate, prompt=prompt,
            window=_reflect_window(),
            now=_dt.date.today().isoformat(),
            max_new_tokens=cfg["max_new_tokens"],
            til_max_age_days=cfg["til_max_age_days"],
            max_claims=cfg["max_claims"],
            # The attributed-report channel. Non-zero widens the candidate list to
            # `READING_FACETS` AND budgets the render for what that admits — the two must
            # move together or the pass is offered claims nothing will carry.
            max_reports=cfg["max_reports"],
            describe_source=_describe_source)
    except Exception as e:
        traceback.print_exc()
        return {"text": "", "skipped": "error", "error": str(e)}


def generate_gist_text(record: dict, kind: str, *,
                       on_chunk: Optional[Callable] = None) -> dict:
    """Generate one text's recap and RETURN it. Writes nothing. Blocking (GPU thread).

    Split out of :func:`run_gist_pass` so the recap can be produced where the operator can
    still see it. The manual wander flow is a **dry run** — two passes are generated, shown,
    and only persisted if Apply is pressed — and the recap was breaking that contract by
    being generated *inside* Apply: an operator pressed a button meaning "keep what I just
    read" and got a third generation they never saw, whose prose then reaches live chat
    (`rag_engine._render_til_nomination`) and reach-out messages
    (`outreach._source_material`). So the wander flow now generates it as a third preview
    pass and Apply only writes what was shown.

    The backlog and news paths keep calling :func:`run_gist_pass`, which is this plus the
    write — there is no operator in either, so nothing is being reviewed and generating at
    the point of writing is correct there.
    """
    if _runtime.model is None:
        return {"skipped": "no_model"}
    try:
        system_prompt = (_PROMPTS_DIR / "til_gist_prompt.txt").read_text(
            encoding="utf-8").strip()
    except Exception:
        return {"skipped": "no_prompt"}          # degrade to previous behaviour, silently
    if not system_prompt:
        return {"skipped": "no_prompt"}

    content, truncated_input = til_gist.build_reading_content(
        record, kind,
        til_facts.block_budget_chars(int(_runtime.context_length or 32768)))
    if not content:
        return {"skipped": "no_text"}

    # Stage 1: what does this text touch that is already on record? Appended to the SYSTEM
    # prompt rather than to the reading content, so it lands where live chat puts it —
    # standing knowledge in the framing, ahead of the material of the moment — and the
    # reading content stays exactly what `build_reading_content` produced.
    facts = _fetch_reading_facts(record, kind)
    if facts.get("text"):
        system_prompt = (system_prompt + "\n\n"
                         + _load_reading_block_template().replace("{facts}",
                                                                  facts["text"]))

    activity_log.set_ambient_label(f"til_gist:{kind}")
    generate_fn = _make_sync_reflect_generate(_get_rag())
    raw = generate_fn(
        content, system_prompt,
        temperature=0.7, top_p=0.95,
        # A few hundred characters of product against a whole article of input. The budget
        # is for the thinking, not the answer — the thought ceiling holds back 30% of it
        # for the recap itself, which is ample.
        max_new_tokens_setting="4096",
        disable_rag=True,                        # a recap is of the text, nothing else
        # ON, unlike the protocol pass: this one emits prose, which is the shape the
        # verbatim loop guard is written for. The reason that pass turns it off — a fixed
        # `[fact] (about: …)` prefix filling the guard's window — does not exist here.
        on_chunk=on_chunk,
    )
    return {"raw": raw or "", "text": til_gist.clean_gist(raw or ""),
            "truncated_input": truncated_input,
            "truncated": bool(getattr(generate_fn, "last_truncated", False)),
            "facts_picked": facts.get("picked") or [],
            "facts_chars": len(facts.get("text") or ""),
            "facts_sources": facts.get("sources") or [],
            "facts_skipped": facts.get("skipped") or ""}


def write_gist_text(gen: dict, record: dict, kind: str, snippet_path, *,
                    run_id: str = "") -> dict:
    """Persist an already-generated recap (:func:`generate_gist_text`). No GPU.

    The write half, so a reviewed recap can be committed without regenerating it. Reports
    ``rejected`` for a generation `clean_gist` refused, which leaves the snippet in the
    backlog to be retried — the same outcome as never having generated one.
    """
    if not gen or gen.get("skipped"):
        return {"skipped": gen.get("skipped", "no_gist") if gen else "no_gist"}
    written = til_gist.write_gist(snippet_path, record=record, kind=kind,
                                  text=gen.get("raw", ""), run_id=run_id,
                                  truncated=bool(gen.get("truncated_input")))
    text = gen.get("text") or ""
    return {"chars": len(text), "written": str(written) if written else "",
            "rejected": bool(gen.get("raw") and not text)}


def run_gist_pass(record: dict, kind: str, snippet_path, *, run_id: str = "") -> dict:
    """Generate AND write the prose recap for one fetched text. Blocking (GPU thread).

    :func:`generate_gist_text` + :func:`write_gist_text`, for the callers with no operator
    in the loop — the backlog drain and the news ingestion phase. Nothing is being reviewed
    there, so generating at the point of writing is right; the manual wander flow splits
    the two so Apply persists only what was shown.

    The sibling of :func:`run_facts_pass` over the same material, and the two are
    deliberately separate generations rather than one pass emitting both: a protocol is
    exhaustive and a recap is selective, and asking for both at once gets a recap of the
    protocol. They also read the text differently — the protocol in parts (facts extract
    independently), the recap whole (see `til_gist.build_reading_content`).

    Best-effort like its sibling: a failure here costs a recap and never the reading it
    came from.
    """
    gen = generate_gist_text(record, kind)
    if gen.get("skipped"):
        return {"skipped": gen["skipped"]}
    out = write_gist_text(gen, record, kind, snippet_path, run_id=run_id)
    # What stage 1 did, reported whether or not it found anything: a recap written under
    # two facts and one written under none read the same, so an unreported channel is one
    # whose silence nobody notices. `facts_skipped` separates the ordinary empty answers
    # (`picked_nothing` — the expected result for most texts) from the broken ones
    # (`no_tree`, `generate_failed`, …).
    out.update({"truncated_input": gen.get("truncated_input", False),
                "truncated": gen.get("truncated", False),
                "facts_picked": gen.get("facts_picked") or [],
                "facts_chars": gen.get("facts_chars", 0),
                "facts_sources": gen.get("facts_sources") or [],
                "facts_skipped": gen.get("facts_skipped") or ""})
    return out


def _record_gist(record: dict, kind: str, snippet_path, *, run_id: str = "",
                 emit: Optional[Callable] = None) -> None:
    """Best-effort :func:`run_gist_pass`, swallowing everything. Never fails a caller."""
    try:
        res = run_gist_pass(record, kind, snippet_path, run_id=run_id)
    except Exception:
        traceback.print_exc()
        return
    if emit is not None and res.get("chars"):
        try:
            emit("pass_progress", phase="ingestion",
                 message=f"Recap: wrote {res['chars']} chars for the {kind} text.")
        except Exception:
            pass
    print(f"[til] gist ({kind}): {res}", flush=True)


# Backlog texts that produced nothing writable this process, keyed ("facts"|"gist", path).
# Both drains are oldest-first with a small per-run cap, so a text whose pass reliably
# yields nothing sits at the HEAD and re-runs its failing generation every drain — observed
# live (2026-08-15 → 08-18): two lookups texts consumed 2 of the 3 protocol slots on three
# consecutive runs while the backlog grew 32 → 38 → 48. In memory rather than on disk,
# mirroring `background_reflection._backfill_unproductive`: the usual cause is the prompt
# or the model, and a server restart is how those change.
_DRAIN_UNPRODUCTIVE: set = set()


def drain_gist_backlog(limit: int = 1, *, run_id: str = "") -> dict:
    """Recap up to *limit* snippets that have none. Blocking (GPU thread).

    The recap's counterpart of :func:`drain_facts_backlog`, and on any existing box the
    backlog IS the corpus — this artifact did not exist until now, so unlike the protocol
    (which at least runs at fetch time for new material) there is no material anywhere that
    already has one.
    """
    backlog = [(p, k) for p, k in til_gist.list_backlog(_TIL_SNIPPETS_DIR)
               if ("gist", str(p)) not in _DRAIN_UNPRODUCTIVE]
    if not backlog:
        return {"skipped": "empty_backlog"}
    done: list = []
    for path, kind in backlog[:max(1, int(limit))]:
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            _DRAIN_UNPRODUCTIVE.add(("gist", str(path)))     # unreadable ⇒ same wedge
            continue
        res = run_gist_pass(record, kind, path, run_id=run_id)
        if res.get("skipped") == "no_text":
            # A per-TEXT condition wearing a skip's shape: returning here (as a box-level
            # skip does) would let one empty snippet stop every drain behind it.
            _DRAIN_UNPRODUCTIVE.add(("gist", str(path)))
            continue
        if res.get("skipped"):
            return {"skipped": res["skipped"], "done": done,
                    "remaining": len(backlog) - len(done)}
        if not res.get("written"):
            # Rejected by the sanitizer, or the write failed: either way the slot bought
            # no file, and the file is what takes a snippet out of the queue.
            _DRAIN_UNPRODUCTIVE.add(("gist", str(path)))
        done.append({"snippet": Path(path).name, "kind": kind,
                     "chars": res.get("chars", 0), "rejected": res.get("rejected", False)})
    return {"done": done, "remaining": len(backlog) - len(done)}


def drain_facts_backlog(limit: int = 1, *, run_id: str = "") -> dict:
    """Record the protocol for up to *limit* snippets that have none. Blocking (GPU thread).

    Why this lane needs a backlog where chats did not: a chat is reflected once, soon
    after it happens, by a pass that was going to run anyway, so wiring the protocol into
    that pass covers the corpus. An article is read exactly once, at fetch, and never
    looked at again — so without this, the protocol would exist only for material fetched
    after the day it shipped and every article already on disk would stay unrecorded.
    It also picks up anything a parser fix superseded (`til_facts.has_facts` folds in the
    shared staleness probe), so this lane self-heals exactly as the chat one does.

    One snippet per call by default: it runs on the single GPU executor thread, and this
    is maintenance that must never be the reason an interactive turn waits.
    """
    backlog = [(p, k) for p, k in til_facts.list_backlog(_TIL_SNIPPETS_DIR)
               if ("facts", str(p)) not in _DRAIN_UNPRODUCTIVE]
    if not backlog:
        return {"skipped": "empty_backlog"}
    done: list = []
    for path, kind in backlog[:max(1, int(limit))]:
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            _DRAIN_UNPRODUCTIVE.add(("facts", str(path)))    # unreadable ⇒ same wedge
            continue
        res = run_facts_pass(record, kind, path, run_id=run_id)
        if res.get("skipped") == "no_text":
            # Per-text, not box state — see the gist drain's twin branch.
            _DRAIN_UNPRODUCTIVE.add(("facts", str(path)))
            continue
        if res.get("skipped"):
            return {"skipped": res["skipped"], "done": done,
                    "remaining": len(backlog) - len(done)}
        if not res.get("written"):
            # 0 facts, a cut generation, or a failed write: the slot bought no file, so
            # the text would sit at the drain's oldest-first head and fail identically
            # next run. Retired for the process; a restart (new prompt, new model)
            # retries it.
            _DRAIN_UNPRODUCTIVE.add(("facts", str(path)))
        done.append({"snippet": Path(path).name, "kind": kind,
                     "facts": res.get("facts", 0),
                     "cut": res.get("cut_parts", 0)})
    return {"done": done, "remaining": len(backlog) - len(done)}


def _ingestion_chunk(emit: Callable) -> Callable:
    """on_chunk that streams a learning pass into the Sleep tab as it generates.

    Emits ``phase_progress`` text deltas (already batched by the generate helper, so
    ~one event per ~80 chars, not per token) — the same channel the classic
    reflection streams on, so ingestion shows Ava reflecting live, like the manual
    Learn/Wander/Resolve buttons did during debugging."""
    def on_chunk(delta: str) -> None:
        if delta:
            emit("phase_progress", phase="ingestion", text=delta)
    return on_chunk

def ingest_news(emit: Callable) -> None:
    """Fetch the day's Wikipedia current-events digest; learn + apply if it changed."""
    budget = _read_wiki_budget()
    try:
        digest = _run_til_fetch()   # yesterday's digest; writes the snippet to til/
    except Exception as e:
        emit("pass_warning", phase="ingestion", message=f"News fetch failed: {e}")
        return
    date = digest.get("date") or ""
    if date and date == budget.get("last_news_date"):
        emit("pass_progress", phase="ingestion",
             message=f"News unchanged ({date}) — skipping.")
        return
    emit("pass_progress", phase="ingestion",
         message=f"News {date}: reflecting on the digest…")
    try:
        learned = _run_learning_pass(digest.get("text", ""), date, _ingestion_chunk(emit))
        if learned.get("contract_problem"):
            # See the twin branch in `ingest_lookup` for why this is a named event.
            emit("pass_warning", phase="ingestion",
                 message=("News learning pass abandoned its contract: "
                          f"{learned['contract_problem']}"))
        counts = _apply_learning_text_live(learned["text"], f"til:{date}" if date else "til",
                                           truncated=learned.get("truncated", False))
    except Exception as e:
        traceback.print_exc()
        emit("pass_warning", phase="ingestion", message=f"News learning failed: {e}")
        return
    budget["last_news_date"] = date
    _write_wiki_budget(budget)
    emit("pass_progress", phase="ingestion",
         message=(f"News {date}: applied {counts.get('rag', 0)} RAG, "
                  f"{counts.get('evict', 0)} resolved, {counts.get('weights', 0)} weights."))
    # The protocol of the digest — everything it stated, not just what was kept above.
    # Last, and best-effort: the learning output has already reached live memory, so a
    # failure here costs the record and nothing else. See `core.til_facts`.
    if digest.get("path"):
        runid = f"til:{date}" if date else "til"
        _record_facts_protocol(digest, "news", digest["path"], run_id=runid, emit=emit)
        # And the recap beside it: this digest is the single most-cited source id in the
        # live ask pool, so it is the text a raised question most often needs to carry.
        _record_gist(digest, "news", digest["path"], run_id=runid, emit=emit)

def pair_subjects(bound: list, unbound: list, extracted: list) -> tuple:
    """``(subjects, {subject: [questions]})`` — what to fetch, and who asked for it.

    Pure, and extracted from ``ingest_lookup`` because it was silently wrong in two ways
    that no observable behaviour distinguished from "the article was worth re-reading".

    **A subject may be asked about by several questions.** The map used to keep the FIRST
    and drop the rest, so a second ask about the same thing never reached the
    `write_lookup` marker, stayed `lookupable`, and had its article re-fetched on every
    ingestion — which re-extracted the same subject and dropped the marker again. An
    unbounded loop, entered by any two asks on one topic. Observed with the Gaza "Yellow
    Line": two asks, one marked, the other re-fetching the same page on 07-31, again on
    07-31 and again on 08-09; four more asks were in that state when this was fixed.

    **Position beats substring.** The extractor is handed the unbound questions in order
    and answers in order, so when the counts line up ``unbound[i]`` IS the question a
    subject came from. The old code preferred a substring scan — the first question
    *containing* the subject string — and kept the positional index only as a fallback,
    which is backwards: the scan lands on a different question whenever two of them
    mention the same thing, so a marker could retire the wrong ask and leave the right one
    to re-fetch. The scan survives for the unaligned case, where there is no position to
    trust.

    Subject order is preserved and each is listed once, so the fetch is unchanged: one
    article per subject, however many questions want it.
    """
    subj_to_q: dict = {}
    subjects: list = []

    def add(subject, question):
        subject = (subject or "").strip()
        if not subject or not question:
            return
        if subject not in subj_to_q:
            subj_to_q[subject] = []
            subjects.append(subject)
        if question not in subj_to_q[subject]:
            subj_to_q[subject].append(question)

    for r in (bound or []):
        add(r.get("lookup"), r.get("content"))

    uq = [r.get("content") for r in (unbound or [])]
    aligned = len(extracted or []) == len(uq)
    for idx, s in enumerate(extracted or []):
        q = uq[idx] if aligned else None
        if q is None:
            q = next((c for c in uq if c and (s or "").lower() in c.lower()), None)
        add(s, q)
    return subjects, subj_to_q


def _lookup_source_id(found: list) -> str:
    """Provenance id for what a lookup learning pass just read.

    This lane reads a DIGEST of several fetched articles in one pass, so in general no
    single id can name its source, and the pass has always filed everything under the bare
    string ``lookup`` — an id that resolves to nothing, leaving 3 live asks on this box
    unable to recover the text behind them.

    The **one-article case is exact and common**, so it is fixed here: the digest is then a
    single text with a snippet on disk, and its `<kind>/<stem>` id resolves like any other
    (`til_gist.resolve_source`). Several articles still yield ``lookup``, unchanged and
    still unresolvable — the honest answer, since attributing a joint reading to whichever
    article came first would put the wrong text behind a question. Fixing THAT means
    reading each article in its own pass, which is a change to what ingestion costs rather
    than to what it records, and so is deliberately not bundled here.
    """
    paths = [rec.get("path") for _subject, rec, _q in (found or [])
             if isinstance(rec, dict) and rec.get("path")]
    if len(paths) == 1:
        try:
            # Kind from the file's own parent dir, which is the rule `graph.read` already
            # uses for these protocols — and not a literal, since a literal is exactly how
            # `SOURCE_KINDS` came to say `lookup` for a directory named `lookups`.
            return til_facts.input_id(paths[0], Path(paths[0]).parent.name)
        except Exception:
            return "lookup"
    return "lookup"


def ingest_lookup(emit: Callable) -> None:
    """Resolve open [ask:search]: bound titles direct, rest via the clean base; apply."""
    mem = ReflectionMemory(_MEMORY_DIR)
    recs = [{"content": (q.get("content") or "").strip(),
             "key": q.get("key") or "",
             "lookup": (q.get("lookup") or "").strip()}
            for q in mem.lookupable_questions()]
    recs = [r for r in recs if r["content"]]
    if not recs:
        emit("pass_progress", phase="ingestion",
             message="Lookup: no open [ask:search] questions.")
        return

    bound = [r for r in recs if r["lookup"]]
    unbound = [r for r in recs if not r["lookup"]]
    extracted: list = []
    if unbound:
        emit("pass_progress", phase="ingestion",
             message=f"Lookup: extracting {len(unbound)} subject(s) on the clean base...")
        try:
            extracted = _run_extract_subjects_sync(
                [r["content"] for r in unbound],
                on_log=lambda m: emit("pass_progress", phase="ingestion", message=m))
        except Exception as e:
            traceback.print_exc()
            emit("pass_warning", phase="ingestion",
                 message=f"Subject extraction failed: {e}")
            extracted = []
    subjects, subj_to_q = pair_subjects(bound, unbound, extracted)
    if not subjects:
        emit("pass_progress", phase="ingestion", message="Lookup: no subjects to fetch.")
        return

    try:
        fetched = _fetch_articles_sync(subjects)
    except Exception as e:
        traceback.print_exc()
        emit("pass_warning", phase="ingestion", message=f"Article fetch failed: {e}")
        return

    key_by_content = {r["content"]: r["key"] for r in recs}
    found: list = []
    attempted: dict = {}
    for subject, rec in fetched:
        qs = subj_to_q.get(subject) or []
        if rec is not None:
            found.append((subject, rec, qs))
        # EVERY question that asked about this subject is marked, not just the one that
        # happened to be first: one fetch of one article answers them all or none of
        # them, and re-fetching the identical page cannot change that. This is the fix
        # for the loop — see `_add`.
        for q in qs:
            attempted[q] = (subject, rec is not None, rec["title"] if rec else "")

    # Fetch-once markers (live), so an article that doesn't resolve a question
    # leaves it open but no longer re-fetched.
    writer = _get_reflection_writer()
    runid = f"lookup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for content, (subject, found_bool, title) in attempted.items():
        key = key_by_content.get(content)
        if key:
            writer.write_lookup(key=key, subject=subject, found=found_bool,
                                title=title, run_id=runid)

    if not found:
        emit("pass_progress", phase="ingestion",
             message="Lookup: no articles found for the open questions.")
        return
    emit("pass_progress", phase="ingestion",
         message=f"Lookup: reflecting on {len(found)} answer(s)…")
    try:
        digest = _assemble_lookup_digest([r["content"] for r in recs], found)
        learned = _run_lookup_learning_pass(digest, _ingestion_chunk(emit))
        if learned.get("contract_problem"):
            # The apply below is harmless on a contract-less text (zero items parse),
            # so the check's whole value is the NAMED event: without it a pass that
            # wrote an essay instead of its sections reports "applied 0 RAG" — the
            # same line an uneventful digest produces.
            emit("pass_warning", phase="ingestion",
                 message=("Lookup learning pass abandoned its contract: "
                          f"{learned['contract_problem']}"))
        counts = _apply_learning_text_live(learned["text"], _lookup_source_id(found),
                                           truncated=learned.get("truncated", False))
    except Exception as e:
        traceback.print_exc()
        emit("pass_warning", phase="ingestion", message=f"Lookup learning failed: {e}")
        return
    emit("pass_progress", phase="ingestion",
         message=(f"Lookup: applied {counts.get('rag', 0)} RAG, "
                  f"{counts.get('evict', 0)} resolved."))


# --------------------------------------------------------------------------- #
# GPU-free self-test                                                           #
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    """Run: ``python -m core.til_wander``. Covers the pure lookup-pairing logic.

    Only the parts that need no GPU, no model and no data on disk — which is where the
    re-fetch loop lived: a mapping bug with no observable symptom except an article being
    fetched again, and nothing in the pipeline treats that as an error.
    """
    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    print("pair_subjects — one fetch per subject, every asker marked")
    q1 = 'What is the "Yellow Line" in the context of the Gaza Strip?'
    q2 = 'Is the "Yellow Line" in Gaza a physical barrier or a coordinate-based line?'
    unbound = [{"content": q1, "lookup": ""}, {"content": q2, "lookup": ""}]
    subjects, m = pair_subjects([], unbound, ["Yellow Line (Gaza)", "Yellow Line (Gaza)"])
    check("one subject, so one fetch", subjects, ["Yellow Line (Gaza)"])
    check("...but BOTH asks are recorded against it — the re-fetch loop",
          m["Yellow Line (Gaza)"], [q1, q2])

    print("\nthe positional pairing, which the substring scan used to override")
    a = "Tell me about the Board of Peace and the Yellow Line."
    b = "What is the Yellow Line?"
    subjects, m = pair_subjects([], [{"content": a, "lookup": ""},
                                     {"content": b, "lookup": ""}],
                                ["Board of Peace", "Yellow Line"])
    check("each subject lands on the question it was extracted from",
          (m["Board of Peace"], m["Yellow Line"]), ([a], [b]))
    # Unaligned counts leave no position to trust, so the scan is the fallback it was
    # always meant to be.
    _, m2 = pair_subjects([], [{"content": a, "lookup": ""},
                               {"content": b, "lookup": ""}], ["Yellow Line"])
    check("unaligned falls back to the substring scan", m2["Yellow Line"], [a])

    print("\nbound questions, ordering and the degenerate inputs")
    subjects, m = pair_subjects([{"content": "q-bound", "lookup": "Kyiv"}],
                                [{"content": "q-free", "lookup": ""}], ["Devs"])
    check("a bound question uses its own title", m["Kyiv"], ["q-bound"])
    check("subject order is preserved (bound first, then extracted)",
          subjects, ["Kyiv", "Devs"])
    check("a duplicate ask is listed once",
          pair_subjects([{"content": "q", "lookup": "X"},
                         {"content": "q", "lookup": "X"}], [], [])[1]["X"], ["q"])
    check("an empty subject is not fetched", pair_subjects([], [], [""])[0], [])
    check("a subject with no question is not fetched",
          pair_subjects([], [{"content": "", "lookup": ""}], ["X"])[0], [])
    check("nothing in, nothing out", pair_subjects([], [], []), ([], {}))

    print("\n_assemble_lookup_digest — every asker sees its answer")
    rec = {"title": "Yellow Line (Gaza Strip)", "source_url": "http://x", "text": "body"}
    digest = _assemble_lookup_digest([q1, q2], [("Yellow Line (Gaza)", rec, [q1, q2])])
    check("both questions are echoed, so both can be [resolved]",
          digest.count("QUESTION:"), 2)
    check("...against one copy of the article", digest.count("ARTICLE —"), 1)
    check("a single question still renders (the pre-list shape)",
          _assemble_lookup_digest([q1], [("S", rec, q1)]).count("QUESTION:"), 1)
    check("no question falls back to the subject",
          "(about S)" in _assemble_lookup_digest([], [("S", rec, [])]), True)

    print("\n_lookup_source_id — provenance only when one text was read")
    check("one article names itself",
          _lookup_source_id([("S", {"path": "/a/b/lookups/x.txt"}, [])]),
          "lookups/x.txt")
    check("several articles stay unresolvable rather than misattributed",
          _lookup_source_id([("S", {"path": "/a/lookups/x.txt"}, []),
                             ("T", {"path": "/a/lookups/y.txt"}, [])]), "lookup")
    check("no article, no id", _lookup_source_id([]), "lookup")

    print("\nlearning_contract_problem — a failed pass must not read as a quiet page")
    structured = ("<think>weighing what to keep.</think>\n"
                  "## WEIGHTS\n\n## RAG\n- [fact] a thing (trigger: cue)\n\n## RESOLVED\n")
    check("a structured reflection passes", learning_contract_problem(structured), "")
    # The prompt permits keeping nothing — empty sections are still the contract.
    empty_ok = "<think>nothing here.</think>\n## WEIGHTS\n\n## RAG\n\n## RESOLVED\n"
    check("empty sections still pass", learning_contract_problem(empty_ok), "")
    # A lone tagged line with no headers is partial compliance, not abandonment.
    check("a bare tagged line passes",
          learning_contract_problem("</think>[ask:user] Ну и как тебе это?"), "")
    # The observed failure (2026-08-18): a persona essay in place of the sections.
    essay = ("<think>I should apply my analytical lens.</think>\n"
             "Знаешь, в этом материале я нахожу почти византийскую глубину — "
             "полноценная семиотическая карта человеческого желания.")
    check("an essay is named, not passed",
          "prose" in learning_contract_problem(essay), True)
    # Drafted sections INSIDE the think do not satisfy a contract the answer abandoned.
    check("sections drafted in the think do not count",
          "prose" in learning_contract_problem(
              "<think>## RAG\n- [fact] draft</think>Просто эссе о дружбе."), True)
    check("no answer region is its own named failure",
          "answer region" in learning_contract_problem("<think>cut mid-thought"), True)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
