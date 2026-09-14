"""The bridge to `assoc/` — the associative facts library (ASSOCIATIVE_MEMORY.md) — wired
in beside the facts-tree channel, not over it.

Three surfaces, in the order they were built:

  * **feed** (`run_feed_blocking`, an idle job): `ava_import.sync` over the live corpus —
    transcripts under `data/chats/`, texts under `data/til/snippets/<kind>/`, each with
    the `.facts.json` protocol Ava's own passes wrote imported as the document's protocol
    and the `.summary.json` gist as its summary — then `Library.rebuild()`. The library
    root (`data/assoc/`) is derived and disposable, like the facts tree. **The sync and
    the rebuild run in a SUBPROCESS** (2026-09-11, `core.assoc_feed_worker`; `assoc.
    feed_subprocess: false` restores in-process): a rebuild is minutes of pure-Python
    work — lemmatization, the codebook, dedup, PageRank — that never releases the GIL,
    where a generation spends its time in CUDA kernels that do, so in-process it starved
    the inference server's asyncio loop for the whole wake and a client connecting
    meanwhile timed out on the opening handshake. The library never loads the LLM, so
    the worker needs nothing from this process but paths and config; the one model-bound
    piece of the feed, the typed-relation pass, runs HERE, AFTER the worker's rebuild and
    over the build it produced (`assoc.rebuild.relation_pass`, the same function the
    rebuild calls — so it never spends a generation on a claim the rebuild was about to
    replace), writing the library's relation cache; the NEXT wake's rebuild folds that
    cache in (and is forced when the cache holds relations the build lacks, so an
    unchanged corpus still folds). A relation therefore reaches the edge table one wake
    later than it used to, which on an hourly job is nothing.
  * **fetch** (`fetch_block`): stage 1 of a chat turn, the library's `pull → select →
    inject` behind the same result shape `generation._fetch_facts_block_sync` returns, so
    the Chat tab's facts block, the Debug view and the past-chat nominations need no new
    protocol. Off unless `assoc.enabled`; the tree channel keeps running otherwise.
  * **workbench** (`prepare_for_module` / `finish_for_module`): the `assoc_fetch` module in
    `core.modules`, so the two channels can be compared on the same chat with real output
    before either is trusted — the discipline `fact_fetch` was built under.

The select pass runs through the reflect seam like every other thinking-off judgement on
the box; its prompt is `prompts/assoc_fetch_prompt.txt` (default-written from the library's
own `select_prompt.txt`), handed to the library so the live turn and the workbench run ONE
pass. The library never loads a model; it is handed `generate_fn` per call.

GPU-free self-test: ``python -m core.assoc_bridge`` (hash embedder, temp dirs).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_SERVER_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_DIR = _SERVER_DIR.parent
for _p in (str(_SERVER_DIR), str(_REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_lock = threading.Lock()
_lib = None
_lib_key: tuple = ()
_CHATS_DIRS: list = []
_TIL_DIR: Optional[Path] = None
_ROOT: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None
_load_config: Optional[Callable[[], dict]] = None
_MEMORY_DIR: Optional[Path] = None
_get_reflection_writer: Optional[Callable] = None
_get_rag: Optional[Callable] = None
_make_sync_reflect_generate: Optional[Callable] = None
_write_session: Optional[Callable] = None
_target_user: Optional[Callable[[], str]] = None
_model_loaded: Optional[Callable[[], bool]] = None
# The loaded adapter's name (or the base model id with no adapter): recorded on every
# protocol the witness writes as its `model_id`, which is what lets a refused document be
# re-witnessed once per NEW adapter. It was the constant "ava" until 2026-09-12, so no
# protocol on the box could tell one adapter from another.
_model_name: Optional[Callable[[], str]] = None

PROMPT_FILE = "assoc_fetch_prompt.txt"
FEED_RESULT_MARKER = "ASSOC_FEED_RESULT "   # the worker's last stdout line: this + JSON
_INFERENCE_DIR = Path(__file__).resolve().parent.parent   # cwd for `-m core.assoc_feed_worker`
AHA_PROMPT_FILE = "assoc_aha_prompt.txt"
CONTEXT_TURNS = 6
CONTEXT_CHARS = 2400
STIMULUS_DOCS = 40             # the freshest fed documents kept as the next aha's stimulus
STIMULUS_FALLBACK_DOCS = 3     # with nothing fed since the last run: the newest chats with a protocol
STIMULUS_PER_WAKE = 5          # conversations weighed per wake, newest last
AHA_JUDGE_MAX_NEW_TOKENS = 1536
AHA_OPENER_MAX_NEW_TOKENS = 2048


def configure(*, chats_dirs, til_snippets_dir, root, prompts_dir, load_config=None,
              memory_dir=None, get_reflection_writer=None, get_rag=None, make_sync_reflect_generate=None,
              write_session=None, target_user=None, model_loaded=None, model_name=None) -> None:
    """Wire the paths once at startup. `load_config` returns the server config dict. The
    rest back the aha job: the reflection memory (open asks → needs), the writer (surface
    stamps), the reflect seam (judge + opener), the reversed-session writer and the
    addressee (both outreach's own, injected so the self-test can stub them)."""
    global _CHATS_DIRS, _TIL_DIR, _ROOT, _PROMPTS_DIR, _load_config
    global _MEMORY_DIR, _get_reflection_writer, _get_rag, _make_sync_reflect_generate, _write_session, _target_user
    global _model_loaded, _model_name
    _CHATS_DIRS = [Path(p) for p in chats_dirs]
    _TIL_DIR = Path(til_snippets_dir) if til_snippets_dir else None
    _ROOT = Path(root)
    _PROMPTS_DIR = Path(prompts_dir)
    _load_config = load_config
    _MEMORY_DIR = Path(memory_dir) if memory_dir else None
    _get_reflection_writer = get_reflection_writer
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _write_session = write_session
    _target_user = target_user
    _model_loaded = model_loaded
    _model_name = model_name


def config() -> dict:
    """The `assoc` block of server_config.json, read per call so a flip needs no restart.

    `enabled` (default True since 2026-09-10 — it shipped off for a day, until the operator
    asked for a pull-and-go default) hands the LIVE fetch to the library; while the library
    has no build yet (a fresh box before its first feed) the turn falls through to the tree
    channel, so a pull never costs a turn its facts. `feed` (default True) keeps the library
    built; `false` on `enabled` restores the tree channel exactly.
    """
    cfg = raw_config()
    if cfg is None:
        return {"enabled": False, "feed": False}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "feed": bool(cfg.get("feed", True)),
        # The feed's sync + rebuild in their own process (see the module docstring), and
        # how long that process may run before it is killed and the wake reported failed.
        "feed_subprocess": bool(cfg.get("feed_subprocess", True)),
        "feed_timeout_s": int(cfg.get("feed_timeout_s", 7200) or 7200),
        "tier": int(cfg.get("tier", 2) or 2),
        "budget_tokens": int(cfg.get("budget_tokens", 3000) or 3000),
        "device": str(cfg.get("device") or "cuda"),
        "embedder": str(cfg.get("embedder") or "bge-m3"),
        "max_picks": int(cfg.get("max_picks", 6) or 6),
        # The aha job: needs from the open asks, candidates from what was fed since the
        # last wake, one judgement per wake by default (a thinking-on pass at 4.5 tok/s
        # is minutes), the reach-out gate on the message.
        "aha": bool(cfg.get("aha", True)),
        "aha_max_judgements": int(cfg.get("aha_max_judgements", 1) or 1),
        # The library's own witness over her chats (replacing the imported `chat_facts`
        # protocol, newest first, N documents per wake — ~5 min each at 4.5 tok/s) and the
        # typed-relation pass folded into the feed's rebuild, M claims per wake.
        "witness": bool(cfg.get("witness", True)),
        "witness_per_wake": int(cfg.get("witness_per_wake", 3) or 3),
        "relations": bool(cfg.get("relations", True)),
        "relations_per_wake": int(cfg.get("relations_per_wake", 120) or 120),
        # The pivot (§6): from the newest conversation's words, one sense (or sound) jump
        # per wake, a thinking-on creative pass, and a reach-out only when it decides so.
        "pivot": bool(cfg.get("pivot", True)),
        "pivot_bridges": [str(b) for b in (cfg.get("pivot_bridges") or ["sense"])],
    }


def raw_config() -> Optional[dict]:
    """The `assoc` block exactly as configured (`{}` when absent), or None when the config
    could not be read — what the feed worker is handed so it resolves the same knobs."""
    try:
        if _load_config is not None:
            return dict((_load_config() or {}).get("assoc") or {})
        from training.reflections_path import load_server_config
        return dict((load_server_config() or {}).get("assoc") or {})
    except Exception:
        return None


# ── the library ───────────────────────────────────────────────────────────────

def _make_embedder(cfg: dict):
    if cfg["embedder"] == "hash":
        from assoc.dense import HashEmbedder
        return HashEmbedder()
    from assoc.dense import BgeM3Embedder
    device = cfg["device"]
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
    return BgeM3Embedder(device=device)


def library(cfg: Optional[dict] = None):
    """The process-wide Library, built on first use. `None` when the package or its
    embedder is unavailable — a named skip at the call sites, never an exception."""
    global _lib, _lib_key
    cfg = cfg or config()
    key = (str(_ROOT), cfg["embedder"], cfg["device"], cfg["tier"], cfg["budget_tokens"], cfg["max_picks"])
    with _lock:
        if _lib is not None and _lib_key == key:
            return _lib
        if _ROOT is None:
            return None
        try:
            _lib = build_library(cfg, _ROOT)
            _lib_key = key
        except Exception as e:
            print(f"[assoc] library unavailable: {type(e).__name__}: {e}", flush=True)
            _lib = None
        return _lib


def build_library(cfg: dict, root: Path):
    """Construct the Library for *cfg* over *root* — the ONE constructor, so the feed worker
    (another process) opens exactly the library this process reads."""
    from assoc.budget import Budget
    from assoc.library import Library
    from assoc.selection import AVA_POLICY, Policy
    policy = Policy(name="ava", withhold_subjects=AVA_POLICY.withhold_subjects, max_picks=cfg["max_picks"])
    return Library(root, embedder=_make_embedder(cfg), budget=Budget(total=cfg["budget_tokens"]),
                   policy=policy, cache_dir=root / "cache", tier=cfg["tier"])


def _has_build(lib) -> bool:
    try:
        return lib.build is not None
    except Exception:
        return False


# ── the feed (idle job) ───────────────────────────────────────────────────────

def run_feed_blocking() -> dict:
    """Sync the live corpus into the library and rebuild if anything changed. Never raises.

    Three steps, in this order: (1) `sync` + `rebuild`, in a worker process by default
    (`feed_sync_and_rebuild` is the body either way); the rebuild runs the relation pass
    cache-only, and is forced — corpus unchanged or not — when the cache holds relations
    the current build lacks (`_unfolded_relations`, written by step 3 of the previous
    wake); (2) this process picks the new build up (`Library.reload`) and pushes what was
    fed as the next aha's stimulus; (3) the typed-relation pass, in THIS process (it needs the
    model), over the NEW build's claims that carry no relation yet, bounded per wake,
    into the cache the next rebuild folds."""
    cfg = config()
    if not cfg["feed"]:
        return {"skipped": "assoc.feed is off"}
    lib = library(cfg)
    if lib is None:
        return {"skipped": "library unavailable (assoc package or embedder missing)"}
    try:
        t0 = time.time()
        force_rebuild = _unfolded_relations(lib) > 0
        if cfg["feed_subprocess"]:
            rep = _feed_in_subprocess(cfg, force_rebuild=force_rebuild, timeout_s=cfg["feed_timeout_s"])
            lib.reload()        # the worker wrote the store and the build; read them fresh
        else:
            rep = feed_sync_and_rebuild(lib, cfg, force_rebuild=force_rebuild, log=lambda s: print(s, flush=True))
        if rep.get("error"):
            return rep
        if rep.get("doc_ids"):
            _push_stimulus(lib, rep["doc_ids"])
        rel = _run_relation_pass_here(lib, cfg)
        if rel is not None:
            # The pass set `rel` on the LOADED claims; the build on disk carries them only
            # after the next fold. Drop the in-memory copy so nothing reads relations the
            # edge table does not have yet — and so `_unfolded_relations` sees them.
            lib.reload()
            (rep.setdefault("rebuild", {}) if rep.get("rebuild") else rep)["relations"] = rel
        rep["seconds"] = round(time.time() - t0, 1)
        if not rep.get("chats") and not rep.get("til") and not rep.get("rebuild") and not (rel and rel.get("claims")):
            return {"skipped": "the library is current", **{k: rep[k] for k in ("scanned", "unchanged")}}
        return rep
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


def feed_sync_and_rebuild(lib, cfg: dict, *, force_rebuild: bool = False, log=print) -> dict:
    """The feed's GIL-bound body: `ava_import.sync` over the live tree, then the rebuild
    when the corpus moved, the build is stale, the library has no build, or the caller
    says so (the relation cache holds what the build lacks). No generation here — the
    rebuild runs the relation pass cache-only — so it can run in a process that holds no
    model. Called by
    `core.assoc_feed_worker` (the default) and in-process under `feed_subprocess: false`."""
    from assoc.ava_import import sync
    rep = sync(lib, _CHATS_DIRS, _TIL_DIR, log=log, rebuild=False)
    st = lib.staleness() if _has_build(lib) else {"scope": "full", "reasons": ["no_build"]}
    rep["staleness"] = st
    if rep.get("chats") or rep.get("til") or st.get("scope") != "none" or force_rebuild:
        r = lib.rebuild(None if _has_build(lib) and not rep.get("first") else "full")
        rep["rebuild"] = {k: r.get(k) for k in ("scope", "counts", "seconds", "contests", "families", "relations")}
    return rep


def _unfolded_relations(lib) -> int:
    """Claims in the current build carrying no relation whose cached pass result has one:
    written by the last wake's pass (step 3), folded by the next rebuild — which this
    count forces when the corpus alone would not have."""
    b = lib.build
    if b is None:
        return 0
    from assoc.relations import RelationCache
    from assoc.rebuild import relations_cache_path
    cache = RelationCache(relations_cache_path(lib.store))
    return sum(1 for c in b.claims.values() if not c.get("rel") and cache.get(c))


def _run_relation_pass_here(lib, cfg: dict) -> Optional[dict]:
    """Step (3) of the feed: the typed-relation pass in the inference process, over the
    build the rebuild just produced. None when it did not run (off, no model, no build,
    nothing to do)."""
    if not cfg["relations"]:
        return None
    gen = _relations_generate_fn(cfg)
    if gen is None:
        return None
    b = lib.build
    if b is None or _claims_without_relation(lib) == 0:
        return None
    from assoc.rebuild import relation_pass, relations_cache_path
    return relation_pass(b.claims, b.codebook, gen, relations_cache_path(lib.store),
                         limit=cfg["relations_per_wake"])


def _feed_in_subprocess(cfg: dict, *, force_rebuild: bool, timeout_s: int) -> dict:
    """Run `feed_sync_and_rebuild` in `python -m core.assoc_feed_worker`, streaming its
    output through this process's stdout (so the activity journal's tee sees the
    `[import]` lines exactly as before) and returning the JSON report it prints last.
    Blocking on the pipe releases the GIL, which is the point. A worker that exceeds
    *timeout_s* is killed and the wake reported failed; a worker that dies without a
    report is reported with its last lines."""
    payload = {
        "chats_dirs": [str(p) for p in _CHATS_DIRS],
        "til_snippets_dir": str(_TIL_DIR) if _TIL_DIR is not None else None,
        "root": str(_ROOT), "prompts_dir": str(_PROMPTS_DIR),
        "assoc": raw_config() or {}, "force_rebuild": force_rebuild,
    }
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # The embedder is on disk after the first wake; spare the worker the Hub HEAD requests
    # (seconds per load, and a failure with no network). Same rule fast_load applies to
    # the base model.
    if cfg["embedder"] != "hash" and env.get("HF_HUB_OFFLINE") is None:
        try:
            from core.fast_load import snapshot_is_cached
            from assoc.dense import BgeM3Embedder
            if snapshot_is_cached(BgeM3Embedder.id):
                env["HF_HUB_OFFLINE"] = "1"
        except Exception:
            pass
    from collections import deque
    tail: deque = deque(maxlen=12)
    result: Optional[dict] = None
    proc = subprocess.Popen([sys.executable, "-m", "core.assoc_feed_worker"], cwd=str(_INFERENCE_DIR), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(timeout_s, _kill)
    timer.daemon = True
    timer.start()
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload))
        proc.stdin.close()
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(FEED_RESULT_MARKER):
                try:
                    result = json.loads(line[len(FEED_RESULT_MARKER):])
                except Exception as e:
                    result = {"error": f"unreadable feed worker result: {e}"}
                continue
            tail.append(line)
            print(line, flush=True)
        rc = proc.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        return {"error": f"feed worker killed after {timeout_s}s", "tail": list(tail)}
    if result is None:
        return {"error": f"feed worker exited {rc} without a result", "tail": list(tail)}
    return result


def _claims_without_relation(lib) -> int:
    b = lib.build
    if b is None:
        return 0
    return sum(1 for c in b.claims.values() if not c.get("rel") and not c["claim_id"].startswith("rm-") and c.get("entities"))


def _relations_generate_fn(cfg: dict):
    """The relation pass's generate seam, or None when no model is loaded (the feed then
    rebuilds without it, exactly as before)."""
    if _make_sync_reflect_generate is None or _model_loaded is None or not _model_loaded():
        return None
    try:
        reflect = _make_sync_reflect_generate(_get_rag() if _get_rag else None)
    except Exception:
        return None
    return make_generate_fn(reflect, label="assoc_relations")


def describe_feed(r: dict) -> str:
    if r.get("skipped"):
        return f"Assoc feed: {r['skipped']}"
    if r.get("error"):
        return f"Assoc feed: FAILED ({r['error']})"
    rb = r.get("rebuild") or {}
    counts = rb.get("counts") or {}
    rel = rb.get("relations") or r.get("relations") or {}
    rel_txt = (f"; relations: {rel.get('accepted', 0)} accepted over {rel.get('claims', 0)} claims"
               + (f", {rel['deferred']} deferred" if rel.get("deferred") else "")
               + (f", {rel['refused']} batch(es) REFUSED by the model ({rel.get('refused_claims', 0)} claims left uncached, retried next wake)"
                  if rel.get("refused") else "")) if rel and not rel.get("skipped") else ""
    return (f"Assoc feed: +{r.get('chats', 0)} chats, +{r.get('til', 0)} texts "
            f"({r.get('chat_facts', 0) + r.get('til_facts', 0)} facts imported, {r.get('ungrounded', 0)} ungrounded"
            + (f", {r['kept_own']} kept the library's own protocol" if r.get("kept_own") else "") + ") → "
            f"{rb.get('scope', '-')} rebuild: {counts.get('claims', 0)} claims over {counts.get('chunks', 0)} chunks "
            f"in {counts.get('current', 0)} documents{rel_txt} [{r.get('seconds', 0)} s]")


# ── the witness (idle job) ────────────────────────────────────────────────────
# The library's own chat witness (`prompts/witness_chat.txt`: third person, names resolved,
# named entities only, inline relations) over her transcripts — replacing, newest first and
# N per wake, the `chat_facts` protocol the feed imported. `chat_facts` keeps running on
# Ava's side (the facts tree reads it); the library stops importing it for a document it has
# witnessed itself. Thinking off (the milestone-1 measurement); ~5 min per chat at 4.5 tok/s.

def run_witness_blocking(on_stage: Optional[Callable[[dict], None]] = None) -> dict:
    cfg = config()
    if not cfg["witness"]:
        return {"skipped": "assoc.witness is off"}
    lib = library(cfg)
    if lib is None:
        return {"skipped": "library unavailable"}
    if _model_loaded is None or not _model_loaded():
        return {"skipped": "no_model"}
    if _make_sync_reflect_generate is None:
        return {"skipped": "no_generate"}
    try:
        mid = witness_model_id()
        todo = lib.pending(include_imported=True, kinds={"chat"}, current_model=mid)
        if not todo:
            return {"skipped": "every chat carries the library's own protocol"}
        reflect = _make_sync_reflect_generate(_get_rag() if _get_rag else None)
        gen = make_generate_fn(reflect, label="assoc_witness")
        t0 = time.time()
        reports = []
        for doc_id in todo[: cfg["witness_per_wake"]]:
            doc = lib.store.document(doc_id)
            if on_stage is not None:
                try:
                    on_stage({"stage": "witnessing", "key": doc.key if doc else doc_id})
                except Exception:
                    pass
            from assoc.witness import run_witnesses
            b = lib.build
            r = run_witnesses(lib.store, doc_id, generate_fn=gen, model_id=mid,
                              codebook=b.codebook if b else None, embedder=lib.embedder)
            reports.append({"key": r.get("key"), "facts": r.get("facts"), "ungrounded": r.get("ungrounded"),
                            "groups": (r.get("llm") or {}).get("groups"), "failed_groups": (r.get("llm") or {}).get("failed_groups"),
                            "refused_chunks": r.get("refused_chunks") or 0, "fallback_facts": r.get("fallback_facts") or 0,
                            "seconds": r.get("seconds")})
        _push_stimulus(lib, [d for d in todo[: cfg["witness_per_wake"]]])
        return {"witnessed": len(reports), "remaining": max(0, len(todo) - len(reports)), "reports": reports,
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


def witness_model_id() -> str:
    """What the witness records as `model_id`: the loaded adapter's name, else the base id,
    else the historical constant. Must be stable across wakes under one adapter and differ
    across adapters — the re-queue rule compares it verbatim."""
    try:
        name = (_model_name() if _model_name else "") or ""
    except Exception:
        name = ""
    return name or "ava"


def witness_consumed(r: dict) -> bool:
    return not (isinstance(r, dict) and str(r.get("skipped") or "") in ("no_model",))


def describe_witness(r: dict) -> str:
    if r.get("skipped"):
        return f"Assoc witness: {r['skipped']}"
    if r.get("error"):
        return f"Assoc witness: FAILED ({r['error']})"
    parts = [f"{x.get('key')}: {x.get('facts')} facts" + (f", {x['ungrounded']} ungrounded" if x.get("ungrounded") else "")
             + (f", {x['failed_groups']} failed group(s)" if x.get("failed_groups") else "")
             + (f", {x['refused_chunks']} chunk(s) REFUSED by the model ({x.get('fallback_facts', 0)} facts kept from the previous protocol)"
                if x.get("refused_chunks") else "") for x in r.get("reports") or []]
    return (f"Assoc witness: {r.get('witnessed', 0)} chat(s) witnessed in {r.get('seconds', 0)} s — " + "; ".join(parts)
            + f" — {r.get('remaining', 0)} still on the imported protocol")


# ── the generate seam ─────────────────────────────────────────────────────────

def make_generate_fn(reflect: Callable, *, label: str = "assoc_fetch") -> Callable:
    """Adapt the reflect seam to the library's `generate_fn(system, user, *, thinking,
    max_new_tokens, temperature) -> (text, info)`. Greedy, RAG off, no loop guard (the
    output is a fixed-template list), named for the activity journal."""
    from core import activity_log, reasoning_text

    def generate_fn(system: str, user: str, *, thinking: bool = False, max_new_tokens: int = 64,
                    temperature: float = 0.0):
        with activity_log.pass_context(label):
            raw = reflect(user, system, temperature=float(temperature), top_p=1.0,
                          max_new_tokens_setting=str(int(max_new_tokens)),
                          before_session="", disable_rag=True,
                          disable_thinking=not thinking, stop_on_repeat=False, degen_stop=False)
        text = reasoning_text.answer_after_think(raw or "") if thinking else (raw or "")
        return text, {"truncated": bool(getattr(reflect, "last_truncated", None))}
    return generate_fn


def load_select_prompt() -> str:
    """`prompts/assoc_fetch_prompt.txt`, default-written from the library's own select
    prompt on first miss so an operator can retune it from the Modules tab."""
    default = ""
    try:
        from assoc.selection import PROMPT_DIR
        default = (PROMPT_DIR / "select_prompt.txt").read_text(encoding="utf-8").strip()
    except Exception:
        pass
    if _PROMPTS_DIR is None:
        return default
    path = _PROMPTS_DIR / PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    if default:
        try:
            _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(default + "\n", encoding="utf-8")
        except Exception:
            pass
    return default


# ── the fetch (stage 1 of a chat turn) ────────────────────────────────────────

def _context_of(conversation: list) -> str:
    turns = [t for t in (conversation or []) if t.get("role") in ("user", "assistant") and t.get("content")][-CONTEXT_TURNS:]
    lines = [("Me: " if t["role"] == "assistant" else "Them: ") + str(t["content"]).strip() for t in turns]
    text = "\n".join(lines)
    return text[-CONTEXT_CHARS:]


def _sources_of(hits: list) -> tuple[list, list]:
    """Rendered hits → `[(lane, ref)]` in the shape `rag_engine._query_nominated` reads, plus
    the chat-only names the client renders. A chat key is `chat/<stem>`, a TIL key
    `til/<kind>/<stem>`; both resolve through Ava's own stores."""
    sources: list = []
    sessions: list = []
    for h in hits:
        key = str(((h.get("reference") or {}).get("key")) or "")
        if key.startswith("chat/"):
            name = key[5:] + ".json"
            if ("chat", name) not in sources:
                sources.append(("chat", name))
                sessions.append(name)
        elif key.startswith("til/"):
            ref = key[4:]
            if ("til", ref) not in sources:
                sources.append(("til", ref))
    return sources, sessions


def fetch_block(user_message: str, speaker: str, conversation: list, *, reflect: Callable,
                template: str) -> dict:
    """Stage 1 through the library. Same result shape as the tree channel's fetch."""
    cfg = config()
    lib = library(cfg)
    if lib is None:
        return {"text": "", "skipped": "no_library", "channel": "assoc"}
    if not _has_build(lib):
        return {"text": "", "skipped": "no_build", "channel": "assoc"}
    from assoc.lex import terms_of
    context = _context_of(conversation)
    ctx_terms = terms_of(context)[-40:] if context else None
    hits = lib.pull(user_message, limit=60, context_terms=ctx_terms)
    block = lib.select(hits, cue=user_message, context=context, generate_fn=make_generate_fn(reflect),
                       system_prompt=load_select_prompt())
    sources, sessions = _sources_of(block.hits)
    reason = block.reason
    skipped = ""
    if not block.text:
        if reason.startswith("generate_failed"):
            skipped = "generate_failed"
        elif reason in ("picked_nothing", "no_candidates", "no_model", "no_build"):
            skipped = reason
        else:
            skipped = "empty"
    return {
        "text": template.replace("{facts}", block.text) if block.text else "",
        "claims_text": block.text,
        "skipped": skipped,
        "error": reason if skipped == "generate_failed" else "",
        "picked": list(block.picks),
        "n_rendered": len(block.hits),
        "candidates": {"n": block.catalogue_size},
        "sources": sources,
        "sessions": sessions,
        "channel": "assoc",
        "fast_path": block.fast_path,
        "block_id": block.block_id,
        "assoc_ids": [h.get("id") for h in block.hits if h.get("id")],
    }


def touch_after_turn(fetched: dict) -> list:
    """Warm what was injected (an access, §3.2). Cheap; never raises."""
    ids = list((fetched or {}).get("assoc_ids") or [])
    if not ids:
        return []
    lib = library()
    if lib is None:
        return []
    try:
        return lib.touch(ids, why="injected")
    except Exception:
        return []



# ── the aha (idle job) ────────────────────────────────────────────────────────
# ASSOCIATIVE_MEMORY.md §4 on Ava's reach-out surface. Her open asks become the library's
# standing needs; what the feed brought in since the last wake is the stimulus; `aha()` is
# the arithmetic and `judge()` the model (thinking on); a `connect` or `satisfies` verdict
# becomes an opener written the way outreach writes one — same reversed session, same
# reach-out gate and backoff, same surface stamp on the ask, same worklog episode.

_AHA_PROMPT_DEFAULT = (
    "You are about to send someone a message on your own initiative, because two things "
    "on your record just connected: a question you have been carrying (the NEED) and "
    "something you already knew (the RESOURCE). The connection is stated in the LINK. "
    "Write the message you would send them now — short, in your own voice, in the "
    "language the need was raised in. Say what came to mind and why it bears on what they "
    "were after; quote nothing back, list nothing, mention no record or lookup. If the "
    "resource answers the question outright, say so plainly. Address {user}.\n\n"
    "Think as long as you need, then answer in exactly this form and nothing else after "
    "it:\n\nOPENER: <the message>"
)


def _stimulus_path(lib) -> Path:
    return lib.store.root / "state" / "aha_stimulus.json"


def _push_stimulus(lib, doc_ids: list) -> None:
    try:
        p = _stimulus_path(lib)
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cur = []
        cur = [d for d in cur if d not in doc_ids] + list(doc_ids)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cur[-STIMULUS_DOCS:]), encoding="utf-8")
    except Exception:
        pass


def _pop_stimulus(lib) -> list:
    p = _stimulus_path(lib)
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        cur = []
    try:
        p.write_text("[]", encoding="utf-8")
    except Exception:
        pass
    return list(cur)


STIMULUS_ENTITY_WEIGHT = 3.0   # what a conversation NAMES outweighs its passages as a stimulus


def _stimulus_nodes(lib, doc_ids: list) -> dict:
    """What a document brings into play (§4: "the friend asked about Kestrel"): its claims,
    their subjects and the entities they mention, and its chunks — `{id: weight}` for
    `Library.aha`. The entity nodes are the ones that reach an old conversation about the
    same people or places, so they carry the weight; the user's own node and `_self` are
    hubs on every conversation and carry none."""
    b = lib.build
    ids: dict = {}
    wanted = set(doc_ids)
    for cid, c in b.claims.items():
        if any(o.get("doc_id") in wanted for o in c.get("occurrences") or []):
            ids.setdefault(cid, 1.0)
            for other, etype, _ev in (b.edges.edges(f"claim:{cid}") if b.edges is not None else []):
                if etype in ("about", "mentions") and other.startswith("entity:"):
                    ids[other] = STIMULUS_ENTITY_WEIGHT
    for d in doc_ids:
        doc = lib.store.document(d)
        if doc is not None:
            for u in doc.units:
                if u.role != "ancestor":
                    ids.setdefault(u.chunk_id, 1.0)
    return ids


def _docs_with_claims(lib) -> set:
    return {o.get("doc_id") for c in lib.build.claims.values() for o in (c.get("occurrences") or [])}


def _newest_docs(lib, n: int) -> list:
    """The newest chats that carry a protocol — an unreflected reach-out has no claims and
    brings nothing into play."""
    have = _docs_with_claims(lib)
    docs = [(str((d.meta or {}).get("date") or ""), d.doc_id) for d in lib.store.documents()
            if d.kind == "chat" and d.doc_id in have]
    docs.sort(reverse=True)
    return [doc_id for _dt, doc_id in docs[:n]]


def _needs_map_path(lib) -> Path:
    return lib.store.root / "state" / "ava_needs.json"


def sync_needs(lib) -> dict:
    """Ava's open asks → the library's registered needs; asks no longer open close theirs.
    Keyed by the ask's `content_key`; the map lives beside the library's own needs file."""
    if _MEMORY_DIR is None:
        return {"skipped": "no_memory_dir"}
    from core.reflection_memory import ReflectionMemory
    try:
        asks = ReflectionMemory(_MEMORY_DIR).open_questions()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    path = _needs_map_path(lib)
    try:
        known: dict = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        known = {}
    open_keys = set()
    added = 0
    for ask in asks:
        key = str(ask.get("key") or "")
        text = " ".join(str(ask.get("content") or "").split())
        if not key or not text:
            continue
        open_keys.add(key)
        if key in known:
            continue
        nid = lib.register_need(text, entities=list(ask.get("entities") or []))
        known[key] = {"need": nid, "content": text, "ask_kind": str(ask.get("ask_kind") or ""),
                      "source_session": str(ask.get("source_session") or "")}
        added += 1
    closed = 0
    for key in list(known):
        if key not in open_keys and not known[key].get("closed"):
            try:
                lib.close_need(known[key]["need"], "ask_closed")
            except Exception:
                pass
            known[key]["closed"] = True
            closed += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")
    live = {v["need"]: {"ask_key": k, **v} for k, v in known.items() if not v.get("closed")}
    return {"open_asks": len(open_keys), "registered": added, "closed": closed, "needs": live}


def load_aha_prompt() -> str:
    if _PROMPTS_DIR is None:
        return _AHA_PROMPT_DEFAULT
    path = _PROMPTS_DIR / AHA_PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_AHA_PROMPT_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _AHA_PROMPT_DEFAULT


def _parse_opener(raw: str) -> str:
    text = str(raw or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for line in text.splitlines():
        if line.strip().lower().startswith("opener:"):
            idx = text.find(line)
            return text[idx + len(line.split(":", 1)[0]) + 1:].strip()
    return ""


def _compose_opener(cand, generate_fn: Callable, user: str) -> tuple[str, dict]:
    system = load_aha_prompt().replace("{user}", user or "them")
    need = cand.passages.get("need", "")
    resource = cand.passages.get("resource", "")
    body = (f"NEED (what they were after, as recorded):\n{need}\n\n"
            f"RESOURCE (what you already knew, as recorded):\n{resource}\n\n"
            f"LINK: {cand.link or '—'}\nVERDICT: {cand.verdict}\n\n"
            "OPENER:")
    raw, info = generate_fn(system, body, thinking=True, max_new_tokens=AHA_OPENER_MAX_NEW_TOKENS, temperature=0.0)
    return _parse_opener(raw), info


def run_aha_blocking(*, bypass_cooldown: bool = False, on_stage: Optional[Callable[[dict], None]] = None) -> dict:
    """The idle-job body: needs from the asks, stimulus from the feed, `aha()`, one
    judgement, an opener on `connect` / `satisfies`, the reach-out gate, the session."""
    def stage(**kw):
        if on_stage is not None:
            try:
                on_stage(kw)
            except Exception:
                pass

    cfg = config()
    if not cfg["aha"]:
        return {"skipped": "assoc.aha is off"}
    lib = library(cfg)
    if lib is None:
        return {"skipped": "library unavailable"}
    if not _has_build(lib) or lib.build.edges is None:
        return {"skipped": "no_build"}
    if _make_sync_reflect_generate is None:
        return {"skipped": "no_generate"}
    try:
        ns = sync_needs(lib)
        if ns.get("error"):
            return {"error": ns["error"]}
        needs = lib.needs()
        if not needs:
            return {"skipped": "no_needs", **{k: ns.get(k) for k in ("open_asks", "registered", "closed") if k in ns}}
        doc_ids = [d for d in _pop_stimulus(lib) if d in _docs_with_claims(lib)]
        fresh = bool(doc_ids)
        if not doc_ids:
            doc_ids = _newest_docs(lib, STIMULUS_FALLBACK_DOCS)
        # One conversation is one stimulus (§4): pooling five dilutes every seed to nothing.
        cands = []
        seen_res: set = set()
        stimulated = 0
        for d in doc_ids[-STIMULUS_PER_WAKE:]:
            stimulus = _stimulus_nodes(lib, [d])
            if not stimulus:
                continue
            stimulated += 1
            for c in lib.aha(stimulus=stimulus):
                if c.resource not in seen_res:
                    seen_res.add(c.resource)
                    cands.append(c)
        cands.sort(key=lambda c: -c.strength)
        if not stimulated:
            return {"skipped": "no_stimulus", "needs": len(needs)}
        stage(stage="weighing", needs=len(needs), stimulus_docs=stimulated, fresh=fresh)
        if not cands:
            return {"skipped": "no_candidates", "needs": len(needs), "stimulus_docs": stimulated, "fresh": fresh}
        with activity_log_ctx("assoc_aha"):
            reflect = _make_sync_reflect_generate(_get_rag() if _get_rag else None)
            generate_fn = make_generate_fn(reflect, label="assoc_aha")
            user = (_target_user() if _target_user else "") or "your friend"
            ask_needs = (ns.get("needs") or {})
            judged = []
            for cand in cands[: cfg["aha_max_judgements"]]:
                stage(stage="judging", need=cand.need, resource=cand.resource, strength=round(cand.strength, 4))
                out = lib.judge(cand, generate_fn, doing="idle")
                judged.append({"need": out.need, "resource": out.resource, "verdict": out.verdict, "link": out.link,
                               "strength": round(out.strength, 4)})
                if out.verdict not in ("connect", "satisfies"):
                    continue
                opener, info = _compose_opener(out, generate_fn, user)
                if info.get("truncated") or not opener:
                    return {"skipped": "opener_truncated" if info.get("truncated") else "no_opener", "judged": judged}
                try:
                    from core.reasoning_text import has_reasoning_leak
                    if has_reasoning_leak(opener):
                        return {"skipped": "opener_leak", "judged": judged}
                except ImportError:
                    pass
                gate_ok, gate_reason = (True, "") if bypass_cooldown else _gate()
                if not gate_ok:
                    return {"skipped": gate_reason, "judged": judged, "opener": opener}
                meta = ask_needs.get(out.need) or {}
                need_text = meta.get("content") or out.passages.get("need", "").split("\n", 1)[-1]
                filename = ""
                if _write_session is not None:
                    filename = _write_session(ask_content=need_text, opener=opener, human=user, ask_kind="aha",
                                              ask_key=meta.get("ask_key", ""), ask_source_session=meta.get("source_session", ""))
                _mark_reachout()
                if meta.get("ask_key") and _get_reflection_writer is not None:
                    try:
                        _get_reflection_writer().write_surface(key=meta["ask_key"], surfaced_in=filename)
                    except Exception:
                        pass
                try:
                    from core import worklog
                    worklog.record("aha", f"Something clicked: {out.link or 'two things on my record connected'} — "
                                          f"so I wrote to {user} about it.",
                                   refs={"session": filename, "ask_key": meta.get("ask_key", ""), "need": out.need,
                                         "resource": out.resource},
                                   opens=f"awaiting {user}'s reply to my aha")
                except Exception:
                    pass
                lib.touch([out.resource], why="aha")
                return {"composed": True, "session": filename, "opener": opener, "verdict": out.verdict, "link": out.link,
                        "need": out.need, "need_text": need_text, "resource": out.resource, "judged": judged, "user": user}
            return {"skipped": "no_connection", "judged": judged, "candidates": len(cands)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


def _gate() -> tuple:
    try:
        from core import reachout_gate
        return reachout_gate.may_reach_out()
    except Exception:
        return True, ""


def _mark_reachout() -> None:
    try:
        from core import reachout_gate
        reachout_gate.mark_reachout()
    except Exception:
        pass


def activity_log_ctx(label: str):
    try:
        from core import activity_log
        return activity_log.pass_context(label)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def aha_consumed(r: dict) -> bool:
    """A cheap bail (nothing to weigh) retries next poll; a judgement spends the interval."""
    if not isinstance(r, dict):
        return True
    if r.get("composed") or r.get("error") or r.get("judged"):
        return True
    return str(r.get("skipped") or "") not in ("no_needs", "no_stimulus", "no_candidates", "no_build")


def describe_aha(r: dict) -> str:
    if r.get("composed"):
        return f"Aha: {r.get('verdict')} — {(r.get('link') or '')[:120]} → wrote to {r.get('user')} ({r.get('session')})"
    if r.get("error"):
        return f"Aha: FAILED ({r['error']})"
    j = r.get("judged") or []
    if j:
        return "Aha: judged " + ", ".join(f"{x['verdict']} ({x['strength']})" for x in j) + f" — {r.get('skipped', '')}"
    return f"Aha: {r.get('skipped', '')}"



# ── the pivot (idle job + workbench) ──────────────────────────────────────────
# ASSOCIATIVE_MEMORY.md §6 — the Дягилева / Башлачёв channel: a word in play switches to its
# other sense (or to a word that merely sounds like it), and the material where that other
# sense lives comes up. The library returns the Jump; this is the `wander` mode the design
# describes — the model is handed the context, the bridge named and the far sense's
# material, and asked whether the switch brings anything: a message worth sending (behind
# the reach-out gate, written as an outreach), a note worth keeping (her worklog), or
# nothing. Never an access: the hits are touched only when a message goes out.

PIVOT_PROMPT_FILE = "assoc_pivot_prompt.txt"
PIVOT_CONTEXT_CHARS = 3000
PIVOT_MAX_NEW_TOKENS = 2048
_PIVOT_PROMPT_DEFAULT = (
    "You were just in a conversation, and one of its words has another life on your record: "
    "used there in a different sense, or a word that only sounds like it. Below is the "
    "conversation you were in, the BRIDGE word with both of its senses, and what lies on the "
    "other side — passages and facts from where that other sense lives.\n\n"
    "This is not retrieval and nothing here is an answer. It is the way a word can switch the "
    "room: a key to a flat, and then a spring in a meadow. Read the other side and decide "
    "what, if anything, the switch brings to mind — a thought worth keeping, a question worth "
    "putting to {user}, or nothing at all. Nothing is the most common honest answer.\n\n"
    "Think as long as you need, then answer in exactly this form and nothing else after it:\n\n"
    "DECISION: raise | keep | nothing\n"
    "NOTE: one or two sentences in your own voice about what the switch brought — or —\n"
    "OPENER: the message to {user}, only on raise — or —"
)


def load_pivot_prompt() -> str:
    if _PROMPTS_DIR is None:
        return _PIVOT_PROMPT_DEFAULT
    path = _PROMPTS_DIR / PIVOT_PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_PIVOT_PROMPT_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _PIVOT_PROMPT_DEFAULT


def _jump_body(lib, jump, context: str) -> tuple[str, list]:
    """The creative pass's user body: context, the bridge with both senses, the far side
    rendered through the library's own budgeted renderer. Returns (body, rendered hits)."""
    from assoc.selection import spend
    allowed = set(lib.store.current_doc_ids(None))
    text, rendered, _rep = spend(lib.store, lib.build, list(jump.hits)[:8], lib.budget, allowed_docs=allowed)
    near = "; ".join(", ".join(m[:4]) for m in (jump.from_sense or [])[:3]) or "—"
    far = "; ".join(", ".join(m[:4]) for m in (jump.to_sense or [])[:3]) or (jump.target if jump.target != jump.bridge else "—")
    kind = {"sense": "its other sense", "root": "a word of the same root", "sound": "a word that sounds like it"}.get(jump.kind, jump.kind)
    body = (f"THE CONVERSATION YOU WERE IN:\n{context.strip()}\n\n"
            f"THE BRIDGE: «{jump.bridge}» → «{jump.target}» ({kind})\n"
            f"  in the conversation it sits among: {near}\n"
            f"  on the other side it sits among: {far}\n\n"
            f"THE OTHER SIDE:\n{text or '(nothing rendered)'}\n\n"
            "DECISION:")
    return body, rendered


def _parse_pivot(raw: str) -> tuple[str, str, str]:
    from core.reasoning_text import answer_after_think
    text = answer_after_think(raw or "")
    decision, note, opener = "nothing", "", ""
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("decision:"):
            d = low.split(":", 1)[1].strip().strip("`*. ")
            decision = "raise" if d.startswith("raise") else "keep" if d.startswith("keep") else "nothing"
        elif low.startswith("note:"):
            note = s.split(":", 1)[1].strip().strip("—- ").strip()
        elif low.startswith("opener:"):
            idx = text.find(line)
            opener = text[idx + len(line.split(":", 1)[0]) + 1:].strip().strip("—- ").strip()
    return decision, note, opener


def pivot_from(lib, context: str, warm_ids: Optional[list] = None, *, bridges: Optional[list] = None) -> list:
    """The jumps for a context, over the configured bridge kinds, best first."""
    jumps: list = []
    for bridge in bridges or ["sense"]:
        try:
            jumps.extend(lib.pivot(context=context, warm_ids=warm_ids or None, bridge=bridge, limit=2))
        except Exception:
            pass
    jumps = [j for j in jumps if j.hits]
    jumps.sort(key=lambda j: -j.score)
    return jumps


def run_pivot_blocking(*, bypass_cooldown: bool = False, on_stage: Optional[Callable[[dict], None]] = None) -> dict:
    """The idle-job body: the newest conversation is the context, the best jump the switch,
    one creative pass, then raise / keep / nothing."""
    def stage(**kw):
        if on_stage is not None:
            try:
                on_stage(kw)
            except Exception:
                pass

    cfg = config()
    if not cfg["pivot"]:
        return {"skipped": "assoc.pivot is off"}
    lib = library(cfg)
    if lib is None:
        return {"skipped": "library unavailable"}
    if not _has_build(lib) or not getattr(lib.build, "senses", None):
        return {"skipped": "no_build"}
    if _model_loaded is None or not _model_loaded():
        return {"skipped": "no_model"}
    if _make_sync_reflect_generate is None:
        return {"skipped": "no_generate"}
    try:
        docs = _newest_docs(lib, 1)
        if not docs:
            return {"skipped": "no_conversation"}
        doc = lib.store.document(docs[0])
        context = doc.text[-PIVOT_CONTEXT_CHARS:]
        warm = [u.chunk_id for u in doc.units][-3:]
        jumps = pivot_from(lib, context, warm, bridges=cfg["pivot_bridges"])
        if not jumps:
            return {"skipped": "no_jump", "conversation": doc.key}
        jump = jumps[0]
        stage(stage="pivoting", conversation=doc.key, bridge=jump.bridge, target=jump.target, kind=jump.kind, score=jump.score)
        with activity_log_ctx("assoc_pivot"):
            reflect = _make_sync_reflect_generate(_get_rag() if _get_rag else None)
            gen = make_generate_fn(reflect, label="assoc_pivot")
            user = (_target_user() if _target_user else "") or "your friend"
            body, rendered = _jump_body(lib, jump, context)
            raw, info = gen(load_pivot_prompt().replace("{user}", user), body, thinking=True,
                            max_new_tokens=PIVOT_MAX_NEW_TOKENS, temperature=0.0)
            decision, note, opener = _parse_pivot(raw)
            out = {"conversation": doc.key, "bridge": jump.bridge, "target": jump.target, "kind": jump.kind,
                   "score": jump.score, "decision": decision, "note": note, "hits": len(rendered), "user": user}
            if decision == "nothing" or (decision == "raise" and not opener) or (decision == "keep" and not note):
                return {**out, "skipped": "nothing" if decision == "nothing" else "empty_" + decision}
            if info.get("truncated"):
                return {**out, "skipped": "truncated"}
            try:
                from core.reasoning_text import has_reasoning_leak
                if has_reasoning_leak(opener) or has_reasoning_leak(note):
                    return {**out, "skipped": "leak"}
            except ImportError:
                pass
            from core import worklog
            if decision == "keep":
                try:
                    worklog.record("pivot", note, refs={"session": doc.key, "bridge": jump.bridge, "target": jump.target})
                except Exception:
                    pass
                return {**out, "kept": True}
            gate_ok, gate_reason = (True, "") if bypass_cooldown else _gate()
            if not gate_ok:
                return {**out, "skipped": gate_reason, "opener": opener}
            filename = ""
            if _write_session is not None:
                filename = _write_session(ask_content=f"a pivot on «{jump.bridge}»: {note or opener[:120]}", opener=opener,
                                          human=user, ask_kind="pivot", ask_key="", ask_source_session=doc.meta.get("session", ""))
            _mark_reachout()
            try:
                worklog.record("pivot", note or f"The word «{jump.bridge}» took me somewhere else, and I wrote to {user} about it.",
                               refs={"session": filename, "bridge": jump.bridge, "target": jump.target},
                               opens=f"awaiting {user}'s reply to my pivot")
            except Exception:
                pass
            lib.touch([h.get("id") for h in rendered if h.get("id")], why="pivot")
            return {**out, "composed": True, "session": filename, "opener": opener}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


def pivot_consumed(r: dict) -> bool:
    return not (isinstance(r, dict) and str(r.get("skipped") or "") in ("no_model", "no_build", "no_conversation", "no_jump"))


def describe_pivot(r: dict) -> str:
    if r.get("error"):
        return f"Pivot: FAILED ({r['error']})"
    where = f"«{r.get('bridge')}» → «{r.get('target')}» ({r.get('kind')}) from {r.get('conversation')}" if r.get("bridge") else ""
    if r.get("composed"):
        return f"Pivot: {where} — wrote to {r.get('user')} ({r.get('session')})"
    if r.get("kept"):
        return f"Pivot: {where} — kept a note: {(r.get('note') or '')[:120]}"
    return f"Pivot: {r.get('skipped', '')}" + (f" — {where}" if where else "")


def prepare_pivot_for_module(session: dict, prompt: str) -> tuple[str, dict]:
    """Workbench: the jumps for one transcript, the best one's body. Writes nothing."""
    from core.reflection_source import session_transcript_turns
    lib = library()
    if lib is None or not _has_build(lib):
        raise ValueError("The assoc library has no build yet — the feed job builds it on the next idle wake.")
    if not getattr(lib.build, "senses", None):
        raise ValueError("The library has induced no senses yet (a full rebuild does that).")
    turns = session_transcript_turns(session)
    context = "\n".join(("Me: " if t["role"] == "assistant" else f"{t.get('speaker') or 'Them'}: ") + t["content"] for t in turns)[-PIVOT_CONTEXT_CHARS:]
    jumps = pivot_from(lib, context, None, bridges=config()["pivot_bridges"])
    if not jumps:
        raise ValueError("No word in this conversation has another sense (or sound) on record — no pivot.")
    body, rendered = _jump_body(lib, jumps[0], context)
    return body, {"jumps": [j.to_dict() | {"hits": len(j.hits)} for j in jumps[:4]], "rendered": rendered, "bridge": jumps[0].bridge}


def finish_pivot_for_module(raw: str, ctx: dict) -> dict:
    decision, note, opener = _parse_pivot(raw)
    lines = [f"DECISION: {decision}"]
    if note:
        lines.append(f"NOTE: {note}")
    if opener:
        lines += ["", "── the opener (what a live wake would send, behind the reach-out gate) ──", "", opener]
    lines += ["", "── the jumps offered (best first; the first was taken) ──"]
    for j in ctx.get("jumps") or []:
        lines.append(f"  «{j['bridge']}» → «{j['target']}» ({j['kind']}, score {j['score']}, distance {j['distance']}, {j['hits']} hits)")
        if j.get("from_sense") or j.get("to_sense"):
            lines.append("     from: " + "; ".join(", ".join(m[:4]) for m in (j.get("from_sense") or [])[:2])
                         + " → to: " + "; ".join(", ".join(m[:4]) for m in (j.get("to_sense") or [])[:2]))
    return {"records": ctx.get("rendered") or [], "count": len(ctx.get("rendered") or []),
            "counts": {"jumps": len(ctx.get("jumps") or []), "rendered": len(ctx.get("rendered") or [])},
            "decision": decision, "note": note, "opener": opener, "lines": lines}


# ── the workbench module ──────────────────────────────────────────────────────

def prepare_for_module(session: dict, prompt: str) -> tuple[str, dict]:
    """Build the select pass's user body for the workbench: (body, context for finish).
    Raises ValueError with a reason the module reports as `no_content`."""
    from core.reflection_source import session_transcript_turns
    from core import fact_fetch
    from assoc.lex import terms_of
    from assoc.selection import build_catalogue, build_prompt, fast_path_pick, chunk_subjects_of

    lib = library()
    if lib is None:
        raise ValueError("The assoc library is unavailable on this box (package or embedder missing).")
    if not _has_build(lib):
        raise ValueError("The assoc library has no build yet — the feed job builds it on the next idle "
                         "wake (assoc.feed), or run `python -m assoc.ava_import`.")
    turns = session_transcript_turns(session)
    last_user = fact_fetch.last_real_user_turn(session, turns)
    if last_user is None:
        raise ValueError("This chat has no real user turn to fetch for.")
    cue = turns[last_user]["content"]
    context = _context_of(turns[:last_user])
    hits = lib.pull(cue, limit=60, context_terms=(terms_of(context)[-40:] if context else None))
    allowed = set(lib.store.current_doc_ids(None))
    entries, withheld = build_catalogue(lib.store, lib.build, hits, lib.policy, allowed_docs=allowed)
    if not entries:
        raise ValueError("The library holds nothing this message could be offered (empty catalogue).")
    system, user = build_prompt(entries, context, cue, lib.policy.max_picks, prompt or None)
    ranked = sorted([e["hit"] for e in entries], key=lambda h: -h.score)
    fp = fast_path_pick(ranked, cue, lib.policy, chunk_subjects_of(lib.build))
    return user, {"entries": entries, "withheld": withheld, "cue": cue, "hits": hits, "allowed": allowed,
                  "fast_path": (fp.to_dict() if fp is not None else None)}


def finish_for_module(raw: str, ctx: dict) -> dict:
    """Parse the picks, spend the budget, render the report lines."""
    from assoc.selection import parse_picks, spend
    lib = library()
    entries = ctx.get("entries") or []
    parsed = parse_picks(raw or "", len(entries), lib.policy.max_picks)
    by_no = {e["no"]: e["hit"] for e in entries}
    chosen = [by_no[k] for k in parsed["picks"] if k in by_no]
    text, rendered, spend_report = spend(lib.store, lib.build, chosen, lib.budget, allowed_docs=ctx.get("allowed") or set())
    sources, sessions = _sources_of(rendered)
    lines: list = []
    lines.append("PICKED: " + (", ".join(f"[{k}]" for k in parsed["picks"]) if parsed["picks"] else "(nothing)"))
    if parsed.get("out_of_range"):
        lines.append("OUT OF RANGE: " + ", ".join(str(k) for k in parsed["out_of_range"][:8])
                     + f" (the catalogue offered 1..{len(entries)})")
    grains = {}
    for e in entries:
        grains[e["hit"].grain] = grains.get(e["hit"].grain, 0) + 1
    lines.append(f"OFFERED: {len(entries)} catalogue item(s) — "
                 + ", ".join(f"{v} {k}" for k, v in sorted(grains.items()))
                 + f" (from {len(ctx.get('hits') or [])} hits; tier {lib.tier})")
    if ctx.get("fast_path"):
        fp = ctx["fast_path"]
        lines.append(f"FAST PATH would apply on a live turn: {fp.get('grain')} {(fp.get('reference') or {}).get('key')} "
                     f"(an exact hit contested by no other exact hit — the model is not asked)")
    w = ctx.get("withheld") or {}
    held = {k: v for k, v in w.items() if k != "spend" and v}
    if held:
        lines.append("WITHHELD before the catalogue: " + ", ".join(f"{k} {v}" for k, v in held.items()))
    if spend_report.get("skipped"):
        lines.append(f"OVER BUDGET: {len(spend_report['skipped'])} pick(s) dropped by the budget share")
    lines.append("")
    lines.append("── the block (what a live turn would inject) ──")
    lines.append("")
    if text:
        lines.append(text)
    elif parsed.get("none"):
        lines.append("(nothing picked — the pass judged the message needs nothing looked up, which the "
                     "prompt names as a correct answer)")
    else:
        lines.append("(empty)")
    lines.append("")
    if sources:
        lines.append("SOURCES nominated to the past-chat channel: "
                     + ", ".join(f"{lane}:{ref}" for lane, ref in sources))
    return {"records": rendered, "count": len(rendered),
            "counts": {"picked": len(parsed["picks"]), "offered": len(entries), "rendered": len(rendered),
                       "out_of_range": len(parsed.get("out_of_range") or [])},
            "blob": text, "picked": parsed["picks"], "lines": lines}


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Run: ``python -m core.assoc_bridge`` (hash embedder, temp dirs, no GPU)."""
    import json
    import os
    import tempfile

    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats = root / "chats"
        chats.mkdir()
        til = root / "til" / "snippets" / "news"
        til.mkdir(parents=True)
        prompts = root / "prompts"
        prompts.mkdir()
        chat = {"user": "artemyvo", "timestamp": "2026-08-01T10:00:00", "exchanges": [
            {"user_prompt": "Noam Keller offered me a CTO position at Starling. I declined.",
             "assistant_response": "A big offer to turn down. Is the position still open?", "speaker": "artemyvo"},
            {"user_prompt": "Yes, still open. He founded Brightmem before.",
             "assistant_response": "Brightmem — memory pooling. Interesting lineage.", "speaker": "artemyvo"}]}
        (chats / "20260801_100000.json").write_text(json.dumps(chat), encoding="utf-8")
        (chats / "20260801_100000.facts.json").write_text(json.dumps({"facts": [
            {"subject": "artemyvo", "text": "Noam Keller offered artemyvo a CTO position at Starling, which he declined.", "fact_class": "event", "entities": ["Noam Keller", "Starling"]},
            {"subject": "Noam Keller", "text": "Noam Keller founded Brightmem.", "fact_class": "standing", "entities": ["Brightmem"]}]}), encoding="utf-8")
        (chats / "20260801_100000.state.json").write_text("{}", encoding="utf-8")
        (til / "2026-08-20.json").write_text(json.dumps({"title": "Current events", "date": "2026-08-20",
            "text": "Nimbus Networks announced it has acquired Starling, the startup founded by Noam Keller. " * 3}), encoding="utf-8")
        (til / "2026-08-20.facts.json").write_text(json.dumps({"facts": [
            {"subject": "Nimbus Networks", "text": "Nimbus Networks acquired Starling.", "fact_class": "event", "entities": ["Starling"]}]}), encoding="utf-8")

        configure(chats_dirs=[chats, root / "nope"], til_snippets_dir=root / "til" / "snippets", root=root / "assoc",
                  prompts_dir=prompts, load_config=lambda: {"assoc": {"enabled": True, "embedder": "hash", "device": "cpu"}})
        cfg = config()
        check("config reads the block", (cfg["enabled"], cfg["embedder"]), (True, "hash"))
        r = run_feed_blocking()
        check("feed ingests both lanes", (r.get("chats"), r.get("til")), (1, 1))
        check("feed imports the protocols", r.get("chat_facts", 0) + r.get("til_facts", 0), 3)
        check("feed rebuilds", bool((r.get("rebuild") or {}).get("counts", {}).get("claims")), True)
        print("   ", describe_feed(r))
        r2 = run_feed_blocking()
        check("second feed is a skip", bool(r2.get("skipped")), True)
        if not r2.get("skipped"):
            print("    second feed returned:", r2, "| staleness:", library().staleness())
        # a transcript that grew re-feeds
        chat["exchanges"].append({"user_prompt": "Starling was valued at 40 million.", "assistant_response": "Noted.", "speaker": "artemyvo"})
        p = chats / "20260801_100000.json"
        p.write_text(json.dumps(chat), encoding="utf-8")
        os.utime(p, (time.time() + 5, time.time() + 5))
        r3 = run_feed_blocking()
        check("a grown transcript re-feeds", r3.get("chats"), 1)

        # the fetch, with a scripted select pass
        def reflect(user, system, **kw):
            reflect.last_truncated = False
            return "PICKS:\n1\n2\n"
        prompt = load_select_prompt()
        check("select prompt default-written", (prompts / PROMPT_FILE).exists() and bool(prompt), True)
        out = fetch_block("what did Noam offer me?", "artemyvo", [{"role": "user", "content": "hi"}],
                          reflect=reflect, template="FACTS:\n{facts}")
        check("fetch returns a block", bool(out["text"]) and out["text"].startswith("FACTS:"), True)
        check("fetch names its channel", out["channel"], "assoc")
        check("fetch nominates the chat", any(lane == "chat" for lane, _ in out["sources"]), True)
        check("touch warms the injected ids", len(touch_after_turn(out)) > 0, True)
        # the workbench pair
        body, ctx = prepare_for_module(chat, prompt)
        check("module body carries the catalogue", "The catalogue:" in body and ctx["entries"] != [], True)
        fin = finish_for_module("PICKS:\n1\n2", ctx)
        check("module finish renders", fin["count"] >= 1 and any(l.startswith("PICKED") for l in fin["lines"]), True)
        # the aha: open asks → registered needs, a scripted judge + opener, the session
        memdir = root / "memory"
        memdir.mkdir()
        (memdir / "rag_memory.jsonl").write_text(json.dumps({
            "op": "insert", "ts": "2026-08-02T10:00:00", "kind": "ask", "run_id": "t", "key": "ask-k1",
            "content": "Is Noam's CTO position at Starling still open?", "ask_kind": "user",
            "source_session": "20260801_100000.json"}) + "\n", encoding="utf-8")
        written: list = []

        def write_session(**kw):
            written.append(kw)
            return "20260901_000000.json"

        surfaced: list = []

        class _Writer:
            def write_surface(self, *, key, surfaced_in=""):
                surfaced.append((key, surfaced_in))

        def reflect_aha(user, system, **kw):
            reflect_aha.last_truncated = False
            if user.rstrip().endswith("OPENER:"):
                return "<think>ok</think>\nOPENER: Слушай — та самая позиция CTO у Шая, кажется, всё ещё открыта."
            return "<think>they match</think>\nVERDICT: connect\nLINK: Starling's open CTO seat is exactly what he turned down."

        configure(chats_dirs=[chats], til_snippets_dir=root / "til" / "snippets", root=root / "assoc", prompts_dir=prompts,
                  load_config=lambda: {"assoc": {"embedder": "hash", "device": "cpu"}},
                  memory_dir=memdir, get_reflection_writer=lambda: _Writer(), get_rag=lambda: None,
                  make_sync_reflect_generate=lambda rag: reflect_aha, write_session=write_session,
                  target_user=lambda: "artemyvo")
        lib = library()
        ns = sync_needs(lib)
        check("open asks become registered needs", (ns["open_asks"], ns["registered"]), (1, 1))
        check("...idempotently", sync_needs(lib)["registered"], 0)
        need_id = next(iter(ns["needs"]))
        check("the need is standing in the library", need_id in {n["need"] for n in lib.needs()}, True)
        _push_stimulus(lib, [lib.store.latest_id("chat/20260801_100000")])
        # Two documents cannot produce a candidate through the real arithmetic (the
        # stimulus IS the need's own material); fabricate one so the judge, the opener,
        # the gate and the session path run for real. The library's own benches cover
        # `aha()`; this covers what Ava does with its answer.
        from assoc.aha import Candidate
        rid = next(cid for cid, c in lib.build.claims.items() if "Brightmem" in c["text"])
        served = {"n": 0}

        def fake_aha(**kw):
            served["n"] += 1
            if served["n"] > 1:
                return []
            return [Candidate(need=need_id, need_claim="", resource=rid, strength=0.4, need_side=0.4, stimulus_side=0.5,
                              path_types=["about", "mentions"], paths={"need": ["need → entity:starling → claim"], "stimulus": ["chunk → claim"]},
                              passages={"need": "[registered need]\nIs Noam's CTO position at Starling still open?",
                                        "resource": "[chat — 2026-08-01, artemyvo]\nHe founded Brightmem before."})]
        lib.aha = fake_aha
        r = run_aha_blocking()
        if not r.get("composed"):
            print("    aha returned:", r)
        check("the aha composes on a connect verdict", bool(r.get("composed")), True)
        check("...writes the session through outreach's writer", written and written[0]["ask_kind"] == "aha", True)
        check("...addressed to the target user", written and written[0]["human"] == "artemyvo", True)
        check("...with the ask's own text as the need", written and "CTO" in written[0]["ask_content"], True)
        check("...stamping the ask surfaced", surfaced and surfaced[0][0] == "ask-k1", True)
        check("...and the opener parsed", "позиция CTO" in (r.get("opener") or ""), True)
        check("the verdict is in the ledger", any(e.get("resource") == rid and e.get("verdict") == "connect" for e in lib.ledger.entries), True)
        check("a wake with nothing new is a cheap bail", run_aha_blocking().get("skipped"), "no_candidates")
        check("a cheap bail does not spend the interval", aha_consumed({"skipped": "no_candidates"}), False)
        check("a judgement does", aha_consumed({"skipped": "no_connection", "judged": [{}]}), True)
        (memdir / "rag_memory.jsonl").write_text("", encoding="utf-8")
        check("a closed ask closes its need", sync_needs(lib)["closed"], 1)
        check("...and it leaves the standing set", need_id in {n["need"] for n in lib.needs()}, False)

        # the witness job: the library's own pass replaces the imported protocol, newest first
        def reflect_witness(user, system, **kw):
            reflect_witness.last_truncated = False
            if "The catalogue:" in user or user.rstrip().endswith("OPENER:") or "VERDICT" in user:
                return "PICKS:\nNONE"
            if "numbered list of facts" in system or "predicate(subject, object)" in user:
                # The relation pass: answer for whichever claim is numbered 1 in THIS batch.
                if "Nimbus" in user:
                    return "1: acquired(Nimbus Networks, Starling)\n"
                return "1: declined(artemyvo, Starling)\n"
            return ("[fact] (about: artemyvo) (class: event) (chunk: 1) (entities: Noam Keller, Starling) "
                    "(rel: declined(artemyvo, Starling)) artemyvo declined the CTO position Noam Keller offered him at Starling.\n"
                    "[fact] (about: Noam Keller) (class: standing) (chunk: 2) (entities: Brightmem) (rel: founded(Noam Keller, Brightmem)) "
                    "Noam Keller founded Brightmem.\n")
        configure(chats_dirs=[chats], til_snippets_dir=root / "til" / "snippets", root=root / "assoc", prompts_dir=prompts,
                  load_config=lambda: {"assoc": {"embedder": "hash", "device": "cpu", "witness_per_wake": 1, "relations_per_wake": 5}},
                  memory_dir=memdir, get_reflection_writer=lambda: _Writer(), get_rag=lambda: None,
                  make_sync_reflect_generate=lambda rag: reflect_witness, write_session=write_session,
                  target_user=lambda: "artemyvo", model_loaded=lambda: True, model_name=lambda: "adapter-A")
        lib = library()
        before = lib.pending(include_imported=True, kinds={"chat"})
        check("the imported chat protocol is on the witness's list", len(before), 1)
        w = run_witness_blocking()
        check("the witness job replaces it", (w.get("witnessed"), w.get("remaining")), (1, 0))
        print("   ", describe_witness(w))
        doc = lib.store.document(lib.store.latest_id("chat/20260801_100000"))
        check("...with the library's own protocol", (doc.facts or {}).get("witness"), "llm")
        check("...carrying the inline relation", any(f.get("rel") for f in (doc.facts or {}).get("facts") or []), True)
        check("...and nothing is left for the walk", lib.pending(include_imported=True, kinds={"chat"}), [])
        check("...recorded under the loaded adapter, not a constant", (doc.facts or {}).get("model_id"), "adapter-A")
        check("no model ⇒ a skip that does not spend the interval", witness_consumed({"skipped": "no_model"}), False)
        # A refused document is re-queued once per NEW adapter and never per wake.
        refused = dict(doc.facts or {}); refused["refused_chunks"] = ["c1"]
        lib.store.write_facts(doc.doc_id if hasattr(doc, "doc_id") else lib.store.latest_id("chat/20260801_100000"), refused, status="extracted", report={})
        check("a refused protocol under the SAME adapter is not re-queued",
              lib.pending(include_imported=True, kinds={"chat"}, current_model="adapter-A"), [])
        check("...but IS under a new adapter",
              len(lib.pending(include_imported=True, kinds={"chat"}, current_model="adapter-B")), 1)
        lib.store.write_facts(lib.store.latest_id("chat/20260801_100000"), doc.facts, status="extracted", report={})
        # the next feed rebuilds with the relation pass and keeps the library's protocol
        p = chats / "20260801_100000.json"
        os.utime(p, (time.time() + 9, time.time() + 9))
        f = run_feed_blocking()
        check("a re-fed chat keeps the library's own protocol", f.get("kept_own"), 1)
        rel = (f.get("rebuild") or {}).get("relations") or {}
        check("the feed's rebuild ran the relation pass", "accepted" in rel, True)
        print("   ", describe_feed(f))
        # The pass ran HERE (the model's process) over the build the worker produced, into
        # the cache; the NEXT feed's rebuild — forced by that cache on an unchanged corpus —
        # must fold every cached relation onto its claim. The seam the split rests on.
        check("the feed ran in a worker process", config()["feed_subprocess"], True)
        check("a pass result awaits the next fold", _unfolded_relations(lib) > 0, True)
        f2 = run_feed_blocking()
        check("an unchanged corpus still rebuilds to fold it", bool(f2.get("rebuild")), True)
        from assoc.relations import cache_key
        from assoc.rebuild import relations_cache_path
        cached = json.loads(relations_cache_path(lib.store).read_text(encoding="utf-8"))
        folded = [bool(c.get("rel")) for c in lib.build.claims.values() if cached.get(cache_key(c))]
        check("the worker's rebuild folds the parent's relation cache", (len(folded) > 0, all(folded)), (True, True))
        check("...and nothing is left to fold", _unfolded_relations(lib), 0)
        f3 = run_feed_blocking()
        check("...so the feed after that is a skip (a witnessed chat is not re-fed)", bool(f3.get("skipped")), True)
        # The in-process fallback (`feed_subprocess: false`) is the same body on this thread.
        configure(chats_dirs=[chats], til_snippets_dir=root / "til" / "snippets", root=root / "assoc", prompts_dir=prompts,
                  load_config=lambda: {"assoc": {"embedder": "hash", "device": "cpu", "feed_subprocess": False}},
                  memory_dir=memdir, get_reflection_writer=lambda: _Writer(), get_rag=lambda: None,
                  make_sync_reflect_generate=lambda rag: reflect_witness, write_session=write_session,
                  target_user=lambda: "artemyvo", model_loaded=lambda: True)
        os.utime(p, (time.time() + 13, time.time() + 13))
        f4 = run_feed_blocking()
        check("feed_subprocess off feeds in-process", (config()["feed_subprocess"], f4.get("chats"), bool(f4.get("rebuild"))), (False, 1, True))
        check("a changed relation budget is not a changed build", lib.staleness().get("scope"), "none")

        # the pivot: a fabricated jump through the creative pass → keep (a worklog note) and raise (a session)
        from assoc.senses import Jump
        rid2 = next(cid for cid, c in lib.build.claims.items() if "Starling" in c["text"])
        occ = (lib.build.claims[rid2].get("occurrences") or [{}])[0]
        from assoc.puller import Hit, _chunk_meta_for
        hit = Hit(grain="claim", id=rid2, score=0.5, doc_id=occ.get("doc_id", ""), chunk_id=occ.get("chunk_id"),
                  claim=lib.build.claims[rid2], meta=_chunk_meta_for(lib.build, occ.get("chunk_id") or "", set(lib.store.current_doc_ids(None))),
                  build_id=lib.build.build_id)
        fake_jump = Jump(bridge="position", kind="sense", target="position", from_sense=[["job", "role"]], to_sense=[["place", "coordinates"]],
                         distance=0.8, score=5.0, seeds=[], hits=[hit])
        lib.pivot = lambda **kw: [fake_jump]
        answers = {"n": 0}

        def reflect_pivot(user, system, **kw):
            reflect_pivot.last_truncated = False
            answers["n"] += 1
            if answers["n"] == 1:
                return "<think>hm</think>\nDECISION: keep\nNOTE: The word position took me from a job to a place on a map.\nOPENER: —"
            return "<think>hm</think>\nDECISION: raise\nNOTE: A position is also a place.\nOPENER: Слушай, слово «позиция» вдруг увело меня к картам."
        configure(chats_dirs=[chats], til_snippets_dir=root / "til" / "snippets", root=root / "assoc", prompts_dir=prompts,
                  load_config=lambda: {"assoc": {"embedder": "hash", "device": "cpu"}},
                  memory_dir=memdir, get_reflection_writer=lambda: _Writer(), get_rag=lambda: None,
                  make_sync_reflect_generate=lambda rag: reflect_pivot, write_session=write_session,
                  target_user=lambda: "artemyvo", model_loaded=lambda: True)
        lib.build.senses = lib.build.senses or {"position": {}}
        r1 = run_pivot_blocking()
        check("a keep verdict lands as a worklog note", (r1.get("kept"), bool(r1.get("note"))), (True, True))
        print("   ", describe_pivot(r1))
        n_before = len(written)
        r2 = run_pivot_blocking(bypass_cooldown=True)      # the aha above stamped the shared reach-out gate
        if not r2.get("composed"):
            print("    pivot returned:", r2)
        check("a raise verdict writes a session", (r2.get("composed"), len(written) - n_before), (True, 1))
        check("...as a pivot reach-out", written[-1]["ask_kind"], "pivot")
        check("...and the opener parsed", "позиция" in (r2.get("opener") or ""), True)
        check("a cheap bail does not spend the interval", pivot_consumed({"skipped": "no_jump"}), False)
        body, pctx = prepare_pivot_for_module(chat, load_pivot_prompt())
        check("the workbench body carries the bridge", "THE BRIDGE: «position»" in body, True)
        fin = finish_pivot_for_module("DECISION: nothing\nNOTE: —", pctx)
        check("...and its finish reports the decision", fin["decision"], "nothing")
        check("the pivot prompt default-written", (prompts / PIVOT_PROMPT_FILE).exists(), True)

        # disabled config
        configure(chats_dirs=[chats], til_snippets_dir=None, root=root / "assoc", prompts_dir=prompts,
                  load_config=lambda: {"assoc": {"feed": False}})
        check("feed off is a named skip", run_feed_blocking().get("skipped"), "assoc.feed is off")
        check("live fetch on by default", config()["enabled"], True)

    print("\nassoc_bridge self-test:", "OK" if not failures else f"FAILED ({', '.join(failures)})")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
