"""Wander → SFT capture (the Ambient Enculturation lane).

When Ava wanders into a random article and reflects on it, that single generation is
captured here as a **durable, keep-forever** training example in the wander corpus. Since
``train_cycle`` now fits a fresh LoRA on the frozen base every build (never resumed), the
corpus is re-consolidated on every build — so a wander's imprint persists across cycles the
way it did under the old resumed-adapter scheme, instead of evaporating after one build. It
is still **not** a decaying ledger anchor (it never enters ``consolidation_anchors.jsonl``);
the structured facts/asks half of a wander is routed separately into ``rag_memory`` by the
writer. The corpus record's ``prompt`` (source article) + ``target`` (reaction) also back the
chat-RAG *wander channel* (``rag_engine``), which fades by wall-clock age — see
AVA_DESIGN_LEGACY.md → *Ambient Enculturation — Style & Language Bleed*.

The point is register/language **bleed**: her own thought carries the article's tongue, so
re-training it every build keeps that language warm in the weights. People-over-wander
weighting is held by the fixed ``WANDER_LR_MULT`` (one rung below a fresh chat) plus the
user-token wander budget that rations how many wanders are captured at all.

Shared by the inference server (capture: :func:`append_example`) and the offline train cycle
(consume: :func:`load_pending`). ``clear_pending`` remains for disaster-recovery/wipe only —
the cycle no longer clears the corpus. Both sides reach ``core/`` on ``sys.path``; the corpus
resolves relative to this module (``server/data/til``, the ordered state home) so neither has
to pass paths around.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# A leading ``<think>…</think>`` + the trailing answer — mirrors
# ``training.render``'s acceptance split so the inference server can judge a captured
# target without importing the training package.
_THINK_RE = re.compile(r"(?s)\s*<think>(.*?)</think>(.*)", re.IGNORECASE)


def looks_trainable(target: str) -> bool:
    """True when *target* would survive ``_render_wander_examples``' guard.

    Requires a single leading ``<think>…</think>`` with a **non-empty** thought, then a
    non-empty answer span carrying **no** further think tags. A *double-think* target —
    gemma-4's native reasoning channel plus a second, prompt-induced literal ``<think>``
    block — fails this (the answer span still holds ``<think>``), which is exactly why
    such a capture never lands. Mirrors ``training.render.trainable_answer`` + ``has_cot``.
    """
    m = _THINK_RE.match(target or "")
    if not m or not (m.group(1) or "").strip():
        return False                      # no CoT, or an empty thought
    answer = (m.group(2) or "").strip()
    if not answer or "<think>" in answer or "</think>" in answer:
        return False                      # empty / double-think answer span
    return True

# server/data/til — the ordered state home (chats already moved to server/data/chats).
# parents[2] from server/inference/core/ is server/.
_TIL_DIR = Path(__file__).resolve().parents[2] / "data" / "til"
# The retired one-shot queue under the old inference/data tree — folded in once on access.
_LEGACY_PATH = Path(__file__).resolve().parents[1] / "data" / "hot" / "wander" / "pending_sft.jsonl"


def corpus_path() -> Path:
    """``server/data/til/wander.jsonl`` — the durable, append-only wander corpus."""
    return _TIL_DIR / "wander.jsonl"


# Back-compat alias — some callers/docs still say "pending".
pending_path = corpus_path


def _migrate_legacy(dest: Path) -> None:
    """One-shot: fold the old one-shot queue (inference/data/hot/wander) into the durable
    corpus, then remove it. No-op (one stat) once migrated. Best-effort."""
    if not _LEGACY_PATH.exists():
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = [ln for ln in _LEGACY_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            with open(dest, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        _LEGACY_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def append_example(*, system_prompt: str, prompt: str, target: str,
                   source: dict | None = None) -> bool:
    """Append one wander reflection to the durable wander corpus.

    *system_prompt* / *prompt* / *target* are the exact system / user / assistant turns of
    the generation, stored so the cycle re-renders them train/inference-parity-faithfully
    (``render.build_messages`` + ``assert_parity``). ``prompt`` (the source article) +
    ``target`` (the reaction) also feed the chat-RAG wander channel. Best-effort: a failure
    here never disrupts the wander flow. Returns True iff a record was written.
    """
    target = (target or "").strip()
    if not target:
        return False
    rec = {
        "ts": datetime.now().isoformat(),
        "system_prompt": system_prompt or "",
        "context": [],          # a wander has no prior turns
        "prompt": prompt or "",
        "speaker": "",
        "target": target,
        "source": source or {},  # provenance (wiki/title/url/lang); not trained, but shown in RAG
    }
    try:
        path = corpus_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _migrate_legacy(path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_all() -> list[dict]:
    """Every corpus record, oldest first, **banned ones included**.

    The raw read. Callers that feed training or RAG want :func:`load_pending` (which drops
    banned records); this exists for the operator-facing paths that must still see a banned
    record in order to show, un-ban, or delete it.
    """
    path = corpus_path()
    _migrate_legacy(path)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue        # skip a corrupt line, keep the rest
    except Exception:
        return []
    return out


def load_pending() -> list[dict]:
    """Trainable wander corpus examples, oldest first. Empty when none / unreadable.

    **Banned** records are dropped here: a wander whose generation came out malformed is
    excluded from the build the moment the operator bans it from the Training review tab
    (``banned: true``, set by :func:`set_banned`), without waiting for the deletion that
    "Rewrite history" performs. The record itself stays on disk until then, so the ban is
    reversible and the row is still reviewable.
    """
    return [r for r in load_all() if not r.get("banned")]


def reaction_for(url: str) -> str:
    """What she wrote about the article at *url* — the ANSWER span only, or ``""``.

    Until 2026-08-13 this corpus had exactly one consumer, and it was training: each
    record is a one-shot SFT example teaching register and language bleed, so she is
    *fit on* her reactions rather than recalling them. (The old chat-RAG wander channel
    injected them too, and is off on every path pending redesign — see the
    ``_INJECT_WANDER`` note in ``core.generation``.) This is the second consumer:
    `outreach._source_material` pairs it with the article's recap, so an ask raised from a
    wandered page carries both what she read and what she made of it at the time.

    **The answer span only, never the thought.** `_THINK_RE` splits the captured target
    exactly as the training guard does, and the leading `<think>` half — 2.6k of the 5k
    characters on this corpus — is her raw reasoning. Her CoT is not injected material
    anywhere on this box, and this is not the place to start.

    Matched on URL rather than on the timestamp, because the two clocks differ: a record's
    `ts` is when the pass ran, while its snippet's stem is stamped when the snippet is
    written at Apply, and the observed gap is minutes (`01:13:29` against `011004`). The
    URL is the article's own identity and both sides carry it verbatim.

    Banned records are excluded (:func:`load_pending`) — the corpus-wide rule for anything
    that feeds training or retrieval, and a capture the operator judged malformed is not
    one to put in front of her.
    """
    target_url = (url or "").strip()
    if not target_url:
        return ""
    for rec in reversed(load_pending()):          # newest first: a re-wander supersedes
        if ((rec.get("source") or {}).get("url") or "").strip() != target_url:
            continue
        m = _THINK_RE.match(rec.get("target") or "")
        answer = (m.group(2) if m else (rec.get("target") or "")).strip()
        if answer and "<think>" not in answer:
            return answer
    return ""


def set_banned(ts: str, banned: bool = True) -> bool:
    """Flag/un-flag the corpus record stamped *ts* as banned from training.

    The wander corpus is a flat append-only JSONL with no per-record id beyond its capture
    timestamp, which is what the training render carries as ``source_session``
    (``wander:<ts>``) — so *ts* is the row identity the review tab hands back. Setting the
    flag rewrites the file in place (atomically); the record is preserved so the ban can be
    lifted and so the operator can still read what was banned. Returns True iff a record
    matched and its flag changed.
    """
    ts = (ts or "").strip()
    if not ts:
        return False
    records = load_all()
    changed = False
    for rec in records:
        if str(rec.get("ts") or "") != ts:
            continue
        if bool(rec.get("banned")) == bool(banned):
            continue
        if banned:
            rec["banned"] = True
        else:
            rec.pop("banned", None)
        changed = True
    if not changed:
        return False
    return _rewrite(records)


def delete_banned() -> list[str]:
    """Drop every banned record from the corpus for good. Returns the deleted timestamps.

    The finalize half of a ban, invoked by the Training review "Rewrite history" action —
    the wander counterpart of deleting a banned chat exchange from its transcript. A wander
    record has no transcript to correct and nothing downstream references it by index, so
    the line simply goes: after this it is out of the training corpus *and* out of the
    chat-RAG wander channel, with nothing left to un-ban.
    """
    records = load_all()
    kept = [r for r in records if not r.get("banned")]
    if len(kept) == len(records):
        return []
    removed = [str(r.get("ts") or "") for r in records if r.get("banned")]
    if not _rewrite(kept):
        return []
    return removed


def _rewrite(records: list[dict]) -> bool:
    """Replace the corpus with *records* (atomic temp-file + rename). Best-effort."""
    import os
    import tempfile
    path = corpus_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return True
    except Exception:
        return False


def clear_pending() -> None:
    """Delete the whole wander corpus — disaster-recovery / wipe only.

    The train cycle NO LONGER calls this: the corpus is keep-forever (re-consolidated every
    from-scratch build). Retained for ``wipe_state.py`` and manual recovery.
    """
    try:
        corpus_path().unlink(missing_ok=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# GPU-free self-test                                                           #
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    """Run: ``python -m core.wander_sft``. Covers the two pure readers over the corpus."""
    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    print("looks_trainable — the guard the build applies")
    check("a well-formed capture trains",
          looks_trainable("<think>reasoned</think>the reaction"), True)
    check("no CoT does not", looks_trainable("just an answer"), False)
    check("an empty thought does not", looks_trainable("<think>  </think>answer"), False)
    check("an empty answer does not", looks_trainable("<think>t</think>   "), False)
    check("a double-think target does not — the observed failure",
          looks_trainable("<think>a</think>x <think>b</think> y"), False)

    print("\nreaction_for — the answer span, matched on the article's URL")
    global load_pending
    real, corpus = load_pending, [
        {"ts": "1", "source": {"url": "http://a"},
         "target": "<think>my reasoning</think>What I made of it."},
        {"ts": "2", "source": {"url": "http://b"}, "target": "no think block at all"},
        {"ts": "3", "source": {"url": "http://a"},
         "target": "<think>later</think>A second reading, later."},
    ]
    load_pending = lambda: list(corpus)
    try:
        check("the thought is never returned, only the answer",
              reaction_for("http://a"), "A second reading, later.")
        check("...and a re-wander of one article supersedes the earlier reading",
              reaction_for("http://a").startswith("A second"), True)
        check("a target with no think block still yields its text",
              reaction_for("http://b"), "no think block at all")
        check("an unknown url yields nothing", reaction_for("http://nope"), "")
        check("an empty url is not a lookup key", reaction_for(""), "")
        check("whitespace is not a url", reaction_for("   "), "")
        # `load_pending` already drops banned records, so this only has to not defeat it.
        load_pending = lambda: [r for r in corpus if not r.get("banned")]
        corpus[2]["banned"] = True
        check("a banned capture falls back to the earlier reading",
              reaction_for("http://a"), "What I made of it.")
        corpus[0]["banned"] = True
        check("...and all-banned yields nothing", reaction_for("http://a"), "")
    finally:
        load_pending = real

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
