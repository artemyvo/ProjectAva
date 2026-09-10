"""Shared cross-module gate on Ava's unprompted reach-outs to the user.

``outreach``, ``synthesis`` and ``checkin`` are three fully independent idle jobs that
each cold-open the user — they write a reversed ``initiated_by:"ava"`` session that lands
in the chat list unprompted. They fire on their own clocks and may well come due at the
same moment, which without a throttle would greet the user with two or three unbidden
messages at once. This tiny neutral module is the single gate they all honor.

It answers ONE question — *may she send an unprompted message right now?* — and folds two
independent limits into it:

**1. A cool-down between messages.** At most one unprompted message per window,
regardless of which job originated it.

**2. Backoff on unanswered openers.** The window is not fixed: it doubles for every
message she has already sent *since the user last spoke and had no reply to*
(:func:`unanswered_streak`), capped at :data:`MAX_COOLDOWN_SECONDS`. This is the
generalization of ``outreach._has_dangling_opener`` — which stops her re-raising a
specific open question she is still awaiting an answer on — to the two jobs that compose
free-form and so have no ask to key a guard to. Without it, a flat cool-down equal to the
jobs' own re-arm interval gates nothing at all: on 2026-07-30/31 check-in cold-opened the
user 15 times in 19 hours, each an unanswered restatement of the same thought, and every
send was within the rules. One message unanswered is ordinary; the fifteenth is a
pathology, and the difference is a thing the sender should be able to notice.

Backoff rather than a hard stop: she is meant to be able to reach out on her own
initiative, so being ignored should make her quieter, not mute. At the default cap she
settles to roughly one message a day while the silence lasts.

**Global or per person.** Every entry point takes an optional ``user``. Unscoped
(``user=None``, what ``outreach``/``synthesis`` pass) it asks the original question: has
*any* unprompted message gone out recently, to anyone? Scoped to a person it asks the same
question of that person's thread alone — their openers, their last turn, their streak.
``checkin`` runs once per user, so it scopes: two people are two conversations, and a
message to one is not a reason to stay silent toward the other, nor is one person's silence
evidence that the other is ignoring her. The scoped view is strictly a *narrowing*, so the
unscoped callers still see everything check-in sends.

It is an **output** gate, not a scheduler gate — a distinction that matters. It was once
wired as an idle-job ``ready`` gate, which coupled the jobs: the gate is shared but each
job re-armed on its own (shorter) interval, so whichever job the scheduler happened to
try first monopolized every window and the rest starved, never running at all. So the
check lives at the point of *sending*: each blocking pass runs to completion — doing its
analysis, filling the question pool, consuming its own interval — and consults
:func:`may_reach_out` only immediately before it would write a session, calling
:func:`mark_reachout` once it actually does (manual debug triggers stamp it too, so an
autonomous reach-out won't pile on right after an operator-run one). A job that loses the
race still did its work; it just doesn't DM.

**Durable, not process-local.** The cool-down was once a bare ``time.monotonic()`` stamp,
on the reasoning that a burst only needs avoiding within one idle stretch. That is no
longer safe now that the window can span a day: a restart would reset a 24-hour backoff to
zero and free the box to resume the burst. Both the streak and the last-reach-out time are
therefore read from the chat corpus on disk, which is the durable record of what she
actually sent. The in-process stamp is kept as a floor, so a session written moments ago
counts even before its file is observed.

Reading the corpus has one consequence worth stating plainly: **deleting a transcript
would otherwise erase the evidence of a reach-out**. Since ``core.chat_worklog`` began
deleting openers that go unanswered past its stale window, the deleter tombstones each one
here first (:func:`record_expired`), and :func:`unanswered_streak` folds those tombstones
in alongside the live files. The order matters — the openers old enough to delete are by
definition the longest-ignored ones, so counting only what remains on disk would make the
window *shrink* the longer she is ignored, which is the exact inversion of the rule.

Kept in its own module so ``outreach`` / ``synthesis`` / ``checkin`` share it without
importing one another (no cycle). Like :mod:`core.activity_log` and :mod:`core.worklog` it
is a *leaf*: it imports nothing from the project, so the small "is this a real user turn?"
predicate below is deliberately duplicated from ``checkin._has_user_turn`` rather than
shared (``outreach`` and ``background_reflection`` each carry their own copy too).

GPU-free self-test: ``python -m core.reachout_gate``.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Base reach-out window: after ANY of outreach/synthesis/check-in writes an unprompted
# session, none of them may cold-open the user again for this long. One message per hour
# is the target frequency when she is being answered. Exposed so every caller reads one
# value instead of each hard-coding it.
COOLDOWN_SECONDS = 3600

# Ceiling on the backoff. Reached at an unanswered streak of 5 (1h → 2 → 4 → 8 → 16 → 24),
# after which she settles to about one unprompted message a day while the silence holds.
MAX_COOLDOWN_SECONDS = 24 * 3600

# Monotonic timestamp of the last reach-out this process wrote, or None. A floor under the
# on-disk reading, so a just-written session counts before its file is read back. The
# per-user map is the same floor for a scoped query; the global one still moves on every
# send, so an unscoped caller keeps seeing every reach-out whoever it was addressed to.
_last_reachout: float | None = None
_last_reachout_by_user: dict[str, float] = {}

# Chat corpus root (injected once at startup); None ⇒ disk is not consulted and the gate
# degrades to the in-process stamp with no backoff.
_CHATS_DIR: Optional[Path] = None

# Append-only log of openers that were DELETED from the corpus after going unanswered
# (``core.chat_worklog.delete_stale_reachouts``, run at the start of a reflection). The
# streak below is read from the chat corpus, so a deleted opener would silently stop
# counting — dropping the backoff window back toward its base and re-enabling the burst
# this module exists to stop. The tombstone is what keeps "she sent this and was ignored"
# true after the transcript is gone; it is small (one line per expired opener) and never
# read for anything but the streak. None ⇒ no tombstones, the pre-deletion behaviour.
_EXPIRED_LOG: Optional[Path] = None


def configure(chats_dir: Any, expired_log: Any = None) -> None:
    """Point the gate at the chat corpus (``hot/chats``). Call once at startup.

    *expired_log* is the append-only tombstone log for openers deleted after going
    unanswered (see :data:`_EXPIRED_LOG`). Optional: without it the gate still works,
    it just forgets a reach-out the moment its transcript is deleted."""
    global _CHATS_DIR, _EXPIRED_LOG
    _CHATS_DIR = Path(chats_dir) if chats_dir else None
    _EXPIRED_LOG = Path(expired_log) if expired_log else None


# ── the on-disk record of what she has already sent ────────────────────────────

def _iter_chats():
    """Yield ``(path, data)`` for every readable transcript. Unreadable files are skipped."""
    if _CHATS_DIR is None:
        return
    try:
        paths = sorted(Path(_CHATS_DIR).glob("*.json"))
    except Exception:
        return
    for path in paths:
        # Deliberately duplicated from `chat_sidecar.SIDECAR_SUFFIXES` — this module is a
        # leaf that imports nothing from the project (see the module docstring), so it
        # carries its own copy. Keep the two in sync: a suffix missing here makes a
        # sidecar read as one of Ava's unanswered openers and skews the backoff window.
        if path.name.endswith((".state.json", ".summary.json", ".facts.json",
                               ".shareml.json")):
            continue
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue


def _iter_expired():
    """Yield each tombstone dict from the expired-opener log. Unreadable lines are skipped."""
    if _EXPIRED_LOG is None:
        return
    try:
        raw = Path(_EXPIRED_LOG).read_text(encoding="utf-8")
    except Exception:
        return
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("stem"):
            yield rec


def record_expired(stem: str, *, user: Any = None, sent_at: float = 0.0,
                   opener: str = "") -> bool:
    """Tombstone one unanswered opener whose transcript is being deleted. Returns whether
    it was written.

    Called by the deletion sweep immediately BEFORE the file goes, so the streak (and
    check-in's "what have I already said into this silence" list) survive the deletion.
    *sent_at* is the opener's send time as a POSIX timestamp — the same clock
    :func:`_sent_at` reads off a live transcript, so a tombstone and a file compare
    directly. *opener* is her message text, kept so ``checkin`` can still quote it back to
    her; it is the only content this log holds. Best-effort: the sweep must not fail on a
    logging hiccup (the cost is a shorter backoff, not a corrupt corpus)."""
    if _EXPIRED_LOG is None or not str(stem or "").strip():
        return False
    rec = {
        "stem": str(stem).strip(),
        "user": str(user or "").strip(),
        "sent_at": float(sent_at or 0.0),
        "opener": str(opener or ""),
        "expired_at": time.time(),
    }
    try:
        path = Path(_EXPIRED_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def expired_openers(user: Any = None) -> list[dict]:
    """Every tombstoned opener, oldest→newest, narrowed to *user* when given.

    Exported for ``checkin._standing_openers``, which quotes her standing unanswered
    messages back into the decision prompt: once the sweep deletes an opener the file is
    gone, but she still said it and still was not answered, so it must keep appearing in
    that list or the pass loses exactly the context that stops it repeating itself."""
    key = user_key(user)
    out = [dict(r) for r in _iter_expired()
           if not key or user_key(r.get("user")) == key]
    out.sort(key=lambda r: float(r.get("sent_at") or 0.0))
    return out


_GENERIC_REFERENTS = {
    "user", "the user", "person", "the person", "they", "them", "someone",
    "the speaker", "speaker", "interlocutor", "the interlocutor", "me", "i",
    "unknown", "n/a", "none",
}


def user_key(name: Any) -> str:
    """Comparison key for a person's name — ``""`` when it names nobody.

    Deliberately duplicated from ``reflection_writer.normalize_person`` — this module is a
    leaf that imports nothing from the project (see the module docstring), as are the two
    predicates below. Keep the rule in step with it: lowercased, whitespace-collapsed,
    stripped of surrounding punctuation, reduced to the FIRST token so "Artemy" and
    "artemy voikhansky" are one person, and generic referents name nobody. Exported
    because ``checkin`` scopes by the same key and the two must agree on who is who."""
    s = " ".join(str(name or "").split()).strip(".,;:!?'\"()[]").lower()
    if not s or s in _GENERIC_REFERENTS:
        return ""
    first = s.split(" ", 1)[0].strip(".,;:!?'\"()[]")
    return "" if first in _GENERIC_REFERENTS else first


def _matches(data: dict, key: str) -> bool:
    """True when *data* is a transcript with the person *key* (``""`` ⇒ match anything)."""
    return not key or user_key(data.get("user")) == key


def _has_user_turn(data: dict) -> bool:
    """True when the transcript contains a real utterance by the *human user*.

    Mirrors ``checkin._has_user_turn``: an unanswered Ava opener and an ``interlocutor:"ai"``
    peer transcript both look like conversation but contain nothing the user said, so
    neither may stand in for the user having spoken."""
    exchanges = data.get("exchanges") or []
    if not exchanges:
        return False
    if (data.get("interlocutor") or "").strip() == "ai":
        return False
    if (data.get("initiated_by") or "").strip() == "ava" and len(exchanges) <= 1:
        return False
    return True


def _is_unanswered_opener(data: dict) -> bool:
    """True for a session Ava opened that the user has not replied to."""
    return ((data.get("initiated_by") or "").strip() == "ava"
            and len(data.get("exchanges") or []) <= 1)


def _sent_at(path: Path, data: dict) -> float:
    """When Ava *sent* this opener, as a POSIX timestamp.

    Deliberately the session's creation time (filename stem, else the JSON timestamp) and
    NOT its mtime: ``ChatLogger`` rewrites the file on every turn, so an opener the user
    later replied to has an mtime of the *reply*, which would read as her having reached
    out more recently than she did."""
    stem = path.stem
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S").timestamp()
    except Exception:
        pass
    try:
        return datetime.fromisoformat((data.get("timestamp") or "").strip()).timestamp()
    except Exception:
        pass
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _last_user_turn_ts(key: str = "") -> Optional[float]:
    """When the user last said anything, as a POSIX timestamp, or None if they never have.

    Uses mtime — the file is rewritten per turn, so for a session carrying a real user turn
    its mtime is when that conversation last moved. *key* narrows the question to one
    person; ``""`` asks it of everyone."""
    best: Optional[float] = None
    for path, data in _iter_chats():
        if not _has_user_turn(data) or not _matches(data, key):
            continue
        try:
            mt = path.stat().st_mtime
        except Exception:
            continue
        if best is None or mt > best:
            best = mt
    return best


def unanswered_streak(user: Any = None) -> int:
    """How many unprompted messages she has sent since the user last spoke, unanswered.

    Counts openers *sent after* the user's last turn, so a message she sent before they
    spoke does not count against her even if that particular opener was never answered —
    they did engage, just elsewhere. With no user turn on record at all, every unanswered
    opener counts. *user* narrows both halves to one person's thread: their openers,
    measured against their last turn — one person ignoring her says nothing about how
    another is answering.

    Tombstoned openers (deleted by the stale-reach-out sweep) count exactly as live ones
    do, under the same cutoff. Without that the sweep would quietly undo the backoff: the
    longest-ignored messages are precisely the ones old enough to be deleted, so the
    streak would fall as the silence lengthened."""
    key = user_key(user)
    last_user = _last_user_turn_ts(key)
    n = 0
    seen: set[str] = set()
    for path, data in _iter_chats():
        if not _is_unanswered_opener(data) or not _matches(data, key):
            continue
        if last_user is not None and _sent_at(path, data) <= last_user:
            continue
        seen.add(path.stem)
        n += 1
    for rec in _iter_expired():
        stem = str(rec.get("stem") or "")
        if stem in seen:
            continue          # tombstoned but still on disk — count it once
        if key and user_key(rec.get("user")) != key:
            continue
        if last_user is not None and float(rec.get("sent_at") or 0.0) <= last_user:
            continue
        seen.add(stem)
        n += 1
    return n


def cooldown_seconds(user: Any = None) -> float:
    """The reach-out window right now: :data:`COOLDOWN_SECONDS` doubled per unanswered
    opener, capped at :data:`MAX_COOLDOWN_SECONDS`. Scoped to *user* when given."""
    streak = unanswered_streak(user)
    if streak <= 0:
        return float(COOLDOWN_SECONDS)
    return float(min(MAX_COOLDOWN_SECONDS, COOLDOWN_SECONDS * (2 ** min(streak, 30))))


def seconds_since_reachout(user: Any = None) -> float:
    """Seconds since Ava last reached out, or ``inf`` if she never has.

    The more recent of the on-disk record and this process's own stamp. A large/infinite
    value means the gate is open. Scoped to *user*, it is the last message sent to that
    person — by any of the three jobs, so a scoped caller still sees an unscoped sibling's
    send when it went to the same person."""
    key = user_key(user)
    elapsed = float("inf")
    stamp = _last_reachout_by_user.get(key) if key else _last_reachout
    if stamp is not None:
        elapsed = time.monotonic() - stamp
    latest: Optional[float] = None
    for path, data in _iter_chats():
        if (data.get("initiated_by") or "").strip() != "ava":
            continue
        if not _matches(data, key):
            continue
        ts = _sent_at(path, data)
        if latest is None or ts > latest:
            latest = ts
    # A tombstoned opener was still a message she sent. It is old by construction (the
    # sweep only deletes past its stale window), so it can only ever be the most recent
    # send on a box where every reach-out has since been deleted — but on that box the
    # alternative is reporting "she has never reached out", which opens the gate wide.
    for rec in _iter_expired():
        if key and user_key(rec.get("user")) != key:
            continue
        ts = float(rec.get("sent_at") or 0.0)
        if ts and (latest is None or ts > latest):
            latest = ts
    if latest is not None:
        elapsed = min(elapsed, max(0.0, time.time() - latest))
    return elapsed


def mark_reachout(user: Any = None) -> None:
    """Record that a reach-out session was just written (starts the shared cool-down).

    Always moves the global stamp, and the addressee's own when *user* names one — so an
    unscoped caller keeps seeing every send while a scoped one sees only its own thread."""
    global _last_reachout
    now = time.monotonic()
    _last_reachout = now
    key = user_key(user)
    if key:
        _last_reachout_by_user[key] = now


def may_reach_out(user: Any = None) -> tuple[bool, str]:
    """May she cold-open the user right now? Returns ``(allowed, reason)``.

    *reason* is empty when allowed, else a short skip tag for the caller's result dict:
    ``reachout_cooldown`` (inside the base window) or ``reachout_backoff`` (inside a
    window widened by unanswered openers — the distinction is worth keeping, since one
    means "a sibling job just sent" and the other means "she is talking into silence").

    *user* asks the question of one person's thread instead of the box as a whole."""
    elapsed = seconds_since_reachout(user)
    window = cooldown_seconds(user)
    if elapsed >= window:
        return True, ""
    return False, ("reachout_cooldown" if window <= COOLDOWN_SECONDS
                   else "reachout_backoff")


# ── GPU-free self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    def _write(root: Path, stem: str, *, initiated_by=None, exchanges=1,
               interlocutor=None, user=None) -> Path:
        doc = {"timestamp": stem, "exchanges": [{"i": i} for i in range(exchanges)]}
        if initiated_by:
            doc["initiated_by"] = initiated_by
        if interlocutor:
            doc["interlocutor"] = interlocutor
        if user:
            doc["user"] = user
        path = root / f"{stem}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        ts = datetime.strptime(stem, "%Y%m%d_%H%M%S").timestamp()
        import os
        os.utime(path, (ts, ts))
        return path

    failures = 0

    def check(label: str, got, want) -> None:
        global failures
        ok = got == want
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        configure(root)

        check("empty corpus ⇒ no streak", unanswered_streak(), 0)
        check("empty corpus ⇒ base window", cooldown_seconds(), float(COOLDOWN_SECONDS))

        # A real user conversation, then three unanswered openers after it.
        _write(root, "20250730_120000", exchanges=3)
        _write(root, "20250730_130000", initiated_by="ava", exchanges=1)
        _write(root, "20250730_140000", initiated_by="ava", exchanges=1)
        _write(root, "20250730_150000", initiated_by="ava", exchanges=1)
        check("three unanswered openers", unanswered_streak(), 3)
        check("window doubles per opener", cooldown_seconds(), float(COOLDOWN_SECONDS * 8))

        # An opener the user answered does not count, and its reply is a user turn that
        # resets everything sent before it.
        _write(root, "20250730_160000", initiated_by="ava", exchanges=2)
        check("answered opener resets the streak", unanswered_streak(), 0)
        check("streak reset ⇒ base window", cooldown_seconds(), float(COOLDOWN_SECONDS))

        # An opener sent AFTER that reply counts again.
        _write(root, "20250730_170000", initiated_by="ava", exchanges=1)
        check("opener after the reply counts", unanswered_streak(), 1)

        # A peer transcript is not the user speaking, so it neither resets the streak nor
        # stands in for a user turn.
        _write(root, "20250730_180000", interlocutor="ai", exchanges=6)
        check("peer transcript is not a user turn", unanswered_streak(), 1)

        # The cap holds however deep the silence gets.
        for h in range(1, 12):
            _write(root, f"20250731_{h:02d}0000", initiated_by="ava", exchanges=1)
        check("deep streak", unanswered_streak(), 12)
        check("window is capped", cooldown_seconds(), float(MAX_COOLDOWN_SECONDS))

        # Historical openers are old, so the gate is open despite the backoff.
        allowed, reason = may_reach_out()
        check("aged corpus ⇒ gate open", (allowed, reason), (True, ""))

        # A fresh send closes it, and names the backoff (not the plain cool-down).
        mark_reachout()
        allowed, reason = may_reach_out()
        check("just sent ⇒ blocked as backoff", (allowed, reason), (False, "reachout_backoff"))

        # With no streak the same block reports the plain cool-down.
        for p in root.glob("*.json"):
            p.unlink()
        _write(root, "20250730_120000", exchanges=3)
        mark_reachout()
        allowed, reason = may_reach_out()
        check("no streak ⇒ blocked as cooldown", (allowed, reason), (False, "reachout_cooldown"))

        # ── scoped to one person ──────────────────────────────────────────────
        # Two people are two conversations: one ignoring her must not silence the other.
        for p in root.glob("*.json"):
            p.unlink()
        _last_reachout = None
        _last_reachout_by_user.clear()
        check("names collapse to one key", user_key("Artemy Voikhansky"), "artemy")
        check("a generic referent names nobody", user_key("the user"), "")

        _write(root, "20250801_090000", exchanges=3, user="Artemy")
        _write(root, "20250801_090500", exchanges=3, user="Dana")
        # Three unanswered openers to Artemy, none to Dana.
        for h in range(10, 13):
            _write(root, f"20250801_{h:02d}0000", initiated_by="ava", exchanges=1,
                   user="artemy voikhansky")
        check("global streak counts everyone", unanswered_streak(), 3)
        check("Artemy's streak is his own", unanswered_streak("Artemy"), 3)
        check("Dana is not ignoring her", unanswered_streak("Dana"), 0)
        check("Dana keeps the base window", cooldown_seconds("Dana"),
              float(COOLDOWN_SECONDS))
        check("Artemy's window has widened", cooldown_seconds("Artemy"),
              float(COOLDOWN_SECONDS * 8))

        # A send to one person does not close the other's gate…
        mark_reachout("Artemy")
        check("Artemy's gate closes", may_reach_out("Artemy"),
              (False, "reachout_backoff"))
        check("Dana's stays open", may_reach_out("Dana"), (True, ""))
        # …but an unscoped caller (outreach/synthesis) still sees it.
        check("the global gate closes too", may_reach_out(), (False, "reachout_backoff"))

        # ── tombstones: a deleted opener still counts ─────────────────────────
        # The stale-reach-out sweep deletes the openers that went longest unanswered.
        # Counting only what is left on disk would shrink the window as the silence
        # lengthened, so the tombstones have to carry the same weight as the files.
        for p in root.glob("*.json"):
            p.unlink()
        _last_reachout = None
        _last_reachout_by_user.clear()
        expired = root / "expired.jsonl"
        configure(root, expired_log=expired)

        _write(root, "20250801_090000", exchanges=3, user="Artemy")
        openers = [f"20250801_{h:02d}0000" for h in range(10, 13)]
        for stem in openers:
            _write(root, stem, initiated_by="ava", exchanges=1, user="Artemy")
        check("three live openers", unanswered_streak("Artemy"), 3)

        # Delete two of them the way the sweep does: tombstone, then unlink.
        for stem in openers[:2]:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S").timestamp()
            record_expired(stem, user="Artemy", sent_at=ts, opener=f"opener {stem}")
            (root / f"{stem}.json").unlink()
        check("streak survives the deletion", unanswered_streak("Artemy"), 3)
        check("so does the widened window", cooldown_seconds("Artemy"),
              float(COOLDOWN_SECONDS * 8))
        check("the global view sees them too", unanswered_streak(), 3)
        check("a tombstone is scoped to its addressee", unanswered_streak("Dana"), 0)
        check("her text is kept for check-in",
              [r["opener"] for r in expired_openers("Artemy")],
              [f"opener {s}" for s in openers[:2]])

        # A tombstone predating the user's last turn does not count against her — same
        # rule as a live file, since they did engage after it.
        _write(root, "20250801_180000", exchanges=3, user="Artemy")
        check("the reply clears live and tombstoned alike",
              unanswered_streak("Artemy"), 0)

        # Tombstoning without unlinking must not double-count.
        ts = datetime.strptime(openers[2], "%Y%m%d_%H%M%S").timestamp()
        record_expired(openers[2], user="Artemy", sent_at=ts, opener="still on disk")
        _write(root, "20250801_190000", initiated_by="ava", exchanges=1, user="Artemy")
        check("a tombstoned-but-present opener counts once",
              unanswered_streak("Artemy"), 1)

        # Unconfigured ⇒ degrades to the in-process stamp, no backoff, no crash.
        configure(None)
        check("unconfigured ⇒ no streak", unanswered_streak(), 0)
        check("unconfigured ⇒ base window", cooldown_seconds(), float(COOLDOWN_SECONDS))

    print("\n" + ("all checks passed" if not failures else f"{failures} FAILURE(S)"))
    raise SystemExit(1 if failures else 0)
