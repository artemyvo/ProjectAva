"""Chat-keyed worklog record sites — shared by every path that closes a conversation.

:mod:`core.worklog` is a pure leaf (it imports nothing and never reads a chat). This module
is the thin layer above it holding the two record sites that key on a CHAT rather than on a
subsystem's own episode, so both live in ONE place instead of inside whichever subsystem
happened to need them first:

    record_conversation()     — one first-person entry when a user conversation is
                                processed, closing the reach-out thread that opened it
    expire_stale_reachouts()  — close the threads an answer can no longer close
    delete_stale_reachouts()  — drop the unanswered openers themselves from the corpus

Why it exists at all. ``record_conversation`` used to be private to
:mod:`core.background_reflection`, which meant the ONLY path that ever closed a reach-out
thread was the background per-chat pass — a subsystem that has never run on a live GPU. On
a box whose chats are reflected by operator Sleep runs the close therefore never fired: the
worklog accumulated one ``opens`` per reach-out and zero closes, so ``open_threads()`` grew
without bound and the "what have I left unfinished" state a deliberation pass reads was
pure noise. Both freeze paths now call the same function.

``expire_stale_reachouts`` covers the other half. An Ava-initiated chat the user never
answered is deliberately skipped un-frozen by *both* reflection paths (a later reply must
still make it reflectable), so it can never reach a close site even in principle — its
thread hangs by construction. This closes it explicitly, after
:data:`DEFAULT_STALE_HOURS`, by RECORDING a real closing episode rather than filtering at
read time: being ignored is information a deliberation pass should have, and the fold stays
an honest op-log instead of a rule applied at every read.

``delete_stale_reachouts`` takes the same judgement to the corpus: once an opener is written
off, the transcript goes too. It is not a conversation and never becomes one — reflection
skips it un-frozen forever, so it sits in the backlog permanently, shows up in the chat list
as something to open, and is indexed into chat RAG, where her own unanswered message can be
retrieved as if it were part of a dialogue. It runs at the start of a reflection run
(``reflection_service``), which is where the rest of the corpus's periodic tidying lives.

**Two on-disk consumers read exactly these files, and deleting them naively breaks both.**
:func:`core.reachout_gate.unanswered_streak` counts unanswered openers to widen the
reach-out window, and ``checkin._standing_openers`` quotes them back into the decision
prompt so the pass can tell its first message from its fifteenth — the two guards added
after she cold-opened the user 15 times in 19 hours. Both key on "openers sent since the
user last spoke", and the ones old enough to delete are the *longest*-ignored, so a plain
deletion would shrink the backoff and blank the memory precisely as the silence lengthened,
reproducing that bug. So the sweep tombstones each opener (stem, addressee, send time and
her message text) via :func:`core.reachout_gate.record_expired` before unlinking it, and
both consumers fold the tombstones in. What is deleted is the *transcript* — the thing that
was never a conversation; what survives is the fact that she said it and was not answered.

GPU-free self-test: ``python -m core.chat_worklog``.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core import worklog
from core.chat_sidecar import is_chat_session_json, sanitize_gist

#: How long an unanswered reach-out may hang before its thread is written off. At the
#: reach-out gate's 24 h backoff ceiling she has already stopped sending, so this outlives
#: roughly two silent windows before anything is closed.
DEFAULT_STALE_HOURS = 48.0

_GIST_CAP = 300   # clip a long recap to a readable snippet for the worklog line


# ── shared predicates ─────────────────────────────────────────────────────────────

def is_unanswered_outreach(session: dict) -> bool:
    """True for an Ava-initiated chat holding only her opener (no reply).

    Deliberately duplicated from ``background_reflection`` /
    ``ReflectionRunner._is_unanswered_outreach`` / ``reachout_gate``, which each carry
    their own copy so none of them has to import a sibling for four lines."""
    if (session.get("initiated_by") or "").strip() != "ava":
        return False
    return len(session.get("exchanges") or []) <= 1


def summarize_gist(text: str) -> str:
    """Turn a per-chat consolidation-summary recap into a short, clean "about X" for the
    worklog, or "" if nothing usable remains.

    The structural half — salvaging the prose prefix of a summary that degenerated into a
    consolidation dump, dropping a leading `### Reflection` header and any text after an
    `<eos>` marker, rejecting a fully structured one — is `chat_sidecar.sanitize_gist`,
    shared with the sidecar writer and the RAG reader so all three agree on what counts as
    prose. This adds only the worklog-specific presentation: collapse to one line and clip
    to a sentence-ish snippet, so the first-person entry never reads as reflection format."""
    out = " ".join(sanitize_gist(text).split())
    if not out:
        return ""
    if len(out) > _GIST_CAP:
        cut = out[:_GIST_CAP]
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        out = (cut[:dot + 1] if dot > 80 else cut.rstrip()) + " …"
    return out


def _thread_for_session(filename: str) -> Optional[dict]:
    """The still-open thread this chat's reach-out opener left hanging, if any."""
    try:
        for t in worklog.open_threads():
            if (t.get("refs") or {}).get("session") == filename:
                return t
    except Exception:
        pass
    return None


# ── record site 1: a conversation was processed ────────────────────────────────────

def record_conversation(filename: str, session: dict, gist: str = "") -> Optional[dict]:
    """Emit a first-person ``conversation`` entry for a reflected user conversation, closing
    the reach-out thread that opened it.

    This is the natural moment Ava actually *processes* what was said, and the answer to
    "when should a chat produce a worklog entry" — a live transcript has no clean close, its
    reflection does. Called from BOTH freeze paths: the background per-chat pass
    (``chat_reflected``) and the foreground runner's full ``reflected_at`` stamp.

    It INCLUDES an Ava-initiated chat the user replied to: outreach/synthesis/check-in only
    logged the *opener* ("I reached out about X, awaiting reply"), so the ensuing
    back-and-forth is a genuine conversation worth its own entry — and it **closes** that
    opener's thread. The only sessions skipped are a still-unanswered opener (nothing was
    said back) and an AI-interlocutor transcript (encounters / served gossip — not the user).

    ``gist`` is the caller's per-chat consolidation recap, used as the "about X"; it is
    sanitized here (a degenerate/structured summary is rejected in favour of the plain
    template, so reflection format never leaks into a first-person line). Best-effort — a
    worklog hiccup must never disturb the reflection.
    """
    try:
        if (session or {}).get("interlocutor") == "ai":
            return None   # a peer, not the user (encounter / served gossip)
        if is_unanswered_outreach(session or {}):
            return None   # just her opener — the reach-out entry already covers it

        user = ((session or {}).get("user") or "").strip() or "the user"
        clean = summarize_gist(gist or "")
        if clean:
            summary = f"I talked with {user}. {clean}"
        else:
            n = len((session or {}).get("exchanges") or [])
            tail = f" ({n} exchange(s))" if n else ""
            summary = f"I talked with {user}{tail} and reflected on our conversation."

        thread = _thread_for_session(filename)
        return worklog.record("conversation", summary, refs={"session": filename},
                              closes=(thread or {}).get("id"))
    except Exception:
        traceback.print_exc()
        return None


# ── record site 2: nothing can close this thread any more ──────────────────────────

def _parse_ts(raw: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat((raw or "").strip())
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _age_hours(entry: dict, now: datetime) -> Optional[float]:
    dt = _parse_ts(entry.get("ts") or "")
    if dt is None:
        return None
    return (now - dt).total_seconds() / 3600.0


def expire_stale_reachouts(chats_dir, max_age_hours: float = DEFAULT_STALE_HOURS) -> list[dict]:
    """Close every reach-out thread an answer can no longer close. Returns the entries written.

    Three cases, all of them "no answer can close this any more":

    * the session is still an unanswered Ava opener ``max_age_hours`` after she sent it;
    * the session is no longer on disk at all (nothing can close a thread pointing at a
      chat that does not exist);
    * the session WAS answered but its sidecar is already frozen — reflection has been and
      gone, so :func:`record_conversation` will never fire for it. This is the backlog left
      by the era when the close was wired only to the background pass; on a live box it is
      most of the fold.

    A thread whose session got a reply and is *not* yet frozen is left alone — the close
    belongs to reflection, which will fire it via :func:`record_conversation`.

    The closing entry takes the *opener's* kind (outreach / synthesis / check-in): it is the
    tail of that episode, not a conversation. Guarded on a non-empty chats dir, so a
    mis-wired path or a mid-wipe box can never mass-close the whole fold. Best-effort.
    """
    written: list[dict] = []
    try:
        d = Path(chats_dir)
        if not d.is_dir():
            return written
        on_disk = {p.name for p in d.glob("*.json") if is_chat_session_json(p)}
        if not on_disk:
            return written   # empty / wiped / mis-wired — never mass-close on this

        from core.chat_sidecar import ChatSidecar
        sidecar = ChatSidecar(d)

        now = datetime.now(timezone.utc)
        for t in worklog.open_threads():
            filename = (t.get("refs") or {}).get("session")
            if not filename:
                continue   # not a chat-keyed thread — out of scope
            age = _age_hours(t, now)
            if age is None or age < max_age_hours:
                continue

            who = ""
            if filename in on_disk:
                try:
                    data = json.loads((d / filename).read_text(encoding="utf-8"))
                except Exception:
                    continue   # unreadable → leave the thread alone rather than guess
                who = (data.get("user") or "").strip()
                if not is_unanswered_outreach(data):
                    # They replied. If the chat is still unfrozen, reflection owns this
                    # close (record_conversation fires when it processes the chat); if it
                    # is already frozen, that moment has passed and nothing else will.
                    try:
                        frozen = (sidecar.is_reflected(filename)
                                  or sidecar.is_chat_reflected(filename))
                    except Exception:
                        continue
                    if not frozen:
                        continue
                    reason = (f"{who or 'They'} did reply to that, and I have long since "
                              f"thought it over — nothing left hanging there.")
                else:
                    reason = (f"I never heard back from {who or 'them'} — it has been about "
                              f"{age:.0f} hours, so I'm letting that go.")
            else:
                reason = ("That conversation is no longer on record, so there is nothing "
                          "left for me to wait on.")

            kind = t.get("kind") if t.get("kind") in ("outreach", "synthesis", "checkin") \
                else "conversation"
            written.append(worklog.record(kind, reason, refs={"session": filename},
                                          closes=t.get("id")))
    except Exception:
        traceback.print_exc()
    return written


# ── record site 3: the unanswered opener leaves the corpus ─────────────────────────

def _sent_at(path: Path, data: dict) -> float:
    """When Ava *sent* this opener, as a POSIX timestamp.

    Deliberately mirrors ``reachout_gate._sent_at`` rather than importing it: the tombstone
    this function's timestamp lands in is compared directly against live transcripts read by
    that module, so the two must resolve a send time identically. Creation time (filename
    stem, else the JSON timestamp), never mtime — ``ChatLogger`` rewrites the file per turn."""
    try:
        return datetime.strptime(path.stem, "%Y%m%d_%H%M%S").timestamp()
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


def _opener_text(session: dict) -> str:
    exchanges = session.get("exchanges") or []
    if not exchanges:
        return ""
    return str(exchanges[0].get("assistant_response") or "").strip()


def delete_stale_reachouts(chats_dir, max_age_hours: float = DEFAULT_STALE_HOURS,
                           exclude: Optional[set] = None) -> list[dict]:
    """Delete every Ava-initiated chat the user has not answered within *max_age_hours*.
    Returns one dict per deleted chat (``{session, user, files, sent_at}``).

    An unanswered opener is the one session shape that can never become anything: both
    reflection paths skip it un-frozen (correctly — a later reply must still make it
    reflectable), so past the point where a reply is realistic it is permanent backlog,
    a dead entry in the chat list, and a passage in the chat RAG index where her own
    unanswered message can be retrieved as though it were dialogue. Past the window it is
    written off, and this removes it.

    Every file the stem owns goes with the transcript (``chat_sidecar.chat_products``) — a
    summary or fact record outliving its chat is a memory of a conversation that does not
    exist. Before each unlink the opener is tombstoned in :mod:`core.reachout_gate`, which
    is what keeps the reach-out backoff and check-in's standing-opener list honest once the
    file is gone; see this module's docstring for why that is load-bearing rather than
    tidiness. A tombstone that cannot be written **aborts that deletion** — losing the
    evidence of a reach-out is worse than keeping a stale transcript.

    *exclude* names sessions that must not be touched whatever their age — the caller passes
    the file the live logger is appending to, since an opener adopted in place is a
    conversation the user is in the middle of answering.

    Each deletion records a first-person worklog episode and closes the reach-out thread
    that opened it. Guarded on a non-empty chats dir, so a mis-wired path or a mid-wipe box
    can never mass-delete. Best-effort throughout: reflection must not fail over cleanup.
    """
    deleted: list[dict] = []
    if max_age_hours <= 0:
        return deleted          # 0 / negative ⇒ the sweep is off
    try:
        d = Path(chats_dir)
        if not d.is_dir():
            return deleted
        transcripts = [p for p in sorted(d.glob("*.json")) if is_chat_session_json(p)]
        if not transcripts:
            return deleted      # empty / wiped / mis-wired — never mass-delete on this

        from core import reachout_gate
        from core.chat_sidecar import chat_products

        skip = {str(s) for s in (exclude or set())}
        now = datetime.now(timezone.utc).timestamp()
        for path in transcripts:
            if path.name in skip:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue        # unreadable → leave it alone rather than guess
            if not is_unanswered_outreach(data):
                continue
            sent_at = _sent_at(path, data)
            age_h = (now - sent_at) / 3600.0 if sent_at else 0.0
            if sent_at <= 0 or age_h < max_age_hours:
                continue

            who = (data.get("user") or "").strip()
            if not reachout_gate.record_expired(path.stem, user=who, sent_at=sent_at,
                                                opener=_opener_text(data)):
                # No tombstone ⇒ no deletion. Dropping the file here would shorten the
                # backoff window and blank the standing-opener list, which is how the
                # repeated-cold-open bug came back.
                continue

            files: list[str] = []
            try:
                for p in chat_products(d, path.stem):
                    if p.exists():
                        p.unlink()
                        files.append(p.name)
                if path.exists():
                    path.unlink()
                files.append(path.name)
            except Exception:
                traceback.print_exc()
                continue

            thread = _thread_for_session(path.name)
            summary = (f"I never heard back from {who or 'them'} about that — it has been "
                       f"about {age_h:.0f} hours, so I have let it go and taken it off "
                       f"my record.")
            try:
                worklog.record("outreach", summary, refs={"session": path.name},
                               closes=(thread or {}).get("id"))
            except Exception:
                pass
            deleted.append({"session": path.name, "user": who, "files": files,
                            "sent_at": sent_at, "age_hours": age_h})
    except Exception:
        traceback.print_exc()
    return deleted


# ── GPU-free self-test ─────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise both record sites: the conversation close, the skips that must NOT record,
    and every branch of the expiry sweep. Run: ``python -m core.chat_worklog``."""
    import tempfile
    from datetime import timedelta

    def check(label, got, want):
        assert got == want, f"{label}: got {got!r}, want {want!r}"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats = root / "chats"
        chats.mkdir()
        worklog.configure(root / "worklog.jsonl")

        def write_chat(name, **kw):
            (chats / name).write_text(json.dumps(kw), encoding="utf-8")

        # ── record_conversation ───────────────────────────────────────────────
        answered = {"user": "Artemy", "initiated_by": "ava",
                    "exchanges": [{}, {}, {}]}
        opener = worklog.record("outreach", "I reached out about X.",
                                refs={"session": "a.json"}, opens="awaiting a reply")
        check("thread hangs before the reply", len(worklog.open_threads()), 1)

        gist = "We talked about crocodiles and what it means to name a thing after them."
        e = record_conversation("a.json", answered, gist=gist)
        assert e and e["summary"] == f"I talked with Artemy. {gist}"
        check("reply closes the opener's thread", e.get("closes"), opener["id"])
        check("no threads left hanging", worklog.open_threads(), [])

        # a structured/degenerate gist must never leak reflection format into the line
        e = record_conversation("b.json", {"user": "Artemy", "exchanges": [{}]},
                                gist="## WEIGHTS\n[fact] something")
        assert e and e["summary"].startswith("I talked with Artemy (1 exchange(s))")

        # skips: peer transcript, and an opener with no reply
        check("peer transcript records nothing",
              record_conversation("c.json", {"interlocutor": "ai", "exchanges": [{}, {}]}), None)
        check("unanswered opener records nothing",
              record_conversation("d.json", {"initiated_by": "ava", "exchanges": [{}]}), None)

        # ── expire_stale_reachouts ────────────────────────────────────────────
        before = worklog.latest_id()
        write_chat("stale.json", user="Artemy", initiated_by="ava", exchanges=[{}])
        write_chat("fresh.json", user="Artemy", initiated_by="ava", exchanges=[{}])
        write_chat("answered.json", user="Artemy", initiated_by="ava", exchanges=[{}, {}])
        # answered AND already reflected — the pre-fix backlog nothing else can close
        write_chat("frozen.json", user="Artemy", initiated_by="ava", exchanges=[{}, {}])
        (chats / "frozen.state.json").write_text(
            json.dumps({"source_session": "frozen.json",
                        "reflected_at": "2026-07-24T00:00:00"}), encoding="utf-8")

        stale = worklog.record("outreach", "old reach-out", refs={"session": "stale.json"},
                               opens="awaiting a reply")
        fresh = worklog.record("checkin", "new reach-out", refs={"session": "fresh.json"},
                               opens="awaiting a reply")
        ansd = worklog.record("synthesis", "answered reach-out",
                              refs={"session": "answered.json"}, opens="awaiting a reply")
        froz = worklog.record("checkin", "answered + reflected reach-out",
                              refs={"session": "frozen.json"}, opens="awaiting a reply")
        gone = worklog.record("outreach", "vanished chat", refs={"session": "gone.json"},
                              opens="awaiting a reply")
        bare = worklog.record("wander", "no session ref", opens="a loop of its own")

        # backdate everything but `fresh`
        old = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        for entry in worklog._entries:
            if entry["id"] in (stale["id"], ansd["id"], froz["id"], gone["id"], bare["id"]):
                entry["ts"] = old

        out = expire_stale_reachouts(chats, max_age_hours=DEFAULT_STALE_HOURS)
        check("unanswered / vanished / already-reflected threads expired",
              sorted(e["closes"] for e in out),
              sorted([stale["id"], froz["id"], gone["id"]]))
        check("expiry keeps the opener's kind", {e["kind"] for e in out},
              {"outreach", "checkin"})
        hanging = {e["id"] for e in worklog.open_threads()}
        check("fresh / unreflected-answered / session-less threads survive", hanging,
              {fresh["id"], ansd["id"], bare["id"]})

        # idempotent: a second sweep finds nothing new to close
        check("sweep is idempotent", expire_stale_reachouts(chats), [])

        # an empty / mis-wired chats dir must never mass-close
        empty = root / "empty"
        empty.mkdir()
        n = worklog.latest_id()
        check("empty chats dir closes nothing", expire_stale_reachouts(empty), [])
        check("missing chats dir closes nothing", expire_stale_reachouts(root / "nope"), [])
        check("no entries written on a guarded sweep", worklog.latest_id(), n)
        assert worklog.latest_id() > before

        # ── delete_stale_reachouts ────────────────────────────────────────────
        from core import reachout_gate

        dchats = root / "dchats"
        dchats.mkdir()
        reachout_gate.configure(dchats, expired_log=root / "expired.jsonl")

        def stem_for(hours_ago: float) -> str:
            return (datetime.now(timezone.utc)
                    - timedelta(hours=hours_ago)).strftime("%Y%m%d_%H%M%S")

        def write_ts(stem, **kw):
            (dchats / f"{stem}.json").write_text(json.dumps(kw), encoding="utf-8")

        old_stem = stem_for(100)       # unanswered, well past the window
        new_stem = stem_for(2)         # unanswered, still fresh
        # The answered / user-opened controls belong to Dana: they are real user turns, and
        # attributing them to Artemy would (correctly) clear his streak, making the
        # tombstone assertion below vacuous.
        ans_stem = stem_for(100)[:-1] + "1"   # old but answered
        usr_stem = stem_for(100)[:-1] + "2"   # old, but the user opened it
        write_ts(old_stem, user="Artemy", initiated_by="ava",
                 exchanges=[{"assistant_response": "Are you still up?"}])
        write_ts(new_stem, user="Artemy", initiated_by="ava",
                 exchanges=[{"assistant_response": "One more thought."}])
        write_ts(ans_stem, user="Dana", initiated_by="ava", exchanges=[{}, {}])
        write_ts(usr_stem, user="Dana", exchanges=[{}])
        # sidecars the deletion must take with it
        for suffix in (".state.json", ".summary.json", ".facts.json"):
            (dchats / f"{old_stem}{suffix}").write_text("{}", encoding="utf-8")

        opener_thread = worklog.record("checkin", "I reached out to Artemy.",
                                       refs={"session": f"{old_stem}.json"},
                                       opens="awaiting a reply")

        out = delete_stale_reachouts(dchats)
        check("only the stale unanswered opener goes",
              [d["session"] for d in out], [f"{old_stem}.json"])
        check("its sidecars go with it", sorted(out[0]["files"]),
              sorted([f"{old_stem}{s}" for s in
                      (".state.json", ".summary.json", ".facts.json", ".json")]))
        check("the transcript is gone", (dchats / f"{old_stem}.json").exists(), False)
        check("fresh / answered / user-opened chats survive",
              sorted(p.name for p in dchats.glob("*.json")),
              sorted([f"{new_stem}.json", f"{ans_stem}.json", f"{usr_stem}.json"]))
        check("the reach-out thread is closed",
              [e["id"] for e in worklog.open_threads() if e["id"] == opener_thread["id"]],
              [])

        # The whole point of the tombstone: the gate must still see the deleted opener.
        check("the deleted opener still counts toward the backoff",
              reachout_gate.unanswered_streak("Artemy"), 2)
        check("its text survives for check-in to quote",
              [r["opener"] for r in reachout_gate.expired_openers("Artemy")],
              ["Are you still up?"])

        check("sweep is idempotent", delete_stale_reachouts(dchats), [])
        check("the active session is exempt whatever its age",
              delete_stale_reachouts(dchats, max_age_hours=0.0001,
                                     exclude={f"{new_stem}.json"}), [])
        check("a zero window disables the sweep",
              delete_stale_reachouts(dchats, max_age_hours=0), [])
        check("empty chats dir deletes nothing", delete_stale_reachouts(empty), [])
        check("missing chats dir deletes nothing",
              delete_stale_reachouts(root / "nope"), [])

        # No tombstone log ⇒ no deletion: losing the evidence of a reach-out is worse
        # than keeping a stale transcript.
        reachout_gate.configure(dchats, expired_log=None)
        write_ts(stem_for(200), user="Artemy", initiated_by="ava",
                 exchanges=[{"assistant_response": "hello?"}])
        check("no tombstone log ⇒ nothing is deleted",
              delete_stale_reachouts(dchats), [])

    print("chat_worklog selftest: OK")


if __name__ == "__main__":
    _selftest()
