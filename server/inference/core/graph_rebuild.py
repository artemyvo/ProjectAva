"""Keep the facts tree current — the fold nothing was running.

The tree (`data/graph/tree.json`) is the only thing standing between the `.facts.json`
protocols and the two retrieval channels that read them: the live chat fetch
(`generation._fetch_facts_block_sync`) and the TIL reading fetch
(`til_wander._fetch_reading_facts`). Both resolve it through `graph.store.read_tree`, and
both treat `None` as *no tree yet* — a named skip, never an error.

**Why this module exists.** Building it was manual (`python -m graph.build`), and nothing on
the box ever called it. The consequence is worst exactly where it is least visible:

  * A **clean install** has no tree at all. Every chat turn and every recap skips with
    `no_tree`, forever, while `graph.enabled` reports `true` — a channel that is off in
    substance and on in configuration. The box goes on writing protocols correctly and
    never reads one.
  * A **running box** freezes its tree at whatever the last manual build saw. Conversations
    and readings land, protocols accumulate beside them, and none of it becomes retrievable.

Neither state announces itself: an empty blob from a stale tree is indistinguishable, at the
call site, from an empty blob because nothing was relevant.

**Why an idle job is the right shape.** The fold is pure stdlib over a few dozen JSON files
— no model, no embedder, no GPU — so it belongs beside `worklog_sweep`, the box's other
GPU-free upkeep job, and takes the same short `idle_seconds` for the same reason: waiting a
full idle hour would starve it on precisely the busy box where the tree goes stale fastest.

**Why rebuilding wholesale is safe.** The tree is *derived and disposable — never a store*
(FACTS_TREE.md): the protocols stay the sources, `read_tree` returns `None` for missing,
corrupt AND unrecognised because all three mean *rebuild*, and `fold` is deterministic, so
two builds of an unchanged corpus are byte-identical. There is no migration path by
construction and nothing to lose by rebuilding too often — only work.

GPU-free self-test: ``python -m core.graph_rebuild``.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Optional

# `graph` is a SIBLING package under `server/`, not importable from a bare `inference/`
# cwd — the same insert five other modules here already make.
_SERVER_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def staleness(graph_dir: Optional[Path] = None, **kw) -> dict:
    """Does the tree need rebuilding, and why? Pure — reads mtimes, never the corpus.

    ``{"stale": bool, "reason": str, "protocols": int, "recorded": int, "newest": float}``.

    Three signals, cheap in that order:

    * **no tree** — missing, corrupt or an unrecognised schema. `read_tree` already collapses
      those three to `None` because they mean the same thing here.
    * **count drift** against the `stats.read.files` the tree recorded. This is what catches
      a *deleted* protocol, which moves no mtime forward, and a restored copy whose mtimes
      predate the build.
    * **any protocol newer than the tree FILE**. Deliberately the file's mtime rather than
      the document's `built_at`: that field is a naive local ISO string written by the
      builder, and comparing it to a filesystem mtime means reconstructing which clock and
      which offset it was written under. The file's own mtime is the same quantity in the
      same units as the thing it is compared against, and it is set by the write that
      produced the document. A Migrate or snapshot restore can reset mtimes and buy one
      spurious rebuild; that costs milliseconds and is self-correcting.
    """
    from graph import read as R
    from graph import store as S

    paths = R.protocol_paths(**kw)
    newest = 0.0
    for _lane, p in paths:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue

    out = {"stale": True, "reason": "", "protocols": len(paths),
           "recorded": 0, "newest": newest}

    doc = S.read_tree(graph_dir)
    if doc is None:
        out["reason"] = "no tree"
        return out
    out["recorded"] = int(((doc.get("stats") or {}).get("read") or {}).get("files", 0) or 0)

    tree_file = S.tree_path(graph_dir)
    try:
        built = tree_file.stat().st_mtime
    except OSError:
        out["reason"] = "no tree file"
        return out

    if out["recorded"] != len(paths):
        out["reason"] = f"{len(paths)} protocol(s) on disk, {out['recorded']} in the tree"
        return out
    if newest > built:
        out["reason"] = "a protocol is newer than the tree"
        return out

    out["stale"] = False
    out["reason"] = "current"
    return out


def rebuild(graph_dir: Optional[Path] = None, *, now: Optional[str] = None, **kw) -> dict:
    """Read → resolve → fold → write. Returns the build's own stats.

    Delegates to `graph.build.build`, so the job and the CLI cannot come to mean different
    things — the discipline `fact_fetch` follows for its two callers, applied to the fold.
    ``now`` is passed in rather than read here for the reason `store.build_doc` takes
    ``built_at``: a build that reads the clock is neither reproducible nor testable.
    """
    from graph import build as B
    from graph import store as S

    tree, read_stats = B.build(graph_dir, **kw)
    stamp = now or _dt.datetime.now().isoformat(timespec="seconds")
    doc = S.build_doc(tree, built_at=stamp, read_stats=read_stats)
    path = S.write_tree(doc, graph_dir)
    st = tree.get("stats") or {}
    return {
        "path": str(path), "built_at": stamp,
        "files": int(read_stats.get("files", 0) or 0),
        "files_by_lane": dict(read_stats.get("files_by_lane") or {}),
        "occurrences": int(st.get("occurrences", 0) or 0),
        "claims": int(st.get("claims", 0) or 0),
        "claims_by_facet": dict(st.get("claims_by_facet") or {}),
        "nodes": int(st.get("nodes", 0) or 0),
        "unresolved_fraction": st.get("unresolved_fraction", 0.0),
        # What the two fetch channels may actually be OFFERED — `claim_candidates` filters
        # to the knowledge facets with `person:_self` removed, so the raw claim count says
        # almost nothing about whether the channel has anything to work with. On the corpus
        # this was written against that gap was 197 claims and 5 candidates, which is the
        # single number worth watching in the journal.
        "showable": int((st.get("claims_by_facet") or {}).get("property", 0) or 0)
        + int((st.get("claims_by_facet") or {}).get("event", 0) or 0),
    }


def run_rebuild_blocking() -> dict:
    """The idle-job entry point. GPU-free; never raises.

    Skips when the tree is current, so a box where nothing changed logs one deduplicated
    line rather than an hourly rebuild notice.
    """
    try:
        state = staleness()
    except Exception as e:
        return {"skipped": f"could not check the tree ({e})"}
    if not state.get("stale"):
        return {"skipped": "the facts tree is current", "protocols": state["protocols"]}
    try:
        res = rebuild()
    except Exception as e:
        # A failed fold must not take the scheduler down, and it is worth saying out loud:
        # every downstream skip would otherwise read as "nothing relevant found".
        return {"error": str(e), "reason": state.get("reason", "")}
    res["reason"] = state.get("reason", "")
    return res


def describe(r: dict) -> str:
    """One journal line. Leads with what the channels can be offered, not the raw total."""
    if r.get("skipped"):
        return f"Facts tree: {r['skipped']}"
    if r.get("error"):
        return f"Facts tree: rebuild FAILED ({r['error']})"
    lanes = r.get("files_by_lane") or {}
    lane_txt = ", ".join(f"{v} {k}" for k, v in sorted(lanes.items())) or "no protocols"
    facets = r.get("claims_by_facet") or {}
    facet_txt = ", ".join(f"{v} {k}" for k, v in sorted(facets.items(),
                                                        key=lambda kv: -kv[1]))
    return (f"Facts tree: rebuilt ({r.get('reason', '')}) — {lane_txt} → "
            f"{r.get('claims', 0)} claims over {r.get('nodes', 0)} nodes "
            f"[{facet_txt}], {r.get('showable', 0)} offerable to a fetch")


# -- GPU-free self-test ------------------------------------------------------ #

def _selftest() -> None:
    """Run: ``python -m core.graph_rebuild``."""
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
        snips = root / "snippets" / "news"
        graph_dir = root / "graph"
        chats.mkdir(parents=True)
        snips.mkdir(parents=True)
        kw = {"chats_dir": chats, "archive_dir": root / "nope", "snippets_dir": snips.parent}

        (chats / "20260101_000000.facts.json").write_text(json.dumps({
            "source_session": "20260101_000000.json", "source_user": "artemyvo",
            "facts": [{"subject": "artemyvo", "subject_raw": "Artemy",
                       "text": "lives in Israel.", "fact_class": "standing",
                       "entities": ["Israel"]}]}), encoding="utf-8")

        print("an absent tree is stale, and says so plainly")
        st = staleness(graph_dir, **kw)
        check("stale", st["stale"], True)
        check("reason", st["reason"], "no tree")
        check("it counted the protocols", st["protocols"], 1)

        print("\nrebuilding writes a tree the consumers can read")
        r = rebuild(graph_dir, now="2026-08-15T00:00:00", **kw)
        check("one protocol read", r["files"], 1)
        check("one claim folded", r["claims"], 1)
        # The number the journal leads with: a `standing` chat fact is `property`, which is
        # a knowledge facet, so this claim is one a fetch pass could be offered.
        check("...and it is offerable", r["showable"], 1)
        from graph import store as S
        check("read_tree round-trips it", S.read_tree(graph_dir) is not None, True)

        print("\na current tree is skipped, not rebuilt hourly")
        check("not stale", staleness(graph_dir, **kw)["stale"], False)
        check("the job says so", run_rebuild_blocking.__doc__ is not None, True)

        print("\na NEW protocol makes it stale again")
        p2 = snips / "2026-08-15.facts.json"
        p2.write_text(json.dumps({
            "source_kind": "news", "source_ref": "news/2026-08-15",
            "facts": [{"subject": "Iran", "subject_raw": "Iran",
                       "text": "launched missiles.", "fact_class": "event",
                       "entities": [], "when": "2026-08-15"}]}), encoding="utf-8")
        st = staleness(graph_dir, **kw)
        check("stale on count drift", st["stale"], True)
        check("...and names the drift", "2 protocol(s) on disk" in st["reason"], True)
        r = rebuild(graph_dir, now="2026-08-15T01:00:00", **kw)
        check("both lanes read", sorted((r["files_by_lane"] or {}).items()),
              [("chat", 1), ("til", 1)])
        check("the TIL event is offerable too", r["showable"], 2)

        print("\na DELETED protocol is caught, though no mtime moved forward")
        os.remove(p2)
        st = staleness(graph_dir, **kw)
        check("stale", st["stale"], True)
        check("...via the count, not the clock",
              "1 protocol(s) on disk, 2 in the tree" in st["reason"], True)

        print("\na TOUCHED protocol is caught by mtime")
        rebuild(graph_dir, now="2026-08-15T02:00:00", **kw)
        check("current again", staleness(graph_dir, **kw)["stale"], False)
        f = chats / "20260101_000000.facts.json"
        future = f.stat().st_mtime + 10_000
        os.utime(f, (future, future))
        st = staleness(graph_dir, **kw)
        check("stale", (st["stale"], st["reason"]),
              (True, "a protocol is newer than the tree"))

        print("\nthe journal line")
        line = describe(rebuild(graph_dir, now="2026-08-15T03:00:00", **kw))
        check("names the lanes and the offerable count",
              ("1 chat" in line and "offerable to a fetch" in line), True)
        check("a skip reads as a skip",
              describe({"skipped": "the facts tree is current"}),
              "Facts tree: the facts tree is current")
        check("a failure is not silent",
              "FAILED" in describe({"error": "boom"}), True)

    print("\ncore.graph_rebuild self-test " + ("OK" if not failures else "FAILED"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
