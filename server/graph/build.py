"""CLI: build the facts tree, and browse it.

    cd server
    .venv/bin/python -m graph.build                    # build + write + summary
    .venv/bin/python -m graph.build --dry-run          # build + summary, write nothing
    .venv/bin/python -m graph.build --nodes            # every node, ranked
    .venv/bin/python -m graph.build --node person:artemyvo
    .venv/bin/python -m graph.build --find artemy
    .venv/bin/python -m graph.build --unresolved       # what the structural tiers could not type
    .venv/bin/python -m graph.build --json             # machine-readable summary

GPU-free and stdlib-only: stage 1 is exact + alias resolution, so a build needs no model, no
embedder and no server running. It reads only the two authoritative dirs and writes only
``data/graph/tree.json``.

``--unresolved`` is the one to look at first on a fresh box. It is the honest residue — the
mentions no free structural signal could type — and it is what the later resolver tiers have
to earn (FACTS_TREE.md §5b).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from graph import fold as F
from graph import nodes as G
from graph import read as R
from graph import store as S


def build(graph_dir=None, **kw) -> tuple:
    """Read → resolve → fold. Returns ``(tree, read_stats)``. Writes nothing."""
    occurrences, read_stats = R.load_occurrences(**kw)
    resolver = G.Resolver(G.load_aliases(S.aliases_path(graph_dir)))
    return F.fold(occurrences, resolver), read_stats


# -- rendering --------------------------------------------------------------- #

def _fmt_summary(tree: dict, read_stats: dict) -> str:
    st = tree["stats"]
    L = []
    L.append("sources")
    L.append(f"  protocols read     {read_stats.get('files', 0)}"
             f"  ({_kv(read_stats.get('files_by_lane') or {})})")
    if read_stats.get("files_unreadable"):
        L.append(f"  unreadable         {read_stats['files_unreadable']}")
    if read_stats.get("dropped_empty"):
        L.append(f"  empty lines dropped {read_stats['dropped_empty']}")
    L.append("")
    L.append("occurrences")
    L.append(f"  total              {st['occurrences']}  ({_kv(st['occurrences_by_lane'])})")
    L.append(f"  about nobody       {st['occurrences_unowned']}")
    L.append("")
    L.append("claims")
    L.append(f"  distinct           {st['claims']}")
    L.append(f"  collapsed          {st['occurrences_collapsed']}  (restatements folded)")
    L.append(f"  corroborated       {st['claims_corroborated']}  (>1 distinct source)")
    for facet in F.FACETS:
        n = st["claims_by_facet"].get(facet, 0)
        if n:
            share = f"{100.0 * n / st['claims']:.0f}%" if st["claims"] else "-"
            note = "  ← not knowledge" if facet in (F.FACET_POSITION, F.FACET_REPORT) else ""
            L.append(f"    {facet:<14} {n:>5}  {share:>4}{note}")
    L.append("")
    L.append("nodes")
    L.append(f"  total              {st['nodes']}  ({_kv(st['nodes_by_type'])})")
    L.append(f"  unresolved         {100.0 * st['unresolved_fraction']:.0f}%"
             f"  of nodes are untyped mentions (see --unresolved)")
    return "\n".join(L)


def _kv(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "-"


def _fmt_nodes(tree: dict, limit: int = 40) -> str:
    ns = sorted(tree["nodes"].values(),
                key=lambda n: (-n["n_claims"], -len(n.get("mentioned_in") or []), n["id"]))
    L = [f"{'node':<44} {'claims':>6} {'occ':>5} {'ment':>5}"]
    for n in ns[:limit]:
        L.append(f"{n['id'][:44]:<44} {n['n_claims']:>6} {n['n_occurrences']:>5} "
                 f"{len(n.get('mentioned_in') or []):>5}")
    if len(ns) > limit:
        L.append(f"... {len(ns) - limit} more")
    return "\n".join(L)


def _fmt_node(tree: dict, ident: str, max_occ: int = 3) -> str:
    doc = {"nodes": tree["nodes"], "claims": tree["claims"],
           "occurrences": tree["occurrences"]}
    n = S.node(doc, ident)
    if not n:
        hits = S.find_nodes(doc, ident.split(":", 1)[-1], limit=8)
        if not hits:
            return f"no such node: {ident}"
        return ("no such node: %s\ndid you mean:\n  %s"
                % (ident, "\n  ".join(h["id"] for h in hits)))

    L = [f"{n['id']}   [{n['type']}]",
         f"  label            {n['label']}",
         f"  surface forms    {', '.join(n['surface_forms']) or '-'}",
         f"  claims           {n['n_claims']} over {n['n_occurrences']} occurrences",
         f"  mentioned in     {len(n.get('mentioned_in') or [])} claims"]
    if G.is_self(ident):
        # Stated here, at the point of reading, because this is where someone would be
        # tempted: FACTS_TREE.md §10 forbids this subtree feeding any injected artifact.
        L.append("  NOTE             self node — browsable and reportable only;")
        L.append("                   feeds no injected artifact (FACTS_TREE.md §10)")
    for facet in F.FACETS:
        cids = (n.get("facets") or {}).get(facet) or []
        if not cids:
            continue
        L.append("")
        flag = "" if facet in F.KNOWLEDGE_FACETS else "   (needs attribution — not knowledge)"
        L.append(f"  [{facet}]  {len(cids)}{flag}")
        for c in S.claims_for(doc, ident, facets=(facet,)):
            mark = "*" if c["n_sources"] > 1 else " "
            when = f"  when={c['when']}" if c.get("when") else ""
            L.append(f"   {mark} {c['text']}")
            L.append(f"       {c['n_sources']} source(s), {c['n_occurrences']} occ, "
                     f"{c['first_asserted']}..{c['last_asserted']}{when}")
            if c.get("mentions"):
                L.append(f"       → {', '.join(c['mentions'][:6])}")
            for o in S.occurrences_for(doc, c)[:max_occ]:
                who = o.get("source_user") or o.get("source_title") or ""
                L.append(f"       · {o['source_ref']}#{o['line_index']}"
                         + (f"  ({who})" if who else ""))
    return "\n".join(L)


def _fmt_unresolved(tree: dict, limit: int = 40) -> str:
    ns = [n for n in tree["nodes"].values() if n["type"] == G.TYPE_UNKNOWN]
    ns.sort(key=lambda n: (-len(n.get("mentioned_in") or []), -n["n_claims"], n["id"]))
    L = [f"{len(ns)} untyped mentions — the residue the structural tiers could not type.",
         "Fix any of them by hand in data/graph/aliases.json, or wait for the embedding",
         "and adjudication tiers (FACTS_TREE.md §5b).", "",
         f"{'mention':<44} {'ment':>5} {'claims':>6}"]
    for n in ns[:limit]:
        L.append(f"{n['key'][:44]:<44} {len(n.get('mentioned_in') or []):>5} "
                 f"{n['n_claims']:>6}")
    if len(ns) > limit:
        L.append(f"... {len(ns) - limit} more")
    return "\n".join(L)


# -- entry point ------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build and browse the facts tree.")
    ap.add_argument("--dry-run", action="store_true", help="build and report, write nothing")
    ap.add_argument("--nodes", action="store_true", help="list nodes, most claims first")
    ap.add_argument("--node", metavar="ID", help="show one node (facets → claims → sources)")
    ap.add_argument("--find", metavar="TEXT", help="search nodes by label or surface form")
    ap.add_argument("--unresolved", action="store_true", help="mentions no tier could type")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--graph-dir", metavar="DIR", help="override data/graph/")
    # Source overrides. The DEFAULT is the authoritative set and stays the only thing a
    # normal build reads; these exist to point the browser at another box's export (or at a
    # fixture), which is a read-only inspection, not a second definition of what counts.
    ap.add_argument("--chats-dir", metavar="DIR")
    ap.add_argument("--snippets-dir", metavar="DIR")
    args = ap.parse_args(argv)

    graph_dir = Path(args.graph_dir) if args.graph_dir else None
    srcs = {}
    if args.chats_dir:
        srcs["chats_dir"] = Path(args.chats_dir)
        # An explicit chats dir means a fixture or an export; its archive sibling, if any,
        # travels with it rather than being taken from this box.
        srcs["archive_dir"] = Path(args.chats_dir).parent / "archive" / "chats"
    if args.snippets_dir:
        srcs["snippets_dir"] = Path(args.snippets_dir)
    tree, read_stats = build(graph_dir, **srcs)

    if args.json:
        print(json.dumps({"stats": tree["stats"], "read": read_stats},
                         ensure_ascii=False, indent=2, sort_keys=True))
    elif args.node:
        print(_fmt_node(tree, args.node))
    elif args.find:
        doc = {"nodes": tree["nodes"], "claims": tree["claims"],
               "occurrences": tree["occurrences"]}
        hits = S.find_nodes(doc, args.find, limit=args.limit)
        print("\n".join(f"{h['id']:<44} {h['n_claims']:>4} claims" for h in hits)
              or f"no node matches {args.find!r}")
    elif args.unresolved:
        print(_fmt_unresolved(tree, args.limit))
    elif args.nodes:
        print(_fmt_nodes(tree, args.limit))
    else:
        print(_fmt_summary(tree, read_stats))

    # Only a plain build writes. Every browse flag is read-only — a command whose job is to
    # show you something should not also replace the file a consumer may be reading. (The
    # build itself is cheap and always runs in memory, so a browse is never stale.)
    browsing = bool(args.node or args.find or args.nodes or args.unresolved or args.json)
    if not args.dry_run and not browsing:
        built_at = _dt.datetime.now().isoformat(timespec="seconds")
        path = S.write_tree(S.build_doc(tree, built_at=built_at, read_stats=read_stats),
                            graph_dir)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
