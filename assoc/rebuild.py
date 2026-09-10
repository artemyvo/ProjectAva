"""Rebuild (ASSOCIATIVE_MEMORY.md §2.8, milestone-1 subset): fold the store into
``index/<build_id>/`` — fit stamps, the glossary, the dense index, the claim table — under
a manifest; swap ``index/current`` atomically; report what changed.

Milestone 1 has two scopes: ``fast`` (append what is new; the glossary and claim table are
cheap enough to recompute whole, the dense index re-embeds only uncached text) and ``full``
(the same from scratch, cache dropped). ``deep`` arrives with the model caches in M3.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from . import kinds as kinds_mod
from .authority import authority as _authority, families as _families
from .budget import Budget, fit_chars_per_token, stamp_fits
from .codebook import CODEBOOK_VERSION, Codebook, vocabulary
from .dedup import apply_merges, nominate, rel_contests, resolve_pairs
from .graph import EdgeTable, build_edges
from .relations import RelationCache, run_relation_pass
from .senses import induce_senses, load_senses, save_senses
from .dense import DenseIndex, EmbeddingCache, HashEmbedder, windows_of
from .fold import fold_claims, supersession
from .glossary import GLOSSARY_VERSION, Glossary
from .lex import LEX_VERSION
from .protocol import PARSER_VERSION
from .store import Store

SPLITTER_VERSION = "split-1"
MANIFEST_SCHEMA = 1
_SEQ = itertools.count(1)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Build:
    """A loaded index build: what the puller reads."""

    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.fits = json.loads((path / "fits.json").read_text(encoding="utf-8"))
        self.glossary = Glossary.load(path / "glossary.json")
        self.dense = DenseIndex.load(path) if (path / "dense.faiss").exists() else None
        claims = json.loads((path / "claims.json").read_text(encoding="utf-8"))
        self.claims: dict = claims["claims"]
        self.by_chunk: dict = claims["by_chunk"]
        self.by_subject: dict = claims["by_subject"]
        # chunk_id -> [doc_id, ...]: an unchanged section keeps its id across versions of a
        # page (§1.4), so one chunk legitimately belongs to several documents.
        self.chunk_doc: dict = json.loads((path / "chunk_doc.json").read_text(encoding="utf-8"))
        self.chunk_meta: dict = json.loads((path / "chunk_meta.json").read_text(encoding="utf-8"))
        self.codebook = Codebook.load(path) if (path / "codebook.json").exists() else None
        self.links: dict = json.loads((path / "links.json").read_text(encoding="utf-8")) if (path / "links.json").exists() else {}
        self.edges = EdgeTable.load(path) if (path / "edges.json").exists() else None
        self.senses: dict = load_senses(path)
        self._need_maps: dict = {}

    @property
    def build_id(self) -> str:
        return self.manifest["build_id"]

    def injectable(self, chunk_id: str) -> bool:
        return bool((self.fits.get(chunk_id) or {}).get("injectable"))

    def chunk_tokens(self, chunk_id: str) -> int:
        return int((self.fits.get(chunk_id) or {}).get("tokens") or 0)


def current_build(store: Store) -> Optional[Build]:
    p = store.root / "index" / "current"
    if not p.exists():
        return None
    bid = p.read_text(encoding="utf-8").strip()
    bp = store.root / "index" / bid
    if not (bp / "manifest.json").exists():
        return None
    try:
        return Build(bp)
    except Exception:
        return None


DEFAULT_KNOBS = {"cell_threshold": 0.72, "cell_margin": 0.04, "tier3_min": 0.86, "drift_forced_frac": 0.2,
                 "drift_doubled_cells": 3, "learned_strength": True, "inferred_edges": False}


def versions(embedder_id: str, knobs: Optional[dict] = None) -> dict:
    return {"splitter": SPLITTER_VERSION, "parser": PARSER_VERSION, "lex": LEX_VERSION,
            "glossary": GLOSSARY_VERSION, "codebook": CODEBOOK_VERSION, "embedder": embedder_id,
            "knobs": _knobs_hash({**DEFAULT_KNOBS, **(knobs or {})})}


def _knobs_hash(knobs: dict) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(knobs, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def staleness(store: Store, embedder_id: str, budget: Budget, knobs: Optional[dict] = None) -> dict:
    """Cheapest signal first (§2.8): no manifest → full; version drift → per layer; count drift."""
    b = current_build(store)
    if b is None:
        return {"scope": "full", "reasons": ["no_build"]}
    reasons: list[str] = []
    want = versions(embedder_id, knobs)
    have = b.manifest.get("versions", {})
    for k, v in want.items():
        if have.get(k) != v:
            reasons.append(f"version:{k}")
    counts = b.manifest.get("counts", {})
    n_docs = len(store.all_doc_ids())
    if counts.get("documents") != n_docs:
        reasons.append("count:documents" + ("-" if counts.get("documents", 0) > n_docs else "+"))
    if b.manifest.get("budget", {}).get("total") != budget.total or b.manifest.get("budget", {}).get("ceiling") != budget.chunk_ceiling:
        reasons.append("budget")
    extracted = sum(1 for d in store.all_doc_ids() if (store.root / "documents" / d / "facts.json").exists())
    if counts.get("extracted") != extracted:
        reasons.append("count:extracted")
    # Drift (§2.8): incremental cell assignment degrades; past a knob the next rebuild reclusters.
    kn = {**DEFAULT_KNOBS, **(knobs or {})}
    drift = b.manifest.get("drift") or {}
    assigned = max(drift.get("assigned", 0), 1)
    if drift.get("since_recluster", 0) and (drift.get("forced", 0) / assigned > kn["drift_forced_frac"]
                                             or drift.get("doubled", 0) >= kn["drift_doubled_cells"]):
        reasons.append("drift")
    if not reasons:
        return {"scope": "none", "reasons": []}
    full = any(r.startswith("version:") or r == "drift" for r in reasons) or any(r.endswith("-") for r in reasons)
    return {"scope": "full" if full else "fast", "reasons": reasons}


def rebuild(store: Store, *, embedder=None, budget: Optional[Budget] = None, scope: str = "fast",
            knobs: Optional[dict] = None, now: Optional[str] = None, cache_dir: Optional[Path] = None,
            generate_fn=None, closed_needs: Optional[set] = None) -> dict:
    embedder = embedder or HashEmbedder()
    budget = budget or Budget()
    t0 = time.time()
    build_id = time.strftime("%Y%m%dT%H%M%S", time.localtime()) + f"-{next(_SEQ):03d}{uuid.uuid4().hex[:4]}"
    staging = store.root / "index" / ("." + build_id)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    prev = current_build(store)

    # Embedding cache: carried across fast rebuilds, dropped on full.
    cache_path = Path(cache_dir or (store.root / "index")) / f"embeddings-{embedder.id.replace('/', '_')}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if scope == "full" and cache_path.exists():
        cache_path.unlink()
    cache = EmbeddingCache(cache_path, embedder.id)

    # Every version of every non-removed key is indexed; the CURRENT set is decided at query
    # time from the scope (§1.2), since a scope naming an old version must still reach it.
    docs = [d for d in store.documents() if not (store.keys().get(d.key) or {}).get("removed")]
    current_ids = set(store.current_doc_ids())
    all_ids = {d.doc_id for d in docs}

    # Per-script token estimate, fitted through the injected tokenizer when there is one.
    if budget.tokenizer is not None:
        sample = [u.text for d in docs for u in d.units if u.role == "primary"][:200]
        budget.chars_per_token = fit_chars_per_token(budget.tokenizer, sample)

    fits: dict = {}
    fit_report = {"primaries": 0, "oversize": 0, "split": 0, "still_oversize": 0, "injectable": 0}
    glossary = Glossary()
    dense = DenseIndex(embedder.dim)
    chunk_doc: dict = {}
    chunk_meta: dict = {}
    dense_ids: list[tuple[str, str]] = []
    dense_texts: list[str] = []
    for d in docs:
        spec = kinds_mod.get(d.kind)
        f = stamp_fits(d.units, budget, split_oversize=spec.split_oversize)
        rep = f.pop("__report__")
        for k in fit_report:
            fit_report[k] += rep.get(k, 0)
        fits.update(f)
        is_current = d.doc_id in current_ids
        for u in d.units:
            doc_entry = {"doc_id": d.doc_id, "version": str(d.meta.get("version") or ""), "current": is_current,
                         "date": str(d.meta.get("date") or ""), "title": d.meta.get("title") or d.key, "key": d.key}
            if u.chunk_id in chunk_doc:
                chunk_doc[u.chunk_id].append(d.doc_id)
                chunk_meta[u.chunk_id]["docs"].append(doc_entry)
                continue                      # already indexed: same text, same place
            chunk_doc[u.chunk_id] = [d.doc_id]
            chunk_meta[u.chunk_id] = {"doc_id": d.doc_id, "key": d.key, "kind": d.kind, "path": u.path, "role": u.role,
                                      "unit_type": u.unit_type, "span": list(u.span), "current": is_current,
                                      "title": d.meta.get("title") or d.key, "version": str(d.meta.get("version") or ""),
                                      "date": str(d.meta.get("date") or ""), "speaker": u.keys.get("speaker") or "",
                                      "ts": u.keys.get("ts") or "", "identity": u.identity, "docs": [doc_entry]}
            if u.role == "ancestor":
                continue
            glossary.add(u.chunk_id, u.text, "chunk")
            if f.get(u.chunk_id, {}).get("injectable") or u.role == "primary":
                for w in windows_of(u.text):
                    dense_ids.append((u.chunk_id, "chunk"))
                    dense_texts.append(w)

    claims = fold_claims(docs, current_ids=all_ids)
    sup = supersession(store, current_ids=all_ids)
    for c in sup["removed"]:
        prev_c = claims["claims"].get(c["claim_id"])
        if prev_c is None:
            claims["claims"][c["claim_id"]] = c
            continue
        # The same removal seen from another platform / edition: one claim, every source.
        prev_c["occurrences"].extend(c["occurrences"])
        prev_c["visible_in"] = sorted(set(prev_c.get("visible_in") or []) | set(c.get("visible_in") or []))
        prev_c["sources"] = sorted(set(prev_c["sources"]) | set(c["sources"]))
        prev_c["n_sources"] = len(prev_c["sources"])
    for c in claims["claims"].values():
        c["visible_in"] = c.get("visible_in") or sorted({o["doc_id"] for o in c["occurrences"]})
    kn = {**DEFAULT_KNOBS, **(knobs or {})}
    aliases = _load_aliases(store)
    _apply_aliases(claims, aliases)

    # ----- L2: the codebook (§2.2). Fast: assign new terms into the previous cells; full:
    # recluster. Term vectors go through the same cache as everything else.
    chunk_texts = [u.text for d in docs for u in d.units if u.role != "ancestor"]
    claim_texts = [c["text"] for c in claims["claims"].values()]
    mentions = [e for c in claims["claims"].values() for e in (c.get("entities") or [])] + \
               [c.get("subject_raw") or "" for c in claims["claims"].values()]
    vocab = vocabulary(chunk_texts + claim_texts, mentions)
    prev_cb = prev.codebook if (prev and scope != "full" and prev.codebook and prev.codebook.embedder_id == embedder.id
                                and abs(prev.codebook.threshold - kn["cell_threshold"]) < 1e-9) else None
    codebook = Codebook(kn["cell_threshold"], kn["cell_margin"], embedder.id)
    if vocab:
        tvecs = cache.get_many(vocab, embedder)
        if prev_cb is not None:
            codebook.cells = [dict(c) for c in prev_cb.cells]
            codebook.assign = dict(prev_cb.assign)
            codebook._members_at_recluster = dict(prev_cb._members_at_recluster)
            codebook.drift = dict(prev_cb.drift)
            codebook.assign_new(vocab, tvecs)
            codebook._build_neighbours()
        else:
            codebook.cluster(vocab, tvecs)
    for d in docs:
        for u in d.units:
            if u.role != "ancestor":
                codebook.index(u.chunk_id, u.text, "chunk")

    # ----- Dedup tiers 2–3 with the equivalence check (§2.5).
    def embed_fn(texts: list[str]):
        return cache.get_many(texts, embedder)

    def resolve(mention: str) -> str:
        m = " ".join(mention.split()).strip().lower()
        m = aliases.get(m, m)
        a = codebook.assign.get(m) or codebook.assign.get(m.split()[-1] if m.split() else m)
        return f"cell:{a[0][0]}" if a else m

    def kind_markers(kind: str) -> frozenset:
        return frozenset({"deprecated", "removed"}) if kind in ("tech_doc", "structured") else frozenset()

    pairs = nominate(claims["claims"], cell_set=codebook.cell_set, embed=embed_fn, tier3_min=kn["tier3_min"])
    resolved = resolve_pairs(claims["claims"], pairs, resolve=resolve, kind_markers=kind_markers)
    remap = apply_merges(claims["claims"], resolved["merged"])
    for cid_list in list(claims["by_chunk"].values()):
        for i, cid in enumerate(cid_list):
            cid_list[i] = remap.get(cid, cid)
    for k, cid_list in list(claims["by_chunk"].items()):
        claims["by_chunk"][k] = list(dict.fromkeys(cid_list))
    for k, cid_list in list(claims["by_subject"].items()):
        claims["by_subject"][k] = list(dict.fromkeys(remap.get(c, c) for c in cid_list))
    contests = [{"a": remap.get(x["a"], x["a"]), "b": remap.get(x["b"], x["b"]), "tier": x["tier"]}
                for x in resolved["contests"] + rel_contests(claims["claims"])]
    contests = [x for x in contests if x["a"] != x["b"] and x["a"] in claims["claims"] and x["b"] in claims["claims"]]
    supers = [{"older": remap.get(x["older"], x["older"]), "newer": remap.get(x["newer"], x["newer"]), "tier": x["tier"]}
              for x in resolved["supersedes"]]
    supers = [x for x in supers if x["older"] != x["newer"] and x["older"] in claims["claims"] and x["newer"] in claims["claims"]]
    for c in claims["claims"].values():
        c["contests"], c["superseded_by"] = [], None
    for x in contests:
        claims["claims"][x["a"]]["contests"].append(x["b"])
        claims["claims"][x["b"]]["contests"].append(x["a"])
    for x in supers:
        claims["claims"][x["older"]]["superseded_by"] = x["newer"]

    # ----- Typed relations (§2.3): per document, over claims with no relation yet, cached.
    rel_report: dict = {"skipped": "no_model"}
    if generate_fn is not None:
        by_doc: dict[str, list[dict]] = {}
        titles: dict[str, str] = {}
        for c in claims["claims"].values():
            if c.get("rel") or c["claim_id"].startswith("rm-"):
                continue
            o = (c.get("occurrences") or [{}])[0]
            by_doc.setdefault(o.get("doc_id", "?"), []).append(c)
            titles[o.get("doc_id", "?")] = o.get("title") or o.get("key") or ""

        def cell_of(mention: str):
            a = codebook.assign.get(mention) or codebook.assign.get(mention.split()[-1] if mention.split() else mention)
            return f"cell:{a[0][0]}" if a else None

        rel_report = run_relation_pass(by_doc, titles, generate_fn, RelationCache(store.root / "state" / "relations_cache.json"),
                                       cell_of=cell_of)

    # ----- Families + authority (§2.5).
    fam_of_doc = _families(docs)
    kind_of_doc = {d.doc_id: d.kind for d in docs}
    auth = _authority(claims["claims"], fam_of_doc, kind_of_doc)
    for cid, c in claims["claims"].items():
        c["n_independent"] = auth["independent"].get(cid, 1)
        c["authority"] = round(auth["claims"].get(cid, 0.0), 4)
        c["families"] = sorted({fam_of_doc.get(o["doc_id"], o["doc_id"]) for o in c["occurrences"]})

    for cid, c in claims["claims"].items():
        glossary.add(cid, c["text"] + " " + " ".join(c.get("entities") or []), "claim")
        codebook.index(cid, c["text"], "claim")
        dense_ids.append((cid, "claim"))
        dense_texts.append(c["text"])
    glossary.finalize()

    if dense_texts:
        vecs = cache.get_many(dense_texts, embedder)
        dense.add(dense_ids, dense_texts, vecs)
    cache.save()

    # Alias proposals: entity mentions that landed in one cell under different nodes.
    mention_nodes: dict[str, str] = {}
    for c in claims["claims"].values():
        if c.get("subject_raw") and c.get("subject"):
            mention_nodes[c["subject_raw"]] = c["subject"]
    proposals = codebook.alias_proposals(mention_nodes) if mention_nodes else []
    _write_json(store.root / "state" / "aliases.proposed.json", {"proposals": proposals, "build": build_id})

    # Write the build.
    json.dump(fits, (staging / "fits.json").open("w", encoding="utf-8"))
    glossary.save(staging / "glossary.json")
    dense.save(staging)
    json.dump({"claims": claims["claims"], "by_chunk": claims["by_chunk"], "by_subject": claims["by_subject"]},
              (staging / "claims.json").open("w", encoding="utf-8"), ensure_ascii=False)
    json.dump(chunk_doc, (staging / "chunk_doc.json").open("w", encoding="utf-8"))
    json.dump(chunk_meta, (staging / "chunk_meta.json").open("w", encoding="utf-8"), ensure_ascii=False)
    codebook.save(staging)
    edges = build_edges(docs, claims["claims"], {"contests": contests, "supersedes": supers}, aliases=aliases,
                        closed_needs=set(closed_needs or ()), inferred=_load_inferred(store) if kn.get("inferred_edges") else None)
    if kn.get("learned_strength", True):
        edges.learn_strength()
    edges.save(staging)
    # Senses (§6): induced from the contexts each polysemous word is used in.
    chunk_text_map = {u.chunk_id: u.text for d in docs for u in d.units if u.role != "ancestor"}
    senses = induce_senses(glossary, codebook, chunk_text_map)
    save_senses(senses, staging)
    n_fams = len(set(fam_of_doc.values()))
    json.dump({"contests": contests, "supersedes": supers, "families": fam_of_doc, "n_families": n_fams,
               "family_authority": auth.get("families", {})}, (staging / "links.json").open("w", encoding="utf-8"))
    extracted = sum(1 for d in docs if d.facts)
    counts = {"documents": len(docs), "current": len(current_ids), "chunks": len(chunk_doc),
              "extracted": extracted, "claims": len(claims["claims"]), "terms": glossary.stats()["terms"],
              "dense_rows": len(dense_texts)}
    counts["claims"] = len(claims["claims"])
    report = {"scope": scope, "counts": counts, "fits": fit_report, "claims": claims["stats"],
              "removed_facts": len(sup["removed"]), "glossary": glossary.stats(), "codebook": codebook.stats(),
              "dedup": resolved["stats"], "contests": len(contests), "supersedes": len(supers),
              "families": n_fams, "alias_proposals": len(proposals), "relations": rel_report, "edges": edges.stats(),
              "senses": {"words": len(senses), "sample": sorted(senses)[:8]},
              "delta": _delta(prev.manifest["counts"] if prev else {}, counts), "seconds": round(time.time() - t0, 2)}
    manifest = {"schema": MANIFEST_SCHEMA, "build_id": build_id, "built_at": now or _now(), "scope": scope,
                "versions": versions(embedder.id, knobs), "counts": counts, "drift": codebook.drift,
                "budget": {"total": budget.total, "ceiling": budget.chunk_ceiling, "chars_per_token": budget.chars_per_token},
                "report": report, "previous": prev.build_id if prev else None}
    json.dump(manifest, (staging / "manifest.json").open("w", encoding="utf-8"), indent=1)
    final = store.root / "index" / build_id
    os.replace(staging, final)
    tmp = store.root / "index" / "current.tmp"
    tmp.write_text(build_id, encoding="utf-8")
    os.replace(tmp, store.root / "index" / "current")
    # Keep the previous build for readers mid-call; drop anything older.
    _prune_builds(store, keep={build_id, prev.build_id if prev else ""})
    return report


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _load_inferred(store: Store) -> list[dict]:
    p = store.root / "state" / "inferred_edges.json"
    try:
        return (json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}).get("edges", [])
    except Exception:
        return []


def _load_aliases(store: Store) -> dict[str, str]:
    """state/aliases.json — hand-promoted `{mention: canonical}` (lowercased both sides)."""
    p = store.root / "state" / "aliases.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in (raw or {}).items()}


def _apply_aliases(claims: dict, aliases: dict[str, str]) -> None:
    """Fold subjects through the alias table so `person:артемий` and `person:artemy` are one
    block for the dedup tiers (a promoted alias, never a guess)."""
    if not aliases:
        return
    for c in claims["claims"].values():
        subj = c.get("subject") or ""
        if ":" in subj:
            ns, key = subj.split(":", 1)
            key2 = aliases.get(key.lower())
            if key2 and key2 != key:
                c["subject_alias_of"] = subj
                c["subject"] = f"{ns}:{key2}"
    by_subject: dict = {}
    for cid, c in claims["claims"].items():
        by_subject.setdefault(c["subject"], []).append(cid) if c.get("subject") else None
    claims["by_subject"] = by_subject


def _delta(prev: dict, cur: dict) -> dict:
    return {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}


def _prune_builds(store: Store, keep: set[str]) -> None:
    for p in (store.root / "index").iterdir():
        if p.is_dir() and not p.name.startswith(".") and p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)
