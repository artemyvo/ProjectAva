"""The `Library` facade — the API of ASSOCIATIVE_MEMORY.md §1.2, milestone-1 subset.

    lib = Library(root, embedder=BgeM3Embedder(), budget=Budget(total=8000))
    lib.ingest(text, kind="tech_doc", meta={"key": url, "version": "2.0", "product": "nimbus"})
    lib.extract_pending(generate_fn)          # the deferred LLM witness (structure/parser ran at ingest)
    lib.rebuild()                              # fast; scope decided by staleness()
    block = lib.inject("how do I rotate the token?", context="", scope={"product": "nimbus"}, generate_fn=gen)

The library never loads a model; `generate_fn` is injected per call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from . import rebuild as rebuild_mod
from .budget import Budget
from .dense import HashEmbedder
from .puller import Hit, explain as _explain, pull as _pull
from .selection import Block, Policy, select_and_render
from .store import Store
from .witness import pending_documents, run_witnesses
from .activation import Activation, BULK_OFFSET_DAYS, hours as _hours, now_hours
from . import aha as aha_mod
from .spread import HOPS, Reached, spread


class Library:
    def __init__(self, root, *, embedder=None, budget: Optional[Budget] = None, policy: Optional[Policy] = None,
                 knobs: Optional[dict] = None, cache_dir=None, tier: int = 2, now_fn: Optional[Callable[[], float]] = None):
        self.store = Store(root)
        self.cache_dir = cache_dir
        self.tier = tier
        self.now_fn = now_fn or now_hours           # the clock is injected (§2.6): hours
        self.activation = Activation(self.store.root / "state" / "activation.db")
        self.ledger = aha_mod.Ledger(self.store.root / "state" / "aha.jsonl")
        self._needs_path = self.store.root / "state" / "needs.json"
        self.embedder = embedder or HashEmbedder()
        self.budget = budget or Budget()
        self.policy = policy or Policy()
        self.knobs = dict(knobs or {})
        self._build = None
        self._build_id = None

    # ----- ingest -----------------------------------------------------------------------
    def ingest(self, text: str, kind: str, meta: Optional[dict] = None, scope: Optional[dict] = None) -> str:
        doc_id = self.store.ingest(text, kind, meta, scope)
        # The model-free witnesses run inline; the LLM witness stays pending (§1.6 two steps).
        run_witnesses(self.store, doc_id, generate_fn=None)
        return doc_id

    def extract_pending(self, generate_fn: Optional[Callable] = None, *, model_id: str = "", limit: Optional[int] = None,
                        window: Optional[int] = None) -> list[dict]:
        reports = []
        b = self.build
        for doc_id in pending_documents(self.store)[: (limit or 10**9)]:
            kw = {"window": window} if window else {}
            reports.append(run_witnesses(self.store, doc_id, generate_fn=generate_fn, model_id=model_id,
                                         codebook=b.codebook if b else None, embedder=self.embedder, **kw))
        return reports

    def remove(self, doc_key: str) -> None:
        self.store.remove(doc_key)

    def document(self, doc_id: str):
        return self.store.document(doc_id)

    # ----- rebuild ----------------------------------------------------------------------
    def staleness(self) -> dict:
        return rebuild_mod.staleness(self.store, self.embedder.id, self.budget, self.knobs)

    def rebuild(self, scope: Optional[str] = None, now: Optional[str] = None, generate_fn: Optional[Callable] = None) -> dict:
        if scope is None:
            st = self.staleness()
            scope = "full" if st["scope"] == "full" else "fast"
        rep = rebuild_mod.rebuild(self.store, embedder=self.embedder, budget=self.budget, scope=scope, knobs=self.knobs, now=now,
                                  cache_dir=self.cache_dir, generate_fn=generate_fn, closed_needs=self._closed_needs())
        self._build = None
        rep["creation_accesses"] = self._record_creation_accesses()
        return rep

    def _record_creation_accesses(self) -> int:
        """§2.6: ingestion records one access at `asserted_at`, once per independent family;
        a bulk-ingested document's at `asserted_at` minus the cold offset."""
        b = self.build
        if b is None:
            return 0
        now_h = self.now_fn()
        n = 0
        fam = (b.links or {}).get("families", {})
        # Chunks too: a passage is as much a thing that came up as the facts read off it —
        # once per family, at the document's own date (bulk: minus the cold offset).
        for chunk_id, docs in b.chunk_doc.items():
            node = f"chunk:{chunk_id}"
            if not b.injectable(chunk_id) or self.activation.has_any(node):
                continue
            seen_fams: set[str] = set()
            for doc_id in docs:
                f = fam.get(doc_id, doc_id)
                if f in seen_fams:
                    continue
                seen_fams.add(f)
                meta = self.store.meta(doc_id) or {}
                at = _hours(meta.get("date") or meta.get("version") or meta.get("ingested_at"), fallback_h=now_h)
                if meta.get("bulk"):
                    at -= BULK_OFFSET_DAYS * 24.0
                self.activation.touch([node], now_h=now_h, at_h=min(at, now_h))
                n += 1
        for cid, c in b.claims.items():
            node = f"claim:{cid}"
            if self.activation.has_any(node):
                continue
            seen_fams: set[str] = set()
            for o in c.get("occurrences") or []:
                f = fam.get(o.get("doc_id"), o.get("doc_id"))
                if f in seen_fams:
                    continue
                seen_fams.add(f)
                meta = self.store.meta(o.get("doc_id")) or {}
                at = _hours(o.get("asserted_at") or meta.get("date") or meta.get("ingested_at"), fallback_h=now_h)
                if meta.get("bulk"):
                    at -= BULK_OFFSET_DAYS * 24.0
                self.activation.touch([node], now_h=now_h, at_h=min(at, now_h))
                n += 1
        return n

    @property
    def build(self):
        cur = (self.store.root / "index" / "current")
        bid = cur.read_text(encoding="utf-8").strip() if cur.exists() else None
        if self._build is None or bid != self._build_id:
            self._build = rebuild_mod.current_build(self.store)
            self._build_id = bid
        return self._build

    # ----- pull / select / inject -------------------------------------------------------
    def pull(self, cue: str, *, scope: Optional[dict] = None, limit: int = 60, context_terms: Optional[list[str]] = None,
             tier: Optional[int] = None, channels: Optional[set] = None, now: Optional[float] = None,
             extra_seeds: Optional[dict] = None) -> list[Hit]:
        b = self.build
        if b is None:
            return []
        return _pull(self.store, b, cue, scope=scope, embedder=self.embedder, limit=limit, context_terms=context_terms,
                     tier=self.tier if tier is None else tier, channels=channels, activation=self.activation,
                     now_h=self.now_fn() if now is None else now, extra_seeds=extra_seeds)

    def select(self, hits: list[Hit], *, cue: str, context: str = "", scope: Optional[dict] = None,
               policy: Optional[Policy] = None, generate_fn: Optional[Callable] = None, force_model: bool = False) -> Block:
        b = self.build
        if b is None:
            return Block("", "", [], {}, "no_build", False, "", 0, [])
        allowed = set(self.store.current_doc_ids(scope))
        return select_and_render(self.store, b, hits, cue=cue, context=context, policy=policy or self.policy,
                                 budget=self.budget, generate_fn=generate_fn, allowed_docs=allowed,
                                 log_path=self.store.root / "state" / "picks.jsonl", force_model=force_model)

    def inject(self, cue: str, *, context: str = "", scope: Optional[dict] = None, policy: Optional[Policy] = None,
               generate_fn: Optional[Callable] = None, limit: int = 60, force_model: bool = False) -> Block:
        from .lex import terms_of
        ctx_terms = terms_of(context)[-40:] if context else None
        hits = self.pull(cue, scope=scope, limit=limit, context_terms=ctx_terms)
        block = self.select(hits, cue=cue, context=context, scope=scope, policy=policy, generate_fn=generate_fn, force_model=force_model)
        if block.hits:
            self.touch([h["id"] for h in block.hits], scope=scope, why="inject")
        return block

    def explain(self, hit: Hit) -> dict:
        return _explain(hit)

    # ----- state: activation, needs, the aha ---------------------------------------------
    def _nodes_for(self, ids: list[str]) -> list[str]:
        """Library ids → activation nodes: a claim id, a chunk id, or a node id as given. An
        access to a claim propagates one step to its subject node (§3.2)."""
        b = self.build
        out: list[str] = []
        for i in ids:
            if i.startswith(("claim:", "chunk:", "person:", "entity:", "ident:", "need:", "doc:")):
                out.append(i)
            elif b is not None and i in b.claims:
                out.append(f"claim:{i}")
                subj = b.claims[i].get("subject")
                if subj:
                    out.append(subj)
            elif b is not None and i in b.chunk_doc:
                out.append(f"chunk:{i}")
            else:
                out.append(i)
        return list(dict.fromkeys(out))

    def touch(self, ids: list[str], *, scope: Optional[dict] = None, why: str = "touch", now: Optional[float] = None) -> list[str]:
        now_h = self.now_fn() if now is None else now
        nodes = self._nodes_for(list(ids))
        if nodes:
            self.activation.touch(nodes, now_h=now_h, scope=scope)
        self._append_state("touches.jsonl", {"ts": now_h, "ids": list(ids), "nodes": nodes, "scope": scope or {}, "why": why})
        return nodes

    def outcome(self, block_id: str, signal: str) -> None:
        self._append_state("outcomes.jsonl", {"ts": time.time(), "block_id": block_id, "signal": signal})

    def _closed_needs(self) -> set[str]:
        try:
            return {e["need"] for e in self._load_needs_file().get("closed", [])}
        except Exception:
            return set()

    def _load_needs_file(self) -> dict:
        try:
            return json.loads(self._needs_path.read_text(encoding="utf-8")) if self._needs_path.exists() else {"registered": [], "closed": []}
        except Exception:
            return {"registered": [], "closed": []}

    def _save_needs_file(self, obj: dict) -> None:
        self._needs_path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")

    def needs(self, scope: Optional[dict] = None) -> list[dict]:
        """Standing needs in scope: the edge table's need nodes (claims with a need predicate,
        `todo`s) minus closed ones, plus needs the application registered."""
        b = self.build
        closed = self._closed_needs()
        out: list[dict] = []
        if b is not None and b.edges is not None:
            allowed = set(self.store.current_doc_ids(scope))
            for nn, info in b.edges.needs.items():
                cid = info.get("claim_id")
                c = b.claims.get(cid) or {}
                if nn in closed or not any(d in allowed for d in c.get("visible_in") or []):
                    continue
                out.append({"need": nn, "claim_id": cid, "text": info.get("text"), "predicate": info.get("predicate"),
                            "subject": info.get("subject"), "object": info.get("object"), "kind": info.get("kind")})
        for r in self._load_needs_file().get("registered", []):
            if r["need"] not in closed and (not scope or all(r.get("scope", {}).get(k) == v for k, v in scope.items() if k in r.get("scope", {}))):
                out.append(r)
        return out

    def register_need(self, text: str, *, scope: Optional[dict] = None, subject: str = "", entities: Optional[list[str]] = None,
                      now: Optional[float] = None) -> str:
        import hashlib
        nid = "need:reg-" + hashlib.sha1(f"{text}\x1f{json.dumps(scope or {}, sort_keys=True)}".encode("utf-8")).hexdigest()[:12]
        obj = self._load_needs_file()
        if not any(r["need"] == nid for r in obj["registered"]):
            obj["registered"].append({"need": nid, "claim_id": None, "text": text, "predicate": "registered", "subject": subject,
                                      "entities": list(entities or []), "scope": dict(scope or {}), "registered_at": self.now_fn() if now is None else now})
            self._save_needs_file(obj)
        return nid

    def close_need(self, need: str, reason: str = "closed", *, now: Optional[float] = None) -> None:
        obj = self._load_needs_file()
        obj["closed"].append({"need": need, "reason": reason, "ts": self.now_fn() if now is None else now})
        self._save_needs_file(obj)

    def _need_map(self, need_node: str) -> Reached:
        """The need side's reach map: 3 hops from the need, cached per build (§4)."""
        b = self.build
        key = (b.build_id, need_node)
        cache = b._need_maps
        if key not in cache:
            full = spread(b.edges, aha_mod.need_seeds(b.edges, need_node, b.claims, b.codebook), k=HOPS["need"])
            # Hop distance from the CORE seeds only: a cell posting that happens to be the
            # resource's own chunk must not make the resource look one hop from the need.
            core = spread(b.edges, aha_mod.need_seeds(b.edges, need_node, b.claims, None), k=HOPS["need"])
            full.hops = {n: core.hops.get(n, 99) for n in full.r}
            cache[key] = full
        return cache[key]

    def aha(self, *, stimulus: Optional[list[str]] = None, cue: Optional[str] = None, scope: Optional[dict] = None,
            now: Optional[float] = None, aha_min: float = aha_mod.AHA_MIN) -> list[aha_mod.Candidate]:
        """The arithmetic only (§4): stimulus side 2 hops from the given nodes / cue, need side
        cached 3 hops per standing need, min across sides, ledger check, no model."""
        b = self.build
        if b is None or b.edges is None:
            return []
        seeds: dict[str, float] = {}
        for n in self._nodes_for(list(stimulus or [])):
            seeds[n] = 1.0
        if cue:
            from .puller import cue_entity_seeds
            for h in self.pull(cue, scope=scope, limit=20, tier=2):
                node = f"{h.grain}:{h.id}" if h.grain in ("claim", "chunk") else None
                if node:
                    seeds[node] = seeds.get(node, 0.0) + h.score
            for node, w in cue_entity_seeds(b, cue).items():
                seeds[node] = seeds.get(node, 0.0) + w
        if not seeds:
            return []
        stim = spread(b.edges, seeds, k=HOPS["stimulus"])
        allowed = set(self.store.current_doc_ids(scope))
        need_maps = {n["need"]: self._need_map(n["need"]) for n in self.needs(scope) if n.get("claim_id")}
        cands = aha_mod.find_candidates(b.edges, b.claims, need_maps, stim, ledger=self.ledger, scope=scope, aha_min=aha_min)
        cands = [c for c in cands if any(d in allowed for d in (b.claims.get(c.resource) or {}).get("visible_in") or [])]
        # One entry per resource (two needs of one subject raise the same resource twice).
        seen: set[str] = set()
        uniq = []
        for c in cands:
            if c.resource in seen:
                continue
            seen.add(c.resource)
            uniq.append(c)
        return [aha_mod.fill_candidate(c, b.claims, self.store, b.edges, need_maps, stim) for c in uniq[:aha_mod.MAX_JUDGEMENTS * 2]]

    # ----- pivots (§6) ---------------------------------------------------------------------
    def pivot(self, context: str = "", *, warm_ids: Optional[list[str]] = None, bridge: str = "sense",
              scope: Optional[dict] = None, limit: int = 3, hops: int = 2) -> list:
        """The word-anchor context switch. *context*: the current cue / turns; *warm_ids*:
        recently touched chunk or claim ids whose words are also in play. Returns Jumps,
        best first. Never an access — the activation state is untouched."""
        from .codebook import text_terms
        from .senses import Jump, anchor_scores, root_bridges, sound_bridges
        b = self.build
        if b is None or b.edges is None:
            return []
        words: dict[str, float] = {}
        for t in text_terms(context):
            words[t] = 1.0
        for i in warm_ids or []:
            text = ""
            if i in b.claims:
                text = b.claims[i]["text"]
            elif i in b.chunk_doc:
                doc = self.store.document(b.chunk_doc[i][0])
                u = doc.unit(i) if doc else None
                text = u.text if u else ""
            for t in text_terms(text):
                words.setdefault(t, 0.5)
        if not words:
            return []
        ctx_sig = b.codebook.cells_of_text(context + " " + " ".join(words)) if b.codebook else {}
        allowed_docs = set(self.store.current_doc_ids(scope))
        allowed_chunks = {cid for cid, docs in b.chunk_doc.items() if b.injectable(cid) and any(d in allowed_docs for d in docs)}
        jumps: list[Jump] = []
        if bridge == "sense":
            for a in anchor_scores(words, b.senses, b.glossary, b.codebook, ctx_sig)[:limit]:
                info = b.senses[a["word"]]
                far, near = info["senses"][a["far"]], info["senses"][a["near"]]
                seeds = {f"chunk:{c}": 1.0 for c in far["postings"] if c in allowed_chunks}
                if not seeds:
                    continue
                jumps.append(Jump(bridge=a["word"], kind="sense", target=a["word"],
                                  from_sense=[m for _c, m in near["cells"]], to_sense=[m for _c, m in far["cells"]],
                                  distance=info["split"], score=a["score"], seeds=list(far["postings"]),
                                  detail=f"far sense activation {a['far_act']}, idf {a['idf']}"))
        else:
            vocab = [t for t in b.glossary.postings if not t.startswith(("http", "www"))]
            for w, act in sorted(words.items(), key=lambda kv: -kv[1])[:20]:
                if act < 1.0 and bridge != "root":
                    pass
                cands = root_bridges(w, vocab) if bridge == "root" else [(v, st, kind) for v, st, kind in sound_bridges(w, vocab)]
                for c in cands[:3]:
                    target, strength = c[0], c[1]
                    if target in words:
                        continue                      # a target already in play is no switch
                    idf = b.glossary.idf(target)
                    seeds = {f"chunk:{i}": 1.0 for i in b.glossary.postings_of(target) if i in allowed_chunks}
                    if not seeds:
                        continue
                    jumps.append(Jump(bridge=w, kind=bridge, target=target, distance=strength, score=round(act * strength * idf, 4),
                                      seeds=[n[6:] for n in seeds], detail=(c[2] if len(c) > 2 else "")))
            jumps.sort(key=lambda j: -j.score)
            jumps = jumps[:limit]
        for j in jumps:
            reached = spread(b.edges, {f"chunk:{c}": 1.0 for c in j.seeds if c in allowed_chunks}, k=hops)
            hits = []
            for node, r in sorted(reached.r.items(), key=lambda kv: -kv[1]):
                if node.startswith("chunk:") and node[6:] in allowed_chunks:
                    from .puller import Hit, _chunk_meta_for
                    m = _chunk_meta_for(b, node[6:], allowed_docs)
                    hits.append(Hit(grain="chunk", id=node[6:], score=r, doc_id=m.get("doc_id", ""), chunk_id=node[6:],
                                    channels={"pivot": r}, meta=m, build_id=b.build_id, path=reached.paths.get(node, [])[:2]))
                elif node.startswith("claim:") and node[6:] in b.claims:
                    from .puller import Hit, _chunk_meta_for
                    c = b.claims[node[6:]]
                    occ = (c.get("occurrences") or [{}])[0]
                    hits.append(Hit(grain="claim", id=node[6:], score=r, doc_id=occ.get("doc_id", ""), chunk_id=occ.get("chunk_id"),
                                    channels={"pivot": r}, claim=c, meta=_chunk_meta_for(b, occ.get("chunk_id") or "", allowed_docs),
                                    build_id=b.build_id))
                if len(hits) >= 8:
                    break
            j.hits = hits
        return jumps

    def judge(self, cand: aha_mod.Candidate, generate_fn: Callable, *, now: Optional[float] = None, doing: str = "") -> aha_mod.Candidate:
        now_h = self.now_fn() if now is None else now
        b = self.build
        need_c = b.claims.get(cand.need_claim) or {}
        res_c = b.claims.get(cand.resource) or {}
        line = (f"the need was recorded {(need_c.get('occurrences') or [{}])[0].get('asserted_at', '?')}, "
                f"the resource {(res_c.get('occurrences') or [{}])[0].get('asserted_at', '?')}")
        out = aha_mod.judge(cand, generate_fn, self.ledger, now=now_h, now_line=line, doing=doing)
        if out.verdict in ("satisfies", "stale"):
            self.close_need(cand.need, out.verdict, now=now_h)
        return out

    def _append_state(self, name: str, rec: dict) -> None:
        p = self.store.root / "state" / name
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
