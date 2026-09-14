"""Re-measure the tier ladder on a REAL corpus (ASSOCIATIVE_MEMORY.md §9, §12): Ava's
protocols imported from the main box. Labels come by construction, not by hand:

  fact → source   cue = a grounded fact's text (mostly English), expected = the chunk it is
                  anchored to (mostly a Russian exchange) — the cross-lingual recall of the
                  passage behind a fact, measured at the CHUNK grain only (the claim itself
                  would be a trivial hit);
  turn → facts    cue = a user turn (Russian), expected = any claim anchored to that chunk.

Run: server/.venv/bin/python -m assoc.bench.measure_real <library_root> [--n 300] [--seed 7]
Writes bench/reports/real_corpus.json.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from ..activation import Activation
from ..budget import Budget
from ..library import Library

REPORT = Path(__file__).parent / "reports" / "real_corpus.json"


def _mrr(ranks):
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def _stats(ranks):
    found = [r for r in ranks if r]
    return {"n": len(ranks), "found": len(found), "mrr": round(_mrr(ranks), 3),
            "top1": round(sum(1 for r in found if r == 1) / max(len(ranks), 1), 3),
            "top3": round(sum(1 for r in found if r <= 3) / max(len(ranks), 1), 3),
            "top10": round(sum(1 for r in found if r <= 10) / max(len(ranks), 1), 3)}


def main(root: str, n: int = 300, seed: int = 7, tiers=(1, 2, 3), bge: bool = True) -> dict:
    embedder = None
    if bge:
        from ..dense import BgeM3Embedder
        import torch
        embedder = BgeM3Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    lib = Library(root, embedder=embedder, budget=Budget(total=8000))
    b = lib.build
    rng = random.Random(seed)
    from ..lex import dominant_script

    # ---- fact → source labels: grounded chat claims whose chunk is in the other script
    chat_claims = [c for c in b.claims.values() if c["kind"] == "chat" and c["occurrences"] and c["occurrences"][0].get("chunk_id")]
    labels_f2s = []
    for c in rng.sample(chat_claims, min(len(chat_claims), n * 3)):
        occ = c["occurrences"][0]
        doc = lib.store.document(occ["doc_id"])
        u = doc.unit(occ["chunk_id"]) if doc else None
        if not u or dominant_script(c["text"]) == dominant_script(u.text):
            continue
        labels_f2s.append((c["text"], occ["chunk_id"], occ["doc_id"]))
        if len(labels_f2s) >= n:
            break
    # ---- turn → facts labels: user turns of chunks that anchor ≥1 claim
    labels_t2f = []
    chunk_ids = [cid for cid, cl in b.by_chunk.items() if cl and cid in b.chunk_doc]
    for cid in rng.sample(chunk_ids, min(len(chunk_ids), n * 2)):
        doc = lib.store.document(b.chunk_doc[cid][0])
        u = doc.unit(cid) if doc else None
        if not u or doc.kind != "chat":
            continue
        turn = u.text.split("\nMe:")[0]
        turn = turn.split(":", 1)[1].strip() if ":" in turn else turn
        if len(turn) < 20:
            continue
        labels_t2f.append((turn[:400], set(b.by_chunk[cid])))
        if len(labels_t2f) >= n:
            break

    # cold activation state for the ladder
    saved = lib.activation
    lib.activation = Activation(Path(root) / "state" / "ladder.db")
    lib._record_creation_accesses()
    out = {"root": root, "embedder": lib.embedder.id, "counts": b.manifest["counts"], "labels": {"fact_to_source": len(labels_f2s), "turn_to_facts": len(labels_t2f)}, "tiers": {}}
    try:
        for tier in tiers:
            t0 = time.time()
            ranks_f2s, ranks_f2s_chunk, ranks_f2s_doc = [], [], []
            same_chat_top = 0
            for cue, chunk_id, doc_id in labels_f2s:
                hits = lib.pull(cue, limit=60, tier=tier)
                ranks_f2s.append(next((h.rank for h in hits if h.grain == "chunk" and h.chunk_id == chunk_id), None))
                chunk_hits = [h for h in hits if h.grain == "chunk"]
                ranks_f2s_chunk.append(next((i + 1 for i, h in enumerate(chunk_hits) if h.chunk_id == chunk_id), None))
                if chunk_hits and chunk_hits[0].doc_id == doc_id:
                    same_chat_top += 1          # the top chunk is from the right conversation
                # conversation rank over the passage grains only (the claim's own hit would be trivial)
                docs_seen: list[str] = []
                for h in hits:
                    if h.grain != "claim" and h.doc_id not in docs_seen:
                        docs_seen.append(h.doc_id)
                ranks_f2s_doc.append(docs_seen.index(doc_id) + 1 if doc_id in docs_seen else None)
            t_f2s = time.time() - t0
            t0 = time.time()
            ranks_t2f, ranks_t2f_claim = [], []
            for cue, claim_ids in labels_t2f:
                hits = lib.pull(cue, limit=60, tier=tier)
                ranks_t2f.append(next((h.rank for h in hits if h.grain == "claim" and h.id in claim_ids), None))
                claim_hits = [h for h in hits if h.grain == "claim"]
                ranks_t2f_claim.append(next((i + 1 for i, h in enumerate(claim_hits) if h.id in claim_ids), None))
            t_t2f = time.time() - t0
            t = {"fact_to_source": {"mixed": _stats(ranks_f2s), "among_chunks": _stats(ranks_f2s_chunk), "conversation": _stats(ranks_f2s_doc),
                                    "top_chunk_same_chat": round(same_chat_top / max(len(labels_f2s), 1), 3)},
                 "turn_to_facts": {"mixed": _stats(ranks_t2f), "among_claims": _stats(ranks_t2f_claim)},
                 "ms_per_pull": round(1000 * (t_f2s + t_t2f) / max(len(labels_f2s) + len(labels_t2f), 1), 1)}
            out["tiers"][f"tier{tier}"] = t
            print(f"tier {tier}: fact→source mixed {t['fact_to_source']['mixed']} | among chunks {t['fact_to_source']['among_chunks']} | "
                  f"conversation {t['fact_to_source']['conversation']}\n        turn→facts mixed {t['turn_to_facts']['mixed']} | among claims "
                  f"{t['turn_to_facts']['among_claims']} | {t['ms_per_pull']} ms/pull", flush=True)
    finally:
        lib.activation = saved
    # ---- the creative and associative surfaces, counted
    out["senses"] = {"words": len(b.senses), "sample": sorted(b.senses)[:12]}
    jumps = lib.pivot("ключ от квартиры и замок на двери") + lib.pivot("работа и деньги")
    out["pivot_sample"] = [j.to_dict() | {"hits": len(j.hits)} for j in jumps[:4]]
    out["needs"] = len(lib.needs())
    out["contests"] = b.manifest["report"].get("contests")
    out["families"] = b.manifest["report"].get("families")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    root = args[0]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 300
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 7
    main(root, n=n, seed=seed, bge="--hash" not in args)
