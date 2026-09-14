"""bench/baseline (§9, §12): the ladder — tier 1 (lexical + dense) against tier 2 (plus the
L2 cells and the recent-turn seeds) on the same labels, the same corpus, the same budget.
Records both into ``bench/reports/baseline.json``. Milestone 2's gate: tier 2 beats tier 1."""

import json
from pathlib import Path

REPORT = Path(__file__).parent / "reports" / "baseline.json"

LABELS = [
    # (cue, scope, accepted answers [(key, path tail)], grain) — several chunks may state one fact
    ("how do I rotate the token?", {"version": "2.0", "platform": "windows"}, [("docs/windows/config.md", "Rotating the token"), ("docs/windows/faq.md", "How do I rotate the token?")], "chunk"),
    ("What is the default connect_timeout?", {"version": "1.0", "platform": "linux"}, [("docs/linux/config.md", "connect_timeout")], "chunk"),
    ("where are the logs written?", {"version": "2.0", "platform": "windows"}, [("docs/windows/faq.md", "Where are the logs?"), ("docs/windows/install.md", "Installing on windows")], "chunk"),
    ("how do I upgrade the gateway", {"version": "2.0", "platform": "linux"}, [("docs/linux/install.md", "Upgrading")], "chunk"),
    ("does it support HTTP/3", {"version": "2.0", "platform": "linux"}, [("docs/linux/faq.md", "Does the gateway support HTTP/3?")], "chunk"),
    ("E142", {"version": "2.0", "platform": "linux"}, [("docs/linux/errors.md", "rows5")], "chunk"),
    ("what calls resolve?", {}, [("demo/client.py", "Client.connect")], "claim"),
    ("Noam Keller offered a position", {}, [("chat-noam", "exchange 0")], "chunk"),
    ("ключ от квартиры", {}, [("chat-ru", "exchange 0")], "chunk"),
    ("Starling acquisition price", {}, [("news/starling", "paragraph0"), ("news/syndicated", "paragraph0"), ("news/mirror", "paragraph0"), ("news/contra", "Starling price disputed")], "chunk"),
    # same fact stated in both languages: either chat is a correct top answer
    ("сестра живёт в Хайфе", {}, [("chat-haifa-en", "exchange 0"), ("chat-haifa-ru", "exchange 0")], "chunk"),
    ("sister works at the Technion", {}, [("chat-haifa-ru", "exchange 1"), ("chat-haifa-en", "exchange 1")], "chunk"),
    # the answer exists ONLY in the other language — where L2 has to carry the whole match
    ("предложили должность CTO в стартапе", {}, [("chat-noam", "exchange 0")], "chunk"),
    ("сколько заплатили за Starling", {}, [("news/starling", "paragraph0"), ("news/syndicated", "paragraph0"), ("news/mirror", "paragraph0"), ("news/contra", "Starling price disputed")], "chunk"),
    ("the key to the apartment was lost", {}, [("chat-ru", "exchange 0")], "chunk"),
    ("the old knife's steel started to rust", {}, [("chat-ru", "exchange 2")], "chunk"),
    ("looking for a job since June", {}, [("chat-kestrel", "exchange 0")], "chunk"),
    ("ищет работу с июня", {}, [("chat-kestrel", "exchange 0")], "chunk"),
]
CHAT_KEYS = {"chat-noam", "chat-ru", "chat-haifa-en", "chat-haifa-ru", "chat-kestrel"}


def _rank(hits, accepted, grain):
    """Grain-agnostic on a chunk label: a claim anchored to the expected chunk is the same
    answer at the finer grain (the catalogue nests it under that passage), so it counts."""
    for h in hits:
        if grain == "claim" and h.grain != "claim":
            continue
        if grain == "chunk" and h.grain not in ("chunk", "claim"):
            continue
        if (h.meta.get("key"), (h.meta.get("path") or [""])[-1]) in accepted:
            return h.rank
    return None


def _score(corpus, tier):
    rows, ranks = [], []
    for cue, scope, accepted, grain in LABELS:
        hits = corpus.pull(cue, scope=scope, limit=60, tier=tier)
        r = _rank(hits, set(accepted), grain)
        key = accepted[0][0]
        ranks.append((key, r))
        rows.append({"cue": cue, "expected": [f"{k} › {t}" for k, t in accepted], "rank": r,
                     "top": [{"grain": h.grain, "key": h.meta.get("key"), "path": (h.meta.get("path") or [])[-1:], "score": round(h.score, 3)} for h in hits[:3]]})

    def mrr(pairs):
        return sum(1.0 / r for _, r in pairs if r) / len(pairs) if pairs else 0.0
    chat = [(k, r) for k, r in ranks if k in CHAT_KEYS]
    return {"mrr": round(mrr(ranks), 3), "top3": round(sum(1 for _, r in ranks if r and r <= 3) / len(ranks), 3),
            "found": sum(1 for _, r in ranks if r), "n": len(ranks), "mrr_chat": round(mrr(chat), 3), "rows": rows}


def test_tier_ladder(corpus, tmp_path):
    from assoc.activation import Activation
    # Tier 3 reads the activation state, which other benches warm (priming). The ladder
    # measures the COLD case: a fresh state holding only the creation accesses.
    saved = corpus.activation
    corpus.activation = Activation(tmp_path / "ladder.db")
    corpus._record_creation_accesses()
    try:
        t1 = _score(corpus, 1)
        t2 = _score(corpus, 2)
        t3 = _score(corpus, 3)
    finally:
        corpus.activation = saved
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({"embedder": corpus.embedder.id, "tier1": t1, "tier2": t2, "tier3": t3}, ensure_ascii=False, indent=1))
    hash_embedder = corpus.embedder.id.startswith("hash")
    assert t1["found"] >= t1["n"] - (4 if hash_embedder else 2), [r for r in t1["rows"] if r["rank"] is None]
    if hash_embedder:
        return          # the stand-in cannot cross languages; the ladder is measured with the real embedder
    # The milestone-2 gate: tier 2 beats tier 1 at equal budget, on the chat corpus at least.
    assert t2["mrr_chat"] >= t1["mrr_chat"], (t1["mrr_chat"], t2["mrr_chat"])
    assert t2["mrr"] >= t1["mrr"], (t1["mrr"], t2["mrr"])
    assert t2["found"] == t2["n"], [r for r in t2["rows"] if r["rank"] is None]
    # Milestone 3's gate (§12): tier 3 is kept as the default only if it beats tier 2 at equal
    # budget; otherwise L4 ships as a research knob. Measured 2026-09-09 on these fixtures:
    # it does NOT beat tier 2 cold (0.88 vs 0.944; equal on chat), so the default is tier 2.
    # Recorded every run; guarded only against a regression of the tier-3 knob itself.
    assert t3["found"] == t3["n"], [r for r in t3["rows"] if r["rank"] is None]
    assert t3["mrr_chat"] >= t2["mrr_chat"] - 0.1, (t2["mrr_chat"], t3["mrr_chat"])
    assert t3["mrr"] >= t2["mrr"] - 0.1, (t2["mrr"], t3["mrr"])
