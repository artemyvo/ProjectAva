"""Shared bench corpus (ASSOCIATIVE_MEMORY.md §9): built once per session in a temp dir with
the model-free embedder; the LLM witness is replaced by scripted protocols. Set
``ASSOC_MODEL=1`` to also run the model-dependent benches (gemma-4-31B in-process)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assoc.budget import Budget                       # noqa: E402
from assoc.library import Library                      # noqa: E402
from assoc.witness import run_witnesses                # noqa: E402
from assoc.bench import fixtures as fx                 # noqa: E402


def scripted_witness(lines: str):
    def gen(system, user, *, thinking, max_new_tokens, temperature):
        return lines
    return gen


CACHE_DIR = Path(__file__).parent / ".cache"


def bench_embedder():
    """BGE-M3 when it is on disk (GPU if free, else CPU), the hash stand-in otherwise or
    under ASSOC_EMBEDDER=hash. Embeddings are cached under bench/.cache across runs."""
    if os.environ.get("ASSOC_EMBEDDER", "bge") == "hash":
        from assoc.dense import HashEmbedder
        return HashEmbedder()
    try:
        from huggingface_hub import try_to_load_from_cache
        if not try_to_load_from_cache("BAAI/bge-m3", "pytorch_model.bin"):
            raise RuntimeError("bge-m3 not cached")
        import torch
        from assoc.dense import BgeM3Embedder
        return BgeM3Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:  # noqa: BLE001
        print(f"[bench] falling back to the hash embedder: {e}")
        from assoc.dense import HashEmbedder
        return HashEmbedder()


_EMBEDDER = None


def shared_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = bench_embedder()
    return _EMBEDDER


def build_corpus(root: Path, *, budget_total: int = 4000) -> Library:
    lib = Library(root, budget=Budget(total=budget_total), embedder=shared_embedder(), cache_dir=CACHE_DIR)
    for p in fx.all_docs():
        lib.ingest(p["text"], "tech_doc", p["meta"])
    lib.ingest(fx.chat_kestrel(), "chat", {"key": "chat-kestrel", "date": "2026-08-14"})
    lib.ingest(fx.chat_noam(), "chat", {"key": "chat-noam", "date": "2026-07-02"})
    lib.ingest(fx.chat_ru(), "chat", {"key": "chat-ru", "date": "2026-06-20"})
    lib.ingest(fx.PY_MODULE, "code", {"key": "demo/client.py", "project": "demo", "version": "abc123"})
    lib.ingest(fx.C_MODULE, "code", {"key": "demo/conn.c", "project": "demo", "version": "abc123"})
    lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/starling", "date": "2026-08-20"})
    run_witnesses(lib.store, lib.store.latest_id("news/starling"), generate_fn=scripted_witness(fx.NEWS_PROTOCOL_LINES_REL))
    run_witnesses(lib.store, lib.store.latest_id("chat-ru"), generate_fn=scripted_witness(fx.chat_ru_protocol_lines()))
    run_witnesses(lib.store, lib.store.latest_id("chat-kestrel"), generate_fn=scripted_witness(fx.CHAT_KESTREL_LINES))
    run_witnesses(lib.store, lib.store.latest_id("chat-noam"), generate_fn=scripted_witness(fx.CHAT_NOAM_LINES))
    # Milestone-2 material: cross-lingual pairs, a syndicated article + mirror + a contradiction,
    # two libraries defining `connect`, an index page, restatement / order / negation pairs.
    import json as _json
    (lib.store.root / "state" / "aliases.json").write_text(_json.dumps(fx.ALIASES, ensure_ascii=False), encoding="utf-8")
    for key, text, kind, meta, lines in (
        ("chat-haifa-en", fx.chat_haifa_en(), "chat", {"date": "2026-05-10"}, fx.CHAT_HAIFA_EN_LINES),
        ("chat-haifa-ru", fx.chat_haifa_ru(), "chat", {"date": "2026-06-01"}, fx.CHAT_HAIFA_RU_LINES),
        ("news/syndicated", fx.ARTICLE_SYNDICATED, "news", {"date": "2026-08-21"}, fx.SYNDICATED_LINES),
        ("news/mirror", fx.ARTICLE_MIRROR, "news", {"date": "2026-08-21"}, fx.SYNDICATED_LINES),
        ("news/contra", fx.ARTICLE_CONTRA, "news", {"date": "2026-08-22"}, fx.CONTRA_LINES),
        ("alpha/client.md", fx.LIB_A_DOC, "tech_doc", {"product": "alpha", "version": "1.0"}, fx.LIB_A_LINES),
        ("beta/client.md", fx.LIB_B_DOC, "tech_doc", {"product": "beta", "version": "1.0"}, fx.LIB_B_LINES),
        ("alpha/index.md", fx.INDEX_PAGE, "tech_doc", {"product": "alpha", "version": "1.0"}, fx.INDEX_LINES),
        ("notes/a.md", fx.RESTATE_DOC_A, "tech_doc", {"product": "nimbus", "version": "2.0"}, fx.RESTATE_A_LINES),
        ("notes/b.md", fx.RESTATE_DOC_B, "tech_doc", {"product": "nimbus", "version": "2.0"}, fx.RESTATE_B_LINES),
        ("notes/c.md", fx.RESTATE_DOC_C, "tech_doc", {"product": "nimbus", "version": "2.0"}, fx.RESTATE_C_LINES),
    ):
        lib.ingest(text, kind, {"key": key, **meta})
        run_witnesses(lib.store, lib.store.latest_id(key), generate_fn=scripted_witness(lines))
    for d in fx.pivot_corpus():       # milestone 4: the pivot corpus (no witness; chunks only)
        lib.ingest(d["text"], d["kind"], {"key": d["key"], "title": d["title"], "date": d["date"]})
    lib.rebuild()
    return lib


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Library:
    return build_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture
def fresh_root(tmp_path) -> Path:
    return tmp_path / "lib"


needs_model = pytest.mark.skipif(not os.environ.get("ASSOC_MODEL"), reason="set ASSOC_MODEL=1 to run model-dependent benches")
