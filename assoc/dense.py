"""The dense index (ASSOCIATIVE_MEMORY.md §2.2 / §3 step 1): an embedder + FAISS over chunk
windows and claim texts. The embedder is injected; `BgeM3Embedder` is the default on the
box and `HashEmbedder` the deterministic offline stand-in for the benches (bag of lemmas,
hashed — lexical similarity, no model, no network). Embeddings are cached by text hash so a
rebuild re-embeds only what is new. The frozen component is the embedder: its id is in the
manifest and changing it invalidates this layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

WINDOW_CHARS, WINDOW_OVERLAP = 6000, 600     # BGE-M3 takes 8k tokens; windows are a safety net


class Embedder(Protocol):
    id: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, model-free embedder: hashed bag of lemmas, L2-normalized."""
    id = "hash-lemma-256"
    dim = 256
    min_score = 0.3        # absolute cosine floor before normalization (§3 step 1)

    def encode(self, texts: list[str]) -> np.ndarray:
        from .lex import terms_of
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for term in terms_of(t):
                h = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
                out[i, (h >> 8) % self.dim] += 0.5
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class BgeM3Embedder:
    id = "BAAI/bge-m3"
    dim = 1024
    min_score = 0.5        # BGE-M3's baseline cosine for unrelated text sits at 0.5–0.6

    def __init__(self, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer("BAAI/bge-m3", device=device or "cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, batch_size=16), dtype="float32")


def windows_of(text: str) -> list[str]:
    if len(text) <= WINDOW_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + WINDOW_CHARS])
        if i + WINDOW_CHARS >= len(text):
            break
        i += WINDOW_CHARS - WINDOW_OVERLAP
    return out


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path, embedder_id: str):
        self.path = path
        self.embedder_id = embedder_id
        self._vecs: dict[str, np.ndarray] = {}
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as z:
                    # NpzFile re-reads a member from the zip on EVERY subscript: bind each
                    # array once (z["vecs"][i] in a loop allocated the whole matrix per key).
                    ok = str(z["embedder_id"]) == embedder_id
                    keys = [str(k) for k in z["keys"]] if ok else []
                    vecs = z["vecs"] if ok else None
                self._vecs = {k: vecs[i] for i, k in enumerate(keys)} if ok else {}
            except Exception as e:
                print(f"[dense] embedding cache {path} not loaded: {e!r}")
                self._vecs = {}

    def get_many(self, texts: list[str], embedder: Embedder) -> np.ndarray:
        missing = [t for t in texts if _h(t) not in self._vecs]
        if missing:
            uniq = list(dict.fromkeys(missing))
            vecs = embedder.encode(uniq)
            for t, v in zip(uniq, vecs):
                self._vecs[_h(t)] = np.asarray(v, dtype="float32")
        return np.stack([self._vecs[_h(t)] for t in texts]) if texts else np.zeros((0, embedder.dim), dtype="float32")

    def save(self) -> None:
        keys = list(self._vecs.keys())
        vecs = np.stack([self._vecs[k] for k in keys]) if keys else np.zeros((0, 1), dtype="float32")
        np.savez(self.path, embedder_id=np.array(self.embedder_id), keys=np.array(keys), vecs=vecs)


class DenseIndex:
    """FAISS inner-product index over (id, window) rows; a hit collapses to its id (max)."""

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.rows: list[dict] = []     # row i -> {id, kind, window}

    def add(self, ids_kinds: list[tuple[str, str]], texts: list[str], vecs: np.ndarray) -> None:
        if len(texts) == 0:
            return
        self.index.add(np.ascontiguousarray(vecs, dtype="float32"))
        for (ident, kind), t in zip(ids_kinds, texts):
            self.rows.append({"id": ident, "kind": kind, "window": t[:80]})

    def search(self, qvec: np.ndarray, *, k: int = 50, kind: Optional[str] = None,
               allowed: Optional[set[str]] = None, floor: float = 0.0) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        kk = min(max(k * 4, 32), self.index.ntotal)
        scores, idx = self.index.search(q, kk)
        best: dict[str, float] = {}
        for sc, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            row = self.rows[int(i)]
            if kind and row["kind"] != kind:
                continue
            if allowed is not None and row["id"] not in allowed:
                continue
            if sc < floor:
                continue
            if row["id"] not in best or sc > best[row["id"]]:
                best[row["id"]] = float(sc)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
        return [{"id": ident, "score": sc} for ident, sc in ranked]

    def save(self, dirpath: Path) -> None:
        import faiss
        dirpath.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dirpath / "dense.faiss"))
        (dirpath / "dense_rows.json").write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(dirpath: Path) -> "DenseIndex":
        import faiss
        index = faiss.read_index(str(dirpath / "dense.faiss"))
        d = DenseIndex(index.d)
        d.index = index
        d.rows = json.loads((dirpath / "dense_rows.json").read_text(encoding="utf-8"))
        return d
