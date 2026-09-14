"""RAG engine — retrieves relevant past exchanges as soft context for inference."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Strip a leading ``<think>…</think>`` off a captured reaction so only her spoken reply
# (the language/style payload) is injected into a chat turn, not the reasoning channel.
_WANDER_THINK_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*", re.IGNORECASE)

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as _exc:
    raise ImportError(
        f"RAG dependency missing: {_exc}\n"
        "Install with: pip install sentence-transformers faiss-cpu numpy"
    ) from _exc

from core.chat_sidecar import ChatSidecar, gist_excerpt, iter_chat_json_files
from core.exchange_anchor import (build_tag_stats, query_tokens, tag_matches,
                                  tag_weight, words_match)
from core.rag_policy import (chunk_text, clipped, memory_available_before, rank_score,
                             recorded_at)
from core.reflection_memory import ReflectionMemory

# Consolidation decay lets a bundle's RAG priority fade as it moves into the weights.
# REBUILD retrofit crossfade (§6): the fade is keyed on the bundle's WALL-CLOCK age (hours
# since the CHAT — the session-file stem), the same clock that ramps its training LR — but on
# a DECOUPLED pace. Verbatim chat reaches hard 0 at `rag_cap_age_h` (~4d), when its gist is
# at peak; persona recall keeps its separate `rag_floor_weight`, facts stay at 1.0, and gist
# later decays to `gist_floor_weight`. Relevance gates on raw cosine (rag_policy.rank_score);
# the nonzero modifiers only order relevant results. The old build-count age + sidecar
# *stage* keying are retired. These helpers live in the sibling ``training`` package; the
# import is guarded so inference still runs if that package is absent.
try:
    import sys as _sys
    _SERVER_DIR = Path(__file__).resolve().parent.parent.parent  # server/
    if str(_SERVER_DIR) not in _sys.path:
        _sys.path.insert(0, str(_SERVER_DIR))
    from training.decay import ConsolidationConfig as _ConsolidationConfig
    from training.decay import fresh_time_weight as _fresh_time_weight
    from training.decay import gist_rag_weight_hours as _gist_rag_weight_hours
    from training.decay import rag_weight_hours as _rag_weight_hours
    from training.decay import (
        recollection_rag_weight_hours as _recollection_rag_weight_hours)
    from training.decay import verbatim_rag_weight_hours as _verbatim_rag_weight_hours
    from training.decay import wall_clock_age_hours as _wall_clock_age_hours
    from training.decay import wander_rag_weight_hours as _wander_rag_weight_hours
    from training.ledger import ConsolidationLedger as _ConsolidationLedger
    from training.reflections_path import load_server_config as _load_server_config
except Exception:  # pragma: no cover - training package optional
    _ConsolidationConfig = _ConsolidationLedger = _load_server_config = None
    _rag_weight_hours = _verbatim_rag_weight_hours = None
    _wall_clock_age_hours = _wander_rag_weight_hours = None
    _gist_rag_weight_hours = None
    _fresh_time_weight = None
    _recollection_rag_weight_hours = None

# The project's existing "these two say the same thing" threshold, borrowed rather than
# re-picked: ``persona_render`` uses it as the injector's own already-expressed test, and
# the training-review persona detector reuses it to recognise what that injector wrote.
# Only the constant is shared — the comparison there is statement-vs-thought, here it is
# line-vs-line — so a change to the number moves both together while each keeps its own
# shape.
try:
    from training.persona_render import _DEDUP_JACCARD as _PERSONA_DEDUP_JACCARD
except Exception:  # pragma: no cover - training package optional
    _PERSONA_DEDUP_JACCARD = 0.6


_EMBED_CHUNK_CHARS = 280
_EMBED_OVERLAP_CHARS = 40
# Injection cap per rendered side of a recalled exchange (retrieval is unaffected — the
# whole turn is embedded as _EMBED_CHUNK_CHARS passages, so nothing becomes unfindable;
# this bounds only how much of a winning exchange reaches the prompt). Raised 1800 → 4000
# on 2026-07-25: 37% of this corpus's turns exceed 1800 chars (median 707, p75 3005, p90
# 3900), so the old cap head-clipped a long recalled turn past its midpoint — cutting off
# exactly the end of the thought, where the conclusion tends to be. 4000 clears p90, so
# the typical long turn now arrives whole. Ceiling on the chat block is _DEFAULT_TOP_K
# exchanges × 2 sides × this.
_CHAT_TURN_DISPLAY_CHARS = 4000
_WANDER_REACTION_DISPLAY_CHARS = 1800

# ── Anchor channel ──────────────────────────────────────────────────────────────
# A per-exchange retrieval ANCHOR (core/exchange_anchor.py) is a generated one-line
# descriptor + normalized tags, written to the chat sidecar during reflection. It is
# not a fourth kind of memory: it is the EXCHANGE-granular counterpart of the gist,
# which is session-granular. Verbatim dies hard at `rag_cap_age_h`, after which a
# whole conversation is represented by one session recap forever; the anchor is what
# keeps a *specific* exchange recallable past that cap.
#
# It therefore does not carry its own payload. It CLAIMS an exchange and injects the
# best representation of it still permitted: the real turns while verbatim is alive,
# the descriptor once it is not — the same verbatim→distilled crossfade the gist
# makes, one level down. Because the claim is what it contributes, the traditional
# chat ranking skips a claimed exchange and advances to its next candidate, so an
# anchor slot always adds a distinct exchange to the block instead of duplicating one.
_ANCHOR_DISPLAY_CHARS = 300

# Anchors are embedded one vector per exchange while verbatim contributes several
# passages per exchange, so in a shared index they are a ~1:5 minority that a fixed
# oversampling window crowds out as the corpus grows. They get their own index so the
# reserved slot is actually fillable at any corpus size.
_ANCHOR_TOP_K = 1
# Dense floor. Below the chat channel's 0.45 because an ABOUT line is a short
# paraphrase matched against a full user turn, which scores lower than the
# passage-vs-passage comparison the chat floor was tuned on.
_ANCHOR_MIN_SCORE = 0.35
# Sparse (tag) admission. Tags exist to catch the rare coined tokens mean-pooled
# embeddings dilute, so they must be able to admit an entry the dense side missed —
# but `tag_weight` measures rarity within the TAG vocabulary, not the language, so a
# common word that is merely rare as a tag (the live corpus's `слушай`) scores as
# distinctive. Requiring either corroboration (two tags) or the module's own
# distinctiveness proxy (a long tag) keeps a single filler tag from admitting alone.
_ANCHOR_TAG_MIN_HITS = 2
_ANCHOR_TAG_SOLO_CHARS = 12
# Total matched-tag weight treated as full sparse confidence. Only a scale for
# comparing against a cosine — the weight itself is unbounded and corpus-relative.
_ANCHOR_TAG_FULL_WEIGHT = 6.0

# ── Fact-nominated recall ───────────────────────────────────────────────────────
# The fourth way into the past-chat block, and the only one whose relevance is not
# decided by a cosine. Stage 1 of a live turn (`generation._fetch_facts_block_sync`)
# already runs a pass that READS the arriving message and picks which recorded facts it
# turns on; every one of those facts came out of a conversation, and the facts tree
# knows which (`graph.blob.chat_sources`). So the pick nominates that conversation for
# recall — retrieval keyed on a fact rather than on embedding distance.
#
# This is not a fifth index and adds no vectors. It is a reserved slot that injects the
# conversation's GIST, which is the box's only session-grained representation of a chat
# and therefore the one that matches the nomination's own grain: `chat_facts` records
# per chat, with no exchange index, so nothing here can honestly point at a turn.
#
# What it buys is the case the cosine channels structurally cannot serve: a conversation
# whose wording has nothing in common with the arriving message — a different language,
# a different subject, the fact being the only thread between them — is unreachable by
# `_query_chat` and by `_query_anchors` alike, however relevant it is. Past
# `rag_cap_age_h` that conversation has no verbatim vectors at all, so a fact recalled
# from it arrived with nothing behind it: one decontextualized sentence.
#
# Tighter than `_CHAT_TURN_DISPLAY_CHARS` (4000) because the slot is ADDITIVE — paid on
# top of `top_k`, like an anchor — and a full gist runs to ~2900 chars on the live
# corpus, which is the persona portrait's whole budget spent on one recollection. The
# excerpt rule is `chat_sidecar.gist_excerpt`, shared with check-in's recap window.
_NOMINATION_DISPLAY_CHARS = 700
# Default cap, overridden by `graph.nominate_max`. One: the fetch is asked for its picks
# most-important first, so the first nomination is the conversation behind the fact that
# mattered most, and a second is already an argument about budget rather than about
# relevance.
_NOMINATION_TOP_K = 1


class _RetrievalEmbedder:
    """Multilingual MiniLM adapter with explicit long-text pooling.

    The multilingual MiniLM encoder has a finite 128-token window. Centralising bounded
    character chunks here keeps every caller multilingual, keeps the model on CPU, and
    prevents sentence-transformers from silently discarding the tail of a long
    English/Russian or mixed-language input.

    The public ``encode`` method is retained for the semantic-comparison callers that
    borrow RagEngine's embedder (branch filtering, persona clustering, fact checks).
    Keeping the replacement in the MiniLM family also preserves the approximate cosine
    range expected by those callers' existing thresholds.
    """

    def __init__(self, model) -> None:
        self._model = model

    @staticmethod
    def _mean_normalized(vectors: "np.ndarray") -> "np.ndarray":
        values = np.asarray(vectors, dtype="float32")
        if values.ndim == 1:
            values = values.reshape(1, -1)
        pooled = values.mean(axis=0, keepdims=True)
        norm = float(np.linalg.norm(pooled))
        if norm > 0.0:
            pooled /= norm
        return pooled.astype("float32")

    def _encode_one(self, text: str) -> "np.ndarray":
        pieces = chunk_text(
            text, max_chars=_EMBED_CHUNK_CHARS,
            overlap_chars=_EMBED_OVERLAP_CHARS,
        ) or [""]
        vectors = self._model.encode(
            pieces, convert_to_numpy=True, normalize_embeddings=True,
        )
        return self._mean_normalized(vectors)

    def encode_query(self, text: str) -> "np.ndarray":
        return self._encode_one(text)

    def encode_passages(self, texts: list[str]) -> "np.ndarray":
        if not texts:
            return np.empty((0, 0), dtype="float32")
        if any(len((text or "").strip()) > _EMBED_CHUNK_CHARS for text in texts):
            return np.vstack([self._encode_one(text or "")[0] for text in texts])
        return np.asarray(self._model.encode(
            [(text or "").strip() for text in texts],
            convert_to_numpy=True, normalize_embeddings=True,
        ), dtype="float32")

    def encode(self, sentences, *, convert_to_numpy: bool = True,
               normalize_embeddings: bool = True, **_kwargs):
        one = isinstance(sentences, str)
        values = [sentences] if one else list(sentences)
        encoded = [self._encode_one(value or "")[0] for value in values]
        matrix = np.vstack(encoded) if encoded else np.empty((0, 0), dtype="float32")
        # Current callers request NumPy, but keep the wrapper unsurprising for a direct
        # single-string call and for callers that omit the sentence-transformers flags.
        if not convert_to_numpy:
            result = matrix.tolist()
            return result[0] if one else result
        return matrix[0] if one else matrix


def _attribution_label(entry: dict) -> str:
    """The ``— about X`` / ``— about X, per Y`` suffix for a recalled fact's label.

    Distilled reflection memory is retrieved by topical similarity with no regard for
    who is currently speaking, so a fact about one person surfaces freely in another
    person's conversation. Unlike the verbatim-chat channel — which renders a
    ``speaker:`` prefix, and which disappears entirely at the 96h cap — reflection
    memory used to arrive as bare prose, leaving the person in front of Ava as the only
    available referent. Past four days that made every recalled fact about a third party
    read as the current speaker's. Naming the subject (and, for hearsay, the person whose
    account it is) is what makes deliberate third-party recall possible instead:
    she can only choose to say "Boris mentioned this" if she knows it was Boris.

    Returns ``""`` for unattributed records — asks, persona, world facts, and everything
    written before attribution existed — so those lines are unchanged.
    """
    about = (entry.get("about") or "").strip()
    if not about:
        return ""
    source = (entry.get("source") or "").strip()
    if entry.get("source_class") == "hearsay" and source:
        # Hearsay is true only as an account, so the label carries its teller. This is
        # the only recall surface that marks the difference, since the hearsay gate
        # keeps such a fact out of the weights and out of host-CoT injection entirely.
        return f" — about {about}, per {source}"
    return f" — about {about}"


def _recollection_label(entry: dict) -> str:
    """Label for a recalled ``[recollection]`` — names the conversation's OWN date.

    A recollection is dated twice over: it was written recently (that is what keeps it at
    full RAG weight) but it is *about* a conversation that may be months old. Rendering
    only the reading would present an old conversation as if it were recent, which is the
    same referent confusion ``_attribution_label`` fixes for third-party facts. The chat's
    date comes from ``origin_ts``; an absent one (a recollection of a self-directed source
    with no parseable chat stem) degrades to the bare label.

    Deliberately an ISO date rather than prose: this corpus is Russian/English/mixed and
    the block is injected verbatim, so a localized month name would be wrong half the time.
    """
    origin = recorded_at(entry.get("origin_ts") or "")
    if origin is None:
        return "looking back"
    return f"looking back on a conversation from {origin.date().isoformat()}"


def _display_words(text: str) -> list:
    return [w for w in re.findall(r"\w+", (text or "").lower(), re.UNICODE)]


def _near_duplicate_display(a: str, b: str,
                            jaccard: float = _PERSONA_DEDUP_JACCARD) -> bool:
    """True when two rendered reflection lines say the same thing.

    Compares the DISPLAY text, not the embedding, and so cannot be done with the index
    vectors: a ``[fact]``/``[recollection]`` is indexed on its *trigger* (what should
    bring it back), so two records with unrelated triggers can still render as the same
    sentence — which is exactly the duplicate that reaches the prompt. Lexical rather
    than a second embedder call, both to keep the chat turn's hot path free of one and
    because near-verbatim restatement in one language is the observed shape (facts are
    distilled in Ava's own idiolect, so her paraphrases of one truth share most of their
    words). Symmetric normalized containment, then word overlap.

    Overlap counts words through ``exchange_anchor.words_match`` rather than raw set
    intersection, for the reason that module already exists: this corpus inflects
    heavily, and two distillations of one truth differ in exactly the endings ("commits"
    / "commit") — a set intersection scores such a pair below the threshold and admits
    both. Matching is greedy and one-to-one, so a repeated word cannot be spent twice.

    Two known misses, both deliberate. A Russian record and its English counterpart
    share no words and both survive. So does a heavier reword ("prefers commits straight
    to main" / "likes committing directly to main rather than using feature branches"),
    which overlaps by about a third — and the threshold that would merge it is low enough
    to start merging genuinely distinct facts, which is the worse failure: a duplicated
    line is noise, a dropped one is a fact silently forgotten. Catching either needs a
    semantic pass. The index's own vectors cannot supply it (facts are indexed by
    trigger, not display text), so it would mean encoding the candidates' display text
    per chat turn — affordable with the multilingual embedder already loaded, and the
    obvious next tier if the lexical rule proves too narrow against the live corpus.
    """
    a_words, b_words = _display_words(a), _display_words(b)
    if not a_words or not b_words:
        return False
    if " ".join(a_words) in " ".join(b_words) or " ".join(b_words) in " ".join(a_words):
        return True
    unmatched = list(b_words)
    matched = 0
    for w in a_words:
        for i, other in enumerate(unmatched):
            if words_match(w, other):
                matched += 1
                unmatched.pop(i)
                break
    union = len(a_words) + len(b_words) - matched
    return bool(union) and matched / union >= jaccard


class RagEngine:
    """
    Builds semantic indexes over (a) saved chat exchanges and (b) the distilled
    reflection memory, and retrieves the most relevant of each as formatted
    system-prompt blocks.
    """

    _EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _DEFAULT_TOP_K = 3
    _MIN_SCORE = 0.45  # cosine-similarity floor (0–1, higher = more similar)
    # Cosine-similarity CEILING for the chat channel only (band-pass, not high-pass).
    # The chat payload injects a past *answer* keyed on a past user prompt OR a past
    # reply of Ava's own (both sides embed — `_exchange_passages`), so a hit that is
    # near-identical to the live query drops the full prior Q+A into context and
    # an induction head copies it verbatim — the "model regenerated an old reply with
    # zero contested tokens" pathology. Such a hit is also the lowest-information one
    # (it restates what the live turn already says). Gate on the RAW cosine, not the
    # age-adjusted score, so an old-but-faded near-duplicate can't launder itself back
    # under the ceiling. Chat only: reflection/wander recall is meant to land verbatim.
    _MAX_SCORE = 0.90

    # Reflection memory is sparse and high-value, so it gets its own (slightly
    # looser) retrieval budget rather than competing with chat excerpts.
    _REFL_TOP_K = 3
    _REFL_MIN_SCORE = 0.25
    # How many candidates to pull per slot before filtering. Every rejection rule below
    # (availability, the near-duplicate ceiling, the subject cap) consumes a candidate
    # without filling a slot, so this width has to cover the worst rejecter or the block
    # silently comes back short — breaking the rule that a rejected candidate frees its slot
    # for the next distinct record rather than shortening the block. The subject cap is that
    # rejecter by a wide margin: on a topic the store knows well, the whole head of the
    # ranking is one subject. Measured over seven probe queries on the live store, the
    # former ×5 left two of them a line short and ×10 filled every slot (×20 and ×40 gained
    # nothing). The search itself is over ~2k vectors, so the width is free.
    _REFL_OVERSAMPLE = 10
    # Cosine-similarity CEILING for the reflection channel — the counterpart of
    # ``_MAX_SCORE``, but it CANNOT apply to every kind here, so it is keyed on what a
    # record is indexed BY (`reflection_memory.embed_text`). A ``[persona]``/``[ask]``/
    # ``[impression]`` embeds on the text it displays, so a hit this close to the live
    # turn is the user restating the line back — the same lowest-information,
    # copy-inviting hit the chat ceiling rejects. A ``[fact]``/``[recollection]`` embeds
    # on its TRIGGER ("what should bring it back"), so a near-exact match is the channel
    # working exactly as designed and a ceiling would reject the best-triggered recall
    # first; those two kinds are exempt (`_REFL_CEILING_KINDS`). Higher than the chat
    # channel's 0.90 because the payload is one distilled sentence rather than a whole
    # prior Q+A, so the copy risk needs a closer match to be real. Gate on the RAW
    # cosine, like the chat ceiling, so a faded record can't launder itself back under it.
    _REFL_MAX_SCORE = 0.95
    _REFL_CEILING_KINDS = ("persona", "ask", "impression", "self_impression")

    # Subject cap: how many of the block's slots may go to records about ONE subject.
    # The near-duplicate rule above rejects a record that RESTATES a chosen one; this
    # rejects a record that merely *shares its subject*, which is a different and — on a
    # mature store — far more common failure. A live corpus held eight distinct facts about
    # one evening ritual (a private paradise / a sensory anchor / a 2 AM peace / …): no
    # merge policy can collapse them, since each states something the others do not, and no
    # wording test can see them, since they say different things. But they rank together on
    # any query that touches the topic, so all three slots went to one subject and
    # everything else Ava knew was crowded out — while the block read as one fact restated.
    # Cap 1 buys breadth at the cost of depth on a focused query ("what do you remember
    # about our balcony evenings?" recalls one of them, not three) — the past-chat channel
    # still carries the conversation itself, and a block of three unrelated things she knows
    # is worth more than three angles on one. Raise via `reflection_block.subject_cap`.
    _REFL_SUBJECT_CAP = 1
    # Cosine at which two records count as the same subject. LOWER than
    # `fact_contradict.SUBJECT_SIM_THRESHOLD` (0.55), which asks the same question of the
    # same embedder, for two reasons — both about what a mistake costs, not about what a
    # subject is. (1) The errors are not symmetric here: a false link only DEFERS a record
    # to a later turn (nothing is written, nothing is lost, it surfaces the next time the
    # topic comes up), while over there a false link merges two live records and evicts one
    # for good — so dedup must be conservative and this can afford not to be. (2) That
    # threshold is applied by average-link over a whole cluster; this is single-link against
    # each already-chosen record, which bites sooner at equal value.
    # Measured on the live store over the balcony set (8 records) against 320 sampled
    # cross-topic pairs: 0.45 catches 68% of same-topic pairs while falsely linking 0.9% of
    # unrelated ones; 0.55 catches only 36%, and 0.40 buys 7 more points of recall for 3x
    # the false links. Comparison is restricted to records sharing an indexing BASIS — a
    # fact/recollection embeds on its trigger, a persona/ask/impression on its display text
    # (`_REFL_CEILING_KINDS`, the same split) — since a cue-against-a-sentence cosine across
    # the two would not be measuring subject agreement at all.
    _REFL_SUBJECT_SIM = 0.45

    # Wander (external articles Ava read on her own) gets a single, deliberately small
    # slot. Semantic relevance is gated on raw cosine; age then changes ordering only.
    # This matters because the decay modifier reaches 0.1, where the old
    # ``cosine * modifier >= threshold`` rule was mathematically impossible to pass.
    _WANDER_TOP_K = 1
    _WANDER_MIN_SCORE = 0.15

    def __init__(self, chats_dir: Path, prompts_dir: Path,
                 memory_dir: Optional[Path] = None,
                 consolidation_dir: Optional[Path] = None,
                 fallback_chats_dir: Optional[Path] = None,
                 fallback_memory_dir: Optional[Path] = None,
                 wander_corpus: Optional[Path] = None) -> None:
        self.chats_dir = chats_dir   # configured live chat corpus
        self.prompts_dir = prompts_dir
        self.fallback_chats_dir = Path(fallback_chats_dir) if fallback_chats_dir is not None else None

        self._lock = threading.Lock()
        self._ready = False
        self._index = None          # faiss.IndexFlatIP, built lazily
        self._entries: list[dict] = []
        self._embedder = None
        self._current_session_file: Optional[Path] = None
        self._sidecar = ChatSidecar(chats_dir, fallback_chats_dir=fallback_chats_dir)

        # Reflection-memory index — folded from data/hot/memory/rag_memory.jsonl.
        self._memory = ReflectionMemory(memory_dir, fallback_memory_dir=fallback_memory_dir) if memory_dir is not None else None
        self._refl_index = None
        self._refl_entries: list[dict] = []
        # embed_text -> normalized (1, dim) float32 vector. The reflection index is
        # rebuilt once per consolidated session, but a memory item's embedding is a
        # pure function of its embed_text and only the handful distilled this session
        # change — so cache vectors and re-encode only the misses (O(new) not O(M)).
        self._refl_embed_cache: dict[str, "np.ndarray"] = {}

        # Consolidation decay: verbatim chat drops to zero at the age cap while persona and
        # gist retain independent floors. The ledger lives in data/hot/consolidation/; the
        # wall-clock knobs (`rag_cap_age_h` etc.) ride on `self._consolidation.wall`.
        self._ledger = None
        self._consolidation = None
        if consolidation_dir is not None and _ConsolidationLedger is not None:
            try:
                self._ledger = _ConsolidationLedger(consolidation_dir)
                cfg = _load_server_config().get("consolidation") if _load_server_config else None
                self._consolidation = _ConsolidationConfig.from_dict(cfg)
            except Exception:
                self._ledger = self._consolidation = None
        # REBUILD retrofit (§6): the age clock is WALL-CLOCK hours since the chat (session
        # stem), evaluated at index build and again at retrieval. A FROZEN bundle follows the
        # linear verbatim fade; an unfrozen chat keeps the gentler fresh-window policy until
        # the same raw-age hard cap removes it at 96h. The freeze timestamp is used only as
        # the slope gate; `_reflected_at_cache` memoizes it per session and is cleared when
        # the chat index is rebuilt.
        self._reflected_at_cache: dict[str, Optional[str]] = {}

        # Wander channel — the durable wander corpus (server/data/til/wander.jsonl); each
        # record is an article Ava read on her own + her reaction. Retrieval is embedded on
        # the source article and faded by wall-clock age since capture. None ⇒ no channel.
        self._wander_corpus = Path(wander_corpus) if wander_corpus is not None else None
        self._wander_index = None
        self._wander_entries: list[dict] = []

        # Anchor channel — one entry per anchored exchange, read from the sidecars in
        # the same pass that collects chat entries (the sidecar is already open there
        # for the gist). Its own index, because anchors are a vector-count minority
        # that a shared oversampling window would crowd out. `_anchor_tag_stats` is the
        # corpus side of the tag IDF weight, computed once per build.
        self._anchor_index = None
        self._anchor_entries: list[dict] = []
        self._anchor_tag_stats: dict = {}
        # Config kill-switch (`anchors.enabled`, default on) so the channel can be cut
        # without a code change while it is still being evaluated on a live corpus.
        self._anchors_enabled = True
        try:
            if _load_server_config is not None:
                anchors_cfg = (_load_server_config() or {}).get("anchors") or {}
                self._anchors_enabled = bool(anchors_cfg.get("enabled", True))
        except Exception:
            self._anchors_enabled = True
        # Same kill-switch shape for the recollection kind (`recollections.enabled`,
        # default on): cuts the channel out of retrieval without touching the revisit
        # pass that PRODUCES it, so a corpus can accumulate readings while they are
        # still being judged. Production is gated separately (revisit overrides).
        self._recollections_enabled = True
        try:
            if _load_server_config is not None:
                recol_cfg = (_load_server_config() or {}).get("recollections") or {}
                self._recollections_enabled = bool(recol_cfg.get("enabled", True))
        except Exception:
            self._recollections_enabled = True
        # And again for the [impression] kind (`impressions.enabled`, default on) — Ava's
        # readings of the people she talks to. Retrieval only: the per-session user-notes
        # pass that PRODUCES them is gated by `overrides.user_notes`, and the standing user
        # portrait folded from them is gated by `user_portrait.enabled`, so a corpus can go
        # on accumulating readings (and portraits) while this channel is cut.
        self._impressions_enabled = True
        try:
            if _load_server_config is not None:
                impr_cfg = (_load_server_config() or {}).get("impressions") or {}
                self._impressions_enabled = bool(impr_cfg.get("enabled", True))
        except Exception:
            self._impressions_enabled = True
        # And once more for `[self_impression]` (`self_impressions.enabled`) — the outside
        # view of herself. This one defaults **off**, unlike every switch above it: nothing
        # in the box asks for the channel (`include_self_impressions` defaults False), the
        # records exist to be folded into `users/_self.json` rather than recalled, and a
        # reading of her own transcripts surfacing inside a live turn is the exact
        # circularity `core.self_portrait` is built to stay out of. Present so the channel
        # can be switched on for evaluation without a code change.
        self._self_impressions_enabled = False
        try:
            if _load_server_config is not None:
                self_cfg = (_load_server_config() or {}).get("self_impressions") or {}
                self._self_impressions_enabled = bool(self_cfg.get("enabled", False))
        except Exception:
            self._self_impressions_enabled = False
        # Subject cap on the reflection block (`reflection_block.subject_cap` /
        # `.subject_sim`, defaults `_REFL_SUBJECT_CAP` / `_REFL_SUBJECT_SIM`). A knob rather
        # than a constant because it trades breadth against depth and the right point
        # depends on how much one topic dominates a given corpus: 0 disables the cap, 2
        # leaves one slot guaranteed to a different subject, 1 makes every slot a distinct
        # subject. Read once at construction like the kill-switches above.
        self._subject_cap = self._REFL_SUBJECT_CAP
        self._subject_sim = self._REFL_SUBJECT_SIM
        try:
            if _load_server_config is not None:
                block_cfg = (_load_server_config() or {}).get("reflection_block") or {}
                self._subject_cap = int(block_cfg.get("subject_cap",
                                                      self._REFL_SUBJECT_CAP))
                self._subject_sim = float(block_cfg.get("subject_sim",
                                                        self._REFL_SUBJECT_SIM))
        except Exception:
            self._subject_cap = self._REFL_SUBJECT_CAP
            self._subject_sim = self._REFL_SUBJECT_SIM

        # How many conversations a turn's fetched facts may recall (`graph.nominate_max`,
        # default `_NOMINATION_TOP_K`; 0 ⇒ the channel is off). Lives under `graph` because
        # the nomination cannot happen without the facts-tree fetch that produces it — a
        # box with `graph.enabled` false never calls this path at all — and a knob rather
        # than a constant because it is the one part of the channel that spends prompt
        # budget: each nomination is an additive passage on top of `top_k`.
        self._nominate_max = _NOMINATION_TOP_K
        try:
            if _load_server_config is not None:
                graph_cfg = (_load_server_config() or {}).get("graph") or {}
                self._nominate_max = max(0, int(graph_cfg.get("nominate_max",
                                                              _NOMINATION_TOP_K)))
        except Exception:
            self._nominate_max = _NOMINATION_TOP_K

        self._prompt_template: str = self._load_prompt_template()
        self._chat_reflect_prompt_template: str = self._load_chat_reflect_prompt_template()
        self._refl_prompt_template: str = self._load_refl_prompt_template()
        self._refl_reflect_prompt_template: str = self._load_refl_reflect_prompt_template()
        self._wander_prompt_template: str = self._load_wander_prompt_template()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_current_session_file(self, path: Optional[Path]) -> None:
        """Exclude the active session file from retrieval to avoid self-reference."""
        with self._lock:
            self._current_session_file = path

    def build_index_async(self) -> None:
        """Kick off index construction in a daemon background thread."""
        t = threading.Thread(target=self._build_index, daemon=True)
        t.start()

    def refresh_chat_index(self) -> None:
        """Rebuild the chat-RAG index (e.g. after sidecar stages advance).

        Reflection-memory index is untouched. Safe to call from a background thread.
        """
        self._build_chat_index()

    def index_new_chats(self) -> bool:
        """Rebuild the chat index iff hot/chats holds transcripts not yet indexed.

        Catches *foreign* chats — transcripts written straight to disk (e.g. merged
        in from another Ava via the Migrate tab's chat sync) that bypassed the
        incremental :meth:`add_exchange` path and so are absent from the live index.
        Cheap in the common case: it only lists filenames and compares them against
        the set of already-indexed sessions; the costly re-embed runs solely when a
        genuinely new transcript is present. Returns True if it rebuilt. Safe to call
        from a background thread.
        """
        try:
            with self._lock:
                indexed = {e.get("source_session", "") for e in self._entries}
                current = (self._current_session_file.name
                           if self._current_session_file else None)

            foreign = False
            for d in (self.fallback_chats_dir, self.chats_dir):
                if d is None or not d.exists():
                    continue
                for p in iter_chat_json_files(d):
                    # (ShareML docs and the other per-stem sidecars are already excluded
                    # by `iter_chat_json_files` — see `chat_sidecar.SIDECAR_SUFFIXES`.
                    # They carry no "exchanges" and are never a source_session, so
                    # letting one through here reads as perpetually-foreign and triggers
                    # an index rebuild on every call.)
                    if p.name == current or p.name in indexed:
                        continue
                    foreign = True
                    break
                if foreign:
                    break

            if not foreign:
                return False
            self._build_chat_index()
            return True
        except Exception:
            return False

    def query(self, prompt: str, top_k: int = _DEFAULT_TOP_K, *,
              before_session: str = "", include_chat: bool = True,
              include_wander: bool = False, include_facts: bool = True,
              include_persona: bool = True, include_asks: bool = True,
              include_anchors: bool = True, include_recollections: bool = True,
              include_impressions: bool = True,
              include_self_impressions: bool = False,
              impressions_exclude_about: str = "",
              nominate_sessions: Optional[list[str]] = None,
              reflect_framing: bool = False) -> str:
        """
        Return a formatted context block for *prompt*, combining past-chat
        excerpts and distilled reflection notes (each in its own labeled
        section). Returns an empty string when nothing relevant is found.

        *before_session* applies a temporal cutoff to the past-chat block: only
        excerpts from sessions *older* than this session filename are eligible — the
        session under review and every session started after it are dropped.
        Reflection passes pass the session being reflected on, so Ava reasons only
        from what she knew up to that moment: never the conversation's own later
        turns, nor any session that happened afterwards. (Chat files are named with
        fixed-width timestamps, so a lexicographic filename compare is chronological.)

        *include_chat* gates the past-chat block. The revision pass sets it False:
        it already carries the conversation context in its prompt, so the chat
        excerpts (the bulk of the injected tokens) are the most redundant and
        expensive part — it keeps only the distilled reflection memory.

        *include_wander* gates the wander channel (an external article Ava read on her own
        + her reaction). Off by default and enabled ONLY on the live-chat / encounter paths:
        reflection and revision passes must stay replay-faithful, so wander — which carries
        no ``source_session`` timeline and would leak material newer than the session under
        review — is never injected there.

        *include_facts* / *include_persona* / *include_asks* gate the reflection-memory
        channels independently (the first two are the Chat tab's Facts / Persona toggles,
        letting the operator A/B a channel's effect on chat). ``include_asks`` was folded
        into ``include_facts`` until 2026-07-28 — asks and facts are both "informational"
        — but they turned out to behave nothing alike in competition: an ask embeds on its
        full conversational question text while a fact embeds on a short topic-label
        ``trigger``, so asks systematically out-score facts for the block's three slots.
        Splitting the gate is what lets live chat drop asks while keeping facts. All three
        default True, so every other caller is unaffected; a reflection/revision pass keeps
        the full block.

        *include_anchors* gates the per-exchange anchor slot inside the past-chat block.
        It rides ``include_chat`` (an anchor's payload IS a past-chat exchange, and it is
        fenced by the same active-session and *before_session* rules), so a caller that
        drops the chat block drops anchors with it.

        *include_recollections* gates the ``[recollection]`` channel — what Ava now makes
        of an old conversation, written by a revisit pass. It sits in the reflection block
        beside facts/asks/persona but carries its own clock (see
        ``_recollection_modifier``), and is additionally cut box-wide by the
        ``recollections.enabled`` config switch.

        *include_impressions* gates the ``[impression]`` channel — Ava's readings of the
        people she talks to, the user-side counterpart of ``[persona]``. It is the
        situational half of a two-part story: the *standing* half is the per-person
        portrait folded from the same records and injected as a prompt block by
        ``generation._current_user_portrait``. Also cut box-wide by the
        ``impressions.enabled`` config switch.

        *impressions_exclude_about* is how a caller that injects a portrait avoids paying
        twice for it: pass the portrait's subject, and impressions about **that person**
        are dropped while impressions about everyone else still compete normally. This is
        deliberately narrower than the ``include_persona=False`` a self-portrait triggers,
        and the asymmetry is real — persona is *only ever* about Ava, so a self-portrait
        supersedes the whole channel, whereas a user portrait covers exactly one of the
        people this channel carries readings of. Suppressing all of them would silence
        what she knows of a third party at the moment the conversation turns to them.

        *nominate_sessions* are chat filenames a caller has independent reason to recall,
        best-first — the fact-nomination channel (see ``_NOMINATION_DISPLAY_CHARS``). Live
        chat passes the conversations behind the facts stage 1 picked for this message, so
        a conversation reachable by no cosine still reaches the block. It rides
        ``include_chat``: the payload IS past-chat recall, so a caller that drops the chat
        block drops nominations with it. Every other caller passes nothing and is
        unaffected — in particular no reflection pass nominates, since replay fidelity is
        decided by *before_session* and a nomination would be a second thing to keep honest.

        *reflect_framing* swaps BOTH block wrappers for their reflection-facing variants
        (``rag_memory_reflect_prompt.txt``, ``rag_reflect_prompt.txt``). The defaults are
        written for live chat and say so: the memory one instructs the reader on how to use
        a note *in a reply* — which is an instruction to write one — and the past-chat one
        closes on "unless clearly useful to the user". Both land in the system message of a
        pass whose contract is a structured judgement or decision, not a reply. Set by the
        reflect-generate factory
        (``generation._make_sync_reflect_generate``), so every reflection pass gets it and
        no chat-shaped path does. The retrieved records are identical either way; only the
        framing prose differs.
        """
        try:
            blocks = []
            if include_chat:
                blocks.append(self._query_chat(prompt, top_k, before_session=before_session,
                                               include_anchors=include_anchors,
                                               nominate_sessions=nominate_sessions,
                                               reflect_framing=reflect_framing))
            blocks.append(self._query_reflection(
                prompt, before_session=before_session,
                include_facts=include_facts, include_persona=include_persona,
                include_asks=include_asks,
                include_recollections=include_recollections,
                include_impressions=include_impressions,
                include_self_impressions=include_self_impressions,
                impressions_exclude_about=impressions_exclude_about,
                reflect_framing=reflect_framing,
            ))
            if include_wander:
                blocks.append(self._query_wander(prompt))
            return "\n\n".join(b for b in blocks if b)
        except Exception:
            return ""

    def _query_chat(self, prompt: str, top_k: int, *, before_session: str = "",
                    include_anchors: bool = True,
                    nominate_sessions: Optional[list[str]] = None,
                    reflect_framing: bool = False) -> str:
        """Retrieve relevant past-chat exchanges as a formatted block.

        Three resolvers, in decreasing precision of what they can point at, each claiming
        what it injects so the next one never repeats it:

        1. **Anchors** — a specific exchange, matched on its descriptor and tags. Resolved
           first and CLAIMS its exchanges, so the traditional ranking skips a claimed one
           and advances to its next candidate. An anchor injects the best representation of
           the exchange it claims (verbatim while that is alive, the descriptor after), so
           the claim can never downgrade what the block would otherwise have carried — it
           only frees a slot.
        2. **Nominations** — a whole conversation, named by the facts stage 1 picked rather
           than matched by any cosine. Resolved second, and skipping a conversation an
           anchor already reached: both would be the same conversation, and the anchor's
           exchange is the finer-grained payload, so a nomination that loses here loses to
           something strictly better. Claims the SESSION, not an exchange.
        3. **The ranking** — whatever embeds nearest, skipping both.
        """
        template = (self._chat_reflect_prompt_template if reflect_framing
                    else self._prompt_template)
        with self._lock:
            if not self._ready:
                return ""
            index = self._index
            entries = list(self._entries) if self._entries else []
            current_session = (
                self._current_session_file.name if self._current_session_file else ""
            )

        anchor_snippets: list[str] = []
        claimed: set = set()
        if include_anchors:
            anchor_snippets, claimed = self._query_anchors(
                prompt, before_session=before_session, current_session=current_session)

        nominated_snippets, nominated = self._query_nominated(
            nominate_sessions, before_session=before_session,
            current_session=current_session,
            anchored={s for s, _ in claimed})
        reserved = anchor_snippets + nominated_snippets

        # Both reserved channels outlive the verbatim one: past `rag_cap_age_h` a chat
        # contributes no verbatim vectors at all, so the chat index can be empty while
        # anchors and nominations still answer — which is exactly the age at which a fact
        # recalled from a conversation would otherwise arrive with nothing behind it.
        if index is None or not entries:
            if not reserved:
                return ""
            return template.replace("{context}", "\n\n".join(reserved))

        embedder = self._get_embedder()
        q_vec = embedder.encode_query(prompt.strip()).astype("float32")

        # Filtering happens after FAISS. Search the full small index when a temporal
        # or active-session fence can remove the nearest neighbours. Otherwise count
        # entries whose live wall-clock modifier is now zero and extend the candidate
        # window by that many vectors: entries can cross the 96h boundary after the
        # index was built, and expired verbatim passages must not crowd an eligible
        # gist out of the oversampled result set.
        current_in_index = bool(current_session) and any(
            e.get("source_session", "") == current_session for e in entries
        )
        expired_vectors = 0
        if not before_session and not current_in_index:
            for entry in entries:
                if entry.get("kind") == "gist":
                    live_modifier = self._gist_modifier(
                        entry.get("source_session", ""))
                else:
                    live_modifier = self._chat_modifier(
                        entry.get("source_session", ""))
                if live_modifier <= 0.0:
                    expired_vectors += 1
        k_search = (
            len(entries) if (before_session or current_in_index)
            else min(len(entries), max(top_k * 16, top_k) + expired_vectors)
        )
        scores, indices = index.search(q_vec, k_search)

        ranked_by_exchange: dict[tuple[str, int], tuple[float, dict]] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            e = entries[int(idx)]
            # The active transcript is already present verbatim in the model's
            # conversation messages. Exclude it at query time as well as build time:
            # an in-place resumed chat can already have vectors in the live index.
            if current_session and e.get("source_session", "") == current_session:
                continue
            # Temporal cutoff: drop the session under reflection and anything newer,
            # so a replay-faithful pass can't retrieve future turns of the same
            # conversation or later sessions as "past" context. Filenames are
            # timestamp-ordered, so >= the cutoff means "this session or after."
            if before_session and e.get("source_session", "") >= before_session:
                continue
            # Already claimed by the anchor slot, which is injecting the best available
            # representation of this exchange: advance to the next candidate rather than
            # spend a second slot on the same one. Gist passages carry negative sentinel
            # indices and so are never claimed — a session recap and a specific exchange
            # of that session are different grains, not the same injection.
            if claimed and (e.get("source_session", ""),
                            int(e.get("exchange_index", -1))) in claimed:
                continue
            # Claimed by a nomination, which injected this conversation's recap: a second
            # slice of the same conversation is what the reserved slot was spent to avoid.
            # Session-wide (not per-exchange) because the nomination's grain is the whole
            # chat — it names a conversation, never a turn within it.
            if nominated and e.get("source_session", "") in nominated:
                continue
            # Near-duplicate ceiling on the RAW cosine (see _MAX_SCORE): a hit too
            # similar to the live query is a verbatim-copy trap, so drop it before the
            # age modifier is even applied.
            if float(score) >= self._MAX_SCORE:
                continue
            # Recompute the time component at retrieval, not only at index build, so the
            # 96h verbatim cutoff and 192h gist floor take effect on a long-running server.
            # Wall-clock age is live.
            if e.get("kind") == "gist":
                modifier = self._gist_modifier(e.get("source_session", ""))
            else:
                modifier = self._chat_modifier(e.get("source_session", ""))
            if modifier <= 0.0:
                continue
            # Relevance gates on the RAW cosine; the nonzero age modifier only orders what
            # passed (rank_score — the wander rule, generalized).
            adjusted = rank_score(float(score), modifier, self._MIN_SCORE)
            if adjusted is not None:
                key = (e.get("source_session", ""), int(e.get("exchange_index", -1)))
                prior = ranked_by_exchange.get(key)
                if prior is None or adjusted > prior[0]:
                    ranked_by_exchange[key] = (adjusted, e)
        ranked = sorted(ranked_by_exchange.values(), key=lambda pair: pair[0], reverse=True)

        snippets = list(reserved)
        for _, e in ranked[:top_k]:
            # A gist passage is a distilled recap, not a verbatim turn — render it as a
            # remembered conclusion, never as a quoted exchange (it has no user/response).
            if e.get("kind") == "gist":
                recap = clipped(e.get("content", ""), _CHAT_TURN_DISPLAY_CHARS)
                if recap:
                    snippets.append(f"(From an earlier conversation, you recall: {recap})")
                continue
            speaker = (e.get("speaker") or "").strip() or "Interlocutor"
            user = clipped(e["user"], _CHAT_TURN_DISPLAY_CHARS)
            response = clipped(e["response"], _CHAT_TURN_DISPLAY_CHARS)
            snippets.append(f"{speaker}: {user}\nMe: {response}")

        if not snippets:
            return ""
        return template.replace("{context}", "\n\n".join(snippets))

    def _query_anchors(self, prompt: str, *, before_session: str = "",
                       current_session: str = "") -> tuple[list[str], set]:
        """Resolve the reserved anchor slot: ``(snippets, claimed exchange keys)``.

        Two independent admission gates, because the dense and sparse halves of an anchor
        fail differently. **Dense** similarity against the ABOUT line fixes the granularity
        and cross-lingual mismatches of passage retrieval (the matched unit is the payload
        unit, and the descriptor is written in the conversation's own language). **Sparse**
        tag overlap catches the rare coined tokens (`крокодильничество`, `RESET`) that
        mean-pooled embeddings dilute away — the case dense retrieval loses by construction,
        so tags must be able to admit an entry the dense side never surfaced.

        The two are fused with ``max``, not a sum: they are alternative evidence for the
        same claim, and an anchor found by a rare tag is not *more* relevant for also being
        weakly similar. Summing would let two mediocre signals outrank one strong one.

        Unlike the chat channel there is no ``_MAX_SCORE`` ceiling here. An anchor is a
        paraphrase and so is never the near-duplicate of a live query that the ceiling
        exists to reject — that rejection is precisely why probing a remembered phrase is
        today guaranteed *not* to retrieve it. The copy risk is handled where it actually
        lives, in the payload (see :meth:`_render_anchor`), not by dropping the hit.
        """
        if not self._anchors_enabled:
            return [], set()
        with self._lock:
            index = self._anchor_index
            entries = list(self._anchor_entries)
            stats = dict(self._anchor_tag_stats)
        if index is None or not entries:
            return [], set()
        text = (prompt or "").strip()
        if not text:
            return [], set()

        # ── sparse: lexical tag overlap over the whole anchor corpus ──────────────
        sparse: dict[int, tuple[float, list]] = {}
        tokens = query_tokens(text)
        corpus = len(entries)
        if tokens:
            for i, e in enumerate(entries):
                hits: list = []
                weight = 0.0
                for tag in e.get("tags") or []:
                    if not tag_matches(tag, tokens):
                        continue
                    w = tag_weight(tag, stats.get(tag, 1), corpus)
                    if w <= 0.0:
                        continue
                    hits.append(tag)
                    weight += w
                if not hits:
                    continue
                # A single short tag never admits alone: `tag_weight` scores rarity within
                # the TAG vocabulary, so a common word of the language that happens to be a
                # rare tag reads as distinctive. Demand corroboration (two tags) or the
                # module's own distinctiveness proxy (a long one).
                if (len(hits) < _ANCHOR_TAG_MIN_HITS
                        and max(len(h) for h in hits) < _ANCHOR_TAG_SOLO_CHARS):
                    continue
                sparse[i] = (min(1.0, weight / _ANCHOR_TAG_FULL_WEIGHT), hits)

        # ── dense: the ABOUT line, one vector per anchored exchange ───────────────
        embedder = self._get_embedder()
        q_vec = embedder.encode_query(text).astype("float32")
        k_search = min(len(entries), max(_ANCHOR_TOP_K * 16, _ANCHOR_TOP_K))
        scores, indices = index.search(q_vec, k_search)
        dense: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                dense[int(idx)] = float(score)

        ranked: list[tuple[float, float, dict]] = []
        for i in set(dense) | set(sparse):
            e = entries[i]
            session = e.get("source_session", "")
            # Same fences as the chat channel: the active transcript is already in the
            # conversation messages, and a replay-faithful pass must not see its own
            # session or anything after it.
            if current_session and session == current_session:
                continue
            if before_session and session >= before_session:
                continue
            d = dense.get(i, 0.0)
            s, hits = sparse.get(i, (0.0, []))
            if d < _ANCHOR_MIN_SCORE and not hits:
                continue
            modifier = self._anchor_modifier(session)
            if modifier <= 0.0:
                continue
            ranked.append((max(d, s) * modifier, d, e))
        ranked.sort(key=lambda r: r[0], reverse=True)

        snippets: list[str] = []
        claimed: set = set()
        for _, d, e in ranked[:_ANCHOR_TOP_K]:
            snippet = self._render_anchor(e, dense=d)
            if not snippet:
                continue
            snippets.append(snippet)
            claimed.add((e.get("source_session", ""), int(e.get("exchange_index", -1))))
        return snippets, claimed

    def _query_nominated(self, sessions: Optional[list[str]], *, before_session: str = "",
                         current_session: str = "",
                         anchored: Optional[set] = None) -> tuple[list[str], set]:
        """Resolve the fact-nomination slot: ``(snippets, claimed sessions)``.

        No search, no scoring, no vectors. A nomination arrives already justified — a pass
        read the arriving message and picked a fact, and this conversation is where that
        fact was established — so the work here is only deciding whether recalling it is
        *permitted* and what it can honestly show.

        The fences are the chat channel's own, and are not optional for arriving by a
        different route: the active transcript is already in the conversation messages, and
        a replay-faithful pass must not see its own session or anything after it. There is
        deliberately no relevance floor (nothing was matched, so there is nothing to
        threshold), no ``_MAX_SCORE`` ceiling (a gist is a paraphrase and cannot be the
        verbatim-copy trap that ceiling rejects) and no age modifier: age is a *ranking*
        prior, and nothing here is competing for a slot. An old conversation is precisely
        the case this channel exists for.
        """
        cap = self._nominate_max
        if cap <= 0 or not sessions:
            return [], set()
        anchored = anchored or set()
        snippets: list[str] = []
        used: set = set()
        for entry in sessions:
            if len(snippets) >= cap:
                break
            # A bare string is the chat lane, for the callers that predate the TIL one and
            # for the narrow `chat_sources` view the client renders.
            lane, name = entry if isinstance(entry, (tuple, list)) else ("chat", entry)
            name = (name or "").strip()
            if not name or (lane, name) in used:
                continue
            if lane == "chat":
                if current_session and name == current_session:
                    continue
                if before_session and name >= before_session:
                    continue
                # An anchor already reached this conversation, at exchange granularity. Its
                # payload is the better one; spending the reserved slot on a recap of the
                # same chat would be paying twice to say less.
                if name in anchored:
                    continue
                snippet = self._render_nomination(name)
            elif lane == "til":
                # No fences: a fetched text has no position in the conversation timeline,
                # so neither the active-session nor the *before_session* rule has anything
                # to say about it. That is also why no reflection pass may nominate at all
                # (`query`) — replay fidelity for this lane would need a clock the lane
                # does not carry, and inventing one here would be the wrong place for it.
                snippet = self._render_til_nomination(name)
            else:
                continue
            if not snippet:
                continue
            snippets.append(snippet)
            used.add((lane, name))
        return snippets, {n for l, n in used if l == "chat"}

    def _render_nomination(self, source_session: str) -> str:
        """Render a nominated conversation as its excerpted gist, or ``""``.

        The gist is read from the sidecar rather than from the chat index, and the
        difference is the point: the index only holds gist passages once
        ``_gist_modifier`` has ramped above zero, so a conversation from this morning —
        reflected, recap written, verbatim still fresh — is simply not in there. Reading
        the file answers "is there a recap of this chat" instead of "is a recap of this
        chat currently competitive", which is the question a nomination is asking. It costs
        one small JSON read on the chat path, bounded by ``graph.nominate_max``.

        Returning ``""`` when there is no gist is the honest failure and the whole of the
        fallback: a chat with no recap has only its verbatim exchanges, and choosing WHICH
        exchange to show would need a grain the fact does not have (``chat_facts`` records
        per conversation, with no exchange index). Guessing one would put a turn in front
        of her that nothing chose. If that chat is fresh its exchanges are still in the
        index and the ranking below can retrieve them on their own merits.
        """
        name = (source_session or "").strip()
        if not name:
            return ""
        try:
            gist = gist_excerpt(self._sidecar.summary_text(name),
                                _NOMINATION_DISPLAY_CHARS)
        except Exception:
            return ""
        if not gist:
            return ""
        # Phrased to say why it is here. The other two chat-block payloads present
        # themselves as things that merely came to mind; this one is the conversation
        # behind the facts printed directly above it in the prompt, and reading it as an
        # unrelated coincidence is the one misreading available.
        return f"(The conversation those facts came out of — you recall: {gist})"

    def _render_til_nomination(self, source_ref: str) -> str:
        """Render a nominated ARTICLE or digest as its excerpted recap, or ``""``.

        The TIL half of the same slot, and it exists only because that lane finally has a
        recap to inject (`core.til_gist`) — until then a fact from a wandered article or a
        news digest nominated nothing, which was the gap named when this channel shipped.

        Framed as *something she read* rather than as something said to her, because the
        distinction is load-bearing on this lane: the approved wiki list is chosen for tone
        and not for truth, so a recap of a humour wiki must not arrive in the register a
        conversation with a person does. The recap itself carries the source's character
        (the prompt asks for it explicitly); this only has to avoid contradicting it.

        **The recap only — never her own reaction to the article**, which is stored beside
        it (`wander_sft.reaction_for`) and IS paired with the recap on the outreach path.
        The asymmetry is deliberate, and it is about what a live turn is. Article + her
        reaction, injected into a chat turn, is precisely what the chat-RAG wander channel
        used to do and it is off on every path pending redesign (`generation._INJECT_WANDER`),
        so adding it here would resurrect a disabled channel through a side door — the same
        material, the same position in the prompt, arriving by a different route. Outreach
        is a different case on its merits: that pass is not answering anyone, it is deciding
        whether to raise a question that came out of that reading, and what she made of the
        text is the subject matter rather than background bleed.

        Import is local, mirroring how this module reaches the rest of `core`: the snippets
        tree is resolved through `training.reflections_path` like every other data root, and
        an unbuilt TIL tree must degrade to no nomination rather than to an import error.
        """
        ref = (source_ref or "").strip()
        if not ref:
            return ""
        try:
            from core import til_gist
            from training.reflections_path import til_snippets_dir
            path = til_gist.resolve_source(til_snippets_dir(), ref)
            if path is None:
                return ""
            gist = gist_excerpt(til_gist.gist_text(path), _NOMINATION_DISPLAY_CHARS)
        except Exception:
            return ""
        if not gist:
            return ""
        return f"(What you were reading when that came up — you recall: {gist})"

    def _anchor_modifier(self, source_session: str) -> float:
        """Ranking prior for an anchor: the envelope of the verbatim and gist curves.

        An anchor is a pointer, not a payload, so it must not inherit either curve on its
        own — the verbatim one would kill it at the 96h cap (exactly where it becomes the
        only per-exchange representation left) and the gist one would suppress it while a
        chat is fresh (throwing away its match-side advantages precisely where most
        retrieval happens). Their maximum holds ~1.0 across the verbatim window, rides the
        gist down to its floor afterwards, and never reaches zero — which mirrors the
        payload switch in :meth:`_render_anchor`, so rank and payload agree by construction.
        """
        return max(self._chat_modifier(source_session),
                   self._gist_modifier(source_session))

    def _render_anchor(self, entry: dict, *, dense: float) -> str:
        """Render a claimed exchange as the best representation still permitted.

        Verbatim while it is alive, the descriptor once it is not — the same crossfade the
        gist makes one level up, at exchange granularity. Because this always injects at
        least what the chat channel would have, an anchor claiming an exchange can never
        downgrade the block; it only frees the traditional slot for a different exchange.

        The one case where a live verbatim is still refused: a dense score above the chat
        channel's near-duplicate ceiling means the query is effectively quoting this
        exchange back, and injecting the turns there is the induction-copy trap. The hit is
        kept — retrieving a deliberately probed memory is the point — but it arrives as the
        descriptor, which cannot be copied verbatim because it was never said.
        """
        user = (entry.get("user") or "").strip()
        response = (entry.get("response") or "").strip()
        verbatim_alive = self._chat_modifier(entry.get("source_session", "")) > 0.0
        if user and response and verbatim_alive and dense < self._MAX_SCORE:
            speaker = (entry.get("speaker") or "").strip() or "Interlocutor"
            return (f"{speaker}: {clipped(user, _CHAT_TURN_DISPLAY_CHARS)}\n"
                    f"Me: {clipped(response, _CHAT_TURN_DISPLAY_CHARS)}")
        about = clipped((entry.get("about") or "").strip(), _ANCHOR_DISPLAY_CHARS)
        if not about:
            return ""
        # Deliberately the gist's phrasing: both are a remembered conclusion rather than a
        # quoted turn, and the model has no use for the distinction between them.
        return f"(From an earlier conversation, you recall: {about})"

    def _subject_index_basis(self, kind: str) -> str:
        """Which text a record's vector was built from — its own display line, or its
        trigger. Mirrors ``reflection_memory.embed_text``, which is the source of truth;
        the split is the same one ``_REFL_CEILING_KINDS`` keys on.
        """
        return "display" if kind in self._REFL_CEILING_KINDS else "trigger"

    def _subject_crowded(self, entry: dict, chosen: list[dict]) -> bool:
        """True when *chosen* already holds ``_REFL_SUBJECT_CAP`` records about the same
        subject as *entry* — the injection-side cap (see ``_REFL_SUBJECT_CAP``).

        Compares the stored index vectors, which are L2-normalised, so the dot product is
        the cosine. Only records sharing an indexing basis are compared: across bases the
        number would be a cue-against-a-sentence and would not mean subject agreement.
        Missing vectors (an entry built before the vectors were carried, or a rebuild race)
        simply never crowd — the cap degrades to the previous behaviour rather than
        rejecting blindly.
        """
        cap = self._subject_cap
        if cap <= 0:
            return False
        vec = entry.get("vec")
        if vec is None:
            return False
        basis = self._subject_index_basis(entry.get("kind", ""))
        same = 0
        for other in chosen:
            if self._subject_index_basis(other.get("kind", "")) != basis:
                continue
            other_vec = other.get("vec")
            if other_vec is None:
                continue
            if float(np.dot(vec, other_vec)) >= self._subject_sim:
                same += 1
                if same >= cap:
                    return True
        return False

    def _query_reflection(self, prompt: str, *, before_session: str = "",
                          include_facts: bool = True,
                          include_persona: bool = True,
                          include_asks: bool = True,
                          include_recollections: bool = True,
                          include_impressions: bool = True,
                          include_self_impressions: bool = False,
                          impressions_exclude_about: str = "",
                          reflect_framing: bool = False) -> str:
        """Retrieve relevant distilled reflection notes as a separate block.

        ``include_facts`` / ``include_persona`` / ``include_asks`` /
        ``include_recollections`` / ``include_impressions`` let a caller drop the
        ``[fact]`` / ``[persona]`` / ``[ask]`` / ``[recollection]`` / ``[impression]``
        channels independently. All default True — the reflection/revision path keeps the
        full block. ``impressions_exclude_about`` additionally drops impressions about ONE
        named person (the subject of a portrait the caller is injecting); see
        :meth:`query`.
        """
        # Resolved through the one identity rule attribution and the portrait share, so
        # "Artemy" here matches an ``about`` of "artemy voikhansky" on the record.
        from core.reflection_writer import normalize_person
        exclude_about = normalize_person(impressions_exclude_about)
        with self._lock:
            index = self._refl_index
            entries = list(self._refl_entries)
        if index is None or not entries:
            return ""

        embedder = self._get_embedder()
        q_vec = embedder.encode_query(prompt.strip()).astype("float32")

        # Historical reflection applies availability after vector search. Search all
        # live notes so future-dated or otherwise ineligible top hits cannot starve the
        # eligible result set (the old top-3-first order returned an empty block).
        k = (
            len(entries) if before_session
            else min(max(self._REFL_TOP_K * self._REFL_OVERSAMPLE, self._REFL_TOP_K),
                     len(entries))
        )
        scores, indices = index.search(q_vec, k)

        ranked: list[tuple[float, dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            e = entries[int(idx)]
            # Per-kind channel gates, independent since 2026-07-28 (asks used to ride the
            # Facts gate). Live chat drops asks and keeps facts; reflection keeps all.
            kind = e.get("kind")
            if kind == "persona":
                if not include_persona:
                    continue
            elif kind == "ask":
                if not include_asks:
                    continue
            elif kind == "recollection":
                # Its own branch, above the catch-all below: that `else` treats any
                # unrecognised kind as a fact, so a new kind without a branch here would
                # silently ride the fact gate (and vanish wherever a caller drops facts).
                if not (include_recollections and self._recollections_enabled):
                    continue
            elif kind == "impression":
                # Same reasoning as the branch above — an impression is not a fact, and
                # riding the fact gate would drop it on exactly the chat turns that inject
                # the standing portrait but keep facts.
                if not (include_impressions and self._impressions_enabled):
                    continue
                # Per-subject suppression: this person's settled reading is already in the
                # prompt as a portrait, so their impressions would only restate it — while
                # readings of anyone else stay in and compete normally.
                if exclude_about and normalize_person(e.get("about")) == exclude_about:
                    continue
            elif kind == "self_impression":
                # Its own branch, and the ONE kind here whose caller default is OFF. It is
                # Ava's reading of her own transcripts (see `core.self_portrait`), folded
                # into the outside-view portrait — an observation instrument, not chat
                # context. Retrieved mid-conversation it would read as a third party's
                # notes about her; the config switch defaults false to match, so turning
                # the channel on is a deliberate act rather than the consequence of
                # producing the records.
                if not (include_self_impressions and self._self_impressions_enabled):
                    continue
            elif not include_facts:
                continue
            # ``source_session`` is provenance, not always a timestamp (TIL/wiki
            # learning uses typed ids). Gate on the insert record's real timestamp,
            # with a legacy chat-filename fallback.
            if not memory_available_before(e, before_session):
                continue
            # Near-duplicate ceiling on the RAW cosine, for the kinds indexed by their
            # own display text only (see _REFL_MAX_SCORE): the line would restate the
            # live turn back at her.
            if (kind in self._REFL_CEILING_KINDS
                    and float(score) >= self._REFL_MAX_SCORE):
                continue
            # Relevance gates on the RAW cosine; the decay modifier only orders what
            # passed (rank_score). The old scaled gate meant a half-consolidated fact
            # had to be twice as relevant to surface at all — and a faded one could
            # never surface, an unintended hard forget of distilled memory.
            adjusted = rank_score(float(score), e.get("modifier", 1.0),
                                  self._REFL_MIN_SCORE)
            if adjusted is None:
                continue
            ranked.append((adjusted, e))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        # Greedy admission with pairwise dedup, replacing a flat ``[:_REFL_TOP_K]``.
        # The slots are contested only by this channel now — since the persona digest
        # replaced the `[persona]` RAG channel in chat and open asks stopped being
        # injected, all three go to facts/impressions — so nothing else dilutes a run of
        # paraphrases of one truth, and `content_key` dedups exact repeats but not
        # rewordings. Three restatements of one line is a repeated pattern in-context
        # that no generation-side guard can see (`stop_on_repeat` watches generated text,
        # `_DegenStop` watches token diversity, `_NoCopyPrevReply` protects only the
        # previous reply), so it is rejected here, at the point of injection. A skipped
        # duplicate frees its slot for the next distinct record rather than shortening
        # the block.
        chosen: list[dict] = []
        for _, e in ranked:
            if len(chosen) >= self._REFL_TOP_K:
                break
            display = e.get("display", "")
            if any(_near_duplicate_display(display, c.get("display", ""))
                   for c in chosen):
                continue
            # …and the subject cap, which catches what the wording test cannot: several
            # DISTINCT records about one subject, each saying something the others do not.
            if self._subject_crowded(e, chosen):
                continue
            chosen.append(e)

        # Render grouped by heading rather than one label per line. A three-fact block
        # otherwise opens every line with the SAME prefix — `- (worth recalling) artemyvo`
        # in the 2026-08-04 capture, since facts are distilled subject-first — and a
        # repeated line template is a list-continuation induction seed that every guard on
        # the box is structurally blind to: the varied tails keep each n-gram window novel
        # so `stop_on_repeat` sees no repeat, the varied clauses keep the distinct-token
        # ratio healthy so `_DegenStop` sees no collapse, and the repetition penalty
        # exempts the prompt by design (2026-07-24). The pattern is IN the prompt and the
        # model continues it. Hoisting the shared label + attribution to a heading leaves
        # the bullets carrying only their own content.
        #
        # A group of one keeps the original inline `- (label) text` form: there is no
        # repetition to break, and that is the shape `rag_memory_prompt.txt` describes.
        # Grouping is keyed on the FULL heading (label + attribution), so notes about
        # different people never collapse under one "about X" — the referent confusion
        # `_attribution_label` exists to prevent. Order of first appearance is preserved,
        # so the best-ranked record still leads the block.
        groups: list[tuple[str, list[str]]] = []
        index: dict[str, int] = {}
        for e in chosen:
            kind = e.get("kind")
            if kind == "ask":
                label = "still wondering"
            elif kind == "recollection":
                label = _recollection_label(e)
            elif kind == "impression":
                # Framed as a reading rather than a datum, because that is what separates
                # this kind from the `[fact]` beside it: "worth recalling" would present a
                # provisional impression of someone with the same authority as something
                # they actually stated. `_attribution_label` adds "— about X" after it.
                label = "how they've come to seem to you"
            elif kind == "self_impression":
                # Only reachable with the channel deliberately switched on. Framed as the
                # OUTSIDE view so it can never be read as something she concluded about
                # herself from the inside — that is the persona digest, and the whole
                # value of this kind is that it is the other one.
                label = "how you have come across, reading yourself back"
            else:
                label = "worth recalling"
            heading = f"{label}{_attribution_label(e)}"
            if heading not in index:
                index[heading] = len(groups)
                groups.append((heading, []))
            groups[index[heading]][1].append(str(e.get("display", "")))

        lines = []
        for heading, items in groups:
            if len(items) == 1:
                lines.append(f"- ({heading}) {items[0]}")
                continue
            lines.append(f"{heading}:")
            lines.extend(f"- {item}" for item in items)

        if not lines:
            return ""
        template = (self._refl_reflect_prompt_template if reflect_framing
                    else self._refl_prompt_template)
        return template.replace("{context}", "\n".join(lines))

    def persona_keys(self, prompt: str, *, before_session: str = "",
                     top_k: int = 2, min_score: Optional[float] = None
                     ) -> list[tuple[str, float]]:
        """The reaction→key bridge: live ``[persona]`` keys most relevant to *prompt*.

        Same persona channel as :meth:`_query_reflection`, but returns ``(key, score)`` pairs
        instead of rendered prose so the counter-evidence producer can attach a push to the
        exact stance(s) the reply expressed. Persona-only (facts/asks excluded), temporally
        cut to *before_session* (same availability clock as recall), raw-relevance gated with
        the decay modifier as a ranking prior (``rank_score``),
        deduped by key (keep the best score), highest-first, capped at *top_k*. ``min_score``
        defaults to the reflection floor but a caller may raise it so a counter attaches only
        to a clearly-relevant stance. Empty when nothing clears the floor / no index."""
        with self._lock:
            index = self._refl_index
            entries = list(self._refl_entries)
        if index is None or not entries:
            return []
        floor = self._REFL_MIN_SCORE if min_score is None else min_score

        embedder = self._get_embedder()
        q_vec = embedder.encode_query((prompt or "").strip()).astype("float32")
        k = len(entries) if before_session else min(max(self._REFL_TOP_K * 5,
                                                        self._REFL_TOP_K), len(entries))
        scores, indices = index.search(q_vec, k)

        best: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            e = entries[int(idx)]
            if e.get("kind") != "persona":
                continue
            if not memory_available_before(e, before_session):
                continue
            adjusted = rank_score(float(score), e.get("modifier", 1.0), floor)
            if adjusted is None:
                continue
            key = e.get("key", "")
            if key and adjusted > best.get(key, -1.0):
                best[key] = adjusted
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:max(0, top_k)]

    def _query_wander(self, prompt: str) -> str:
        """Retrieve at most one relevant wander (source article + reaction) as a block.

        Raw similarity supplies the relevance gate, then age decay changes ranking; a wander
        past the last decay step has weight 0 and was dropped at index-build time. The single
        best bounded source passage + bounded reaction is injected so its phrasing can bleed
        without the full article consuming the prompt budget."""
        with self._lock:
            index = self._wander_index
            entries = list(self._wander_entries)
        if index is None or not entries:
            return ""

        embedder = self._get_embedder()
        q_vec = embedder.encode_query(prompt.strip()).astype("float32")

        # One article now contributes multiple bounded passages. Search all of the
        # small corpus, then de-duplicate by article and inject only the best passage.
        k = len(entries)
        scores, indices = index.search(q_vec, k)

        ranked_by_source: dict[str, tuple[float, dict]] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            e = entries[int(idx)]
            adjusted = rank_score(
                float(score), e.get("modifier", 1.0), self._WANDER_MIN_SCORE,
            )
            if adjusted is not None:
                source_id = e.get("source_id", str(idx))
                prior = ranked_by_source.get(source_id)
                if prior is None or adjusted > prior[0]:
                    ranked_by_source[source_id] = (adjusted, e)
        ranked = list(ranked_by_source.values())
        ranked.sort(key=lambda pair: pair[0], reverse=True)

        snippets = [e["display"] for _, e in ranked[:self._WANDER_TOP_K]]
        if not snippets:
            return ""
        return self._wander_prompt_template.replace("{context}", "\n\n".join(snippets))

    @staticmethod
    def _exchange_passages(user: str, response: str) -> list[tuple[str, str]]:
        """Bounded retrieval passages for one exchange, from BOTH sides of the turn.

        Returns ``(passage_text, embed_source)`` pairs, ``embed_source`` in
        ``{"user", "response"}``. Embedding only the user prompt (the historical
        behavior) made Ava's own replies unsearchable: "what did you tell me about X?"
        could only hit if the *user's past phrasing* happened to match. Both sides now
        contribute passage vectors; a query hit on either collapses back to the same
        displayed exchange (the ``(source_session, exchange_index)`` dedup in
        ``_query_chat``), so this widens recall without duplicating injected text.
        The reasoning channel is excluded by construction — ``assistant_response`` /
        the logged answer never contains the ``<think>`` CoT.
        """
        pairs: list[tuple[str, str]] = []
        for source, text in (("user", user), ("response", response)):
            for passage in chunk_text(
                (text or "").strip(), max_chars=_EMBED_CHUNK_CHARS,
                overlap_chars=_EMBED_OVERLAP_CHARS,
            ):
                pairs.append((passage, source))
        return pairs

    def add_exchange(
        self,
        user_prompt: str,
        assistant_response: str,
        speaker: str = "",
        *,
        source_session: str = "",
        exchange_index: int = -1,
    ) -> None:
        """Incrementally index a non-active exchange after it is logged.

        Live turns are already in the conversation prompt, so indexing the current
        transcript only creates self-RAG duplicates. The next session-boundary rebuild
        picks the now-historical file up in one canonical pass.
        """
        try:
            text = user_prompt.strip()
            if not text or not self._ready:
                return
            with self._lock:
                current_session = (
                    self._current_session_file.name if self._current_session_file else ""
                )
            if current_session and source_session == current_session:
                return

            modifier = self._chat_modifier(source_session)
            if modifier <= 0.0:
                return

            passage_pairs = self._exchange_passages(text, assistant_response)
            if not passage_pairs:
                return
            embedder = self._get_embedder()
            vectors = embedder.encode_passages(
                [p for p, _ in passage_pairs]).astype("float32")

            base_entry = {
                "user": text,
                "response": assistant_response.strip(),
                "speaker": speaker.strip(),
                "source_session": source_session,
                "exchange_index": exchange_index,
                "modifier": modifier,
            }
            entries = [
                dict(base_entry, passage_index=i, embed_source=source)
                for i, (_, source) in enumerate(passage_pairs)
            ]
            with self._lock:
                if self._index is None:
                    idx = faiss.IndexFlatIP(vectors.shape[1])
                    idx.add(vectors)
                    self._index = idx
                    self._entries = entries
                else:
                    self._index.add(vectors)
                    self._entries.extend(entries)

        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def refresh_reflection_memory(self) -> None:
        """Rebuild the reflection-memory index from rag_memory.jsonl.

        Call after a consolidation pass is saved so newly distilled notes (and
        evictions) take effect within the same server lifetime, without a restart.
        """
        self._build_reflection_index()

    def refresh_wander(self) -> None:
        """Rebuild the wander channel from the durable corpus.

        Call after a wander is Applied so the new article becomes retrievable within the
        same server lifetime (no restart). Cheap: the corpus is small and CPU-embedded.
        """
        self._build_wander_index()

    def _load_prompt_template(self) -> str:
        path = self.prompts_dir / "rag_prompt.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return "Past conversation context (soft reference only):\n\n{context}"

    def _load_chat_reflect_prompt_template(self) -> str:
        """``rag_prompt.txt``'s past-chat block, framed for a pass that is not in a chat.

        The chat wrapper closes on "do not cite or repeat them verbatim unless clearly
        useful **to the user**" — which presumes a live turn being served. Every
        `reflection_runner` pass sets ``rag_include_chat=False`` and never sees it, and
        synthesis / check-in / deliberation pass ``disable_rag=True``; the callers that
        DO leave the channel on are outreach ("Reach Out"), til_wander's learn/wander
        passes, and the prompt experiment. Outreach is the sharpest case: a pass deciding
        whether to open a conversation, handed verbatim conversation excerpts under a
        wrapper addressed to someone already mid-conversation. Missing file ⇒ the chat
        template.
        """
        path = self.prompts_dir / "rag_reflect_prompt.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return self._prompt_template

    def _load_refl_prompt_template(self) -> str:
        path = self.prompts_dir / "rag_memory_prompt.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return "Notes from your earlier reflections (soft reference only):\n\n{context}"

    def _load_refl_reflect_prompt_template(self) -> str:
        """The same block, framed for a pass that is judging rather than answering.

        ``rag_memory_prompt.txt`` is written for live chat and says so out loud — "use
        them only if they genuinely fit the moment", "do not announce that you are
        consulting notes", and a closing paragraph about what to ask "as you close a
        reply". A reflection pass gets this block appended to its own instructions, so
        that wording was a standing instruction to produce a reply sitting inside a
        prompt whose whole contract is "output these fields, nothing else" — one of the
        pressures steering a revision pass into re-answering the exchange it was asked
        to judge. Missing file ⇒ fall back to the chat template (previous behaviour).
        """
        path = self.prompts_dir / "rag_memory_reflect_prompt.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return self._refl_prompt_template

    def _load_wander_prompt_template(self) -> str:
        path = self.prompts_dir / "rag_wander_prompt.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ("Something you read on your own recently (soft reference only):\n\n"
                    "{context}")

    def _get_embedder(self):
        if self._embedder is None:
            # Keep RAG embeddings on CPU so index rebuilds never contend with the
            # inference model for VRAM or kernel launch bandwidth. The wrapper applies
            # multilingual MiniLM and pools bounded chunks for long text.
            model = SentenceTransformer(self._EMBED_MODEL, device="cpu")
            self._embedder = _RetrievalEmbedder(model)
        return self._embedder

    def _reflected_at_for_session(self, source_session: str) -> Optional[str]:
        """The reflect-once freeze timestamp of a session's bundle (None if unfrozen).

        Memoized within a rebuild (the timestamp is immutable once set). An unfrozen
        session caches None but the cache is cleared each rebuild, so a session frozen
        between rebuilds is re-read.
        """
        if not source_session:
            return None
        if source_session in self._reflected_at_cache:
            return self._reflected_at_cache[source_session]
        try:
            ra = self._sidecar.load(source_session).get("reflected_at")
        except Exception:
            ra = None
        self._reflected_at_cache[source_session] = ra
        return ra

    def _age_of_session(self, source_session: str) -> Optional[float]:
        """Wall-clock hours since the CHAT (session stem), for a FROZEN bundle; else None.

        Age is clocked from the chat's own timestamp and evaluated at retrieval time (now).
        Returns None for an unreflected chat — it is not yet on the path to weights, so it
        stays at full RAG weight (no fade) until it is frozen (guards the pre-weights gap).
        """
        if _wall_clock_age_hours is None or not source_session:
            return None
        if not self._reflected_at_for_session(source_session):   # frozen gate
            return None
        try:
            return _wall_clock_age_hours(source_session, datetime.now())
        except Exception:
            return None

    def _raw_age_of_session(self, source_session: str) -> Optional[float]:
        """Wall-clock hours since the CHAT, IGNORING the frozen gate (None if unparseable).

        Unlike :meth:`_age_of_session` this does not require the bundle to be frozen, so the
        day-0 fresh-window droop applies to a still-unreflected recent chat — the whole point,
        since a day-0 chat is almost always unreflected and would otherwise sit at 1.0.
        """
        if _wall_clock_age_hours is None or not source_session:
            return None
        try:
            return _wall_clock_age_hours(source_session, datetime.now())
        except Exception:
            return None

    def _chat_modifier(self, source_session: str) -> float:
        """RAG crossfade weight for a chat exchange, from its bundle's WALL-CLOCK age.

        A frozen bundle fades linearly 1.0→0 over ``rag_cap_age_h``; the raw-age hard cap
        applies even if a bundle is still unfrozen, so no verbatim exchange survives past
        96h. Before that cap an unfrozen chat retains the gentler day-0 fresh-window policy.
        """
        if (_verbatim_rag_weight_hours is None or self._consolidation is None
                or not source_session):
            return 1.0
        wall = self._consolidation.wall
        raw_age = self._raw_age_of_session(source_session)
        # Literal channel contract: verbatim chat is gone at the cap even when reflection /
        # training lagged. The gist/weights assumption is operational, not a soft exception.
        if raw_age is not None and (
                wall.rag_cap_age_h <= 0 or raw_age >= wall.rag_cap_age_h):
            return 0.0
        modifier = _verbatim_rag_weight_hours(
            self._age_of_session(source_session), wall)
        if _fresh_time_weight is not None:
            modifier = min(modifier, _fresh_time_weight(raw_age, wall))
        return modifier

    def _recollection_modifier(self, written_ts: str) -> float:
        """RAG weight for a ``[recollection]``, from the age of the READING itself.

        Hours since *written_ts* (the op-log insert time), not since the conversation it
        recalls — a re-derived reading of a two-month-old chat is memory formed today.
        Unparseable/absent ⇒ 1.0 (treated as fresh), matching the curve's own permissive
        default: a recollection is written by a pass that just ran, so a missing stamp is
        a bug rather than evidence of age, and withholding it would silently drop the
        channel.
        """
        if _recollection_rag_weight_hours is None or self._consolidation is None:
            return 1.0
        written = recorded_at(written_ts or "")
        if written is None:
            return 1.0
        age_h = (datetime.now() - written).total_seconds() / 3600.0
        return _recollection_rag_weight_hours(age_h, self._consolidation.wall)

    def _gist_modifier(self, source_session: str) -> float:
        """RAG tent weight for a consolidation-gist passage, from its chat's WALL-CLOCK age.

        The crossfade PARTNER of ``_chat_modifier``: the gist ramps 0→1 as verbatim fades
        to zero over ``rag_cap_age_h``, then decays linearly to ``gist_floor_weight`` exactly
        at ``gist_cap_age_h`` and holds. Uses the same frozen-age clock as the verbatim
        crossfade (``_age_of_session``);
        a gist only exists on a reflected (frozen) session, so the age is normally available,
        and an unknown age yields 0 (``gist_rag_weight_hours`` withholds it).
        """
        if _gist_rag_weight_hours is None or self._consolidation is None or not source_session:
            return 0.0
        return _gist_rag_weight_hours(
            self._age_of_session(source_session), self._consolidation.wall
        )

    def _consolidation_modifiers(self) -> dict:
        """{anchor key: crossfade weight} for PERSONA anchors.

        REBUILD §6: each persona anchor's weight fades on its SOURCE BUNDLE's wall-clock age
        (the chat it was distilled from) — the same clock that ramps that bundle's training LR,
        on the decoupled RAG pace. Empty when the training package is unavailable, so every
        item keeps weight 1.0 (no decay) — inference is never hard-coupled to it.

        Facts are deliberately EXCLUDED (2026-07-23): a fact is a timeless relational truth and
        must not be down-ranked by age, so it holds modifier 1.0 for life (see
        _build_reflection_index / AVA_MEMORY.md §3.4). This method now serves persona only.
        """
        if self._ledger is None or self._consolidation is None or _rag_weight_hours is None:
            return {}
        try:
            wall = self._consolidation.wall
            out: dict[str, float] = {}
            for key, rec in self._ledger.fold().items():
                if rec.get("type") != "persona":
                    continue
                se = rec.get("source_exchange") or {}
                sess = se.get("source_session") or rec.get("source_session") or ""
                out[key] = _rag_weight_hours(self._age_of_session(sess), wall)
            return out
        except Exception:
            return {}

    def _collect_chat_entries(self) -> tuple[list[dict], list[str], list[dict], list[str]]:
        """Load chat exchanges keyed by (session, index) with the age crossfade applied.

        REBUILD §6: the weight is per-BUNDLE (age = wall-clock hours since the chat), so every
        exchange of a session shares one modifier. Verbatim reaches hard zero at
        ``rag_cap_age_h`` and is omitted; gist passages are emitted first and survive at their
        semantic floor.

        Also collects the ANCHOR entries for their own index, in this same pass: the sidecar
        is already open here for the gist, and the transcript is already parsed, which is
        what lets an anchor carry the turns it points at without a second read.
        Returns ``(entries, texts, anchor_entries, anchor_texts)``.
        """
        entries: list[dict] = []
        texts: list[str] = []
        anchor_entries: list[dict] = []
        anchor_texts: list[str] = []
        sidecar_cache: dict[str, dict] = {}

        chat_files = {}
        if self.fallback_chats_dir is not None and self.fallback_chats_dir.exists():
            for p in iter_chat_json_files(self.fallback_chats_dir):
                chat_files[p.name] = p
        if self.chats_dir.exists():
            for p in iter_chat_json_files(self.chats_dir):
                chat_files[p.name] = p

        for chat_file in sorted(chat_files.values(), key=lambda p: p.name):
            with self._lock:
                skip = chat_file == self._current_session_file
            if skip:
                continue
            session_name = chat_file.name
            try:
                data = json.loads(chat_file.read_text(encoding="utf-8"))
                session_user = (data.get("user") or "").strip()
                exchanges = data.get("exchanges", [])
                if not isinstance(exchanges, list):
                    continue
                sidecar_doc = sidecar_cache.get(session_name)
                if sidecar_doc is None:
                    sidecar_doc = self._sidecar.load(session_name)
                    sidecar_cache[session_name] = sidecar_doc

                # Per-exchange anchors (core/exchange_anchor.py): the sidecar's top-level
                # `anchors` map, joined here to its transcript turns so a winning anchor can
                # inject the real exchange while verbatim is still alive. Emitted BEFORE the
                # verbatim short-circuit below for the same reason the gist is — outliving
                # the verbatim window is the whole point of the channel.
                anchors = sidecar_doc.get("anchors") if isinstance(sidecar_doc, dict) else None
                if isinstance(anchors, dict):
                    for raw_idx, rec in anchors.items():
                        if not isinstance(rec, dict):
                            continue
                        try:
                            ex_index = int(raw_idx)
                        except (TypeError, ValueError):
                            continue
                        about = (rec.get("about") or "").strip()
                        tags = [t for t in (rec.get("tags") or []) if (t or "").strip()]
                        if not about and not tags:
                            continue
                        exc = (exchanges[ex_index]
                               if 0 <= ex_index < len(exchanges) else None)
                        exc = exc if isinstance(exc, dict) else {}
                        anchor_entries.append({
                            "about": about,
                            "tags": tags,
                            "user": (exc.get("user_prompt") or "").strip(),
                            "response": (exc.get("assistant_response") or "").strip(),
                            "speaker": ((exc.get("speaker") or "").strip()
                                        or session_user),
                            "source_session": session_name,
                            "exchange_index": ex_index,
                        })
                        # The descriptor is the embed text: one unit per exchange, in the
                        # conversation's own language. A tags-only anchor still indexes (on
                        # its tag list) so the sparse gate can reach it at all.
                        anchor_texts.append(about or ", ".join(tags))

                # Consolidation-gist passages: the distilled prose recap the reflection summary
                # pass wrote to the sidecar. Chunked into passages like verbatim turns, but
                # carried on the SLOW gist tent (`_gist_modifier`) — it ramps up as the
                # verbatim fades and OUTLIVES it. Emitted BEFORE the verbatim
                # `session_modifier <= 0` short-circuit below so the gist survives exactly
                # when the verbatim has dropped out of RAG (the whole point of the crossfade).
                # RAG-only; never trained. Each passage gets a distinct negative sentinel
                # `exchange_index` so it holds its own ranking slot — collapse-to-passage, so a
                # query retrieves the relevant slice of the recap, not the whole 2-3k blob.
                # Lives in its own `<stem>.summary.json` (see `ChatSidecar.read_summary`),
                # which also sanitizes on read and falls back to the legacy
                # `consolidation_summary` key for chats reflected before the split.
                gist_text = self._sidecar.summary_text(session_name)
                if gist_text:
                    gist_modifier = self._gist_modifier(session_name)
                    if gist_modifier > 0.0:
                        for p_idx, passage in enumerate(chunk_text(
                            gist_text, max_chars=_EMBED_CHUNK_CHARS,
                            overlap_chars=_EMBED_OVERLAP_CHARS,
                        )):
                            entries.append({
                                "kind": "gist",
                                "content": passage,
                                "source_session": session_name,
                                "exchange_index": -(p_idx + 1),
                                "passage_index": p_idx,
                                "modifier": gist_modifier,
                            })
                            texts.append(passage)

                # Per-bundle wall-clock age crossfade: one weight for the whole session (age
                # is a bundle property). Verbatim is absent at/after the hard 96h cap.
                session_modifier = self._chat_modifier(session_name)
                if session_modifier <= 0.0:
                    continue
                for exc_idx, exc in enumerate(exchanges):
                    if not isinstance(exc, dict):
                        continue
                    user = (exc.get("user_prompt") or "").strip()
                    resp = (exc.get("assistant_response") or "").strip()
                    speaker = (exc.get("speaker") or "").strip() or session_user
                    if not user:
                        continue
                    modifier = session_modifier
                    base_entry = {
                        "user": user,
                        "response": resp,
                        "speaker": speaker,
                        "source_session": session_name,
                        "exchange_index": exc_idx,
                        "modifier": modifier,
                    }
                    # One exchange contributes several passage vectors — bounded chunks
                    # of the user prompt AND of Ava's reply (`_exchange_passages`) — but
                    # query results are collapsed back to one displayed exchange. This
                    # keeps Russian/English tails and Ava's own side of the dialogue
                    # searchable without duplicating injected text.
                    for passage_index, (passage, source) in enumerate(
                        self._exchange_passages(user, resp)
                    ):
                        entries.append(dict(base_entry, passage_index=passage_index,
                                            embed_source=source))
                        texts.append(passage)
            except Exception:
                continue
        return entries, texts, anchor_entries, anchor_texts

    def _install_anchor_index(self, entries: list[dict], texts: list[str]) -> None:
        """Build/replace the anchor faiss index and its tag-IDF corpus statistics."""
        if not texts:
            with self._lock:
                self._anchor_entries = []
                self._anchor_index = None
                self._anchor_tag_stats = {}
            return

        embedder = self._get_embedder()
        embeddings = embedder.encode_passages(texts).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        with self._lock:
            self._anchor_entries = entries
            self._anchor_index = index
            self._anchor_tag_stats = build_tag_stats(entries)

    def _install_chat_index(self, entries: list[dict], texts: list[str]) -> None:
        """Build/replace the chat faiss index from pre-collected entries."""
        if not texts:
            with self._lock:
                self._entries = []
                self._index = None
                self._ready = True
            return

        embedder = self._get_embedder()
        embeddings = embedder.encode_passages(texts).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        with self._lock:
            self._entries = entries
            self._index = index
            self._ready = True

    def _build_chat_index(self) -> None:
        """Rebuild only the chat-RAG index (age-crossfade-aware)."""
        try:
            self._reflected_at_cache = {}   # re-read freeze stamps against current builds
            entries, texts, anchors, anchor_texts = self._collect_chat_entries()
            # Anchors first: `_install_chat_index` is what flips `_ready`, and a query
            # admitted between the two installs would otherwise see a fresh chat index
            # beside a stale anchor one.
            self._install_anchor_index(anchors, anchor_texts)
            self._install_chat_index(entries, texts)
        except Exception:
            with self._lock:
                self._ready = True

    def _build_index(self) -> None:
        """Load all historical exchanges and build the faiss index."""
        try:
            self._build_chat_index()
            self._build_reflection_index()
            self._build_wander_index()
        except Exception:
            with self._lock:
                self._ready = True

    def _build_reflection_index(self) -> None:
        """Fold rag_memory.jsonl into live items and (re)build their faiss index."""
        if self._memory is None:
            return
        try:
            self._reflected_at_cache = {}   # re-read freeze stamps against current builds
            # {anchor key: age crossfade modifier} for fact/persona items (by source bundle).
            mods = self._consolidation_modifiers()
            entries: list[dict] = []
            texts: list[str] = []
            for rec in self._memory.live_items():
                embed_text = ReflectionMemory.embed_text(rec)
                display = (rec.get("content") or "").strip()
                if not embed_text or not display:
                    continue
                # Persona self-statements decay as their SOURCE BUNDLE consolidates into the
                # weights (REBUILD Phase 3 crossfade): their from_weights RAG copy fades
                # 1.0→`rag_floor_weight` as the bundle's age climbs and then holds — an aged
                # persona sinks in ranking but stays recallable (the drop below only fires
                # under a configured floor of 0, the legacy hard forget).
                #
                # FACTS DO NOT FADE (2026-07-23): a fact is a timeless relational truth, so
                # the wall clock must not down-rank it — it holds modifier 1.0 for life. This
                # deliberately makes forgetting a STALE/false fact an explicit-eviction
                # problem rather than a decay one. Direct corrections are handled by
                # fact_contradict's reversible supersession passes; an uncontradicted false
                # fact still needs an explicit lifecycle decision (see AVA_MEMORY.md
                # §3.4/§7 and AVA_OPEN_PROBLEMS.md → Fact Staleness). Asks never decay either.
                #
                # A [recollection] is the one kind clocked PER RECORD rather than per
                # source bundle: it is what Ava now makes of an old conversation, so its
                # age runs from when that reading was written (`ts`), never from the chat
                # it recalls — clocking it on the chat would exactly undo the point of the
                # kind (see training.decay.recollection_rag_weight_hours).
                #
                # An [impression] does not fade either, and for the fact's reason rather
                # than the persona's: it is a reading of someone who is still that person,
                # so the wall clock is not evidence against it. What DOES supersede it is
                # a later reading — either as a re-insert on the same content_key, or by
                # out-ranking it in the portrait fold, where recency weighting already
                # lives (core.user_digest). Decaying it here as well would double-count.
                modifier = 1.0
                if rec.get("kind") == "persona":
                    modifier = mods.get(rec.get("key"), 1.0)
                    if modifier <= 0.0:
                        continue
                elif rec.get("kind") == "recollection":
                    modifier = self._recollection_modifier(rec.get("ts", ""))
                entries.append({
                    "key": rec.get("key", ""),
                    "kind": rec.get("kind", ""),
                    "display": display,
                    "modifier": modifier,
                    # Provenance of a recollection: WHEN the recalled conversation
                    # happened (the reading's own date is `available_at`). Display only —
                    # it does not enter the weight.
                    "origin_ts": rec.get("origin_ts", ""),
                    "tags": rec.get("tags") or [],
                    "source_session": rec.get("source_session", ""),
                    # Op-log insertion time is the actual availability clock. External
                    # TIL/wiki provenance ids are intentionally not overloaded as time.
                    "available_at": rec.get("ts", ""),
                    # Attribution (facts only): who the item is ABOUT and who SAID it.
                    # Carried to the display line, because a fact about one person
                    # recalled while another is speaking is otherwise indistinguishable
                    # from something the person in front of her said — the recall path
                    # has no other signal of whose it is.
                    "about": rec.get("about", ""),
                    "source": rec.get("source", ""),
                    "source_class": rec.get("source_class", ""),
                })
                texts.append(embed_text)

            if not texts:
                with self._lock:
                    self._refl_index = None
                    self._refl_entries = []
                self._refl_embed_cache = {}
                return

            # Encode only texts not already cached from a prior rebuild; reuse the
            # rest. Encoding is the costly CPU step, so this turns a per-session
            # rebuild from O(M live items) into O(new items this session).
            missing = [t for t in dict.fromkeys(texts) if t not in self._refl_embed_cache]
            if missing:
                embedder = self._get_embedder()
                new_vecs = embedder.encode_passages(missing).astype("float32")
                for t, vec in zip(missing, new_vecs):
                    self._refl_embed_cache[t] = vec.reshape(1, -1)

            embeddings = np.vstack([self._refl_embed_cache[t] for t in texts])
            # Prune cache to the live set so it stays bounded by M, not all texts
            # ever seen (evicted/decayed items would otherwise linger forever).
            self._refl_embed_cache = {t: self._refl_embed_cache[t] for t in texts}

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            # Keep each entry's own vector beside it: the subject cap (see
            # _subject_crowded) needs candidate-vs-candidate cosine, not just
            # candidate-vs-query, and FAISS only answers the latter. Vectors are
            # L2-normalised, so a dot product IS the cosine. ~1.3 MB at 800 live items.
            for entry, vec in zip(entries, embeddings):
                entry["vec"] = vec

            with self._lock:
                self._refl_index = index
                self._refl_entries = entries

        except Exception:
            with self._lock:
                self._refl_index = None
                self._refl_entries = []

    def _build_wander_index(self) -> None:
        """(Re)build the wander faiss index from the durable corpus (server/data/til).

        Each corpus record still within the decay window contributes bounded, overlapping
        SOURCE passages (so retrieval can match material late in a long English/Russian
        article). Each carries a bounded display + stripped reaction and the article's
        age-decay modifier. Records past the last decay step (weight 0) are dropped here.
        Best-effort / GPU-free."""
        corpus = self._wander_corpus
        if corpus is None:
            return
        try:
            wall = self._consolidation.wall if self._consolidation is not None else None
            now = datetime.now()
            entries: list[dict] = []
            texts: list[str] = []
            if corpus.exists():
                for line_number, line in enumerate(
                    corpus.read_text(encoding="utf-8").splitlines()
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # Banned from training by the Training review tab (a malformed
                    # generation): keep it out of retrieval too — the whole point of the
                    # ban is that this text should stop shaping her.
                    if rec.get("banned"):
                        continue
                    src = rec.get("prompt", "") or ""
                    ti = src.find("TITLE:")
                    source_text = (src[ti:] if ti != -1 else src).strip()
                    reaction = _WANDER_THINK_RE.sub("", rec.get("target", "") or "").strip()
                    if not source_text or not reaction:
                        continue
                    # Age decay (wall-clock hours since capture); weight 0 ⇒ drop from RAG.
                    if _wander_rag_weight_hours is not None and wall is not None:
                        age = None
                        if _wall_clock_age_hours is not None:
                            age = _wall_clock_age_hours(rec.get("ts"), now)
                        modifier = _wander_rag_weight_hours(age, wall)
                    else:
                        modifier = 1.0
                    if modifier <= 0.0:
                        continue
                    # The prompt stores TITLE/url before the article body. Repeat that
                    # small provenance header on each passage but never inject the full
                    # source blob: a single wander cannot consume the whole chat budget.
                    header, separator, body = source_text.partition("\n\n")
                    if not separator:
                        header, body = "", source_text
                    source_id = f"{rec.get('ts') or 'legacy'}:{line_number}"
                    reaction_display = clipped(
                        reaction, _WANDER_REACTION_DISPLAY_CHARS,
                        marker="\n...[reaction truncated]",
                    )
                    passages = chunk_text(
                        body, max_chars=_EMBED_CHUNK_CHARS,
                        overlap_chars=_EMBED_OVERLAP_CHARS,
                    )
                    for passage_index, passage in enumerate(passages):
                        retrieval_text = (
                            f"{header}\n\n{passage}" if header else passage
                        )
                        entries.append({
                            "display": (
                                f"{retrieval_text}\n\nMy reaction: {reaction_display}"
                            ),
                            "modifier": modifier,
                            "source_id": source_id,
                            "passage_index": passage_index,
                        })
                        texts.append(retrieval_text)

            if not texts:
                with self._lock:
                    self._wander_index = None
                    self._wander_entries = []
                return

            embedder = self._get_embedder()
            embeddings = embedder.encode_passages(texts).astype("float32")
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            with self._lock:
                self._wander_index = index
                self._wander_entries = entries

        except Exception:
            with self._lock:
                self._wander_index = None
                self._wander_entries = []


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test — the subject cap's admission rules
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise ``_subject_crowded`` directly: it is pure given entries that carry their
    vectors, so it needs no index, no embedder and no data on disk. Built via
    ``object.__new__`` because ``__init__`` reads config and prompt templates off disk,
    none of which this logic touches.
    """
    eng = object.__new__(RagEngine)
    eng._subject_cap = RagEngine._REFL_SUBJECT_CAP
    eng._subject_sim = RagEngine._REFL_SUBJECT_SIM

    def ent(kind: str, vec, display: str = "x") -> dict:
        v = np.asarray(vec, dtype="float32")
        n = float(np.linalg.norm(v))
        return {"kind": kind, "display": display, "vec": v / n if n else v}

    near_a = ent("fact", [1.0, 0.0, 0.0])
    near_b = ent("fact", [0.9, 0.44, 0.0])       # cosine ~0.90 with near_a
    far    = ent("fact", [0.0, 0.0, 1.0])        # orthogonal

    assert not eng._subject_crowded(near_a, []), "nothing chosen yet ⇒ never crowded"
    assert eng._subject_crowded(near_b, [near_a]), "same subject must be capped"
    assert not eng._subject_crowded(far, [near_a]), "a distinct subject must pass"

    # Basis split: a fact embeds on its trigger and an impression on its display text, so
    # even an identical vector is not evidence they share a subject.
    impression = ent("impression", [1.0, 0.0, 0.0])
    assert not eng._subject_crowded(impression, [near_a]), \
        "kinds indexed on different text must not be compared"
    assert eng._subject_crowded(impression, [ent("persona", [1.0, 0.0, 0.0])]), \
        "display-indexed kinds compare with each other"

    # Cap 2 leaves one slot to a second same-subject record but not a third.
    eng._subject_cap = 2
    assert not eng._subject_crowded(near_b, [near_a])
    assert eng._subject_crowded(near_b, [near_a, near_a])
    # Cap 0 disables the rule outright (the documented kill value).
    eng._subject_cap = 0
    assert not eng._subject_crowded(near_b, [near_a, near_a])
    eng._subject_cap = RagEngine._REFL_SUBJECT_CAP

    # A vectorless entry (older rebuild, or a race) degrades to the previous behaviour
    # rather than rejecting blindly — it must never crowd, and never BE crowded.
    novec = {"kind": "fact", "display": "x"}
    assert not eng._subject_crowded(novec, [near_a])
    assert not eng._subject_crowded(near_b, [novec])

    # The oversample must cover the cap: with one slot per subject, a block of TOP_K
    # slots can need TOP_K distinct subjects out of the candidate window.
    assert RagEngine._REFL_OVERSAMPLE >= RagEngine._REFL_TOP_K, \
        "oversample too narrow to backfill a fully-capped block"

    # ── fact-nominated recall ────────────────────────────────────────────────
    # Also pure given a sidecar: no index, no embedder, no scoring — a nomination
    # arrives already justified, so this is entirely about fences and payload.
    class _FakeSidecar:
        def __init__(self, gists):
            self._g = gists

        def summary_text(self, name):
            return self._g.get(name, "")

    nom = object.__new__(RagEngine)
    nom._nominate_max = _NOMINATION_TOP_K
    nom._sidecar = _FakeSidecar({
        "20260701_100000.json": "We talked about beekeeping on the roof.",
        "20260801_100000.json": "We talked about the war.",
        "20260810_100000.json": "Today's conversation.",
        "20260901_100000.json": "A conversation from the future.",
    })

    snips, used = nom._query_nominated(["20260701_100000.json"])
    assert len(snips) == 1 and "beekeeping" in snips[0], snips
    assert used == {"20260701_100000.json"}, used
    assert "those facts came out of" in snips[0], \
        "the payload must say why it is here, not read as a coincidence"

    assert nom._query_nominated(None) == ([], set()), "no nominations, no slot"
    assert nom._query_nominated([]) == ([], set())
    # A chat with no recap yields nothing rather than guessing an exchange to show.
    assert nom._query_nominated(["20260101_000000.json"]) == ([], set())

    # The chat channel's fences hold for a nomination too — arriving by a different
    # route is not a reason to escape them.
    assert nom._query_nominated(["20260810_100000.json"],
                                current_session="20260810_100000.json") == ([], set()), \
        "the active transcript is already in the conversation messages"
    assert nom._query_nominated(["20260901_100000.json"],
                                before_session="20260801_100000.json") == ([], set()), \
        "a replay-faithful pass must not see a later session"
    assert nom._query_nominated(["20260701_100000.json"],
                                before_session="20260801_100000.json")[1] == \
        {"20260701_100000.json"}, "an older session still passes the cutoff"

    # An anchor already reached this conversation at exchange granularity: its payload is
    # strictly better, so the reserved slot is not spent saying less about the same chat.
    assert nom._query_nominated(["20260701_100000.json"],
                                anchored={"20260701_100000.json"}) == ([], set())

    # The cap bounds the slot, and 0 is the documented kill value.
    two = ["20260701_100000.json", "20260801_100000.json"]
    assert len(nom._query_nominated(two)[0]) == 1, "default cap is one conversation"
    nom._nominate_max = 2
    assert len(nom._query_nominated(two)[0]) == 2
    assert nom._query_nominated(two)[0][0].count("beekeeping") == 1, \
        "best-first order is the caller's, preserved"
    nom._nominate_max = 0
    assert nom._query_nominated(two) == ([], set()), "cap 0 turns the channel off"
    nom._nominate_max = _NOMINATION_TOP_K
    # A repeated nomination is one conversation, not two spent slots.
    nom._nominate_max = 2
    assert len(nom._query_nominated(["20260701_100000.json"] * 3)[0]) == 1
    nom._nominate_max = _NOMINATION_TOP_K

    # The payload is bounded: an additive slot must not cost a full gist.
    nom._sidecar = _FakeSidecar({"20260701_100000.json": "Sentence one. " * 400})
    long_snip = nom._query_nominated(["20260701_100000.json"])[0][0]
    assert len(long_snip) <= _NOMINATION_DISPLAY_CHARS + 120, len(long_snip)
    assert _NOMINATION_DISPLAY_CHARS < _CHAT_TURN_DISPLAY_CHARS, \
        "a reserved additive slot must be tighter than a ranked one"

    # A sidecar that raises is a retrieval failure, never a failed turn.
    class _Boom:
        def summary_text(self, name):
            raise RuntimeError("disk on fire")

    nom._sidecar = _Boom()
    assert nom._query_nominated(["20260701_100000.json"]) == ([], set())

    # Typed sources: the two lanes resolve through different stores, so the lane travels
    # with the ref. A bare string still means the chat lane, for the narrow caller.
    nom._sidecar = _FakeSidecar({"20260701_100000.json": "We talked about beekeeping."})
    assert nom._query_nominated([("chat", "20260701_100000.json")])[1] == \
        {"20260701_100000.json"}, "a typed chat ref resolves like a bare one"
    # A TIL ref goes to the snippets tree, which is absent in this test — so it must
    # degrade to no nomination rather than raising on the chat path.
    assert nom._query_nominated([("til", "news/2026-07-30")]) == ([], set())
    assert nom._query_nominated([("bogus", "x")]) == ([], set()), "an unknown lane is skipped"
    # Only chat sessions are CLAIMED: the claim exists to stop the ranking injecting the
    # same conversation twice, and the ranking has no TIL entries to collide with.
    nom._nominate_max = 2
    snips2, used2 = nom._query_nominated([("til", "news/2026-07-30"),
                                          ("chat", "20260701_100000.json")])
    assert used2 == {"20260701_100000.json"}, used2
    nom._nominate_max = _NOMINATION_TOP_K

    print("rag_engine subject-cap + nomination self-test OK")


if __name__ == "__main__":
    _selftest()
