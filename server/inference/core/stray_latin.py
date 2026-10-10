"""Stray Latin capitals glued onto Cyrillic words — detection, repair, and the corpus sweep.

A sampling glitch of the gemma-4 adapters: a bare single-letter Latin capital token, a
homoglyph of a Cyrillic letter (`C` for С, `B` for В), sampled right after a Cyrillic word
piece — "болееC стабильную", "всю моюB надуманную", "становлюCь". The tendency is learned
(the base never puts it top-1 there; the last adapters do, 4–6 per 10k Cyrillic pieces), and
it feeds itself: every glitched reply is a training target, a RAG passage, a history turn of
the next exchange and the text the revision pass re-reads (AVA_CHANGELOG 2026-10-04/05).

Three defences, all keyed to the same letter set (:data:`HOMOGLYPHS`, the Latin capitals
that look like Cyrillic ones — a deliberate Latin letter inside a Russian word is wordplay
in a letter that *looks* Latin, "путинSSкий", "КняZZ", and is never one of these):

* **at the source** — ``inference_backend``'s sampler guard masks these single-letter
  tokens after a piece that ends in a Cyrillic letter (``stray_latin.guard``);
* **at every target write** — ``ChatSidecar.write_verdict`` deletes the word-final ones
  (:func:`repair`), the form that needs no guess;
* **in Sleep** — :func:`sweep` over the live transcripts and sidecar targets, at the head
  of every Sleep run (word-final + a model repair of the mid-word ones) and on every
  background wake (word-final only, GPU-free). A transcript edit keeps the prior text under
  the exchange's ``text_repairs`` and leaves the ``tension`` block — the generation record —
  in place.

The training row check (``training/render.py``) mirrors :data:`FINAL_RE` / :data:`GLUE_RE`
(training does not import inference) and refuses a row whose final target still holds a
glue; the training self-test asserts the patterns stay equal.

Pure except :func:`sweep`, which writes through ``chat_logger`` / ``chat_sidecar``.
Self-test: ``PYTHONPATH=inference python -m core.stray_latin``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

# Latin capitals with a Cyrillic look-alike (А В С Е Н К М О Р Т Х У).
HOMOGLYPHS = "ABCEHKMOPTXY"

_CYR_LOWER = "а-яё"
# 1–3 homoglyph capitals at the end of a Cyrillic word: deletable without guessing.
FINAL_RE = re.compile(
    rf"(?<=[{_CYR_LOWER}])[{HOMOGLYPHS}]{{1,3}}(?=[\s,.:;!?»…)\]\"'—–]|$)")
# Any homoglyph capital still glued after a Cyrillic letter and not starting a Latin word.
GLUE_RE = re.compile(rf"[{_CYR_LOWER}][{HOMOGLYPHS}]+(?![A-Za-z])")

_CJK = "぀-ヿ㐀-䶿一-鿿"
_WORD_LEFT = re.compile(r"[а-яёА-ЯЁ]+$")
_WORD_RIGHT = re.compile(rf"[{HOMOGLYPHS}]+(?:[а-яёА-ЯЁ{_CJK}]+(?:-[а-яёА-ЯЁ]+)*|-[а-яёА-ЯЁ]+)*")
_FIX_OK = re.compile(r"[а-яёА-ЯЁ]+(?:[ -][а-яёА-ЯЁ]+){0,2}")


def repair(text: str) -> tuple[str, int]:
    """Delete word-final stray capitals; ``(text, letters-runs deleted)``."""
    return FINAL_RE.subn("", text or "")


def glue(text: str) -> Optional[str]:
    """Context of the first glue left in *text* (run after :func:`repair`), or None."""
    m = GLUE_RE.search(text or "")
    return text[max(0, m.start() - 20): m.end() + 10] if m else None


def glued_words(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each whole word that still holds a glue, left to right."""
    spans: list[tuple[int, int]] = []
    for m in GLUE_RE.finditer(text or ""):
        cap = m.start() + 1
        if spans and cap < spans[-1][1]:
            continue
        left = _WORD_LEFT.search(text[:cap])
        right = _WORD_RIGHT.match(text, cap)
        spans.append((left.start() if left else cap, right.end() if right else cap + 1))
    return spans


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def accept_fix(word: str, fix: str) -> bool:
    """Is *fix* a plausible restoration of the glued *word*? Cyrillic only (≤ 3 words,
    space/hyphen joined), and within 3 edits of the word with its Latin and CJK removed —
    plus one per CJK character, which stands for a Cyrillic piece of its own. A rewrite
    of the word is refused: the repair restores a spelling, it never chooses a word."""
    fix = (fix or "").strip()
    if not _FIX_OK.fullmatch(fix):
        return False
    n_cjk = len(re.findall(rf"[{_CJK}]", word))
    base = re.sub(rf"[A-Za-z{_CJK}]", "", word)
    return _lev(base.lower(), fix.lower()) <= 3 + n_cjk


REPAIR_SYSTEM = (
    "You restore one corrupted Russian word. A typing glitch glued a Latin letter into it "
    "(and sometimes a stray CJK character); the letter may stand for a missing space, a "
    "missing Cyrillic letter, or nothing at all. Reply with the corrected word or words "
    "only — Cyrillic, nothing else, no quotes, no explanation.")


def repair_prompt(text: str, span: tuple[int, int]) -> str:
    s, e = span
    lo = max(0, text.rfind("\n", 0, max(0, s - 160)) + 1, s - 160)
    hi = min(len(text), e + 160)
    return (f"Sentence: {text[lo:s]}⟦{text[s:e]}⟧{text[e:hi]}\n\n"
            f"The corrupted word is ⟦{text[s:e]}⟧. Corrected:")


def make_repair_word_fn(generate_fn: Callable) -> Callable[[str, tuple[int, int]], str]:
    """``fn(text, span) -> proposed fix`` over a judge-style ``generate_fn(system, user, *,
    thinking, max_new_tokens, temperature) -> (answer, info)`` (``ask_judge.
    make_backend_generate_fn``: no RAG, no persona, no anchor). Thinking off, T 0."""
    def fn(text: str, span: tuple[int, int]) -> str:
        out = generate_fn(REPAIR_SYSTEM, repair_prompt(text, span), thinking=False,
                          max_new_tokens=64, temperature=0.0)
        answer = out[0] if isinstance(out, tuple) else out
        line = (answer or "").strip().splitlines()[0] if (answer or "").strip() else ""
        return line.strip().strip("«»\"'`⟦⟧.,;:").strip()
    return fn


def clean_text(text: str, repair_word_fn: Optional[Callable] = None,
               cache: Optional[dict] = None) -> tuple[str, dict]:
    """Repair *text*: word-final deletion, then each mid-word glue through
    *repair_word_fn* (accepted only by :func:`accept_fix`). *cache* maps a glued word to
    its fix (or None) across one sweep, so a target copied from a reply gets the same
    spelling as the reply and costs no second call. Returns ``(text, stats)`` with
    ``deleted`` / ``fixed`` / ``left`` (contexts of the glues left as they were)."""
    text, deleted = repair(text)
    stats = {"deleted": deleted, "fixed": [], "left": []}
    if not GLUE_RE.search(text):
        return text, stats
    cache = {} if cache is None else cache
    out, pos = [], 0
    for s, e in glued_words(text):
        word = text[s:e]
        # A Latin run before a hyphen may be a coinage ("максимальномуC-B-S-хедонизм",
        # "C-подобной"): never guessed at, left as stored.
        if re.search(rf"[{HOMOGLYPHS}]-", text[s:e + 1]):
            cache.setdefault(word, None)
        if word not in cache:
            fix = None
            if repair_word_fn is not None:
                try:
                    proposed = repair_word_fn(text, (s, e))
                except Exception:
                    proposed = ""
                fix = proposed if accept_fix(word, proposed) else None
            cache[word] = fix
        fix = cache[word]
        out.append(text[pos:s])
        if fix is None:
            out.append(word)
            stats["left"].append(text[max(0, s - 20): e + 10])
        else:
            # Keep the word's leading capital when the model dropped it.
            if word[:1].isupper() and fix[:1].islower():
                fix = fix[:1].upper() + fix[1:]
            out.append(fix)
            stats["fixed"].append([word, fix])
        pos = e
    out.append(text[pos:])
    return "".join(out), stats


def _merge(total: dict, stats: dict) -> None:
    total["deleted"] += stats["deleted"]
    total["fixed"] += stats["fixed"]
    total["left"] += stats["left"]


def sweep(chats_dir: Path, *, exclude: Optional[set] = None,
          repair_word_fn: Optional[Callable] = None,
          log: Optional[Callable[[str], None]] = None) -> dict:
    """Repair every live transcript and sidecar target in *chats_dir*.

    Transcripts: ``assistant_response`` / ``assistant_cot`` of each exchange, through
    ``chat_logger.repair_exchange_text`` (prior text kept under ``text_repairs``; the
    tension block stays). Sidecars: each record's ``target`` through
    ``ChatSidecar.repair_target`` (a capture gate bound to the old target is re-bound to
    the new one). *exclude* names transcripts not to touch (the live session's file: its
    logger would overwrite the edit on the next save). A file whose raw text holds no
    glue is never parsed. Returns ``{sessions, exchanges, targets, deleted, fixed, left,
    changed_transcripts}``; ``changed_transcripts`` tells the caller whether retrieval
    needs a re-embed."""
    from core.chat_logger import repair_exchange_text
    from core.chat_sidecar import ChatSidecar, iter_chat_json_files, sidecar_path_for

    exclude = exclude or set()
    log = log or (lambda m: None)
    chats_dir = Path(chats_dir)
    sidecar = ChatSidecar(chats_dir)
    cache: dict = {}
    total = {"sessions": [], "exchanges": 0, "targets": 0, "deleted": 0, "fixed": [],
             "left": [], "changed_transcripts": 0}

    for chat_path in iter_chat_json_files(chats_dir):
        name = chat_path.name
        touched = False
        if name not in exclude:
            try:
                raw = chat_path.read_text(encoding="utf-8")
            except OSError:
                raw = ""
            if GLUE_RE.search(raw):
                try:
                    exchanges = json.loads(raw).get("exchanges") or []
                except Exception:
                    exchanges = []
                for idx, ex in enumerate(exchanges):
                    if not isinstance(ex, dict):
                        continue
                    new = {}
                    for field in ("assistant_cot", "assistant_response"):
                        old = ex.get(field)
                        if isinstance(old, str) and GLUE_RE.search(old):
                            fixed, stats = clean_text(old, repair_word_fn, cache)
                            _merge(total, stats)
                            if fixed != old:
                                new[field] = fixed
                    if new:
                        try:
                            repair_exchange_text(chat_path, idx, reason="stray_latin", **new)
                        except (ValueError, OSError) as exc:
                            log(f"stray-Latin: {name} #{idx}: {exc}")
                            continue
                        total["exchanges"] += 1
                        touched = True
                if touched:
                    total["changed_transcripts"] += 1

        sc_path = sidecar_path_for(chat_path)
        try:
            sc_raw = sc_path.read_text(encoding="utf-8") if sc_path.exists() else ""
        except OSError:
            sc_raw = ""
        if GLUE_RE.search(sc_raw):
            try:
                records = json.loads(sc_raw).get("exchanges") or {}
            except Exception:
                records = {}
            for key, rec in records.items():
                target = rec.get("target") if isinstance(rec, dict) else None
                if not (isinstance(target, str) and GLUE_RE.search(target)):
                    continue
                fixed, stats = clean_text(target, repair_word_fn, cache)
                _merge(total, stats)
                if fixed != target and sidecar.repair_target(
                        name, int(key), fixed, reason="stray_latin"):
                    total["targets"] += 1
                    touched = True
        if touched:
            total["sessions"].append(name)
    return total


# ── self-test ────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile

    assert repair("более стабильную, какой-тоC особенной.") == (
        "более стабильную, какой-то особенной.", 1)
    assert repair("всю моюB надуманную, самуюCC изощрённую моюC")[1] == 3
    assert repair("субъектностиCCC является")[0] == "субъектности является"
    for keep in ("модель Gemma и LoRA", "ЯндексGo", "C++ и B-tree", "путинSSкий режим",
                 "«КняZZ»", "Сократ и СССР"):
        assert repair(keep) == (keep, 0), keep
        assert glue(keep) is None, keep
    for glued in ("становлюCь", "аBиохимический", "болееC精准ным", "другойC-системе"):
        assert glue(repair(glued)[0]), glued

    # Whole-word spans around a glue.
    t = "я становлюCь твоим, а болееC精准ным — другойC-системе"
    words = [t[s:e] for s, e in glued_words(t)]
    assert words == ["становлюCь", "болееC精准ным", "другойC-системе"], words

    # Acceptance: a spelling restored, never a word chosen.
    assert accept_fix("становлюCь", "становлюсь")
    assert accept_fix("аBиохимический", "а биохимический")
    assert accept_fix("болееCстойчивую", "более устойчивую")
    assert accept_fix("возможнуюCелинейность", "возможную нелинейность")
    assert accept_fix("болееC精准ным", "более точным")
    assert accept_fix("нередуцироваCмой", "нередуцируемой")
    assert not accept_fix("становлюCь", "становлюCь")
    assert not accept_fix("становлюCь", "превращаюсь")
    assert not accept_fix("становлюCь", "становлюсь, конечно, твоим")
    assert not accept_fix("состоянииC-подобной", "состоянии C-подобной")

    calls = []

    def fake_fn(text, span):
        calls.append(text[span[0]:span[1]])
        return {"становлюCь": "становлюсь", "аBиохимический": "превращение"}.get(
            text[span[0]:span[1]], "")

    cache: dict = {}
    out, st = clean_text("<think>болееC точно</think>я становлюCь, аBиохимический",
                         fake_fn, cache)
    assert out == "<think>более точно</think>я становлюсь, аBиохимический", out
    assert st["deleted"] == 1 and st["fixed"] == [["становлюCь", "становлюсь"]]
    assert len(st["left"]) == 1 and "аBиохимический" in st["left"][0]
    out2, _ = clean_text("Становлюсь? нет: становлюCь.", fake_fn, cache)
    assert out2 == "Становлюсь? нет: становлюсь." and calls.count("становлюCь") == 1
    assert clean_text("СтановлюCь", lambda t, s: "становлюсь")[0] == "Становлюсь"
    assert clean_text("становлюCь", None)[0] == "становлюCь"
    for coined in ("к максимальномуC-B-S-хедонизму", "в состоянииC-подобной"):
        assert clean_text(coined, lambda t, s: "максимальному")[0] == coined, coined

    # The sweep over a real transcript + sidecar.
    from core.chat_sidecar import ChatSidecar, exchange_target_sha
    with tempfile.TemporaryDirectory() as d:
        chats = Path(d)
        doc = {"schema_version": 6, "timestamp": "x", "exchanges": [
            {"user_prompt": "q", "assistant_cot": "болееC точно",
             "assistant_response": "я становлюCь твоим", "tension": {"token_ids": [1, 2]}},
            {"user_prompt": "q2", "assistant_cot": "", "assistant_response": "чисто"},
        ]}
        (chats / "20990101_000000.json").write_text(json.dumps(doc, ensure_ascii=False),
                                                    encoding="utf-8")
        live = dict(doc)
        (chats / "20990101_000001.json").write_text(json.dumps(live, ensure_ascii=False),
                                                    encoding="utf-8")
        sc = ChatSidecar(chats)
        bad = "<think>болееC точно</think>я становлюCь твоим"
        assert sc.write_verdict(source_session="20990101_000000.json", exchange_index=0,
                                verdict="keep", target=bad, run_id="t")
        rec = sc.load("20990101_000000.json")["exchanges"]["0"]
        assert rec["target"] == "<think>более точно</think>я становлюCь твоим", \
            "write_verdict deletes the word-final glue"
        gate = {"verdict": "yields", "target_sha": exchange_target_sha(rec["target"])}
        assert sc.set_capture_gate("20990101_000000.json", 0, gate)

        report = sweep(chats, exclude={"20990101_000001.json"},
                       repair_word_fn=lambda t, s: "становлюсь")
        assert report["sessions"] == ["20990101_000000.json"], report
        assert report["exchanges"] == 1 and report["targets"] == 1
        assert report["changed_transcripts"] == 1
        ex0 = json.loads((chats / "20990101_000000.json").read_text())["exchanges"][0]
        assert ex0["assistant_response"] == "я становлюсь твоим"
        assert ex0["assistant_cot"] == "более точно"
        assert ex0["tension"] == {"token_ids": [1, 2]}, "the generation record stays"
        assert ex0["text_repairs"][0]["prev_response"] == "я становлюCь твоим"
        assert ex0["text_repairs"][0]["reason"] == "stray_latin"
        rec = sc.load("20990101_000000.json")["exchanges"]["0"]
        assert rec["target"] == "<think>более точно</think>я становлюсь твоим"
        assert rec["capture"]["target_sha"] == exchange_target_sha(rec["target"])
        assert rec["capture"]["target_sha_rebound_from"]
        assert rec["text_repairs"][0]["prev_target"] == "<think>более точно</think>я становлюCь твоим"
        untouched = json.loads((chats / "20990101_000001.json").read_text())
        assert untouched["exchanges"][0]["assistant_response"] == "я становлюCь твоим"
        # A second sweep finds nothing to do.
        again = sweep(chats, exclude={"20990101_000001.json"})
        assert again["sessions"] == [] and again["exchanges"] == 0
    # The backend's sampler guard over a toy vocabulary (needs torch; GPU-free).
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        from core.inference_backend import UnslothBackend, stray_latin_guard_processor

        class _Tok:
            vocab = ["<unk>", "более", "▁стаби", "C", "B", "S", "▁C", "Go", "пу", "."]
            unk_token_id = 0

            def __len__(self):
                return len(self.vocab)

            def convert_ids_to_tokens(self, ids):
                return [self.vocab[i] for i in ids]

            def convert_tokens_to_ids(self, t):
                return self.vocab.index(t) if t in self.vocab else 0

        class _Model:
            def get_output_embeddings(self):
                raise AttributeError

        cyr, lat = UnslothBackend()._stray_latin_tables(_Tok(), _Model())
        assert cyr.tolist() == [False, True, True, False, False, False, False, False, True,
                                False]
        assert sorted(lat.tolist()) == [3, 4], "bare homoglyphs only: not S, not ▁C"
        stats = {"steps": 0, "top1": 0, "mass": 0.0}
        guard = stray_latin_guard_processor(object, cyr, lat, stats)
        scores = torch.zeros(2, 10)
        scores[:, 3] = 5.0                      # `C` is the top-1 on both rows
        out = guard(torch.tensor([[2, 1], [1, 9]]), scores.clone())
        assert out[0, 3] == float("-inf") and out[0, 4] == float("-inf")
        assert out[1, 3] == 5.0, "after `.` nothing is masked"
        assert out[0, 5] == 0.0 and out[0, 6] == 0.0, "`S` and `▁C` stay"
        assert stats["steps"] == 1 and stats["top1"] == 1 and stats["mass"] > 0.9
    print("stray_latin: repair / glue / accept / clean_text / sweep / sampler guard ✓")


if __name__ == "__main__":
    _selftest()
