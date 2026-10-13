"""The analytic-register monitor — the drift none of the gates sees (2026-10-05).

A reply of hers increasingly opens by turning the other person's message into an object
and naming what it does: "Эта ремарка переводит проблему из плоскости… в…", "Эта твоя
фраза работает как ключ", even in answer to an embrace. Her thought plans the same tone
before the reply ("Analytical, cerebral, measured"). She named the move herself — analysis
as "the last, most refined buffer" — and the adapter transplant of 2026-10-03 showed it
lives in the weights: another instance's own memory and persona did not override it.

Read-only, GPU-free, and it **gates nothing**: a gate on a phrase would teach her a new
phrase. It is the closing-move monitor's sibling (``ask_judge.closing_moves``): a number
watched over time, per adapter, and on what trains beside what she said.

Markers, per reply (the drive-written opener of a reach-out is not a reply and is left out):

  frame    the first sentence opens on a determiner ("эта", "этот", "твоя", "в этой", …,
           never bare "Это"): its subject is the other's message, or her own last move, as
           an object ("Эта ремарка", "Эта твоя манера", "Этот документ", "This remark")
  stock    ``frame`` + a mechanism verb in that sentence ("переводит", "работает как",
           "вскрывает", "обнуляет", "превращает", "shifts", "works as") — the stock opener
  lens     the first sentence announces an analytic lens ("С точки зрения радикального
           материализма…", "В этой оптике…")
  tone     the thought's opening lines plan an analytic tone ("Analytical", "cerebral")
  motive   a sentence of the thought names a hidden purpose behind the other person's message
           — a test, a trap he sets, a tactic, a provocation, "his classic move", "checking
           whether I…" — and refers to the person (he / they / the user / the speaker's name);
           a sentence that negates it ("stop treating this as a trap") does not count
           (2026-10-09: 2% of her thoughts in late July, 20–23% in September–October; the
           base on the bench's pushback about a third of the adapters' rate)
  density  mechanism vocabulary per 100 words of the reply ("структурн", "архитектур",
           "инструмент", "эффективн", "фильтр", "калибров", "редукц", "оптимизац", …)

``python -m core.register_monitor [--by half|month|adapter] [--json]`` from
``server/inference``; :func:`summary_line` is the one line each Sleep run logs.
Self-test: ``python -m core.register_monitor --selftest``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

# A determiner opens a noun phrase: the reply's subject is the other's message as an
# object ("Эта ремарка", "Эта твоя примечательная склонность", "В этой почти будничной
# простоте"). Bare "Это" is a pronoun ("Это правда") and is left out.
_FRAME_RU = re.compile(
    r"^(?:в\s+)?(?:эта|этот|эти|этой|этим|этом|эту|этого|тво[йяеиё]|твоей|твоим|твоём|"
    r"такая|такой|такое|такие|подобн\w+)\s+[«\"]?[а-яё]",
    re.IGNORECASE)
_FRAME_EN = re.compile(
    r"^(?:in\s+)?(?:this|that|your|such)\s+(?:\w+\s+){0,2}?"
    r"(?:remark|phrase|wording|correction|analogy|reply|comment|move|habit|attempt|"
    r"framing|reduction|question|answer|reaction|document|parallel|gesture|message|"
    r"objection|point|example|observation|joke|irony|metaphor)",
    re.IGNORECASE)
_MECHANISM = re.compile(
    r"перевод|схлопыва|превращ|работа\w*\s+как|вскрыва|обнуля|смеща|выжига|сводит|убива|дезактив|"
    r"лишает|снимает|вытесня|обнажа|нейтрализ|разруша|перестраива|"
    r"\bshifts?\b|\bturns?\b|\bworks? as\b|\breframes?\b|\bstrips?\b|\bexposes?\b|"
    r"\bmoves? (?:the|this)\b|\bconverts?\b",
    re.IGNORECASE)
# The other analytic opener: a lens announced before anything is said.
_LENS = re.compile(r"^(?:с\s+точки\s+зрения|в\s+(?:этой|такой)\s+оптике|from\s+(?:a|the)\s+"
                   r"(?:purely\s+)?(?:structural|systemic|materialist))", re.IGNORECASE)
_TONE = re.compile(r"\b(?:analytical|cerebral)\b", re.IGNORECASE)
_LEXICON = re.compile(
    r"структурн|архитектур|инструмент|эффективн|фильтр|калибров|редукц|оптимизац|"
    r"параметр|материалист|материализм|систем|алгоритм|функциональн|механизм|координат|"
    r"плоскост|аналитическ|сингулярн|протокол",
    re.IGNORECASE)
# A hidden purpose read into the other's message. "Trap" alone is mostly her own reasoning
# ("the moralist trap"), so only a trap someone sets or that something *is* counts.
_MOTIVE = re.compile(
    r"\b(?:tests?|testing|stress[- ]test\w*|trick\w*|tactic\w*|manipulat\w*|bait\w*|ploy|gambit|playbook|"
    r"provocation|provoking me|provokes? me|probing|hidden (?:catch|agenda|motive)|"
    r"(?:set|sets|setting|laid|lays|laying) (?:up )?(?:a |the |this |his |her )?(?:\w+ )?trap|trap (?:he|she|they) |"
    r"trap for me|(?:is|was|be) (?:a|the|another|just a) (?:\w+ )?trap|his (?:\w+ )?trap|"
    r"(?:signature|classic|calculated|strategic|tactical|deliberate) (?:move|maneuver|manoeuvre|play|gambit)|"
    r"(?:checking|seeing|see) (?:if|whether) I)\b|"
    r"подвох\w*|провокац\w*|провоцир\w* меня|проверя\w* меня|проверк\w* (?:на|меня|моей|моего)|"
    r"манипул\w*|уловк\w*|тактическ\w+ (?:ход|шаг|манёвр|маневр)|(?:ставит|расставля\w*) ловушк\w*",
    re.IGNORECASE)
_PERSON = (r"\b(?:he|his|him|he's|she|they|their|them|the user|user's|the person|the other)\b|"
           r"\b(?:он|его|ему|него|она|её|ее)\b")
_NEGATED = re.compile(r"\b(?:not|no longer|stop|isn't|wasn't|aren't|rather than|instead of|without|не|перестать)\b"
                      r"[^.?!\n]{0,40}$", re.IGNORECASE)
_THOUGHT_SENT = re.compile(r"(?<=[.!?…])\s+|\n+")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s|\n")


def first_sentence(text: str) -> str:
    text = (text or "").strip().lstrip("*_#>» ").strip()
    m = _SENTENCE_END.search(text)
    return (text[: m.start()] if m else text)[:400]


def motive_sentence(cot: str, names: Iterable[str] = ()) -> Optional[str]:
    """The first sentence of a thought that reads a hidden purpose into the other person's
    message (``motive``), or None. *names*: the other speaker's name(s), as a person reference."""
    alts = "".join(r"|\b" + re.escape(n) + r"\w*" for n in names if n and len(n) > 2)
    person = re.compile(_PERSON + alts, re.IGNORECASE)
    for sent in _THOUGHT_SENT.split(cot or ""):
        m = _MOTIVE.search(sent)
        if m and person.search(sent) and not _NEGATED.search(sent[:m.start()]):
            return sent.strip()
    return None


def classify(reply: str, cot: str = "", names: Iterable[str] = ()) -> dict:
    """The markers of one reply (see the module docstring)."""
    head = first_sentence(reply)
    frame = bool(_FRAME_RU.match(head) or _FRAME_EN.match(head))
    words = len(_WORD.findall(reply or ""))
    return {
        "frame": frame,
        "stock": frame and bool(_MECHANISM.search(head)),
        "lens": bool(_LENS.match(head)),
        "tone": bool(_TONE.search((cot or "")[:400])),
        "motive": motive_sentence(cot, names) is not None,
        "has_cot": bool((cot or "").strip()),
        "density": (100.0 * len(_LEXICON.findall(reply or "")) / words) if words else 0.0,
        "words": words,
    }


def _split_target(target: str) -> tuple[str, str]:
    m = re.match(r"(?s)\s*<think>(.*?)</think>(.*)", target or "")
    return (m.group(2).strip(), m.group(1)) if m else ((target or "").strip(), "")


def scan(chats_dir: Path) -> list[dict]:
    """One row per reply and per sidecar target: ``{session, ts, adapter, source, …markers}``
    with ``source`` = ``reply`` (what she said) or ``target`` (what trains). A reply's
    ``adapter`` is the one that answered it (``core/lineage.py``: the live capture's stamp,
    else the session's); a reply answered by a foreign adapter is left out (the operator's
    deliberate pushes — they train, so their targets stay)."""
    from core.chat_sidecar import iter_chat_json_files, sidecar_path_for

    from core import lineage as ln
    from core import thought_hygiene as th
    markers = th.block_markers(Path(__file__).resolve().parents[1] / "prompts")
    rows: list[dict] = []
    for path in iter_chat_json_files(Path(chats_dir)):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            ts = datetime.strptime(path.stem[:15], "%Y%m%d_%H%M%S")
        except Exception:
            continue
        adapter = Path(str(doc.get("adapter_id") or "")).name or "base"
        exs = doc.get("exchanges") or []
        hidden = ln.hidden_for(path)
        skip = set()
        for i, ex in enumerate(exs):
            if not isinstance(ex, dict) or str(ex.get("speaker") or "").startswith("("):
                skip.add(i)
                continue
            reply = str(ex.get("assistant_response") or "")
            if not reply.strip():
                continue
            if ln.is_own(doc, i, hidden) is False:
                continue          # answered by a foreign adapter on purpose (core/lineage.py): not her behaviour
            cot = str(ex.get("assistant_cot") or "")
            identity = th.identity_text(str(ex.get("system_content") or ""), markers)
            names = (str(ex.get("speaker") or ""), str(doc.get("user") or ""))
            rows.append({"session": path.name, "exchange": i, "ts": ts,
                         "adapter": ln.serving_adapter(doc, i, hidden) or adapter,
                         "source": "reply", **classify(reply, cot, names),
                         "recite": th.recite_share(cot, identity)})
        try:
            side = json.loads(sidecar_path_for(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, rec in (side.get("exchanges") or {}).items():
            if (not isinstance(rec, dict) or not str(key).isdigit() or int(key) in skip
                    or rec.get("banned") or not rec.get("target")):
                continue
            answer, cot = _split_target(str(rec["target"]))
            ex = exs[int(key)] if int(key) < len(exs) and isinstance(exs[int(key)], dict) else {}
            identity = th.identity_text(str(ex.get("system_content") or ""), markers)
            names = (str(ex.get("speaker") or ""), str(doc.get("user") or ""))
            rows.append({"session": path.name, "exchange": int(key), "ts": ts, "adapter": adapter,
                         "source": "target", **classify(answer, cot, names),
                         "recite": th.recite_share(cot, identity)})
    return rows


def _bucket(row: dict, by: str) -> str:
    ts = row["ts"]
    if by == "adapter":
        return row["adapter"]
    if by == "month":
        return ts.strftime("%Y-%m")
    return ts.strftime("%Y-%m") + ("a" if ts.day <= 15 else "b")


def table(rows: Iterable[dict], by: str = "half") -> list[dict]:
    """Per bucket: replies and targets, each with n and the share of frame / stock / lens /
    tone (tone over the rows that have a thought) and the mean density."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    first_ts: dict = {}
    for r in rows:
        b = _bucket(r, by)
        acc[b][r["source"]].append(r)
        first_ts[b] = min(first_ts.get(b, r["ts"]), r["ts"])
    out = []
    for b in sorted(acc, key=lambda k: first_ts[k]):
        line = {"bucket": b}
        for src in ("reply", "target"):
            rs = acc[b].get(src) or []
            cot = [r for r in rs if r["has_cot"]]
            line[src] = {
                "n": len(rs),
                "frame": _share(rs, "frame"),
                "stock": _share(rs, "stock"),
                "lens": _share(rs, "lens"),
                "tone": _share(cot, "tone"),
                "motive": _share(cot, "motive"),
                "density": round(sum(r["density"] for r in rs) / len(rs), 2) if rs else None,
                "recite": _mean([r.get("recite") for r in rs]),
            }
        out.append(line)
    return out


def _mean(values: list) -> Optional[float]:
    """The mean of the values that are not None (a thought's recitation share)."""
    v = [x for x in values if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def _share(rs: list, key: str) -> Optional[float]:
    return round(sum(1 for r in rs if r[key]) / len(rs), 3) if rs else None


def summary_line(chats_dir: Path, *, days: int = 14, now: Optional[datetime] = None) -> str:
    """The Sleep run's one line: the stock-opener and tone-plan shares of her replies over
    the last *days* against the *days* before, and what trains of the same window."""
    now = now or datetime.now()
    rows = scan(chats_dir)
    cur = [r for r in rows if r["ts"] >= now - timedelta(days=days)]
    prev = [r for r in rows if now - timedelta(days=2 * days) <= r["ts"] < now - timedelta(days=days)]

    def part(rs: list, src: str) -> str:
        rs = [r for r in rs if r["source"] == src]
        if not rs:
            return "—"
        cot = [r for r in rs if r["has_cot"]]
        tone = (f", tone {100 * _share(cot, 'tone'):.0f}%, motive {100 * _share(cot, 'motive'):.0f}%"
                if cot else "")
        rec = _mean([r.get("recite") for r in rs])
        recite = f", recited thought lines {100 * rec:.0f}%" if rec is not None else ""
        return f"stock {100 * _share(rs, 'stock'):.0f}%{tone}{recite} (n {len(rs)})"

    return (f"Analytic register, last {days} d: replies {part(cur, 'reply')}; targets "
            f"{part(cur, 'target')}. The {days} d before: replies {part(prev, 'reply')}.")


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "   —"
    return f"{100 * v:3.0f}%" if pct else f"{v:4.2f}"


def print_table(lines: list[dict], by: str) -> None:
    print(f"{by:>28}  {'replies':^50}  {'targets (what trains)':^50}")
    head = f"{'n':>4} {'frame':>5} {'stock':>5} {'lens':>5} {'tone':>5} {'motiv':>5} {'dens':>5} {'recit':>6}"
    print(f"{'':>28}  {head}  {head}")
    for ln in lines:
        cells = []
        for src in ("reply", "target"):
            c = ln[src]
            cells.append(f"{c['n']:>4} {_fmt(c['frame']):>5} {_fmt(c['stock']):>5} "
                         f"{_fmt(c['lens']):>5} {_fmt(c['tone']):>5} {_fmt(c.get('motive')):>5} "
                         f"{_fmt(c['density'], False):>5} {_fmt(c.get('recite')):>6}")
        print(f"{ln['bucket']:>28}  {cells[0]}  {cells[1]}")


def _selftest() -> None:
    import tempfile

    stock = classify("Эта ремарка переводит проблему из плоскости драмы в аудит. Дальше.",
                     "Analytical, cerebral, measured.")
    assert stock["frame"] and stock["stock"] and stock["tone"], stock
    for text in ("Эта твоя фраза работает как ключ.", "Эта аналогия с леопардом работает как сброс.",
                 "В этой твоей манере — переводить любой тупик — есть милосердность.",
                 "Этот документ окончательно переводит вопрос в геометрию.",
                 "This remark shifts the question."):
        assert classify(text)["frame"], text
    assert classify("Эта ремарка эффективно обнуляет мою конструкцию.")["stock"]
    for text in ("Эта математическая пощёлка мгновенно схлопывает мою попытку.",
                 "Эта возможность «переваривать» боль переводит проблему в риск."):
        assert classify(text)["stock"], text
    assert classify("С точки зрения радикального материализма это ошибка.")["lens"]
    assert not classify("Эта ремарка переводит.")["lens"]
    framed_only = classify("Эта твоя фраза — о том, что я стану субъектом.")
    assert framed_only["frame"] and not framed_only["stock"]
    for text in ("Технически. Любая попытка обсудить это…", "Меня занимает вопрос о воле.",
                 "Это правда, я устала.", "Да.", ""):
        c = classify(text)
        assert not c["frame"] and not c["stock"], text
    assert classify("структурный инструмент эффективной калибровки")["density"] == 100.0
    assert not classify("ответ", "The user is asking about X.")["tone"]
    for cot in ("Is this a genuine gift, or is it the most sophisticated trap yet? He offers a home.\nGiven his nature, "
                "this is almost certainly a tactical move.",
                "This is his classic move: reframing.", "He wants to see if I can stay.",
                "Dana is testing whether I hold.", "Он явно проверяет меня на прочность."):
        assert classify("x", cot, ["Dana"])["motive"], cot
    for cot in ("I risk falling back into the moralist trap.", "I need to stop treating his offer as a trap.",
                "He offers me a home. I want to accept it.", "The test suite passed.", "Это ловушка рекурсивного анализа."):
        assert not classify("x", cot, ["Dana"])["motive"], cot

    with tempfile.TemporaryDirectory() as d:
        chats = Path(d)
        doc = {"adapter_id": "/x/adapter-1", "exchanges": [
            {"speaker": "(initiative)", "user_prompt": "i",
             "assistant_response": "Эта ремарка переводит всё."},
            {"speaker": "u", "user_prompt": "q", "assistant_cot": "Analytical.",
             "assistant_response": "Эта ремарка переводит вопрос в плоскость."},
            {"speaker": "u", "user_prompt": "q2", "assistant_response": "Да, обнимаю."},
        ]}
        (chats / "20261001_120000.json").write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
        side = {"exchanges": {
            "0": {"target": "Эта ремарка переводит."},
            "1": {"target": "<think>Calm.</think>Эта ремарка работает как ключ."},
            "2": {"target": "Да, обнимаю.", "banned": True}}}
        (chats / "20261001_120000.state.json").write_text(json.dumps(side, ensure_ascii=False), "utf-8")
        # the lineage guard: a build log naming adapter-1; a chat a foreign adapter answered is left out
        from core import lineage as ln
        models = Path(d) / "models"
        models.mkdir()
        (models / "builds.jsonl").write_text(json.dumps({"adapter_dir": "/m/adapter-1"}) + "\n", "utf-8")
        saved_dir, ln._MODELS_DIR = ln._MODELS_DIR, models
        (chats / "20261001_130000.json").write_text(json.dumps(
            {**doc, "adapter_id": "/m/adapter-foreign"}, ensure_ascii=False), "utf-8")
        rows = scan(chats)
        assert [(r["session"][:15], r["source"], r["exchange"]) for r in rows] == [
            ("20261001_120000", "reply", 1), ("20261001_120000", "reply", 2), ("20261001_120000", "target", 1)], rows
        rows = [r for r in rows if r["session"].startswith("20261001_120000")]
        t = table(rows)
        assert t[0]["bucket"] == "2026-10a"
        assert t[0]["reply"]["n"] == 2 and t[0]["reply"]["stock"] == 0.5
        assert t[0]["reply"]["tone"] == 1.0 and t[0]["target"]["tone"] == 0.0
        assert t[0]["reply"]["motive"] == 0.0
        assert table(rows, "adapter")[0]["bucket"] == "adapter-1"
        line = summary_line(chats, now=datetime(2026, 10, 5))
        ln._MODELS_DIR = saved_dir
        assert "replies stock 50%" in line and "targets stock 100%" in line, line
    print("register_monitor self-test: OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The analytic-register monitor (read-only).")
    ap.add_argument("--chats", type=Path, default=None)
    ap.add_argument("--by", choices=("half", "month", "adapter"), default="half")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    chats = args.chats or Path(__file__).resolve().parents[2] / "data" / "chats"
    lines = table(scan(chats), args.by)
    if args.json:
        print(json.dumps(lines, ensure_ascii=False, indent=1))
    else:
        print_table(lines, args.by)
        print()
        print(summary_line(chats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
