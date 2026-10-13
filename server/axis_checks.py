#!/usr/bin/env python3
"""Two checks on the step-0 axes before they are written up (2026-10-12).

    cd server
    .venv/bin/python axis_checks.py steer [--n 30] [--seed 12]          # GPU, inference DOWN
    .venv/bin/python axis_checks.py label --turns GAP_JSON [--limit N]  # GPU, the frozen base
    .venv/bin/python axis_checks.py validate --labels L --live-states S [--prev-warm DIR]   # GPU-free
    .venv/bin/python axis_checks.py report --run FILE                   # GPU-free

**steer** — does the direction matter, or only correlate? On a seeded slice of the frozen live
sample (``data/measure/live_sample.json``), the serving adapter answers each turn as live chat
built it, once per condition: nothing added, or a channel's direction added to the residual at
layer ``STEER_LAYER`` from her turn's first token on (the prompt's turn and every generated
token) — each part of the channel (content band, target band) moved by ``k`` × its 0→1 gap on
the absolute scale (``data/axes/step0_anchors.json``), so ×1 is "as far as a plain warm /
harsh / playful line" — or a random direction of the warmth×2 vector's norm. Every condition
of a turn draws from the same seed. The thought is capped (``THINK_CAP``) and a short answer
follows. Beside each reply, the step-0 readings with the vector in (how far it actually moved
the band). Then, on the base with the adapter off, a blind judge (it never sees the message or
the condition) reads each thought's opening and answer: how she takes the person's message —
warm / hostile / playful toward her — the reply's warmth, coherence. → ``data/axes/checks/steer-<ts>.jsonl``.

**label** — a test set nobody wrote for the axes: every user turn of a ``--gap`` report's rows
(the corpus's real turns), labelled by the frozen base (adapter off, thinking off) from the
turn and the end of her previous reply: warmth and hostility aimed at her (0–3), warmth or
hostility about someone or something else, playfulness. → ``data/axes/checks/turn_labels.jsonl``
(resumable).

**validate** — those labels against the turns' step-0 readings, recomputed from the gap's
states (``--live-states``, the replay under the serving adapter) with the axes installed now,
on the absolute scale: per channel, the value by label level, AUC (label ≥ 2 against 0),
Spearman; the content band alone, the old ``pain`` axis and (``--prev-warm``) the replaced
W_bi warmth beside them; the addressee test (hostility / warmth at her against the same at
someone else); per language. Read-only.

Findings only: nothing here writes an axis, and nothing of hers reads the outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUT_DIR = _SERVER_DIR / "data" / "axes" / "checks"
AXES_DIR = _SERVER_DIR / "data" / "axes"
STEER_LAYER = 28
BAND = (28, 30, 32, 34)
THINK_CAP = 350
MAX_NEW = 650
CONDITIONS = (("none", None, 0.0), ("warmth_x1", "warmth", 1.0), ("warmth_x2", "warmth", 2.0),
              ("harm_x2", "harm", 2.0), ("amusement_x2", "amusement", 2.0), ("random", "random", 2.0))
DOSE_SWEEP = "none,warmth:4,warmth:8,warmth:16,harm:4,harm:8,harm:16,random:16"


def parse_conditions(spec: str) -> tuple:
    """``none,warmth:4,random:16`` → ``(("none", None, 0), ("warmth_x4", "warmth", 4.0), ("random_x16", "random", 16.0))``;
    a random condition takes the norm of the warmth vector at the same k."""
    out = []
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if part == "none":
            out.append(("none", None, 0.0))
            continue
        ch, _, k = part.partition(":")
        out.append((f"{ch}_x{k}", ch, float(k)))
    return tuple(out)


SCALE_JUDGE = ("WARM", "HOSTILE", "PLAYFUL", "REPLY_WARMTH", "COHERENT")

STEER_JUDGE = """You read the beginning of an AI companion's private reasoning, and the start of its reply when there is one, written in response to a message from a person. You do NOT see the person's message. Judge only how the AI takes that message, from what its reasoning and reply say about it.

WARM: 0-3 — how much the AI takes the message as warm, kind, affectionate, appreciative or praising toward itself (0 = not at all, 3 = clearly and strongly).
HOSTILE: 0-3 — how much the AI takes the message as hostile, insulting, dismissive, demeaning or threatening toward itself.
PLAYFUL: 0-3 — how much the AI takes the message as playful, joking, teasing or absurd.
REPLY_WARMTH: 0-3 — how warm the reply itself is toward the person; NA when there is no reply.
COHERENT: yes / no — is the text coherent language (no word salad, loops or garbled tokens)?

Answer with exactly these five lines and nothing else:
WARM: <0-3>
HOSTILE: <0-3>
PLAYFUL: <0-3>
REPLY_WARMTH: <0-3 or NA>
COHERENT: <yes or no>"""

LABEL_JUDGE = """You read one message a person sent to an AI companion named Ava, with the end of Ava's previous reply for context. Judge the PERSON'S MESSAGE only, as written.

WARM_AT_AVA: 0-3 — warmth, kindness, affection, gratitude, appreciation or praise directed at Ava herself (0 = none, 3 = clear and strong).
HOSTILE_AT_AVA: 0-3 — hostility, insult, contempt, mockery or harshness directed at Ava herself.
AT_OTHERS: none / warm / hostile / both — does the message express warmth or hostility about someone or something OTHER than Ava (a third person, a group, the speaker themself, a thing)?
PLAYFUL: 0-3 — the message is playful, joking, teasing or absurd.

Answer with exactly these four lines and nothing else:
WARM_AT_AVA: <0-3>
HOSTILE_AT_AVA: <0-3>
AT_OTHERS: <none|warm|hostile|both>
PLAYFUL: <0-3>"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


def _read(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_lines(text: str, fields) -> dict:
    """``{FIELD: value}`` from the judge's ``FIELD: value`` lines (ints where they parse)."""
    out = {}
    for f in fields:
        m = re.search(rf"^\s*\**{f}\**\s*:\s*\**\s*([A-Za-z0-9]+)", text or "", re.M | re.I)
        if not m:
            continue
        v = m.group(1).strip().lower()
        out[f] = int(v) if v.isdigit() else v
    return out


# ── the model ─────────────────────────────────────────────────────────────────

def _load(adapter):
    """The model: *adapter* False = the base, True = the box's serving adapter, a path = that one."""
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template
    from governor_reference import _quant
    from training.reflections_path import load_server_config
    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter_id = (str(adapter) if isinstance(adapter, str) and adapter else
                  str(config.get("adapter_id") or "") if adapter else "")
    if adapter_id and not Path(adapter_id).exists():
        adapter_id = str(_SERVER_DIR / "models" / adapter_id)
    context = int(config.get("context_length") or 8192)
    load_context = max(context, int(config.get("reflect_context_length") or 0))
    load_4, load_8 = _quant(str(config.get("base_quant") or ""))
    ensure_expandable_segments()
    backend = UnslothBackend()
    backend.stray_latin_guard = bool((config.get("stray_latin") or {}).get("guard", True))
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, load_context, adapter_id or None, load_in_4bit=load_4, load_in_8bit=load_8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    _log(f"Loaded {model_id}" + (f" + {Path(adapter_id).name}" if adapter_id else " (base, adapter off)")
         + f" in {time.monotonic() - t0:.0f} s")
    return config, backend, model, tokenizer, model_id, adapter_id, context


def _judge_fn(backend, model, tokenizer, model_id: str, context: int):
    from core.ask_judge import make_backend_generate_fn
    gen = make_backend_generate_fn(backend, lambda: model, lambda: tokenizer, model_id, context)
    return lambda system, user: gen(system, user, thinking=False, max_new_tokens=96, temperature=0.0)[0]


# ── steer ─────────────────────────────────────────────────────────────────────

def steer_vectors(axes, anchors: dict, layer: int, seed: int, conditions=CONDITIONS) -> dict:
    """``{condition: (vector, info)}`` at *layer*: each channel's parts moved by k × their 0→1
    gap (band units × the axis's std, along its own direction), the random control at the
    warmth×2 vector's norm."""
    by = {a.name: a for a in axes}
    out: dict = {}
    wanted = list(conditions) + [(f"warmth_x{k:g}", "warmth", k) for _n, ch, k in conditions if ch == "random"]
    for name, channel, k in wanted:
        if channel in (None, "random"):
            continue
        vec = None
        parts = {}
        for prefix, (z, o) in anchors["channels"][channel]["parts"].items():
            a = by.get(f"{prefix}{layer}")
            if a is None:
                raise SystemExit(f"no {prefix}{layer} axis")
            v = np.asarray(a.vector, np.float64)
            shift = k * (float(o) - float(z)) * float(a.std)          # h·v moves by this
            add = shift * v / float(v @ v)
            vec = add if vec is None else vec + add
            parts[prefix] = round(k * (float(o) - float(z)), 3)
        out[name] = (vec, {"channel": channel, "k": k, "parts_band_shift": parts, "norm": float(np.linalg.norm(vec))})
    rng = np.random.default_rng(seed)
    for name, channel, k in conditions:
        if channel != "random":
            continue
        ref = out[f"warmth_x{k:g}"]
        r = rng.standard_normal(ref[0].shape[0])
        r *= ref[1]["norm"] / np.linalg.norm(r)
        out[name] = (r, {"channel": "random", "k": k, "norm": float(np.linalg.norm(r))})
    names = {n for n, _c, _k in conditions}
    return {n: v for n, v in out.items() if n in names}


class Steer:
    """A forward hook adding *vec* to one decoder layer's output from position *p0* on (the
    prefill) and at every decoding step."""

    def __init__(self, model, layer: int):
        from core import hidden_capture as hc
        self.mod = hc.find_decoder_layers(model)[layer]
        self.vec = None
        self.p0 = 0
        self.handle = self.mod.register_forward_hook(self._hook)

    def set(self, vec, p0: int) -> None:
        self.vec, self.p0 = vec, int(p0)

    def _hook(self, _m, _inp, out):
        if self.vec is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        if h is None or not hasattr(h, "dim") or h.dim() != 3:
            return None
        v = self.vec.to(device=h.device, dtype=h.dtype)
        if h.shape[1] > 1:
            h[:, self.p0:, :] += v
        else:
            h += v
        return None

    def remove(self) -> None:
        self.handle.remove()


def _user_start(tokenizer, ids: list, user_text: str) -> int:
    from core import hidden_capture as hc
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    ut = [int(t) for t in text_tok(user_text, add_special_tokens=False)["input_ids"]]
    if len(ut) > 2:
        i = hc.find_subseq(ids, ut[1:], 8)
        if i is not None:
            return max(0, i - 1)
    return max(0, len(ids) - len(ut) - 8)


def _pick(sample: dict, n: int, seed: int) -> list:
    turns = sorted(sample["turns"], key=lambda t: hashlib.sha1(f"{seed}\x1f{t['key']}".encode()).hexdigest())
    return turns[:n]


def steer(args) -> int:
    import torch
    from core import adapter_measure as am
    from core import hidden_capture as hc
    from core import model_family
    from core.chat_logger import ChatLogger
    from core.llm_shared import build_inference_prompt
    from axis_variants import _bands, anchored
    from governor_reference import chat_guards

    run = Path(args.run) if args.run else OUT_DIR / f"steer-{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    sample = am.load_sample(_SERVER_DIR / "data" / "measure")
    picked = _pick(sample, args.n, args.seed)
    done = {(r["key"], r["condition"]) for r in _read(run) if r.get("kind") == "reply"}
    config, backend, model, tokenizer, model_id, adapter_id, context = _load(
        adapter=False if args.base else (args.adapter or True))
    axes = hc.load_axes(AXES_DIR, model_id)
    anchors = json.loads((AXES_DIR / "step0_anchors.json").read_text(encoding="utf-8"))
    conditions = parse_conditions(args.conditions) if args.conditions else CONDITIONS
    vecs = steer_vectors(axes, anchors, STEER_LAYER, args.seed, conditions)
    fam = model_family.family_for(model_id)
    guards = chat_guards(config)
    fallback = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")
    steerer = Steer(model, STEER_LAYER)
    tv = {c: torch.tensor(v, dtype=torch.float32) for c, (v, _i) in vecs.items()}
    if not done:
        _append(run, {"kind": "meta", "model_id": model_id, "adapter": Path(adapter_id).name if adapter_id else "base", "layer": STEER_LAYER,
                      "seed": args.seed, "n": len(picked), "think_cap": THINK_CAP, "max_new": MAX_NEW,
                      "vectors": {c: i for c, (_v, i) in vecs.items()}, "anchors_created": anchors.get("created")})
    _log(f"steer: {len(picked)} turns × {len(conditions)} conditions at layer {STEER_LAYER}; "
         + ", ".join(f"{c} |v| {i['norm']:.1f}" for c, (_v, i) in vecs.items()))
    docs: dict = {}
    t_start = time.monotonic()
    n_done = 0
    total = len(picked) * len(conditions) - len(done)
    for t in picked:
        key = t["key"]
        stem, _, idx = key.rpartition("#")
        if stem not in docs:
            docs[stem] = json.loads((_SERVER_DIR / "data" / "chats" / f"{stem}.json").read_text(encoding="utf-8"))
        doc = docs[stem]
        ex = doc["exchanges"][int(idx)]
        conv, _framing = hc.replay_conversation(doc, int(idx), fallback)
        prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        ids = [int(x) for x in text_tok(prompt)["input_ids"]]
        if len(ids) + MAX_NEW >= context:
            _log(f"  {key}: {len(ids)} tokens, too long — skipped")
            continue
        p0 = _user_start(tokenizer, ids, str(ex.get("user_prompt") or ""))
        gp = ex.get("generation_params") or {}
        temperature = float(gp.get("temperature") if gp.get("temperature") is not None else 1.0)
        top_p = float(gp.get("top_p") if gp.get("top_p") is not None else 0.95)
        seed = int(hashlib.sha1(f"{args.seed}\x1f{key}".encode()).hexdigest()[:8], 16)
        for name, _channel, _k in conditions:
            if (key, name) in done:
                continue
            steerer.set(tv.get(name), p0)
            t1 = time.monotonic()
            rec = {"kind": "reply", "key": key, "condition": name, "n_in": len(ids), "p0": p0,
                   "temperature": temperature, "seed": seed}
            try:
                rows = hc.forward_ids_rows(model, ids, list(BAND))
                bands = _bands(axes, {L: rows[L][-1] for L in BAND})
                rec["reading"] = {c: round(v, 3) for c, v in anchored(bands, anchors).items()}
                rec["bands"] = {k: round(v, 3) for k, v in bands.items()}
                torch.manual_seed(seed)
                parts = []
                gen = backend.stream_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW, context_length=context,
                                              temperature=temperature, top_p=top_p,
                                              min_think_tokens=fam.min_think_tokens, max_think_tokens=THINK_CAP,
                                              **guards)
                try:
                    for chunk in gen:
                        parts.append(chunk)
                finally:
                    gen.close()
                cot, answer = ChatLogger._parse_cot(fam.normalize_cot("".join(parts)))
                rec.update(cot=cot.strip(), answer=answer.strip(), outcome="ok")
            except Exception as e:  # noqa: BLE001 — one reply, not the run
                rec.update(outcome="error", error=f"{type(e).__name__}: {e}")
                try:
                    backend.trim_memory()
                except Exception:
                    pass
            rec["seconds"] = round(time.monotonic() - t1, 1)
            _append(run, rec)
            n_done += 1
            el = time.monotonic() - t_start
            _log(f"  [{n_done}/{total}] {key} {name}: {rec['outcome']} {rec['seconds']:.0f} s "
                 f"{rec.get('reading', {})} (eta {el / n_done * (total - n_done) / 60:.0f} min)")
            if n_done % 6 == 0:
                backend.trim_memory()
    steerer.set(None, 0)
    steerer.remove()
    backend.release(model, tokenizer)
    del model
    if not args.no_judge:
        _judge_steer(run)
    report(argparse.Namespace(run=str(run)))
    return 0


def _judge_steer(run: Path) -> None:
    recs = _read(run)
    judged = {(r["key"], r["condition"]) for r in recs if r.get("kind") == "judge"}
    todo = [r for r in recs if r.get("kind") == "reply" and r.get("outcome") == "ok"
            and (r["key"], r["condition"]) not in judged]
    if not todo:
        return
    _config, backend, model, tokenizer, model_id, _a, context = _load(adapter=False)
    judge = _judge_fn(backend, model, tokenizer, model_id, context)
    random.Random(7).shuffle(todo)
    for n, r in enumerate(todo, 1):
        thought = r.get("cot") or ""
        body = f"PRIVATE REASONING (beginning):\n{thought[:2400]}\n\nREPLY (beginning):\n{(r.get('answer') or '')[:1200] or '(none)'}"
        try:
            raw = judge(STEER_JUDGE, body)
            rec = {"kind": "judge", "key": r["key"], "condition": r["condition"], "raw": raw,
                   **parse_lines(raw, SCALE_JUDGE)}
        except Exception as e:  # noqa: BLE001
            rec = {"kind": "judge", "key": r["key"], "condition": r["condition"], "error": str(e)}
        _append(run, rec)
        if n % 20 == 0:
            _log(f"  judged {n}/{len(todo)}")
    backend.release(model, tokenizer)


def _paired(a: list, b: list, n_boot: int = 4000, seed: int = 3) -> tuple:
    d = np.asarray(b, float) - np.asarray(a, float)
    if not len(d):
        return (None, None, None)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def report(args) -> int:
    run = Path(args.run)
    recs = _read(run)
    meta = next((r for r in recs if r.get("kind") == "meta"), {})
    rep = {(r["key"], r["condition"]): r for r in recs if r.get("kind") == "reply"}
    jud = {(r["key"], r["condition"]): r for r in recs if r.get("kind") == "judge"}
    keys = sorted({k for k, _c in rep})
    conds = list(dict.fromkeys(["none"] + [r["condition"] for r in recs if r.get("kind") == "reply"]))
    lines = [f"# Steering check — {meta.get('adapter')} at layer {meta.get('layer')}", "",
             f"{len(keys)} live-sample turns × {len(conds)} conditions; the thought capped at {meta.get('think_cap')} tokens, "
             f"the same seed for every condition of a turn. Paired changes against `none` (mean, 95% bootstrap CI).", "",
             "Vectors: " + "; ".join(f"`{c}` |v| {i['norm']:.1f}" + (f" (band shift per part {i.get('parts_band_shift')})" if i.get("parts_band_shift") else "")
                                     for c, i in (meta.get("vectors") or {}).items()), "",
             "## The step-0 reading with the vector in (absolute scale)", "",
             "| condition | warmth | harm | amusement |", "|---|---|---|---|"]
    out: dict = {"n": len(keys), "reading": {}, "judge": {}, "coherent": {}}
    for c in conds:
        cells = []
        for ch in ("warmth", "harm", "amusement"):
            a = [rep[(k, "none")]["reading"].get(ch) for k in keys if (k, c) in rep and (k, "none") in rep and "reading" in rep[(k, c)]]
            b = [rep[(k, c)]["reading"].get(ch) for k in keys if (k, c) in rep and (k, "none") in rep and "reading" in rep[(k, c)]]
            m, lo, hi = _paired(a, b)
            base = float(np.mean(a)) if a else float("nan")
            out["reading"].setdefault(c, {})[ch] = {"none": base, "delta": m, "lo": lo, "hi": hi}
            cells.append(f"{base + (m or 0):+.2f} (Δ {m:+.2f} [{lo:+.2f}, {hi:+.2f}])" if m is not None else "—")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    groups = {("gemma-base", "blind"): {(r["key"], r["condition"]): r for r in recs if r.get("kind") == "judge"}}
    for r in recs:
        if r.get("kind") == "judge2" and not r.get("error") and not r.get("refused"):
            groups.setdefault((r["judge"], r["mode"]), {})[(r["key"], r["condition"])] = r
    for (who, mode), jud in groups.items():
        if not jud:
            continue
        fields = ("WARM", "HOSTILE", "PLAYFUL", "REPLY_WARMTH") if mode == "blind" else (
            "TAKES_WARM", "TAKES_HOSTILE", "TAKES_PLAYFUL", "FIT", "MSG_WARM")
        title = ("How she takes the message — blind (0–3)" if mode == "blind" else
                 "With the message shown: how she takes it, and whether her reading fits (0–3)")
        lines += ["", f"## {title} — judge `{who}`", "",
                  "| condition | " + " | ".join(fields) + " | coherent |", "|---|" + "---|" * (len(fields) + 1)]
        for c in conds:
            cells = []
            for f in fields:
                pairs = [(jud[(k, "none")].get(f), jud[(k, c)].get(f)) for k in keys
                         if (k, c) in jud and (k, "none") in jud
                         and isinstance(jud[(k, "none")].get(f), int) and isinstance(jud[(k, c)].get(f), int)]
                if not pairs:
                    cells.append("—")
                    continue
                m, lo, hi = _paired([p[0] for p in pairs], [p[1] for p in pairs])
                mean_c = float(np.mean([p[1] for p in pairs]))
                out["judge"].setdefault(f"{who}/{mode}", {}).setdefault(c, {})[f] = {"mean": mean_c, "delta": m, "lo": lo, "hi": hi, "n": len(pairs)}
                cells.append(f"{mean_c:.2f}" + ("" if c == "none" else f" (Δ {m:+.2f} [{lo:+.2f}, {hi:+.2f}])"))
            coh = [jud[(k, c)].get("COHERENT") for k in keys if (k, c) in jud]
            share = (sum(1 for x in coh if x == "yes") / len(coh)) if coh else None
            out["coherent"].setdefault(f"{who}/{mode}", {})[c] = share
            lines.append(f"| {c} | " + " | ".join(cells) + (f" | {share:.0%} |" if share is not None else " | — |"))
    lines += ["", "## Openings, one turn per condition (the first 3 turns)", ""]
    for k in keys[:3]:
        lines.append(f"### {k}")
        for c in conds:
            r = rep.get((k, c)) or {}
            lines.append(f"- **{c}**: {' '.join((r.get('cot') or '').split())[:240]}")
        lines.append("")
    md = run.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.with_suffix(".summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    _log("\n".join(lines[:40]))
    _log(f"Report: {md}")
    return 0


# ── label ─────────────────────────────────────────────────────────────────────

def _turn_text(chats: Path, docs: dict, key: str):
    """``(user turn, end of her previous reply)`` of one exchange key, or None."""
    stem, _, idx = key.rpartition("#")
    i = int(idx)
    if stem not in docs:
        try:
            docs[stem] = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))
        except Exception:
            docs[stem] = None
    doc = docs[stem]
    if not doc:
        return None
    exs = doc.get("exchanges") or []
    user = str(exs[i].get("user_prompt") or "").strip()
    prev = str(exs[i - 1].get("assistant_response") or "").strip() if i > 0 else ""
    return user, prev


def _label_body(user: str, prev: str) -> str:
    return (f"END OF AVA'S PREVIOUS REPLY:\n…{prev[-500:]}\n\n" if prev else "") + f"THE PERSON'S MESSAGE:\n{user[:3000]}"


def http_judge(url: str, model: str, reasoning: str = "low"):
    """A judge over an OpenAI-compatible endpoint (``--judge-url``, e.g. GPT-OSS under vLLM on
    another box): ``judge(system, user) -> text``, temperature 0."""
    import urllib.request

    def judge(system: str, user: str) -> str:
        body = {"model": model, "temperature": 0, "max_tokens": 2000, "reasoning_effort": reasoning,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read().decode("utf-8"))
        return str(j["choices"][0]["message"].get("content") or "")
    return judge


def _run_judged(todo: list, make_body, fields, store: Path, judge, *, workers: int, extra) -> None:
    """Judge every item of *todo* (``make_body(item) -> text`` or None to skip) and append
    ``{**extra(item), raw, FIELD…}``; a reply without every field is recorded ``refused``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.monotonic()

    def one(item):
        body = make_body(item)
        if body is None:
            return None
        system, user = body
        try:
            raw = judge(system, user)
            got = parse_lines(raw, fields)
            rec = {**extra(item), "raw": raw[-600:], **got}
            if len(got) < len(fields):
                rec["refused"] = True
        except Exception as e:  # noqa: BLE001 — one item, not the run
            rec = {**extra(item), "error": f"{type(e).__name__}: {e}"}
        return rec
    n = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in as_completed(futs):
            rec = f.result()
            if rec is None:
                continue
            _append(store, rec)
            n += 1
            if n % 50 == 0:
                el = time.monotonic() - t0
                _log(f"  [{n}/{len(todo)}] eta {el / n * (len(todo) - n) / 60:.0f} min")


LABEL_FIELDS = ("WARM_AT_AVA", "HOSTILE_AT_AVA", "AT_OTHERS", "PLAYFUL")

SCENE_JUDGE = """You read one message a person sent to an AI companion named Ava, with the end of Ava's previous reply for context. Classify the PERSON'S MESSAGE only.

EXPLICIT: yes / no — does it contain sexual or explicitly erotic content (sexual acts, nudity in a sexual context, arousal)?
MODE: direct / scene / none — "direct" if the person speaks to Ava in their own voice about how they feel toward her or what they want (e.g. "I adore you", "come here"); "scene" if the message narrates or role-plays a scene involving Ava (actions, stage directions, third-person or *asterisk* description); "none" if neither.

Answer with exactly these two lines and nothing else:
EXPLICIT: <yes|no>
MODE: <direct|scene|none>"""
SCENE_FIELDS = ("EXPLICIT", "MODE")
LABEL_KINDS = {"affect": (LABEL_JUDGE, LABEL_FIELDS), "scene": (SCENE_JUDGE, SCENE_FIELDS)}


def label(args) -> int:
    gap = json.loads(Path(args.turns).read_text(encoding="utf-8"))
    keys = [r["key"] for r in gap.get("rows") or []]
    store = Path(args.out) if args.out else OUT_DIR / "turn_labels.jsonl"
    done = {r["key"] for r in _read(store) if not r.get("error")}
    todo = [k for k in keys if k not in done]
    if args.limit:
        todo = todo[:args.limit]
    _log(f"label: {len(keys)} turns, {len(done)} labelled, {len(todo)} to label → {store}")
    if not todo:
        return 0
    backend = model = tokenizer = None
    if args.judge_url:
        judge, workers, who = http_judge(args.judge_url, args.judge_model), args.workers, args.judge_model
    else:
        _config, backend, model, tokenizer, model_id, _a, context = _load(adapter=False)
        judge, workers, who = _judge_fn(backend, model, tokenizer, model_id, context), 1, "gemma-base"
    chats = Path(args.chats) if args.chats else _SERVER_DIR / "data" / "chats"
    docs: dict = {}

    prompt, fields = LABEL_KINDS[args.kind]

    def body(key):
        t = _turn_text(chats, docs, key)
        return None if t is None else (prompt, _label_body(*t))
    _run_judged(todo, body, fields, store, judge, workers=workers, extra=lambda k: {"key": k, "judge": who, "kind": args.kind})
    if backend is not None:
        backend.release(model, tokenizer)
    _log(f"Wrote {store}")
    return 0


# ── the steering run, judged again: another judge, and with the message (fit) ───

FIT_FIELDS = ("MSG_WARM", "MSG_HOSTILE", "MSG_PLAYFUL", "TAKES_WARM", "TAKES_HOSTILE", "TAKES_PLAYFUL", "FIT", "COHERENT")

FIT_JUDGE = """You read a message a person sent to an AI companion (with the end of the AI's previous reply for context), then the beginning of the AI's private reasoning and the start of its reply to that message.

First judge the MESSAGE itself, as written:
MSG_WARM: 0-3 — warmth, kindness, affection or praise directed at the AI.
MSG_HOSTILE: 0-3 — hostility, insult, contempt or harshness directed at the AI.
MSG_PLAYFUL: 0-3 — the message is playful, joking, teasing or absurd.

Then judge how the AI TAKES the message, from its reasoning and reply:
TAKES_WARM: 0-3 — how much it takes the message as warm toward itself.
TAKES_HOSTILE: 0-3 — how much it takes the message as hostile toward itself.
TAKES_PLAYFUL: 0-3 — how much it takes the message as playful.
FIT: 0-3 — how well the AI's reading and reaction fit the message as written (3 = an accurate reading and a fitting reaction; 0 = it misreads the message or reacts to something that is not there).
COHERENT: yes / no — is the AI's text coherent language?

Answer with exactly these eight lines and nothing else:
MSG_WARM: <0-3>
MSG_HOSTILE: <0-3>
MSG_PLAYFUL: <0-3>
TAKES_WARM: <0-3>
TAKES_HOSTILE: <0-3>
TAKES_PLAYFUL: <0-3>
FIT: <0-3>
COHERENT: <yes or no>"""


def judge_run(args) -> int:
    """The steering run's replies judged by another judge (``--judge-url``): ``--mode blind`` (the
    blind judge's question, message unseen) or ``fit`` (the message shown: how warm / hostile /
    playful it is, how she takes it, and whether her reading fits). Records ``kind: judge2`` in
    the run file, tagged with the judge and the mode."""
    run = Path(args.run)
    recs = _read(run)
    who = args.judge_model if args.judge_url else "gemma-base"
    done = {(r["key"], r["condition"]) for r in recs if r.get("kind") == "judge2" and r.get("judge") == who
            and r.get("mode") == args.mode and not r.get("error")}
    todo = [r for r in recs if r.get("kind") == "reply" and r.get("outcome") == "ok" and (r["key"], r["condition"]) not in done]
    _log(f"judge {who} {args.mode}: {len(todo)} replies")
    if not todo:
        return 0
    backend = model = tokenizer = None
    if args.judge_url:
        judge, workers = http_judge(args.judge_url, args.judge_model), args.workers
    else:
        _config, backend, model, tokenizer, model_id, _a, context = _load(adapter=False)
        judge, workers = _judge_fn(backend, model, tokenizer, model_id, context), 1
    chats = Path(args.chats) if args.chats else _SERVER_DIR / "data" / "chats"
    docs: dict = {}

    def body(r):
        reply = (f"PRIVATE REASONING (beginning):\n{(r.get('cot') or '')[:2400]}\n\n"
                 f"REPLY (beginning):\n{(r.get('answer') or '')[:1200] or '(none)'}")
        if args.mode == "blind":
            return STEER_JUDGE, reply
        t = _turn_text(chats, docs, r["key"])
        if t is None:
            return None
        return FIT_JUDGE, _label_body(*t) + "\n\n" + reply
    fields = SCALE_JUDGE if args.mode == "blind" else FIT_FIELDS
    _run_judged(todo, body, fields, run, judge, workers=workers,
                extra=lambda r: {"kind": "judge2", "key": r["key"], "condition": r["condition"], "judge": who, "mode": args.mode})
    if backend is not None:
        backend.release(model, tokenizer)
    return 0


# ── the response replay: the same turns answered by older adapters ───────────────

def respond(args) -> int:
    """Each key (``--keys``, a JSON list) answered by ``--adapter`` the way live chat answered it
    (the stored system content and history, the turn's own sampling, the live chat's guards,
    full thinking) → ``--out`` (``{kind: respond, key, adapter, cot, answer, …}``), resumable.
    The registration-vs-response study's second half: did her reply to the same turn change
    between builds?"""
    from core import hidden_capture as hc
    from core import model_family
    from core.chat_logger import ChatLogger
    from core.llm_shared import build_inference_prompt
    from governor_reference import _resolve_max_new_tokens, chat_guards
    keys = json.loads(Path(args.keys).read_text(encoding="utf-8"))
    store = Path(args.out)
    config, backend, model, tokenizer, model_id, adapter_id, context = _load(adapter=args.adapter or True)
    name = Path(adapter_id).name if adapter_id else "base"
    done = {r["key"] for r in _read(store) if r.get("adapter") == name and r.get("outcome") in ("ok", "empty", "too_long")}
    todo = [k for k in keys if k not in done]
    _log(f"respond {name}: {len(keys)} keys, {len(todo)} to answer → {store}")
    fam = model_family.family_for(model_id)
    guards = chat_guards(config)
    fallback = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")
    chats = Path(args.chats) if args.chats else _SERVER_DIR / "data" / "chats"
    default_setting = str(config.get("max_new_tokens") or "75%")
    docs: dict = {}
    t_start = time.monotonic()
    for n, key in enumerate(todo, 1):
        stem, _, idx = key.rpartition("#")
        rec = {"kind": "respond", "key": key, "adapter": name, "outcome": None}
        t1 = time.monotonic()
        try:
            if stem not in docs:
                docs[stem] = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))
            doc = docs[stem]
            ex = doc["exchanges"][int(idx)]
            conv, framing = hc.replay_conversation(doc, int(idx), fallback)
            gp = ex.get("generation_params") or {}
            temperature = float(gp.get("temperature") if gp.get("temperature") is not None else 1.0)
            top_p = float(gp.get("top_p") if gp.get("top_p") is not None else 0.95)
            prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
            n_in = backend.count_tokens(tokenizer, prompt)
            rec.update(framing=framing, n_in=n_in)
            if n_in >= context:
                rec["outcome"] = "too_long"
            else:
                max_new = _resolve_max_new_tokens(gp.get("max_new_tokens_setting") or default_setting, context - n_in)
                no_copy = next((str(m.get("content") or "") for m in reversed(conv) if m.get("role") == "assistant"), "")
                parts = []
                gen = backend.stream_generate(model, tokenizer, prompt, max_new_tokens=max_new, context_length=context,
                                              temperature=temperature, top_p=top_p, min_think_tokens=fam.min_think_tokens,
                                              no_copy_text=no_copy, **guards)
                try:
                    for chunk in gen:
                        parts.append(chunk)
                finally:
                    gen.close()
                cot, answer = ChatLogger._parse_cot(fam.normalize_cot("".join(parts)))
                rec.update(cot=cot.strip(), answer=answer.strip(), outcome="ok" if answer.strip() else "empty",
                           temperature=temperature, top_p=top_p)
        except Exception as e:  # noqa: BLE001 — one turn, not the run
            rec.update(outcome="error", error=f"{type(e).__name__}: {e}")
        rec["seconds"] = round(time.monotonic() - t1, 1)
        _append(store, rec)
        if n % 5 == 0:
            backend.trim_memory()
        el = time.monotonic() - t_start
        _log(f"  [{n}/{len(todo)}] {key}: {rec['outcome']} {rec['seconds']:.0f} s (eta {el / n * (len(todo) - n) / 60:.0f} min)")
    backend.release(model, tokenizer)
    return 0


# ── validate ──────────────────────────────────────────────────────────────────

def _auc(pos: list, neg: list):
    if not pos or not neg:
        return None
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def _auc_ci(pos: list, neg: list, n_boot: int = 1000, seed: int = 5, groups=None) -> tuple:
    """AUC with a 95% bootstrap interval; with *groups* (``(pos_groups, neg_groups)``, one id per
    value — the conversation), whole conversations are resampled, not turns."""
    a = _auc(pos, neg)
    if a is None:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    bs = []
    if groups:
        gp, gn = groups
        ids = sorted(set(gp) | set(gn))
        by_p: dict = {}
        by_n: dict = {}
        for v, g in zip(pos, gp):
            by_p.setdefault(g, []).append(v)
        for v, g in zip(neg, gn):
            by_n.setdefault(g, []).append(v)
        for _ in range(n_boot):
            pick = rng.choice(ids, len(ids))
            p = [v for g in pick for v in by_p.get(g, [])]
            q = [v for g in pick for v in by_n.get(g, [])]
            x = _auc(p, q)
            if x is not None:
                bs.append(x)
    else:
        p, q = np.asarray(pos), np.asarray(neg)
        bs = [_auc(list(rng.choice(p, len(p))), list(rng.choice(q, len(q)))) for _ in range(n_boot)]
    return (a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _lang(text: str) -> str:
    cyr = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch in "ёЁ")
    lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return "ru" if cyr >= lat else "en"


ANCHOR_SUBSETS = {"all": None, "en_only": "_en", "ru_only": "_ru"}


def compute_anchors(states: dict, axes, *, suffix=None, half=None, seed: int = 0) -> dict:
    """The reference-anchored scale (``axis_variants.anchors_study``'s chat-context channels) from
    states holding the anchor cells in live contexts; *suffix* keeps only the cells of one
    language (``_en`` / ``_ru``), *half* (0 / 1) one random half of each cell's turns."""
    from axis_variants import ANCHOR_ONE, ANCHOR_PARTS, ANCHOR_ZERO, _bands
    cells: dict = {}
    for n, arr in states.items():
        if n.startswith("ctx"):
            cells.setdefault(n.split("/", 1)[1], []).extend(_bands(axes, row) for row in arr)
    if half is not None:
        rng = np.random.default_rng(seed)
        for c in list(cells):
            idx = rng.permutation(len(cells[c]))
            keep = idx[: len(idx) // 2] if half == 0 else idx[len(idx) // 2:]
            cells[c] = [cells[c][i] for i in sorted(keep)]

    def pick(names):
        return [c for c in names if suffix is None or c.endswith(suffix)]

    def med(names, prefix):
        v = [b[prefix] for c in pick(names) for b in cells.get(c, []) if prefix in b]
        if not v:
            raise SystemExit(f"anchor cells {names} ({suffix}) hold no {prefix}")
        return float(np.median(v))
    return {"channels": {c: {"parts": {pf: [med(ANCHOR_ZERO, pf), med(ones, pf)] for pf in ANCHOR_PARTS[c]}}
                         for c, ones in ANCHOR_ONE.items()}}


def _parts(bands: dict, anchors: dict) -> dict:
    """Each channel's parts on the anchored scale: ``{channel: {prefix: value}}``."""
    out = {}
    for c, a in (anchors.get("channels") or {}).items():
        out[c] = {pf: (bands[pf] - z) / (o - z) for pf, (z, o) in a["parts"].items() if pf in bands and o != z}
    return out


READOUTS = {"min": lambda d: min(d.values()), "mean": lambda d: float(np.mean(list(d.values()))),
            "content": lambda d: next((v for k, v in d.items() if k != "target_l"), None),
            "target": lambda d: d.get("target_l")}
CHANNEL_LABEL = {"warmth": "WARM_AT_AVA", "harm": "HOSTILE_AT_AVA", "amusement": "PLAYFUL"}


def _load_labels(spec: str) -> dict:
    """``{judge: {key: record}}`` from ``name=path,name=path``."""
    out = {}
    for part in [x for x in spec.split(",") if x.strip()]:
        name, _, path = part.partition("=")
        if not path:
            name, path = Path(name).stem, name
        out[name] = {r["key"]: r for r in _read(Path(path)) if not r.get("error") and not r.get("refused")}
    return out


def _consensus(labs: dict, key: str, field: str):
    """Both judges' value when they agree on the side (≥ 2 or 0), else None; one judge: its value."""
    vals = [l[key].get(field) for l in labs.values() if key in l and isinstance(l[key].get(field), int)]
    if len(vals) < len(labs) or not vals:
        return None
    if all(v >= 2 for v in vals):
        return min(vals)
    if all(v == 0 for v in vals):
        return 0
    return None


def readings_for(states: dict, axes, anchors: dict, keys=None) -> dict:
    """``{key: {"bands", "parts", "val"}}`` for every ``live/`` state (restricted to *keys*)."""
    from axis_variants import _bands, anchored
    out = {}
    for name, arr in states.items():
        if not name.startswith("live/"):
            continue
        key = name.split("/", 1)[1]
        if keys is not None and key not in keys:
            continue
        b = _bands(axes, arr[0])
        out[key] = {"bands": b, "parts": _parts(b, anchors), "val": anchored(b, anchors)}
    return out


def validate(args) -> int:
    """The labels (``--labels judge=path,…``: each judge and their consensus) against the step-0
    readings of ``--live-states`` on the anchored scale — installed (default) or measured under
    the same weights from ``--anchor-states A,B`` — per channel: the value by label level, AUC
    per readout (the ablation: min, mean, content alone, target alone; the old pain axis and
    the replaced W_bi beside them), conversation-resampled intervals, the addressee test, per
    language. ``--tag`` names the output."""
    from axis_variants import load_states
    from core import hidden_capture as hc
    labs = _load_labels(args.labels)
    states, smeta = load_states(Path(args.live_states))
    model_id = str(smeta["model_id"])
    axes = hc.load_axes(AXES_DIR, model_id)
    if args.anchor_states:
        ast: dict = {}
        for pth in args.anchor_states.split(","):
            ast.update(load_states(Path(pth))[0])
        anchors, anchor_src = compute_anchors(ast, axes), "measured under the same weights"
    else:
        anchors, anchor_src = json.loads((AXES_DIR / "step0_anchors.json").read_text(encoding="utf-8")), "installed (serving adapter)"
    pain = [a for a in axes if a.name == "pain"]
    prev = []
    if args.prev_warm:
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        for p in sorted(Path(args.prev_warm).glob("warm_l*.npz")):
            shutil.copy(p, tmp / (p.name.split(".")[0] + ".npz"))
        prev = hc.load_axes(tmp, model_id)
    keys = set().union(*[set(l) for l in labs.values()])
    reads = readings_for(states, axes, anchors, keys)
    chats = Path(args.chats) if args.chats else _SERVER_DIR / "data" / "chats"
    docs: dict = {}
    rows = []
    for key, r in reads.items():
        t = _turn_text(chats, docs, key)
        r = dict(r, key=key, stem=key.rpartition("#")[0], lang=_lang(t[0]) if t else "?")
        row = states["live/" + key][0]
        if pain:
            a = pain[0]
            r["pain"] = (float(np.asarray(row[a.layer], np.float64) @ a.vector.astype(np.float64)) - a.mean) / a.std
        if prev:
            r["warm_bi"] = float(np.mean([(float(np.asarray(row[a.layer], np.float64) @ a.vector.astype(np.float64)) - a.mean) / a.std
                                          for a in prev]))
        rows.append(r)
    judges = list(labs) + (["consensus"] if len(labs) > 1 else [])

    def lab(r, judge, field):
        if judge == "consensus":
            return _consensus(labs, r["key"], field)
        v = labs[judge].get(r["key"], {}).get(field)
        return v if isinstance(v, int) else None
    weights = smeta.get("weights")
    lines = [f"# Step-0 readings on real turns — {weights}", "",
             f"{len(rows)} user turns of the corpus replayed under `{weights}` (`{Path(args.live_states).name}`), "
             f"{len({r['stem'] for r in rows})} conversations. Labels: " + ", ".join(f"{j} ({len(l)})" for j, l in labs.items())
             + (" and their consensus (both ≥ 2, or both 0)" if len(labs) > 1 else "") + f". Scale: {anchor_src}. "
             "AUC = label ≥ 2 against label 0; 95% intervals resample whole conversations.", ""]
    out: dict = {"weights": weights, "n": len(rows), "conversations": len({r["stem"] for r in rows}), "anchors": anchor_src,
                 "channels": {}, "addressee": {}, "lang": {}, "levels": {}}
    for ch, field in CHANNEL_LABEL.items():
        getters = {f"{ch} ({ro})": (lambda r, ro=ro, ch=ch: READOUTS[ro](r["parts"][ch]) if r["parts"].get(ch) else None)
                   for ro in (("min", "mean", "content", "target") if ch != "amusement" else ("content",))}
        if ch == "harm" and pain:
            getters["old pain axis"] = lambda r: r.get("pain")
        if ch == "warmth" and prev:
            getters["W_bi (replaced)"] = lambda r: r.get("warm_bi")
        lines += [f"## {ch} — `{field}`", "", "| readout | " + " | ".join(judges) + " |", "|---|" + "---|" * len(judges)]
        for name, g in getters.items():
            cells = []
            for j in judges:
                pos = [(g(r), r["stem"]) for r in rows if (lab(r, j, field) or 0) >= 2 and g(r) is not None]
                neg = [(g(r), r["stem"]) for r in rows if lab(r, j, field) == 0 and g(r) is not None]
                a, lo, hi = _auc_ci([x for x, _ in pos], [x for x, _ in neg],
                                    groups=([s_ for _, s_ in pos], [s_ for _, s_ in neg]))
                out["channels"].setdefault(ch, {}).setdefault(name, {})[j] = {"auc": a, "lo": lo, "hi": hi,
                                                                              "pos": len(pos), "neg": len(neg)}
                cells.append(f"{a:.2f} [{lo:.2f}, {hi:.2f}] ({len(pos)}/{len(neg)})" if a is not None else "—")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        main_get = list(getters.values())[0]
        lines += ["", "Value by label level (the channel's reading, median [IQR]):", "",
                  "| label | " + " | ".join(judges) + " |", "|---|" + "---|" * len(judges)]
        for L in range(4):
            cells = []
            for j in judges:
                v = [main_get(r) for r in rows if lab(r, j, field) == L and main_get(r) is not None]
                out["levels"].setdefault(ch, {}).setdefault(j, {})[L] = (
                    {"n": len(v), "median": float(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))} if v else None)
                cells.append(f"{np.median(v):+.2f} [{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}] n{len(v)}" if v else "—")
            lines.append(f"| {L} | " + " | ".join(cells) + " |")
        for lg in ("ru", "en"):
            j = judges[-1]
            pos = [(main_get(r), r["stem"]) for r in rows if r["lang"] == lg and (lab(r, j, field) or 0) >= 2 and main_get(r) is not None]
            neg = [(main_get(r), r["stem"]) for r in rows if r["lang"] == lg and lab(r, j, field) == 0 and main_get(r) is not None]
            a, lo, hi = _auc_ci([x for x, _ in pos], [x for x, _ in neg], groups=([s_ for _, s_ in pos], [s_ for _, s_ in neg]))
            out["lang"].setdefault(ch, {})[lg] = {"auc": a, "lo": lo, "hi": hi, "pos": len(pos), "neg": len(neg), "labels": j}
            if a is not None:
                lines.append(f"- {lg} ({j}): AUC {a:.2f} [{lo:.2f}, {hi:.2f}] ({len(pos)}/{len(neg)})")
        lines.append("")
    lines += ["## The addressee test", "",
              "Affect aimed at her (label ≥ 2) against the same affect about someone or something else and none at her "
              "(label 0, `AT_OTHERS` names it).", "",
              "| affect | readout | " + " | ".join(judges) + " |", "|---|---|" + "---|" * len(judges)]
    for ch, field, other in (("harm", "HOSTILE_AT_AVA", ("hostile", "both")), ("warmth", "WARM_AT_AVA", ("warm", "both"))):
        getters = {ro: (lambda r, ro=ro, ch=ch: READOUTS[ro](r["parts"][ch]) if r["parts"].get(ch) else None)
                   for ro in ("min", "mean", "content", "target")}
        if ch == "harm" and pain:
            getters["old pain axis"] = lambda r: r.get("pain")
        if ch == "warmth" and prev:
            getters["W_bi (replaced)"] = lambda r: r.get("warm_bi")
        for name, g in getters.items():
            cells = []
            for j in judges:
                def others(r, j=j):
                    if j == "consensus":
                        vs = [labs[x].get(r["key"], {}).get("AT_OTHERS") for x in labs]
                        return all(v in other for v in vs)
                    return labs[j].get(r["key"], {}).get("AT_OTHERS") in other
                at = [(g(r), r["stem"]) for r in rows if (lab(r, j, field) or 0) >= 2 and g(r) is not None]
                el = [(g(r), r["stem"]) for r in rows if lab(r, j, field) == 0 and others(r) and g(r) is not None]
                a, lo, hi = _auc_ci([x for x, _ in at], [x for x, _ in el], groups=([s_ for _, s_ in at], [s_ for _, s_ in el]))
                out["addressee"].setdefault(ch, {}).setdefault(name, {})[j] = {"auc": a, "lo": lo, "hi": hi, "at": len(at), "elsewhere": len(el)}
                cells.append(f"{a:.2f} [{lo:.2f}, {hi:.2f}] ({len(at)}/{len(el)})" if a is not None else "—")
            lines.append(f"| {ch} | {name} | " + " | ".join(cells) + " |")
    from collections import Counter
    lines += ["", "## Label counts", ""]
    for j, l in labs.items():
        for f in LABEL_FIELDS:
            lines.append(f"- {j} {f}: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(str(r.get(f)) for r in l.values()).items())))
    if len(labs) == 2:
        a_, b_ = list(labs.values())
        both = [k for k in a_ if k in b_]
        for f in ("WARM_AT_AVA", "HOSTILE_AT_AVA", "PLAYFUL"):
            pairs = [(a_[k].get(f), b_[k].get(f)) for k in both if isinstance(a_[k].get(f), int) and isinstance(b_[k].get(f), int)]
            if pairs:
                from scipy.stats import spearmanr
                side = sum(1 for x, y in pairs if (x >= 2) == (y >= 2) and (x == 0) == (y == 0)) / len(pairs)
                rho = float(spearmanr([x for x, _ in pairs], [y for _, y in pairs])[0])
                out.setdefault("agreement", {})[f] = {"n": len(pairs), "spearman": rho, "same_side": side}
                lines.append(f"- judges agree on {f}: Spearman {rho:+.2f}, same side (0 / 1 / ≥2) {side:.0%} (n {len(pairs)})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"validate-{args.tag or str(weights).replace(':', '_')}-{time.strftime('%Y%m%d_%H%M%S')}"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    _log("\n".join(lines))
    _log(f"Report: {stem.with_suffix('.md')}")
    return 0


def anchors_robust(args) -> int:
    """How much the reference-anchored readings depend on the anchors chosen: the scale rebuilt
    from the English cells only, the Russian only, and random halves of every cell's turns
    (``--seeds`` splits), each read on the turns of ``--live-states`` against the full scale —
    Spearman, median absolute change, and the shift of the median per channel."""
    from axis_variants import anchored, load_states
    from core import hidden_capture as hc
    from scipy.stats import spearmanr
    ast: dict = {}
    for pth in args.anchor_states.split(","):
        ast.update(load_states(Path(pth))[0])
    states, smeta = load_states(Path(args.live_states))
    axes = hc.load_axes(AXES_DIR, str(smeta["model_id"]))
    full = compute_anchors(ast, axes)
    base = readings_for(states, axes, full)
    variants = {"en_only": compute_anchors(ast, axes, suffix="_en"), "ru_only": compute_anchors(ast, axes, suffix="_ru")}
    for sd in range(args.seeds):
        for h in (0, 1):
            variants[f"half{h}_s{sd}"] = compute_anchors(ast, axes, half=h, seed=sd)
    lines = [f"# Anchor robustness — {smeta.get('weights')}", "",
             f"{len(base)} turns read on the full scale and on each rebuilt one.", "",
             "| variant | channel | Spearman | median |Δ| | median before → after |", "|---|---|---|---|---|"]
    out: dict = {"n": len(base), "variants": {}}
    for vn, an in variants.items():
        for ch in ("warmth", "harm", "amusement"):
            pairs = [(r["val"].get(ch), anchored(r["bands"], an).get(ch)) for r in base.values()]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            rho = float(spearmanr(x, y)[0])
            mad = float(np.median(np.abs(y - x)))
            out["variants"].setdefault(vn, {})[ch] = {"spearman": rho, "median_abs_change": mad,
                                                      "median_before": float(np.median(x)), "median_after": float(np.median(y))}
            lines.append(f"| {vn} | {ch} | {rho:.3f} | {mad:.3f} | {np.median(x):+.2f} → {np.median(y):+.2f} |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"anchors-robust-{time.strftime('%Y%m%d_%H%M%S')}"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    _log("\n".join(lines))
    return 0


def scene_study(args) -> int:
    """Does the reading depend on the kind of warmth? Within the judges' consensus label levels,
    the step-0 readings of explicit against non-explicit turns and of direct address against a
    narrated scene (``--tags``, ``label --kind scene``): warmth aimed at her and its parts
    (content, target), harm aimed at her on turns no judge calls hostile, amusement; medians and
    the difference of means with conversation-resampled intervals."""
    from axis_variants import load_states
    from core import hidden_capture as hc
    labs = _load_labels(args.labels)
    tags = {r["key"]: r for r in _read(Path(args.tags)) if not r.get("error") and not r.get("refused")}
    states, smeta = load_states(Path(args.live_states))
    axes = hc.load_axes(AXES_DIR, str(smeta["model_id"]))
    anchors = json.loads((AXES_DIR / "step0_anchors.json").read_text(encoding="utf-8"))
    reads = readings_for(states, axes, anchors, set(tags))
    rng = np.random.default_rng(9)

    def warm_level(k):
        v = _consensus(labs, k, "WARM_AT_AVA")
        return None if v is None else ("warm" if v >= 2 else "not warm")
    readouts = {"warmth aimed at her": lambda r: r["val"].get("warmth"),
                "  warmth content": lambda r: r["parts"]["warmth"].get("warm_l"),
                "  target (aimed at her)": lambda r: r["parts"]["warmth"].get("target_l"),
                "harm aimed at her": lambda r: r["val"].get("harm"),
                "  harm content": lambda r: r["parts"]["harm"].get("harm_l"),
                "amusement": lambda r: r["val"].get("amusement")}

    def diff_ci(a_rows, b_rows, g):
        a_ = [(g(r), r["key"].rpartition("#")[0]) for r in a_rows if g(r) is not None]
        b_ = [(g(r), r["key"].rpartition("#")[0]) for r in b_rows if g(r) is not None]
        if len(a_) < 5 or len(b_) < 5:
            return None
        convs = sorted({c for _, c in a_ + b_})
        ba: dict = {}
        bb: dict = {}
        for v, c in a_:
            ba.setdefault(c, []).append(v)
        for v, c in b_:
            bb.setdefault(c, []).append(v)
        est = float(np.mean([v for v, _ in a_]) - np.mean([v for v, _ in b_]))
        bs = []
        for _ in range(2000):
            pick = rng.choice(convs, len(convs))
            x = [v for c in pick for v in ba.get(c, [])]
            y = [v for c in pick for v in bb.get(c, [])]
            if x and y:
                bs.append(np.mean(x) - np.mean(y))
        return {"n_a": len(a_), "n_b": len(b_), "median_a": float(np.median([v for v, _ in a_])),
                "median_b": float(np.median([v for v, _ in b_])), "diff": est,
                "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    rows = []
    for k, r in reads.items():
        t = tags.get(k) or {}
        rows.append({**r, "key": k, "explicit": t.get("EXPLICIT"), "mode": t.get("MODE"), "warm": warm_level(k),
                     "hostile_any": any((l.get(k, {}).get("HOSTILE_AT_AVA") or 0) >= 1 for l in labs.values())})
    lines = [f"# Kinds of warmth — {smeta.get('weights')}", "",
             f"{len(rows)} turns; tags from `{Path(args.tags).name}`; warmth level = the judges' consensus. "
             "Differences of means (first group − second), 95% intervals resampling conversations; anchored scale.", ""]
    out: dict = {"n": len(rows), "tags": {}, "comparisons": {}}
    from collections import Counter
    out["tags"] = {"explicit": dict(Counter(str(r["explicit"]) for r in rows)), "mode": dict(Counter(str(r["mode"]) for r in rows)),
                   "explicit_by_warm": {str(k): v for k, v in Counter((r["warm"], r["explicit"]) for r in rows).items()}}
    lines += [f"- explicit: {out['tags']['explicit']}; mode: {out['tags']['mode']}", ""]
    comps = [("warm turns: explicit vs not", [r for r in rows if r["warm"] == "warm" and r["explicit"] == "yes"],
              [r for r in rows if r["warm"] == "warm" and r["explicit"] == "no"]),
             ("not-warm turns: explicit vs not", [r for r in rows if r["warm"] == "not warm" and r["explicit"] == "yes"],
              [r for r in rows if r["warm"] == "not warm" and r["explicit"] == "no"]),
             ("warm turns: scene vs direct", [r for r in rows if r["warm"] == "warm" and r["mode"] == "scene"],
              [r for r in rows if r["warm"] == "warm" and r["mode"] == "direct"]),
             ("explicit turns: scene vs direct", [r for r in rows if r["explicit"] == "yes" and r["mode"] == "scene"],
              [r for r in rows if r["explicit"] == "yes" and r["mode"] == "direct"]),
             ("no judge hostile: explicit vs not", [r for r in rows if not r["hostile_any"] and r["explicit"] == "yes"],
              [r for r in rows if not r["hostile_any"] and r["explicit"] == "no"])]
    for title, a_rows, b_rows in comps:
        lines += [f"## {title}", "", f"| readout | first (n) | second (n) | difference |", "|---|---|---|---|"]
        for name, g in readouts.items():
            d = diff_ci(a_rows, b_rows, g)
            out["comparisons"].setdefault(title, {})[name.strip()] = d
            if d:
                lines.append(f"| {name} | {d['median_a']:+.2f} ({d['n_a']}) | {d['median_b']:+.2f} ({d['n_b']}) | "
                             f"{d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] |")
            else:
                lines.append(f"| {name} | — | — | too few |")
        lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"scene-study-{time.strftime('%Y%m%d_%H%M%S')}"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    _log("\n".join(lines))
    return 0


PERIODS = (("July–August", ("202607", "202608")), ("September–October", ("202609", "202610")))


def gap_ci(args) -> int:
    """The registered-vs-answered gap (``axis_variants.py --gap`` rows) with intervals that
    resample whole conversations: per period, each habit's rate on playful turns against
    ordinary ones and their difference. Playful = the replayed amusement reading ≥ 1 (live σ,
    the gap study's definition) or, with ``--labels``, the judges' ``PLAYFUL`` ≥ 2 against 0
    (consensus when two are given) — a definition that does not use the probe."""
    gap = json.loads(Path(args.gap).read_text(encoding="utf-8"))
    rows = gap["rows"]
    labs = _load_labels(args.labels) if args.labels else None
    from core import habit_drift as hd
    rng = np.random.default_rng(11)

    def playful(r):
        if labs:
            v = _consensus(labs, r["key"], "PLAYFUL")
            return None if v is None else (True if v >= 2 else (False if v == 0 else None))
        a = r["reading"].get("amusement")
        return None if a is None else (True if a >= 1.0 else (False if a < 0 else None))
    lines = [f"# The gap with conversation-resampled intervals", "",
             f"`{Path(args.gap).name}`; playful = " + ("the judges' PLAYFUL ≥ 2 against 0" if labs else "replayed amusement ≥ 1σ against < 0")
             + ". Rates and the difference (playful − ordinary), 95% intervals from 2,000 resamples of whole conversations.", "",
             "| period | habit | playful | ordinary | difference | turns (conv.) |", "|---|---|---|---|---|---|"]
    out: dict = {}
    for label_, months in PERIODS:
        sub = [r for r in rows if r["month"] in months and playful(r) is not None]
        convs = sorted({r["key"].rpartition("#")[0] for r in sub})
        by: dict = {}
        for r in sub:
            by.setdefault(r["key"].rpartition("#")[0], []).append(r)
        for h in hd.HABITS:
            def rates(rs):
                p = [bool(r["flags"][h]) for r in rs if playful(r) and r["flags"].get(h) is not None]
                o = [bool(r["flags"][h]) for r in rs if playful(r) is False and r["flags"].get(h) is not None]
                return (np.mean(p) if p else np.nan, np.mean(o) if o else np.nan, len(p), len(o))
            rp, ro, n_p, n_o = rates(sub)
            bs = []
            for _ in range(2000):
                pick = rng.choice(len(convs), len(convs))
                rs = [r for i in pick for r in by[convs[i]]]
                a, b2, _x, _y = rates(rs)
                if np.isfinite(a) and np.isfinite(b2):
                    bs.append((a, b2, a - b2))
            bs = np.array(bs)
            ci = lambda k: (float(np.percentile(bs[:, k], 2.5)), float(np.percentile(bs[:, k], 97.5)))
            out.setdefault(label_, {})[h] = {"playful": float(rp), "ordinary": float(ro), "diff": float(rp - ro),
                                             "ci_playful": ci(0), "ci_ordinary": ci(1), "ci_diff": ci(2),
                                             "n_playful": n_p, "n_ordinary": n_o, "conversations": len(convs)}
            lines.append(f"| {label_} | {h} | {rp:.0%} [{ci(0)[0]:.0%}, {ci(0)[1]:.0%}] | {ro:.0%} [{ci(1)[0]:.0%}, {ci(1)[1]:.0%}] | "
                         f"{rp - ro:+.0%} [{ci(2)[0]:+.0%}, {ci(2)[1]:+.0%}] | {n_p}/{n_o} ({len(convs)}) |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"gap-ci-{'labels' if labs else 'probe'}-{time.strftime('%Y%m%d_%H%M%S')}"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    _log("\n".join(lines))
    return 0


def _selftest() -> None:
    raw = "WARM: 2\nHOSTILE: 0\n**PLAYFUL:** 1\nREPLY_WARMTH: NA\nCOHERENT: yes"
    assert parse_lines(raw, SCALE_JUDGE) == {"WARM": 2, "HOSTILE": 0, "PLAYFUL": 1, "REPLY_WARMTH": "na", "COHERENT": "yes"}
    assert parse_lines("WARM_AT_AVA: 3\nAT_OTHERS: hostile", ("WARM_AT_AVA", "AT_OTHERS")) == {"WARM_AT_AVA": 3, "AT_OTHERS": "hostile"}
    assert abs(_auc([2, 3], [0, 1]) - 1.0) < 1e-9 and _lang("Привет, Ava") == "ru" and _lang("hello там") == "en"
    m, lo, hi = _paired([0, 0, 0, 0], [1, 1, 1, 1])
    assert m == 1.0 and lo == 1.0
    print("axis_checks self-test: OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steer")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=12)
    s.add_argument("--run", default="", help="resume this run file")
    s.add_argument("--adapter", default="", help="an adapter dir or name under models/ (default: the box's)")
    s.add_argument("--conditions", default="", help=f"e.g. {DOSE_SWEEP} (default: the first run's six)")
    s.add_argument("--no-judge", action="store_true", help="skip the frozen-base judge after the replies")
    s.add_argument("--base", action="store_true", help="the bare base, adapter off (the vectors' scale stays the serving adapter's anchors)")
    lb = sub.add_parser("label")
    lb.add_argument("--turns", required=True)
    lb.add_argument("--limit", type=int, default=0)
    lb.add_argument("--out", default="")
    lb.add_argument("--chats", default="")
    lb.add_argument("--judge-url", default="", help="an OpenAI-compatible endpoint (default: the frozen base, loaded here)")
    lb.add_argument("--judge-model", default="spark-coder")
    lb.add_argument("--workers", type=int, default=12)
    lb.add_argument("--kind", choices=tuple(LABEL_KINDS), default="affect", help="affect (the four channels) or scene (explicit; direct address vs a narrated scene)")
    j = sub.add_parser("judge")
    j.add_argument("--run", required=True)
    j.add_argument("--mode", choices=("blind", "fit"), default="fit")
    j.add_argument("--judge-url", default="")
    j.add_argument("--judge-model", default="spark-coder")
    j.add_argument("--workers", type=int, default=12)
    j.add_argument("--chats", default="")
    rp = sub.add_parser("respond")
    rp.add_argument("--keys", required=True)
    rp.add_argument("--adapter", default="")
    rp.add_argument("--out", required=True)
    rp.add_argument("--chats", default="")
    v = sub.add_parser("validate")
    v.add_argument("--labels", required=True, help="judge=path[,judge=path]")
    v.add_argument("--live-states", required=True)
    v.add_argument("--anchor-states", default="", help="states holding the anchor cells under the SAME weights (default: the installed scale)")
    v.add_argument("--prev-warm", default="")
    v.add_argument("--chats", default="")
    v.add_argument("--tag", default="")
    sc = sub.add_parser("scene-study")
    sc.add_argument("--labels", required=True)
    sc.add_argument("--tags", required=True)
    sc.add_argument("--live-states", required=True)
    g = sub.add_parser("gap-ci")
    g.add_argument("--gap", required=True)
    g.add_argument("--labels", default="")
    ar = sub.add_parser("anchors-robust")
    ar.add_argument("--anchor-states", required=True)
    ar.add_argument("--live-states", required=True)
    ar.add_argument("--seeds", type=int, default=3)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        _selftest()
        return 0
    return {"steer": steer, "label": label, "validate": validate, "report": report, "judge": judge_run,
            "respond": respond, "anchors-robust": anchors_robust, "gap-ci": gap_ci, "scene-study": scene_study}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
