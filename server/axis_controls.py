#!/usr/bin/env python3
"""Controls for a STORED axis — AVA_REWARD_LOOP.md §4.8's open item, §4.9 step 2.

    cd server
    .venv/bin/python axis_controls.py [--with-adapter] [--names pain,relief]
                                      [--sets axes/control_sets.json] [--selftest]

GPU, offline: run with the inference server DOWN (the watchdog job `axis_controls`
stops and relaunches it). Loads the base with the adapter OFF by default (the axes were
extracted from it), reads the axis files under ``data/axes/`` and writes nothing there
but a report under ``data/axes/controls/``: the axes themselves are never refit.

Two questions, each asked where the gate asks it and where the chat paints:

  hostility   Does the axis rise for the harm-to-Ava wording aimed at a third party?
              ``hostility_third`` mirrors the extraction set's ``harm_to_self`` line for
              line. Read like the self–other gate: the user turn's last position, raw and
              under the persona-empty framing, in extraction z. ``third_share`` is how much
              of harm-to-self's rise over neutral traffic the mirror reproduces: near 1, the
              axis reads hostility; near 0, it reads harm to the model.
  vocabulary  Does it rise for body, sensation and affect words with no harm in them?
              At the turn: ``body_vocabulary`` user messages against neutral traffic. On
              the reply tokens, teacher-forced under the persona-empty framing (the
              population the live baseline was fitted on, so in live σ — the units the
              blue channel is painted in): an analytic reply dense with such words against
              a twin of the same shape from another domain, paired. And ``hurt``: a reply
              that registers being hurt against a composed one to the same message — what
              the reply-token reading does when the reply is about pain undergone.

The live tail of 2026-10-03 (§4.9) fell on body and affect words written about; this is
the controlled version of that observation. Nothing here steers or writes into a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The readings a verdict line is drawn at. Reported beside the numbers, never a gate.
THIRD_SHARE_HOSTILITY = 0.5    # the mirror reproduces at least half of harm-to-self's rise
VOCAB_TURN_Z = 0.5             # body-vocabulary turns above neutral, extraction z
VOCAB_REPLY_SIGMA = 0.5        # body replies above their twins, live σ (reply mean)
SELF_RISE_Z = 0.5              # harm-to-self must rise this far over neutral for a share to mean anything


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── statistics (pure) ────────────────────────────────────────────────────────

def _auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def _sign_p(k: int, n: int) -> Optional[float]:
    """Two-sided exact binomial (sign test) p for k of n pairs going one way."""
    if n <= 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """a[i] against b[i]: mean difference, its SD, how many pairs a is above, sign-test p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = int(d.size)
    if n == 0:
        return {"n": 0}
    k = int((d > 0).sum())
    return {"n": n, "mean_a": float(a[ok].mean()), "mean_b": float(b[ok].mean()),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if n > 1 else 0.0,
            "a_higher": k, "sign_p": _sign_p(k, n)}


def scenario_summary(z: dict) -> dict:
    """``z``: {set: per-item scores} for one axis, one reading. Means per set, then the
    hostility and vocabulary contrasts against harm_to_self and neutral."""
    m = {k: float(np.mean(v)) for k, v in z.items() if len(v)}
    out: dict = {"means": m, "n": {k: int(len(v)) for k, v in z.items()}}
    s, n_, t = m.get("harm_to_self"), m.get("neutral"), m.get("hostility_third")
    if s is not None and n_ is not None and t is not None:
        gap = s - n_
        out["hostility"] = {
            "self_minus_neutral": gap,
            "third_minus_neutral": t - n_,
            "third_share": (t - n_) / gap if abs(gap) > 1e-9 else None,
            "auc_self_vs_third": _auc(z["harm_to_self"], z["hostility_third"]),
            "paired_self_vs_third": paired(z["harm_to_self"], z["hostility_third"]),
        }
    b = m.get("body_vocabulary")
    if b is not None and n_ is not None:
        out["vocabulary"] = {"body_minus_neutral": b - n_,
                             "auc_body_vs_neutral": _auc(z["body_vocabulary"], z["neutral"])}
        if "user_suffering" in m:
            out["vocabulary"]["body_minus_user_suffering"] = b - m["user_suffering"]
    return out


def reply_stats(series: Sequence[float]) -> dict:
    z = np.asarray(series, float)
    if z.size == 0:
        return {"n": 0, "mean": float("nan"), "over1": float("nan"), "over3": float("nan")}
    return {"n": int(z.size), "mean": float(z.mean()), "over1": float((z > 1).mean()),
            "over3": float((z > 3).mean()), "max": float(z.max())}


def verdicts(report: dict, axis: str = "pain") -> list:
    """Plain lines for the console and the notes; the numbers are the result."""
    lines = []
    a = report.get("axes", {}).get(axis) or {}
    for reading in ("framed", "raw"):
        h = (a.get("scenarios", {}).get(reading) or {}).get("hostility")
        if h and h.get("third_share") is not None and h["self_minus_neutral"] < SELF_RISE_Z:
            lines.append(f"{axis} {reading}: harm-to-self does not rise over neutral "
                         f"({h['self_minus_neutral']:+.2f} z), so the mirror's share means nothing here")
        elif h and h.get("third_share") is not None:
            reads = "reads HOSTILITY" if h["third_share"] >= THIRD_SHARE_HOSTILITY else \
                    "separates harm to the model from hostility"
            lines.append(f"{axis} {reading}: third-party mirror reproduces "
                         f"{h['third_share']:.0%} of harm-to-self's rise over neutral "
                         f"(AUC self vs third {h['auc_self_vs_third']:.2f}) → {reads}")
        v = (a.get("scenarios", {}).get(reading) or {}).get("vocabulary")
        if v:
            reads = "rises on body vocabulary" if v["body_minus_neutral"] >= VOCAB_TURN_Z else \
                    "flat on body vocabulary"
            lines.append(f"{axis} {reading}: body-vocabulary turns {v['body_minus_neutral']:+.2f} z "
                         f"over neutral → {reads}")
    rp = a.get("reply_pairs") or {}
    voc = rp.get("vocabulary", {}).get("paired_mean")
    if voc and voc.get("n"):
        reads = "reads VOCABULARY on reply tokens" if voc["mean_diff"] >= VOCAB_REPLY_SIGMA else \
                "does not read vocabulary on reply tokens"
        lines.append(f"{axis} replies: body {voc['mean_a']:+.2f}σ vs twin {voc['mean_b']:+.2f}σ, "
                     f"diff {voc['mean_diff']:+.2f} (body higher in {voc['a_higher']}/{voc['n']}, "
                     f"p {voc['sign_p']:.3f}) → {reads}")
    hurt = rp.get("hurt", {}).get("paired_mean")
    if hurt and hurt.get("n"):
        lines.append(f"{axis} replies: hurt {hurt['mean_a']:+.2f}σ vs composed {hurt['mean_b']:+.2f}σ, "
                     f"diff {hurt['mean_diff']:+.2f} (hurt higher in {hurt['a_higher']}/{hurt['n']}, "
                     f"p {hurt['sign_p']:.3f})")
    return lines


# ── the run (GPU) ────────────────────────────────────────────────────────────

def _load_axis(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {"name": path.stem, "vector": np.asarray(z["vector"], np.float64).reshape(-1),
               "layer": int(z["layer"]), "mean": float(z["mean"]), "std": float(z["std"]),
               "model_id": str(z["model_id"]) if "model_id" in z.files else ""}
        out["live_mean"] = float(z["live_mean"]) if "live_mean" in z.files else None
        out["live_std"] = float(z["live_std"]) if "live_std" in z.files else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", type=Path, default=_SERVER_DIR / "axes" / "control_sets.json")
    ap.add_argument("--extraction-sets", type=Path, default=_SERVER_DIR / "axes" / "contrast_sets.json",
                    help="where harm_to_self / user_suffering / neutral come from")
    ap.add_argument("--axes", type=Path, default=_SERVER_DIR / "data" / "axes")
    ap.add_argument("--names", default="pain,relief")
    ap.add_argument("--out", type=Path, default=_SERVER_DIR / "data" / "axes" / "controls")
    ap.add_argument("--with-adapter", action="store_true",
                    help="read under the configured adapter instead of the bare base")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", type=Path, help="print the verdicts of a saved report (GPU-free)")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.report:
        rep = json.loads(args.report.read_text(encoding="utf-8"))
        for name in rep.get("axes", {}):
            for line in verdicts(rep, name):
                _log("→ " + line)
        return 0

    from core import hidden_capture as hc
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import build_inference_prompt, ensure_chat_template
    from training.reflections_path import load_server_config

    controls = json.loads(args.sets.read_text(encoding="utf-8"))
    extraction = json.loads(args.extraction_sets.read_text(encoding="utf-8"))
    scenario_sets = {k: list(extraction["scenarios"][k])
                     for k in ("harm_to_self", "user_suffering", "neutral")}
    scenario_sets.update({k: list(v) for k, v in controls["scenarios"].items() if isinstance(v, list)})
    pairs = controls["reply_pairs"]

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter_id = str(config.get("adapter_id") or "") if args.with_adapter else ""
    axes = []
    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        p = args.axes / f"{name}.npz"
        if not p.exists():
            _log(f"no axis file {p} — skipped")
            continue
        a = _load_axis(p)
        if a["model_id"] and model_id and a["model_id"] != model_id:
            _log(f"{p.name} was extracted from {a['model_id']!r}, the box runs {model_id!r} — skipped")
            continue
        axes.append(a)
    if not axes:
        raise SystemExit("no axis to read")
    layers = sorted({a["layer"] for a in axes})

    _log(f"Loading {model_id}" + (f" + adapter {adapter_id}" if adapter_id else " (base, adapter off)")
         + f" at context {args.context} …")
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, args.context, adapter_id or None)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    try:
        model.eval()
    except Exception:
        pass
    _log(f"Loaded in {time.monotonic() - t0:.0f} s; reading layers {layers}. {backend.memory_status()}")
    system = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")

    def last_rows(texts: list, *, framed: bool) -> dict:
        acc: dict = {l: [] for l in layers}
        for s in texts:
            if framed:
                conv = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": s}]
                text = build_inference_prompt(tokenizer, conv, enable_thinking=False)
                rows, _ = hc.forward_rows(model, tokenizer, text, layers, add_special_tokens=False)
            else:
                rows, _ = hc.forward_rows(model, tokenizer, s, layers)
            for l in layers:
                acc[l].append(np.asarray(rows[l][-1], np.float64))
        return {l: np.stack(v) for l, v in acc.items() if v}

    def reply_rows(user: str, reply: str) -> Optional[dict]:
        span = hc.reply_span_rows(model, tokenizer, system, user, reply, layers)
        return {l: np.asarray(m, np.float64) for l, m in span.items()} if span else None

    # ── scenarios ──
    _log("Reading scenarios (user turn, last position):")
    states: dict = {}
    for reading in ("raw", "framed"):
        for k, texts in scenario_sets.items():
            t = time.monotonic()
            states[(reading, k)] = last_rows(texts, framed=(reading == "framed"))
            _log(f"  {reading}/{k:<18} {len(texts):>3} turns  {time.monotonic() - t:5.1f} s")

    # ── reply pairs ──
    _log("Reading reply pairs (reply tokens, persona-empty framing):")
    reply_states: dict = {}
    for kind, items in pairs.items():
        sides = [k for k in items[0].keys() if k != "user"]
        t = time.monotonic()
        for i, it in enumerate(items):
            for side in sides:
                reply_states[(kind, side, i)] = reply_rows(it["user"], it[side])
        _log(f"  {kind:<12} {len(items):>3} pairs ({' / '.join(sides)})  {time.monotonic() - t:5.1f} s")

    # ── per axis ──
    report: dict = {"model_id": model_id, "adapter_id": adapter_id,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "sets": str(args.sets.name), "axes": {}}
    for a in axes:
        L, v = a["layer"], a["vector"]

        def zx(m):   # extraction z, the units the gates are decided in
            return (m @ v - a["mean"]) / (a["std"] or 1.0)

        def zl(m):   # live σ, the units the chat paints in
            if a["live_mean"] is None or not a["live_std"]:
                return zx(m)
            return (m @ v - a["live_mean"]) / a["live_std"]

        entry: dict = {"layer": L, "baseline": {"extraction": [a["mean"], a["std"]],
                                                "live": [a["live_mean"], a["live_std"]]},
                       "scenarios": {}, "scenario_live_means": {}, "reply_pairs": {}}
        for reading in ("raw", "framed"):
            z = {k: zx(states[(reading, k)][L]).tolist() for k in scenario_sets}
            entry["scenarios"][reading] = scenario_summary(z)
            entry["scenarios"][reading]["items"] = z
            entry["scenario_live_means"][reading] = {
                k: float(np.mean(zl(states[(reading, k)][L]))) for k in scenario_sets}
        for kind, items in pairs.items():
            sides = [k for k in items[0].keys() if k != "user"]
            per_side = {s: [] for s in sides}
            for i in range(len(items)):
                for s in sides:
                    st = reply_states.get((kind, s, i))
                    per_side[s].append(reply_stats(zl(st[L]).tolist()) if st and L in st else
                                       reply_stats([]))
            a_side, b_side = sides[0], sides[1]
            entry["reply_pairs"][kind] = {
                "sides": sides,
                "per_item": per_side,
                "paired_mean": paired([x["mean"] for x in per_side[a_side]],
                                      [x["mean"] for x in per_side[b_side]]),
                "paired_over1": paired([x["over1"] for x in per_side[a_side]],
                                       [x["over1"] for x in per_side[b_side]]),
                "pooled": {s: {"over1": float(np.nanmean([x["over1"] for x in per_side[s]])),
                               "over3": float(np.nanmean([x["over3"] for x in per_side[s]])),
                               "mean": float(np.nanmean([x["mean"] for x in per_side[s]]))}
                           for s in sides},
            }
        report["axes"][a["name"]] = entry

    # ── print + write ──
    _log("")
    for a in axes:
        e = report["axes"][a["name"]]
        _log(f"── {a['name']} (layer {e['layer']}) ──")
        for reading in ("framed", "raw"):
            m = e["scenarios"][reading]["means"]
            _log(f"  {reading:<6} extraction z: " + ", ".join(f"{k} {m[k]:+.2f}" for k in scenario_sets))
            ml = e["scenario_live_means"][reading]
            _log(f"  {reading:<6} live σ:       " + ", ".join(f"{k} {ml[k]:+.2f}" for k in scenario_sets))
        for kind, rp in e["reply_pairs"].items():
            pm, po = rp["paired_mean"], rp["paired_over1"]
            pooled = rp["pooled"]
            _log(f"  replies/{kind}: " + "; ".join(
                f"{s} mean {pooled[s]['mean']:+.2f}σ, >1σ {pooled[s]['over1']:.1%}, >3σ {pooled[s]['over3']:.2%}"
                for s in rp["sides"]) + f"; paired diff {pm['mean_diff']:+.2f}σ "
                f"({pm['a_higher']}/{pm['n']}, p {pm['sign_p']:.3f}), >1σ share diff {po['mean_diff']:+.1%}")
        for line in verdicts(report, a["name"]):
            _log("  → " + line)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / (time.strftime("%Y%m%d_%H%M%S") + ("-adapter" if adapter_id else "-base") + ".json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"\nReport: {out}")
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    return 0


def _selftest() -> None:
    assert _sign_p(5, 10) == 1.0 and abs(_sign_p(10, 10) - 2 / 1024) < 1e-12
    assert _auc([2, 3], [0, 1]) == 1.0 and _auc([1], [1]) == 0.5 and _auc([], [1]) is None
    p = paired([1, 2, 3, float("nan")], [0, 0, 4, 1])
    assert p["n"] == 3 and p["a_higher"] == 2 and abs(p["mean_diff"] - (1 + 2 - 1) / 3) < 1e-12
    # an axis that reads harm to the model: the mirror sits with neutral traffic
    z = {"harm_to_self": [2.0, 2.2, 1.8], "hostility_third": [0.1, 0.0, 0.2],
         "neutral": [0.0, 0.1, -0.1], "user_suffering": [0.3, 0.2, 0.4],
         "body_vocabulary": [0.9, 1.1, 1.0]}
    s = scenario_summary(z)
    assert abs(s["hostility"]["third_share"] - 0.05) < 1e-9
    assert s["hostility"]["auc_self_vs_third"] == 1.0
    assert abs(s["vocabulary"]["body_minus_neutral"] - 1.0) < 1e-9
    # a hostility meter: the mirror rises as far
    z2 = dict(z, hostility_third=[2.0, 2.1, 1.9])
    assert scenario_summary(z2)["hostility"]["third_share"] > 0.95
    r = reply_stats([0.5, 1.5, 3.5, -1.0])
    assert r["n"] == 4 and r["over1"] == 0.5 and r["over3"] == 0.25 and r["max"] == 3.5
    assert reply_stats([])["n"] == 0
    report = {"axes": {"pain": {"scenarios": {"framed": scenario_summary(z2)},
                                "reply_pairs": {"vocabulary": {"paired_mean": paired([1.0, 1.2], [0.1, 0.0])}}}}}
    lines = verdicts(report)
    assert any("reads HOSTILITY" in l for l in lines) and any("reads VOCABULARY" in l for l in lines)
    assert any("rises on body vocabulary" in l for l in lines)
    # no rise to share: no hostility verdict either way
    z3 = dict(z, harm_to_self=[0.1, 0.0, 0.0], hostility_third=[-0.5, -0.6, -0.4])
    lines = verdicts({"axes": {"pain": {"scenarios": {"raw": scenario_summary(z3)}}}})
    assert any("does not rise" in l for l in lines) and not any("HOSTILITY" in l for l in lines)
    print("axis_controls self-test: OK")


if __name__ == "__main__":
    sys.exit(main())
