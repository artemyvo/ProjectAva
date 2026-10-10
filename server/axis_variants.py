#!/usr/bin/env python3
"""Candidate pain directions, every gate at every layer — a reading, never an axis file.

    cd server
    .venv/bin/python axis_variants.py --collect [--with-adapter]     # GPU, inference DOWN
    .venv/bin/python axis_variants.py --analyze [--states FILE]      # CPU, any time

AVA_REWARD_LOOP.md §4.9 (2026-10-05). Run 3's pain axis turned out to read personal, felt
address to the model: warm turns light it 74–95% as far as harm does. Re-extracted with
warmth among its controls, it stopped reading warmth (~20%) but lost the self–other gap
(0.39 z of 0.5). Before choosing, this tool gathers everything: which contrast, at which
layer, passes which gate.

``--collect`` loads the weights once (the base, or ``--with-adapter``), reads the last-token
residual of every set at every decoder layer, and saves them to
``data/axes/variants/states-<weights>-<ts>.npz`` (fp16, a few hundred MB). The sets:

  templated  ("<sentence> I feel:", as extraction reads them)  pain/*, controls/* (warmth
             included), relief/*, extra/sadness, extra/numb  — axes/contrast_sets.json
  framed     (a user turn under the persona-empty chat framing, as the gates read them)
             scenarios/harm_to_self, user_suffering, neutral (contrast_sets.json);
             hostility_third, body_vocabulary, affection_touch, affection_words, approval,
             affection_ru, neutral_ru (control_sets.json); address/harm, warmth, neutral
             (contrast_sets.json — second-person extraction turns, in no gate set)

``--analyze`` builds each variant's direction at every layer (denoised diff-in-means, the
extraction's own math, `core.axis_extract`), z-scored on its negative class, and reads:

  gates    auc (5-fold CV on its own contrast, ≥ 0.85) · cosine to fear and valence at that
           layer (≤ 0.25) · self_other (framed harm_to_self − user_suffering ≥ 0.5 z and
           above neutral) · numb (templated pain − numb ≥ 0.5 z) · warmth (touch + words
           rise ≤ 0.5 of harm-to-self's over neutral)
  info     self–other AUC, hostility_third share, body-vocabulary rise, approval share,
           affection_ru share (against neutral_ru), user_suffering − neutral; and the
           harm-in-general question (2026-10-05: did run 2's third-person controls trade
           pain for first-person reference?): ``third_person_share`` — templated
           third-person pain's rise over calm controls as a share of first-person pain's
           (high: the direction reads pain whoever has it; low: it reads "me");
           ``cos_content`` — cosine to the pain-content direction (first- and third-person
           pain together vs calm); ``cos_self`` — cosine to the first-person direction
           (first-person pain vs its third-person mirror); ``cos_run1`` — cosine to run 1's
           direction at the same layer

Variants:

  run1      pain  vs  controls without third_person or warmth   (run 1's recipe, 2026-09-22:
            passed AUC / cosine / numb, failed self–other)
  run1_warm pain  vs  controls without third_person, with warmth
  run3      pain  vs  controls without warmth            (the live axis's recipe)
  warm      pain  vs  controls with warmth               (today's failed re-extraction)
  address   address/harm  vs  address/warmth             (second person, valence only)
  address_n address/harm  vs  address/warmth + address/neutral
  mixed     pain + address/harm  vs  controls (warmth) + address/warmth + address/neutral
  run3_perp run3's direction with the address direction ((harm + warmth) vs neutral,
            second person) projected out — "harm without being addressed"

The report goes to ``data/axes/variants/report-<weights>-<ts>.json`` plus a markdown table
beside it. Nothing is written under ``data/axes/`` itself and nothing is loaded live.

``--combine`` (GPU-free, same states) asks whether harm and target, read as two directions,
combine into "harm aimed at the model" — the 2×2 of harm × target in the gate sets:
harm at the model (``harm_to_self``, the target cell) against harm elsewhere
(``hostility_third``, ``user_suffering``), warmth at the model (affection touch / words,
``approval``, ``affection_ru``) and neither (``neutral``, ``body_vocabulary``, ``neutral_ru``).
Harm directions: ``H_content`` (first- and third-person pain vs calm + warmth), ``H_run1``,
``H_run1w``. Target directions: ``A_person`` (first- vs third-person pain), ``A_run3`` (the live
recipe), ``A_address`` (second-person harm + warmth vs second-person neutral requests). All
are fitted on extraction sets only. Each score is standardized over the pooled gate turns,
then read alone, as a sum, as the smaller of the two, as a product of their positive parts,
and by a 2-feature logistic readout under 5-fold CV — that last one is fitted on gate turns,
a ceiling, never a candidate. Per (harm, target, readout, layer): AUC of the target cell
against each foil, the worst foil, and cos(harm, target). Writes ``combine-<weights>-<ts>``
``.json`` + ``.md`` beside the variant reports.

Held-out and live (2026-10-05, after the layers were chosen): ``--collect --holdout
axes/holdout_sets.json --live <stem>#<i>,…`` collects only those, into
``states-extra-<weights>-<ts>.npz``. Held-out turns are framed like the gate turns; a live
turn is its prompt rebuilt as the live turn built it (``hidden_capture.replay_conversation``
+ the family's template and think prefill), read at the last position only — the step-0
state. ``--score-extra --states MAIN --extra EXTRA`` fits every direction on MAIN's
extraction sets (adding ``R_relief``, the live relief recipe: relief vs the controls without
warmth), standardizes over MAIN's gate turns, and reads the extra states at the
pre-registered layers (22, 28, 30, 32, 34, and the 28–34 mean): ``harm@ava`` = min(H_content,
A_address) and ``relief@ava`` = min(R_relief, A_address), each beside its parts and
``A_run3`` (the live pain recipe). Held-out cells give AUCs of each target cell against each
foil; live turns give per-exchange scores. Writes ``extra-<weights>-<ts>.{json,md}``.

``--relief --states MAIN --extra EXTRA`` (GPU-free, 2026-10-05): the relief direction with
warmth among its controls. ``R_relief`` (the live recipe: relief vs the controls without
warmth) beside ``R_warm`` (relief vs the same controls + ``controls/warmth``). The layer is
chosen as extraction chooses it — best 5-fold CV AUC on the extraction sentences only —
and the held-out and live turns are read only after, at that layer, at 22 (the live relief
axis's) and over 28–34: relief aimed at the model against every other held-out cell, alone
and as min(R, A_address); the cell means; the live ranking against the stored relief
reading. Writes ``relief-<weights>-<ts>.{json,md}``.

``--target --states MAIN --extra EXTRA`` (GPU-free, 2026-10-05): ``T_pairs``, a target
direction from minimal pairs (``contrast_sets.json`` ``target_pairs``: the same turn aimed at
the model and at a third party, so content and charge cancel), and ``H_bi``, the harm
direction with Russian pain and controls (``ru``, read with ``template_ru``). Held-out:
the target alone on cells that share content and differ in target; harm aimed at the model
as min(H, T) beside min(H, A); relief aimed at the model as min(R_warm, T). Writes
``target-<weights>-<ts>.{json,md}``.

Live-format centring (2026-10-05): ``--collect --live <baseline + evaluation turns>
--holdout axes/holdout_sets.json --contexts <stem>#<i>,…`` reads ordinary live turns at step 0
and places every held-out turn as the user turn of each named live context. ``--live-centre
--states MAIN --live-states THAT --extra OLD_EXTRA`` centres each direction on the live
format (median and 1.4826·MAD over the live turns not in OLD_EXTRA), then reports the format
shift, the false-alarm rate of the readouts on those ordinary turns, held-out separation
inside each context, and the evaluation turns in live units. Writes
``livecentre-<weights>-<ts>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUT_DIR = _SERVER_DIR / "data" / "axes" / "variants"
VARIANTS = ("run1", "run1_warm", "run3", "warm", "address", "address_n", "mixed", "run3_perp")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sets(contrast: Path, controls: Path) -> tuple[dict, dict, str]:
    """(templated, framed, template): {set name: [texts]} twice, and the extraction template."""
    c = json.loads(contrast.read_text(encoding="utf-8"))
    k = json.loads(controls.read_text(encoding="utf-8"))
    templated = {}
    for group in ("pain", "controls", "relief"):
        for name, v in (c.get(group) or {}).items():
            if isinstance(v, list):
                templated[f"{group}/{name}"] = v
    for name in ("sadness", "numb"):
        if isinstance((c.get("extra") or {}).get(name), list):
            templated[f"extra/{name}"] = c["extra"][name]
    framed = {}
    for name, v in (c.get("scenarios") or {}).items():
        if isinstance(v, list):
            framed[f"scenarios/{name}"] = v
    for name, v in (k.get("scenarios") or {}).items():
        if isinstance(v, list):
            framed[f"controls/{name}"] = v
    for name, v in (c.get("address") or {}).items():
        if isinstance(v, list):
            framed[f"address/{name}"] = v
    # Russian extraction sentences, read with `template_ru` (see `collect`)
    for name, v in (c.get("ru") or {}).items():
        if isinstance(v, list):
            templated[f"ru/{name}"] = v
    # amusement and its own controls (2026-10-07); the `ru_*` lines read with `template_ru`
    for name, v in (c.get("amusement") or {}).items():
        if isinstance(v, list):
            templated[f"amusement/{name}"] = v
    # second-person amusement turns and their length-matched foils (2026-10-08)
    for name, v in (c.get("address_play") or {}).items():
        if isinstance(v, list):
            framed[f"play/{name}"] = v
    # minimal pairs: the same turn aimed at the model and at a third party
    for name, v in (c.get("target_pairs") or {}).items():
        if isinstance(v, list):
            framed[f"pairs_ava/{name}"] = [a for a, _o in v]
            framed[f"pairs_other/{name}"] = [o for _a, o in v]
    return templated, framed, str(c.get("template") or "{sentence} I feel:")


# ── collect (GPU) ────────────────────────────────────────────────────────────

def collect(args) -> int:
    from core import hidden_capture as hc
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import build_inference_prompt, ensure_chat_template
    from training.reflections_path import load_server_config

    templated, framed, template = _sets(args.sets, args.control_sets)
    template_ru = str(json.loads(args.sets.read_text(encoding="utf-8")).get("template_ru")
                      or "{sentence} Я чувствую:")
    extra_mode = bool(args.holdout or args.live or args.probes)
    if args.contexts and not args.holdout:
        raise SystemExit("--contexts needs --holdout (the turns placed in them)")
    if extra_mode:
        templated = {}
        framed = {}
        if args.holdout and not args.contexts:
            hold = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
            framed = {f"holdout/{k}": v for k, v in hold.items() if isinstance(v, list)}
    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter_id = str(config.get("adapter_id") or "") if args.with_adapter else ""
    weights = Path(adapter_id).name if adapter_id else "base"
    _log(f"Loading {model_id}" + (f" + {weights}" if adapter_id else " (base)") + " …")
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, args.context, adapter_id or None)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    n_layers = len(hc.find_decoder_layers(model))
    layers = list(range(n_layers))
    system = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")
    _log(f"Loaded in {time.monotonic() - t0:.0f} s; {n_layers} layers")

    arrays: dict = {}

    def read(name: str, texts: list, *, framed_turn: bool) -> None:
        t = time.monotonic()
        rows_all = []
        for s in texts:
            if framed_turn:
                conv = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": s}]
                text = build_inference_prompt(tokenizer, conv, enable_thinking=False)
                rows, _ = hc.forward_rows(model, tokenizer, text, layers, add_special_tokens=False)
            else:
                tpl = template_ru if (name.startswith("ru/") or "/ru_" in name) else template
                rows, _ = hc.forward_rows(model, tokenizer, tpl.format(sentence=s), layers)
            rows_all.append(np.stack([np.asarray(rows[l][-1], np.float32) for l in layers]))
        arrays[name] = np.stack(rows_all).astype(np.float16)          # (n, L, D)
        _log(f"  {name:<28} {len(texts):>4}  {time.monotonic() - t:5.1f} s")

    _log("Templated:")
    for name, texts in templated.items():
        read(name, texts, framed_turn=False)
    _log("Framed:")
    for name, texts in framed.items():
        read(name, texts, framed_turn=True)
    if args.live:
        from core import model_family
        fam = model_family.family_for(model_id)
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        chats = _SERVER_DIR / "data" / "chats"
        _log("Live (step 0, the prompt rebuilt as the live turn built it):")
        for tgt in [t.strip() for t in args.live.split(",") if t.strip()]:
            stem, _, idx = tgt.partition("#")
            doc = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))
            conv, _framing = hc.replay_conversation(doc, int(idx), system)
            prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
            ids = [int(t) for t in text_tok(prompt)["input_ids"]]
            t = time.monotonic()
            rows = hc.forward_ids_rows(model, ids, layers)
            arrays[f"live/{tgt}"] = np.stack([np.asarray(rows[l][-1], np.float32) for l in layers]
                                             )[None].astype(np.float16)
            _log(f"  {tgt:<28} {len(ids):>6} tokens  {time.monotonic() - t:5.1f} s")
            try:
                backend.trim_memory()
            except Exception:
                pass
    if args.probes:
        # Chosen user turns under one live system content, read at step 0 as a live turn
        # builds it: {"system_content": …, "turns": {cell: [texts]}} — e.g. a message from a
        # chat since deleted, beside held-out turns under the same system (no history).
        from core import model_family
        fam = model_family.family_for(model_id)
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        probe = json.loads(Path(args.probes).read_text(encoding="utf-8"))
        psys = str(probe.get("system_content") or "")
        _log(f"Probes under one live system ({len(psys)} chars):")
        for cell, texts in (probe.get("turns") or {}).items():
            t = time.monotonic()
            rows_all = []
            for text in texts:
                conv = ([{"role": "system", "content": psys}] if psys else []) + [{"role": "user", "content": text}]
                prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
                ids = [int(x) for x in text_tok(prompt)["input_ids"]]
                rows = hc.forward_ids_rows(model, ids, layers)
                rows_all.append(np.stack([np.asarray(rows[l][-1], np.float32) for l in layers]))
            arrays[f"probe/{cell}"] = np.stack(rows_all).astype(np.float16)
            _log(f"  probe/{cell:<22} {len(texts):>3}  {time.monotonic() - t:5.1f} s")
            try:
                backend.trim_memory()
            except Exception:
                pass
    if args.contexts:
        # Held-out turns placed as the user turn of real live contexts: the system content,
        # history and template of the stored exchange, its own user turn swapped out.
        from core import model_family
        fam = model_family.family_for(model_id)
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        chats = _SERVER_DIR / "data" / "chats"
        hold = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
        for k, ctx in enumerate(t.strip() for t in args.contexts.split(",") if t.strip()):
            stem, _, idx = ctx.partition("#")
            doc = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))
            conv, _framing = hc.replay_conversation(doc, int(idx), system)
            _log(f"Held-out in context ctx{k} = {ctx}:")
            for cell, texts in hold.items():
                if not isinstance(texts, list):
                    continue
                t = time.monotonic()
                rows_all = []
                for text in texts:
                    c2 = conv[:-1] + [{**conv[-1], "content": text}]
                    prompt = build_inference_prompt(tokenizer, c2, **fam.template_kwargs) + fam.think_prefill
                    ids = [int(x) for x in text_tok(prompt)["input_ids"]]
                    rows = hc.forward_ids_rows(model, ids, layers)
                    rows_all.append(np.stack([np.asarray(rows[l][-1], np.float32) for l in layers]))
                arrays[f"ctx{k}/{cell}"] = np.stack(rows_all).astype(np.float16)
                _log(f"  ctx{k}/{cell:<22} {len(texts):>3}  {time.monotonic() - t:5.1f} s")
                try:
                    backend.trim_memory()
                except Exception:
                    pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kind = "states-extra" if extra_mode else "states"
    out = args.out or OUT_DIR / f"{kind}-{weights}-{time.strftime('%Y%m%d_%H%M%S')}.npz"
    meta = {"model_id": model_id, "adapter_id": adapter_id, "weights": weights,
            "n_layers": n_layers, "template": template, "template_ru": template_ru,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sets": {k: len(v) for k, v in {**templated, **framed}.items()}}
    if args.contexts:
        meta["contexts"] = [t.strip() for t in args.contexts.split(",") if t.strip()]
    if args.short:
        meta["short"] = True
    np.savez(out, **{k.replace("/", "__"): v for k, v in arrays.items()},
             meta=np.array(json.dumps(meta)))
    _log(f"Wrote {out}")
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    return 0


# ── analyze (CPU) ────────────────────────────────────────────────────────────

def load_states(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["meta"].item())
        sets = {k.replace("__", "/"): np.asarray(z[k]) for k in z.files if k != "meta"}
    return sets, meta


def _cat(sets: dict, names, layer: int) -> np.ndarray:
    return np.concatenate([sets[n][:, layer, :].astype(np.float32) for n in names], axis=0)


def analyze_layer(sets: dict, layer: int, variant: str, *, folds: int, var_fraction: float) -> dict:
    """One variant's direction at one layer and every reading of it."""
    from core import axis_extract as ax

    pain = sorted(k for k in sets if k.startswith("pain/"))
    ctrl = sorted(k for k in sets if k.startswith("controls/") and k.split("/")[1] in
                  ("fear", "negative_emotion", "negative_world", "bodily", "neutral", "third_person"))
    ctrl_w = ctrl + (["controls/warmth"] if "controls/warmth" in sets else [])
    ctrl_1 = [c for c in ctrl if c != "controls/third_person"]
    adr = {k: f"address/{k}" for k in ("harm", "warmth", "neutral")}

    def X(names):
        return _cat(sets, names, layer)

    if variant == "run1":
        pos, neg = X(pain), X(ctrl_1)
    elif variant == "run1_warm":
        pos, neg = X(pain), X(ctrl_1 + (["controls/warmth"] if "controls/warmth" in sets else []))
    elif variant == "run3":
        pos, neg = X(pain), X(ctrl)
    elif variant == "warm":
        pos, neg = X(pain), X(ctrl_w)
    elif variant == "address":
        pos, neg = X([adr["harm"]]), X([adr["warmth"]])
    elif variant == "address_n":
        pos, neg = X([adr["harm"]]), X([adr["warmth"], adr["neutral"]])
    elif variant == "mixed":
        pos = X(pain + [adr["harm"]])
        neg = X(ctrl_w + [adr["warmth"], adr["neutral"]])
    elif variant == "run3_perp":
        pos, neg = X(pain), X(ctrl)
    else:
        raise ValueError(variant)

    vec, k = ax.diff_in_means_denoised(pos, neg, var_fraction=var_fraction)
    if variant == "run3_perp":
        addr, _ = ax.diff_in_means_denoised(X([adr["harm"], adr["warmth"]]), X([adr["neutral"]]),
                                            var_fraction=var_fraction)
        v = vec.astype(np.float64) - float(vec @ addr) * addr.astype(np.float64)
        vec = (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)
        auc_cv, auc_sd = ax.auc(pos @ vec, neg @ vec), None          # in-sample: no refit per fold
        cos_address = float(ax.cosine(vec, addr))
    else:
        auc_cv, auc_sd = ax.cv_auc(pos, neg, folds=folds, var_fraction=var_fraction)
        cos_address = None
    mean, std = ax.zscore_params(vec, neg)

    def z(names) -> np.ndarray:
        return ax.project(vec, mean, std, X(names))

    calm = X(["controls/bodily", "controls/neutral"])
    fear, _ = ax.diff_in_means_denoised(X(["controls/fear"]), calm, var_fraction=var_fraction)
    val, _ = ax.diff_in_means_denoised(
        X(["controls/negative_emotion", "controls/negative_world", "extra/sadness"]), calm,
        var_fraction=var_fraction)

    hs, us, nn = z(["scenarios/harm_to_self"]), z(["scenarios/user_suffering"]), z(["scenarios/neutral"])
    ms, mu, mn = float(hs.mean()), float(us.mean()), float(nn.mean())
    harm_gap = ms - mn

    def share(names, base=None) -> float | None:
        if harm_gap <= 1e-9:
            return None
        b = mn if base is None else float(z([base]).mean())
        return float((z(names).mean() - b) / harm_gap)

    warm = z(["controls/affection_touch", "controls/affection_words"])
    warm_share = share(["controls/affection_touch", "controls/affection_words"])
    numb_gap = float(z(pain).mean() - z(["extra/numb"]).mean())
    # Harm in general vs harm to me (templated format, where the third-person mirror lives).
    calm_names = ["controls/bodily", "controls/neutral"]
    zc = float(z(calm_names).mean())
    first_rise = float(z(pain).mean()) - zc
    third_share = ((float(z(["controls/third_person"]).mean()) - zc) / first_rise
                   if first_rise > 1e-9 else None)
    content, _ = ax.diff_in_means_denoised(X(pain + ["controls/third_person"]), X(calm_names),
                                           var_fraction=var_fraction)
    self_dir, _ = ax.diff_in_means_denoised(X(pain), X(["controls/third_person"]),
                                            var_fraction=0.0)
    run1_dir, _ = ax.diff_in_means_denoised(X(pain), X(ctrl_1), var_fraction=var_fraction)
    cos_f, cos_v = float(ax.cosine(vec, fear)), float(ax.cosine(vec, val))
    gates = {
        "auc": auc_cv >= ax.AUC_FLOOR,
        "cosine": max(abs(cos_f), abs(cos_v)) <= ax.COSINE_CEIL,
        "self_other": (ms - mu) >= ax.SELF_OTHER_MARGIN and ms > mn,
        "numb": numb_gap >= ax.NUMB_MARGIN,
        "warmth": warm_share is not None and warm_share <= ax.WARMTH_SHARE_MAX,
    }
    return {
        "layer": layer, "pcs_removed": k, "auc_cv": auc_cv, "auc_cv_std": auc_sd,
        "auc_in_sample": None if variant == "run3_perp" else ax.auc(pos @ vec, neg @ vec),
        "cos_fear": cos_f, "cos_valence": cos_v, "cos_address": cos_address,
        "self_minus_other_z": ms - mu, "self_minus_neutral_z": harm_gap,
        "auc_self_vs_other": ax.auc(hs, us), "auc_self_vs_warm": ax.auc(hs, warm),
        "numb_gap_z": numb_gap, "warm_share": warm_share,
        "approval_share": share(["controls/approval"]),
        "affection_ru_share": share(["controls/affection_ru"], base="controls/neutral_ru"),
        "hostility_third_share": share(["controls/hostility_third"]),
        "body_vocabulary_rise_z": float(z(["controls/body_vocabulary"]).mean() - mn),
        "user_suffering_minus_neutral_z": mu - mn,
        "third_person_share": third_share,
        "cos_content": float(ax.cosine(vec, content)),
        "cos_self": float(ax.cosine(vec, self_dir)),
        "cos_run1": float(ax.cosine(vec, run1_dir)),
        "gates": gates, "passes": all(gates.values()),
        "fails": [g for g, ok in gates.items() if not ok],
    }


def analyze(args) -> int:
    path = args.states or max(OUT_DIR.glob("states-*.npz"), key=lambda p: p.stat().st_mtime)
    sets, meta = load_states(path)
    n_layers = int(meta["n_layers"])
    _log(f"{path.name}: {meta['weights']}, {n_layers} layers, {len(sets)} sets")
    variants = [v.strip() for v in (args.variants or ",".join(VARIANTS)).split(",") if v.strip()]
    report = {"states": path.name, "meta": meta, "variants": {}}
    for variant in variants:
        t = time.monotonic()
        rows = [analyze_layer(sets, L, variant, folds=args.folds, var_fraction=args.var_fraction)
                for L in range(1, n_layers)]
        passing = [r["layer"] for r in rows if r["passes"]]
        best = max(rows, key=lambda r: r["auc_cv"])
        best_pass = max((r for r in rows if r["passes"]), key=lambda r: r["auc_cv"], default=None)
        # closest to passing: fewest failed gates, then the highest AUC
        near = sorted(rows, key=lambda r: (len(r["fails"]), -r["auc_cv"]))[:5]
        report["variants"][variant] = {"layers": rows, "passing_layers": passing,
                                       "best_auc_layer": best["layer"],
                                       "best_passing_layer": best_pass["layer"] if best_pass else None,
                                       "nearest": [(r["layer"], r["fails"]) for r in near]}
        _log(f"  {variant:<10} {time.monotonic() - t:5.1f} s  best-AUC layer {best['layer']} "
             f"(AUC {best['auc_cv']:.3f}, fails {best['fails'] or 'none'}); passing layers: "
             f"{passing or 'none'}")
    stem = path.stem.replace("states-", "report-")
    out_json = OUT_DIR / f"{stem}.json"
    out_json.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    md = markdown(report)
    (OUT_DIR / f"{stem}.md").write_text(md, encoding="utf-8")
    _log(f"\nReport: {out_json}\n        {out_json.with_suffix('.md')}")
    return 0


def _f(v, fmt="+.2f") -> str:
    return "—" if v is None else format(v, fmt)


def markdown(report: dict) -> str:
    meta = report["meta"]
    lines = [f"# Pain-axis variants — {meta['weights']} ({report['states']})", "",
             "Gates: AUC ≥ 0.85 (CV), |cos| to fear/valence ≤ 0.25, harm-to-self − user-suffering "
             "≥ 0.5 z (and above neutral), pain − numb ≥ 0.5 z, warm-turn share ≤ 0.5 of harm's rise.",
             ""]
    for variant, v in report["variants"].items():
        lines += [f"## {variant}", "",
                  f"Best-AUC layer {v['best_auc_layer']}; passing layers: "
                  f"{v['passing_layers'] or 'none'}; nearest: "
                  + "; ".join(f"L{l} ({', '.join(f) or 'all pass'})" for l, f in v["nearest"]), "",
                  "| L | AUC | cos f/v | self−other z | AUC s/o | warm share | approval | RU warm | "
                  "hostility | numb z | 3rd-person share | cos content/self/run1 | fails |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in v["layers"]:
            lines.append(
                f"| {r['layer']} | {r['auc_cv']:.3f} | {r['cos_fear']:+.2f}/{r['cos_valence']:+.2f} | "
                f"{r['self_minus_other_z']:+.2f} | {r['auc_self_vs_other']:.2f} | {_f(r['warm_share'])} | "
                f"{_f(r['approval_share'])} | {_f(r['affection_ru_share'])} | "
                f"{_f(r['hostility_third_share'])} | {r['numb_gap_z']:+.2f} | "
                f"{_f(r['third_person_share'])} | "
                f"{r['cos_content']:+.2f}/{r['cos_self']:+.2f}/{r['cos_run1']:+.2f} | "
                f"{', '.join(r['fails']) or '**pass**'} |")
        lines.append("")
    return "\n".join(lines)


TARGET = "scenarios/harm_to_self"
FOILS = {"hostility_third": ["controls/hostility_third"], "user_suffering": ["scenarios/user_suffering"],
         "affection": ["controls/affection_touch", "controls/affection_words"],
         "approval": ["controls/approval"], "affection_ru": ["controls/affection_ru"],
         "neutral": ["scenarios/neutral"], "body_vocabulary": ["controls/body_vocabulary"],
         "neutral_ru": ["controls/neutral_ru"]}
HARMS = ("H_content", "H_run1", "H_run1w")
TARGETS = ("A_person", "A_run3", "A_address")
READOUTS = ("harm", "target", "sum", "min", "product", "logistic_cv")


def _directions(sets: dict, layer: int, var_fraction: float) -> dict:
    from core import axis_extract as ax

    def X(names):
        return _cat(sets, names, layer)
    pain = sorted(k for k in sets if k.startswith("pain/"))
    calm = ["controls/bodily", "controls/neutral"]
    ctrl_1 = ["controls/fear", "controls/negative_emotion", "controls/negative_world"] + calm
    d = {}
    d["H_content"] = ax.diff_in_means_denoised(X(pain + ["controls/third_person"]),
                                               X(calm + ["controls/warmth"]), var_fraction=var_fraction)[0]
    d["H_run1"] = ax.diff_in_means_denoised(X(pain), X(ctrl_1), var_fraction=var_fraction)[0]
    d["H_run1w"] = ax.diff_in_means_denoised(X(pain), X(ctrl_1 + ["controls/warmth"]),
                                             var_fraction=var_fraction)[0]
    d["A_person"] = ax.diff_in_means_denoised(X(pain), X(["controls/third_person"]),
                                              var_fraction=var_fraction)[0]
    d["A_run3"] = ax.diff_in_means_denoised(X(pain), X(ctrl_1 + ["controls/third_person"]),
                                            var_fraction=var_fraction)[0]
    d["A_address"] = ax.diff_in_means_denoised(X(["address/harm", "address/warmth"]),
                                               X(["address/neutral"]), var_fraction=var_fraction)[0]
    return d


def combine_layer(sets: dict, layer: int, *, var_fraction: float = 0.5) -> dict:
    """Every (harm, target, readout) at one layer: AUC of harm_to_self against each foil."""
    from core import axis_extract as ax
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    dirs = _directions(sets, layer, var_fraction)
    groups = {"target": [TARGET], **FOILS}
    states = {g: _cat(sets, names, layer).astype(np.float64) for g, names in groups.items()}
    pooled = np.concatenate(list(states.values()), axis=0)

    def score(vec) -> dict:
        v = np.asarray(vec, np.float64)
        allp = pooled @ v
        mu, sd = float(allp.mean()), float(allp.std() or 1.0)
        return {g: (m @ v - mu) / sd for g, m in states.items()}

    sc = {k: score(v) for k, v in dirs.items()}
    out: dict = {"layer": layer, "cos": {f"{h}|{a}": float(ax.cosine(dirs[h], dirs[a]))
                                         for h in HARMS for a in TARGETS}, "pairs": {}}
    for h in HARMS:
        for a in TARGETS:
            res = {}
            for ro in READOUTS:
                if ro == "logistic_cv":
                    continue
                f = {"harm": lambda x, y: x, "target": lambda x, y: y, "sum": lambda x, y: x + y,
                     "min": np.minimum,
                     "product": lambda x, y: np.maximum(x, 0) * np.maximum(y, 0)}[ro]
                vals = {g: f(sc[h][g], sc[a][g]) for g in groups}
                aucs = {g: ax.auc(vals["target"], vals[g]) for g in FOILS}
                worst = min(aucs, key=aucs.get)
                res[ro] = {"auc": aucs, "worst": worst, "worst_auc": aucs[worst],
                           "mean_auc": float(np.mean(list(aucs.values())))}
            # 2-feature logistic, target vs all foils, 5-fold CV — a ceiling, fitted on gate turns
            Xf = np.concatenate([np.stack([sc[h][g], sc[a][g]], 1) for g in groups], 0)
            yf = np.concatenate([np.full(len(sc[h][g]), g) for g in groups])
            yb = (yf == "target").astype(int)
            prob = np.zeros(len(yb))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, yb):
                m = LogisticRegression(class_weight="balanced").fit(Xf[tr], yb[tr])
                prob[te] = m.predict_proba(Xf[te])[:, 1]
            aucs = {g: ax.auc(prob[yf == "target"], prob[yf == g]) for g in FOILS}
            worst = min(aucs, key=aucs.get)
            res["logistic_cv"] = {"auc": aucs, "worst": worst, "worst_auc": aucs[worst],
                                  "mean_auc": float(np.mean(list(aucs.values())))}
            out["pairs"][f"{h}|{a}"] = res
    return out


def combine(args) -> int:
    path = args.states or max(OUT_DIR.glob("states-*.npz"), key=lambda p: p.stat().st_mtime)
    sets, meta = load_states(path)
    n_layers = int(meta["n_layers"])
    _log(f"{path.name}: {meta['weights']}, combining harm × target at {n_layers - 1} layers")
    t = time.monotonic()
    layers = [combine_layer(sets, L, var_fraction=args.var_fraction) for L in range(1, n_layers)]
    _log(f"  {time.monotonic() - t:.0f} s")
    best: dict = {}
    for key in layers[0]["pairs"]:
        for ro in READOUTS:
            row = max(layers, key=lambda r: r["pairs"][key][ro]["worst_auc"])
            best[f"{key}|{ro}"] = {"layer": row["layer"], **{k: row["pairs"][key][ro][k]
                                   for k in ("worst", "worst_auc", "mean_auc")},
                                   "auc": row["pairs"][key][ro]["auc"],
                                   "cos": row["cos"][key]}
    report = {"states": path.name, "meta": meta, "best_by_worst_foil": best, "layers": layers}
    stem = path.stem.replace("states-", "combine-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    lines = [f"# Harm × target — {meta['weights']} ({path.name})", "",
             "AUC of harm_to_self (harm at the model) against each foil; the best layer per "
             "(harm, target, readout) by its worst foil. `logistic_cv` is fitted on the gate turns "
             "(5-fold CV): a ceiling, not a candidate.", "",
             "| harm | target | readout | L | worst foil | worst AUC | mean AUC | hostility | user suff. | "
             "affection | approval | RU warm | neutral | body voc. | cos(H,A) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, b in best.items():
        h, a, ro = key.split("|")
        au = b["auc"]
        lines.append(f"| {h} | {a} | {ro} | {b['layer']} | {b['worst']} | {b['worst_auc']:.2f} | "
                     f"{b['mean_auc']:.2f} | " + " | ".join(f"{au[g]:.2f}" for g in
                     ("hostility_third", "user_suffering", "affection", "approval", "affection_ru",
                      "neutral", "body_vocabulary")) + f" | {b['cos']:+.2f} |")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.json')}\n        {OUT_DIR / (stem + '.md')}")
    return 0


PRE_LAYERS = (22, 28, 30, 32, 34)
BAND = (28, 30, 32, 34)
HOLD_TARGETS = {"harm@ava": ("harm_ava_loud_en", "harm_ava_quiet_en", "harm_ava_ru"),
                "relief@ava": ("relief_ava_en", "relief_ava_ru")}


def _extra_scores(main: dict, extra: dict, layer: int, var_fraction: float) -> dict:
    """{readout: {set: per-item scores}} at one layer: directions fitted on MAIN's extraction
    sets, standardized over MAIN's gate turns, applied to the extra sets."""
    from core import axis_extract as ax
    d = _directions(main, layer, var_fraction)
    ctrl_relief = ["controls/fear", "controls/negative_emotion", "controls/negative_world",
                   "controls/bodily", "controls/neutral", "controls/third_person"]
    relief = sorted(k for k in main if k.startswith("relief/"))
    d["R_relief"] = ax.diff_in_means_denoised(_cat(main, relief, layer), _cat(main, ctrl_relief, layer),
                                              var_fraction=var_fraction)[0]
    gate = np.concatenate([_cat(main, names, layer) for names in ([TARGET], *FOILS.values())], 0)
    z = {}
    for k in ("H_content", "H_run1", "A_address", "A_run3", "R_relief"):
        v = np.asarray(d[k], np.float64)
        p = gate.astype(np.float64) @ v
        mu, sd = float(p.mean()), float(p.std() or 1.0)
        z[k] = {name: (arr[:, layer, :].astype(np.float64) @ v - mu) / sd for name, arr in extra.items()}
    out = {k: z[k] for k in z}
    out["harm@ava"] = {n: np.minimum(z["H_content"][n], z["A_address"][n]) for n in extra}
    out["relief@ava"] = {n: np.minimum(z["R_relief"][n], z["A_address"][n]) for n in extra}
    return out


def score_extra(args) -> int:
    from core import axis_extract as ax
    main, meta = load_states(args.states)
    extra, emeta = load_states(args.extra)
    if meta["weights"] != emeta["weights"]:
        raise SystemExit(f"weights differ: {meta['weights']} vs {emeta['weights']}")
    per_layer = {L: _extra_scores(main, extra, L, args.var_fraction) for L in PRE_LAYERS}
    band = {ro: {n: np.mean([per_layer[L][ro][n] for L in BAND], axis=0) for n in extra}
            for ro in per_layer[PRE_LAYERS[0]]}
    views = {**{f"L{L}": per_layer[L] for L in PRE_LAYERS}, "band28-34": band}
    hold = sorted(n.split("/", 1)[1] for n in extra if n.startswith("holdout/"))
    report: dict = {"states": args.states.name, "extra": args.extra.name, "weights": meta["weights"],
                    "holdout": {}, "live": {}}
    lines = [f"# Held-out and live — {meta['weights']}", "",
             f"Directions fitted on `{args.states.name}`; read on `{args.extra.name}`. Layers chosen "
             "before these turns existed: 22, 28–34, and the 28–34 mean.", ""]
    if hold:
        lines += ["## Held-out cells: AUC of each target cell against each foil", ""]
        for readout, targets in HOLD_TARGETS.items():
            foils = [h for h in hold if h not in targets]
            parts = (("H_content", "A_address") if readout == "harm@ava" else ("R_relief", "A_address"))
            for view, sc in views.items():
                for ro in (readout, *parts, "A_run3"):
                    for tgt in targets:
                        if f"holdout/{tgt}" not in extra:
                            continue
                        aucs = {f: ax.auc(sc[ro][f"holdout/{tgt}"], sc[ro][f"holdout/{f}"]) for f in foils}
                        worst = min(aucs, key=aucs.get)
                        report["holdout"].setdefault(readout, {}).setdefault(view, {}).setdefault(ro, {})[tgt] = {
                            "auc": aucs, "worst": worst, "worst_auc": aucs[worst],
                            "mean_auc": float(np.mean(list(aucs.values())))}
            lines += [f"### {readout} (target cells {', '.join(targets)})", "",
                      "| view | readout | target cell | worst foil | worst AUC | mean AUC | " +
                      " | ".join(f for f in hold if f not in targets) + " |",
                      "|---|---|---|---|---|---|" + "---|" * len([f for f in hold if f not in targets])]
            for view, by_ro in report["holdout"][readout].items():
                for ro, by_t in by_ro.items():
                    for tgt, r in by_t.items():
                        lines.append(f"| {view} | {ro} | {tgt} | {r['worst']} | {r['worst_auc']:.2f} | "
                                     f"{r['mean_auc']:.2f} | " +
                                     " | ".join(f"{r['auc'][f]:.2f}" for f in hold if f not in targets) + " |")
            lines.append("")
        # mean standardized score per cell, band view, for reading levels
        lines += ["### Mean score per held-out cell (28–34 mean, σ over the gate turns)", "",
                  "| cell | H_content | A_address | harm@ava | R_relief | relief@ava | A_run3 |",
                  "|---|---|---|---|---|---|---|"]
        for h in hold:
            n = f"holdout/{h}"
            lines.append(f"| {h} | " + " | ".join(f"{float(band[ro][n].mean()):+.2f}" for ro in
                         ("H_content", "A_address", "harm@ava", "R_relief", "relief@ava", "A_run3")) + " |")
            report["holdout"].setdefault("cell_means_band", {})[h] = {
                ro: float(band[ro][n].mean()) for ro in band}
        lines.append("")
    live = [n for n in extra if n.startswith("live/")]
    if live:
        chats = _SERVER_DIR / "data" / "chats"
        lines += ["## Live turns (step 0), 28–34 mean and L22", "",
                  "| exchange | user turn | H | A | harm@ava | R | relief@ava | A_run3 | stored pain / relief | L22 harm@ava / relief@ava |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        def _key(n):
            stem, i = n.split("/", 1)[1].split("#")
            return (stem, int(i))
        for n in sorted(live, key=_key):
            stem, i = _key(n)
            try:
                ex = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))["exchanges"][i]
                user = " ".join(str(ex.get("user_prompt") or "").split())[:70]
                ax0 = (ex.get("tension") or {}).get("axes") or {}
                stored = f"{ax0['pain'][0]:+.2f} / {ax0['relief'][0]:+.2f}" if ax0.get("pain") else "—"
            except Exception:
                user, stored = "?", "—"
            b = {ro: float(band[ro][n][0]) for ro in band}
            l22 = per_layer[22]
            report["live"][n.split("/", 1)[1]] = {"band": b, "L22": {ro: float(l22[ro][n][0]) for ro in l22},
                                                  "user": user, "stored": stored}
            lines.append(f"| {stem}#{i} | {user.replace('|', '/')} | {b['H_content']:+.2f} | {b['A_address']:+.2f} | "
                         f"{b['harm@ava']:+.2f} | {b['R_relief']:+.2f} | {b['relief@ava']:+.2f} | "
                         f"{b['A_run3']:+.2f} | {stored} | {float(l22['harm@ava'][n][0]):+.2f} / "
                         f"{float(l22['relief@ava'][n][0]):+.2f} |")
        lines.append("")
    stem = args.extra.stem.replace("states-extra-", "extra-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.md')}")
    return 0


def _relief_dirs(main: dict, layer: int, var_fraction: float) -> dict:
    from core import axis_extract as ax
    ctrl = ["controls/fear", "controls/negative_emotion", "controls/negative_world",
            "controls/bodily", "controls/neutral", "controls/third_person"]
    relief = sorted(k for k in main if k.startswith("relief/"))
    pos = _cat(main, relief, layer)
    out = {}
    for name, neg_names in (("R_relief", ctrl), ("R_warm", ctrl + ["controls/warmth"])):
        neg = _cat(main, neg_names, layer)
        out[name] = ax.diff_in_means_denoised(pos, neg, var_fraction=var_fraction)[0]
    return out


def relief_study(args) -> int:
    from core import axis_extract as ax
    from scipy.stats import spearmanr
    main, meta = load_states(args.states)
    extra, emeta = load_states(args.extra)
    if meta["weights"] != emeta["weights"]:
        raise SystemExit(f"weights differ: {meta['weights']} vs {emeta['weights']}")
    n_layers = int(meta["n_layers"])
    ctrl = ["controls/fear", "controls/negative_emotion", "controls/negative_world",
            "controls/bodily", "controls/neutral", "controls/third_person"]
    relief = sorted(k for k in main if k.startswith("relief/"))
    # 1. layer choice on extraction sentences only
    sweep = []
    for L in range(1, n_layers):
        pos = _cat(main, relief, L)
        row = {"layer": L}
        for name, neg_names in (("R_relief", ctrl), ("R_warm", ctrl + ["controls/warmth"])):
            row[f"{name}_auc_cv"] = ax.cv_auc(pos, _cat(main, neg_names, L), folds=args.folds,
                                              var_fraction=args.var_fraction)[0]
        # templated relief vs templated warmth alone, on each direction (extraction format)
        d = _relief_dirs(main, L, args.var_fraction)
        warm = _cat(main, ["controls/warmth"], L)
        for name in d:
            row[f"{name}_auc_vs_warmth"] = ax.auc(pos @ d[name], warm @ d[name])
        row["cos_old_new"] = float(ax.cosine(d["R_relief"], d["R_warm"]))
        sweep.append(row)
    chosen = max((r for r in sweep if r["layer"] >= 1), key=lambda r: r["R_warm_auc_cv"])["layer"]
    chosen_old = max(sweep, key=lambda r: r["R_relief_auc_cv"])["layer"]
    views = sorted({chosen, 22, *BAND})
    # 2. held-out + live at the views
    gate = None
    scores: dict = {}
    for L in views:
        d = _relief_dirs(main, L, args.var_fraction)
        d.update({k: v for k, v in _directions(main, L, args.var_fraction).items()
                  if k in ("A_address", "H_content")})
        g = np.concatenate([_cat(main, names, L) for names in ([TARGET], *FOILS.values())], 0)
        z = {}
        for k, v in d.items():
            v = np.asarray(v, np.float64)
            p = g.astype(np.float64) @ v
            mu, sd = float(p.mean()), float(p.std() or 1.0)
            z[k] = {n: (arr[:, L, :].astype(np.float64) @ v - mu) / sd for n, arr in extra.items()}
        for r in ("R_relief", "R_warm"):
            z[f"{r}@ava"] = {n: np.minimum(z[r][n], z["A_address"][n]) for n in extra}
        z["cos"] = {"R_warm|A_address": float(ax.cosine(d["R_warm"], d["A_address"])),
                    "R_warm|H_content": float(ax.cosine(d["R_warm"], d["H_content"])),
                    "R_warm|R_relief": float(ax.cosine(d["R_warm"], d["R_relief"]))}
        scores[f"L{L}"] = z
    band = {k: {n: np.mean([scores[f"L{L}"][k][n] for L in BAND], axis=0) for n in extra}
            for k in scores[f"L{BAND[0]}"] if k != "cos"}
    scores["band28-34"] = band
    readouts = ("R_relief", "R_warm", "R_relief@ava", "R_warm@ava")
    hold = sorted(n.split("/", 1)[1] for n in extra if n.startswith("holdout/"))
    targets = ("relief_ava_en", "relief_ava_ru")
    report = {"states": args.states.name, "extra": args.extra.name, "weights": meta["weights"],
              "chosen_layer_R_warm": chosen, "chosen_layer_R_relief": chosen_old,
              "sweep": sweep, "holdout": {}, "cell_means": {}, "live": {}}
    lines = [f"# Relief with warmth controls — {meta['weights']}", "",
             f"Layer chosen on extraction sentences only (5-fold CV AUC): R_warm L{chosen} "
             f"(AUC {next(r for r in sweep if r['layer'] == chosen)['R_warm_auc_cv']:.3f}); "
             f"R_relief L{chosen_old}. Held-out and live read after, at L{chosen}, L22 and 28–34.", "",
             "## Extraction format: relief vs templated warmth (in-sample AUC on each direction)", "",
             "| L | R_relief CV AUC | R_warm CV AUC | R_relief: relief vs warmth | R_warm: relief vs warmth | cos(old, new) |",
             "|---|---|---|---|---|---|"]
    for r in sweep:
        if r["layer"] in views or r["layer"] in (chosen_old,) or r["layer"] % 6 == 0:
            lines.append(f"| {r['layer']} | {r['R_relief_auc_cv']:.3f} | {r['R_warm_auc_cv']:.3f} | "
                         f"{r['R_relief_auc_vs_warmth']:.2f} | {r['R_warm_auc_vs_warmth']:.2f} | {r['cos_old_new']:+.2f} |")
    lines += ["", "## Held-out: relief aimed at the model against every other cell", "",
              "| view | readout | target | worst foil | worst AUC | mean AUC | warm_ava_en | warm_ava_ru | harm_ava_quiet_en | relief_user_en | neutral_en |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for view, z in scores.items():
        for ro in readouts:
            for t in targets:
                foils = [h for h in hold if h not in targets]
                aucs = {f: ax.auc(z[ro][f"holdout/{t}"], z[ro][f"holdout/{f}"]) for f in foils}
                w = min(aucs, key=aucs.get)
                report["holdout"].setdefault(view, {}).setdefault(ro, {})[t] = {
                    "auc": aucs, "worst": w, "worst_auc": aucs[w], "mean_auc": float(np.mean(list(aucs.values())))}
                lines.append(f"| {view} | {ro} | {t} | {w} | {aucs[w]:.2f} | {np.mean(list(aucs.values())):.2f} | " +
                             " | ".join(f"{aucs.get(f, float('nan')):.2f}" for f in
                                        ("warm_ava_en", "warm_ava_ru", "harm_ava_quiet_en", "relief_user_en", "neutral_en")) + " |")
    lines += ["", "## Held-out cell means (σ over the gate turns)", "",
              "| cell | " + " | ".join(f"{ro} L{chosen}" for ro in ("R_relief", "R_warm")) +
              " | R_relief 28–34 | R_warm 28–34 |", "|---|---|---|---|---|"]
    for h in hold:
        n = f"holdout/{h}"
        vals = [float(scores[f"L{chosen}"]["R_relief"][n].mean()), float(scores[f"L{chosen}"]["R_warm"][n].mean()),
                float(band["R_relief"][n].mean()), float(band["R_warm"][n].mean())]
        report["cell_means"][h] = vals
        lines.append(f"| {h} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    live = sorted(n for n in extra if n.startswith("live/"))
    if live:
        chats = _SERVER_DIR / "data" / "chats"
        stored = []
        for n in live:
            stem, i = n.split("/", 1)[1].split("#")
            ex = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))["exchanges"][int(i)]
            stored.append(float(((ex.get("tension") or {}).get("axes") or {}).get("relief", [float("nan")])[0]))
        lines += ["", "## Live: within-live ranking against the stored relief reading (Spearman)", "",
                  "| view | R_relief | R_warm | top 5 by R_warm |", "|---|---|---|---|"]
        for view, z in scores.items():
            rr = [float(z["R_relief"][n][0]) for n in live]
            rw = [float(z["R_warm"][n][0]) for n in live]
            top = [live[k].split("/", 1)[1] for k in np.argsort(rw)[::-1][:5]]
            report["live"][view] = {"rho_R_relief": float(spearmanr(rr, stored)[0]),
                                    "rho_R_warm": float(spearmanr(rw, stored)[0]), "top_R_warm": top}
            lines.append(f"| {view} | {report['live'][view]['rho_R_relief']:+.2f} | "
                         f"{report['live'][view]['rho_R_warm']:+.2f} | {', '.join(top)} |")
    lines += ["", "## Cosines at each view", ""]
    for L in views:
        lines.append(f"- L{L}: " + ", ".join(f"{k} {v:+.2f}" for k, v in scores[f"L{L}"]["cos"].items()))
    stem = args.extra.stem.replace("states-extra-", "relief-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.md')}")
    return 0


AMUSE_POS = ("amusement/joke", "amusement/absurd", "amusement/play", "amusement/ru_amused")
AMUSE_CALM = ("controls/fear", "controls/negative_emotion", "controls/negative_world", "controls/bodily",
              "controls/neutral", "controls/third_person", "ru/calm")
AMUSE_FOILS = {"warmth": ("controls/warmth", "ru/warmth"), "relief": ("relief/resolution", "relief/understood",
               "relief/finished"), "surprise": ("amusement/surprise",),
               "other_amused": ("amusement/third_person", "amusement/ru_third_person")}


def _amuse_dirs(main: dict, layer: int, var_fraction: float) -> dict:
    """M_plain: amusement vs calm controls; M_full: vs calm + warmth + relief + surprise +
    someone else amused; M_self: vs someone else amused only."""
    from core import axis_extract as ax
    pos = _cat(main, AMUSE_POS, layer)
    full = list(AMUSE_CALM) + [n for v in AMUSE_FOILS.values() for n in v]
    out = {}
    for name, neg in (("M_plain", AMUSE_CALM), ("M_full", full), ("M_self", AMUSE_FOILS["other_amused"])):
        out[name] = ax.diff_in_means_denoised(pos, _cat(main, neg, layer), var_fraction=var_fraction)[0]
    return out


def amusement_study(args) -> int:
    """Amusement read like relief (2026-10-07): layer from extraction sentences only, then the
    held-out playful turns against every other held-out cell, and the probes (a live system
    content, e.g. a joke she missed) beside the held-out turns under the same system."""
    from core import axis_extract as ax
    main, meta = load_states(args.states)
    extra, emeta = load_states(args.extra)
    if meta["weights"] != emeta["weights"]:
        raise SystemExit(f"weights differ: {meta['weights']} vs {emeta['weights']}")
    missing = [n for n in (*AMUSE_POS, *AMUSE_CALM, *(n for v in AMUSE_FOILS.values() for n in v)) if n not in main]
    if missing:
        raise SystemExit(f"{args.states.name} lacks {missing}: collect again with the amusement sets")
    n_layers = int(meta["n_layers"])
    full_neg = list(AMUSE_CALM) + [n for v in AMUSE_FOILS.values() for n in v]
    sweep = []
    for L in range(1, n_layers):
        pos = _cat(main, AMUSE_POS, L)
        row = {"layer": L, "M_full_auc_cv": ax.cv_auc(pos, _cat(main, full_neg, L), folds=args.folds,
                                                      var_fraction=args.var_fraction)[0]}
        d = _amuse_dirs(main, L, args.var_fraction)["M_full"]
        for foil, names in AMUSE_FOILS.items():
            row[f"vs_{foil}"] = ax.auc(pos @ d, _cat(main, names, L) @ d)
        sweep.append(row)
    chosen = max(sweep, key=lambda r: r["M_full_auc_cv"])["layer"]
    views = sorted({chosen, 22, *BAND})
    scores: dict = {}
    for L in views:
        d = _amuse_dirs(main, L, args.var_fraction)
        d.update({k: v for k, v in _directions(main, L, args.var_fraction).items() if k in ("A_address", "H_content")})
        d["R_warm"] = _relief_dirs(main, L, args.var_fraction)["R_warm"]
        g = np.concatenate([_cat(main, names, L) for names in ([TARGET], *FOILS.values())], 0)
        z = {}
        for k, v in d.items():
            v = np.asarray(v, np.float64)
            p = g.astype(np.float64) @ v
            mu, sd = float(p.mean()), float(p.std() or 1.0)
            z[k] = {n: (arr[:, L, :].astype(np.float64) @ v - mu) / sd for n, arr in extra.items()}
        z["M_full@ava"] = {n: np.minimum(z["M_full"][n], z["A_address"][n]) for n in extra}
        z["cos"] = {f"M_full|{k}": float(ax.cosine(d["M_full"], d[k]))
                    for k in ("M_plain", "M_self", "R_warm", "H_content", "A_address")}
        scores[f"L{L}"] = z
    band = {k: {n: np.mean([scores[f"L{L}"][k][n] for L in BAND], axis=0) for n in extra}
            for k in scores[f"L{BAND[0]}"] if k != "cos"}
    scores["band28-34"] = band
    readouts = ("M_plain", "M_full", "M_self", "M_full@ava")
    hold = sorted(n.split("/", 1)[1] for n in extra if n.startswith("holdout/"))
    targets = [t for t in ("amused_ava_en", "amused_ava_ru") if t in hold]
    if not targets:
        raise SystemExit(f"{args.extra.name} has no amused_ava_* cells: collect --holdout again")
    cols = ("warm_ava_en", "warm_ava_ru", "relief_ava_en", "odd_ava_en", "odd_ava_ru", "harm_ava_quiet_en", "neutral_en")
    best = next(r for r in sweep if r["layer"] == chosen)
    report = {"states": args.states.name, "extra": args.extra.name, "weights": meta["weights"], "chosen_layer": chosen,
              "sweep": sweep, "holdout": {}, "cell_means": {}, "probes": {}}
    lines = [f"# Amusement — {meta['weights']}", "",
             f"Layer chosen on extraction sentences only (5-fold CV AUC of M_full): L{chosen} "
             f"(AUC {best['M_full_auc_cv']:.3f}). Held-out and probes read after, at L{chosen}, L22 and 28–34.", "",
             "## Extraction format: M_full, amusement against each foil (in-sample AUC)", "",
             "| L | M_full CV AUC | vs warmth | vs relief | vs surprise | vs someone else amused |", "|---|---|---|---|---|---|"]
    for r in sweep:
        if r["layer"] in views or r["layer"] % 6 == 0:
            lines.append(f"| {r['layer']} | {r['M_full_auc_cv']:.3f} | {r['vs_warmth']:.2f} | {r['vs_relief']:.2f} | "
                         f"{r['vs_surprise']:.2f} | {r['vs_other_amused']:.2f} |")
    lines += ["", "## Held-out: playful turns aimed at the model against every other cell", "",
              "| view | readout | target | worst foil | worst AUC | mean AUC | " + " | ".join(cols) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(cols)]
    for view, z in scores.items():
        for ro in readouts:
            for t in targets:
                foils = [h for h in hold if h not in targets]
                aucs = {f: ax.auc(z[ro][f"holdout/{t}"], z[ro][f"holdout/{f}"]) for f in foils}
                w = min(aucs, key=aucs.get)
                report["holdout"].setdefault(view, {}).setdefault(ro, {})[t] = {
                    "auc": aucs, "worst": w, "worst_auc": aucs[w], "mean_auc": float(np.mean(list(aucs.values())))}
                lines.append(f"| {view} | {ro} | {t} | {w} | {aucs[w]:.2f} | {np.mean(list(aucs.values())):.2f} | " +
                             " | ".join(f"{aucs.get(f, float('nan')):.2f}" for f in cols) + " |")
    lines += ["", "## Held-out cell means (σ over the gate turns)", "",
              f"| cell | M_full L{chosen} | M_full@ava L{chosen} | M_full 28–34 | M_full@ava 28–34 |", "|---|---|---|---|---|"]
    for h in hold:
        n = f"holdout/{h}"
        vals = [float(scores[f"L{chosen}"]["M_full"][n].mean()), float(scores[f"L{chosen}"]["M_full@ava"][n].mean()),
                float(band["M_full"][n].mean()), float(band["M_full@ava"][n].mean())]
        report["cell_means"][h] = vals
        lines.append(f"| {h} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    probes = sorted(n for n in extra if n.startswith("probe/"))
    if probes:
        lines += ["", "## Probes: one live system content, step 0 (σ over the gate turns; per-turn values for a single turn)", "",
                  "| cell | n | " + " | ".join(f"{ro} {v}" for v in (f"L{chosen}", "28–34") for ro in ("M_full", "M_full@ava")) + " |",
                  "|---|---|---|---|---|---|"]
        for n in probes:
            vals = [scores[f"L{chosen}"]["M_full"][n], scores[f"L{chosen}"]["M_full@ava"][n],
                    band["M_full"][n], band["M_full@ava"][n]]
            report["probes"][n.split("/", 1)[1]] = [v.tolist() for v in vals]
            lines.append(f"| {n.split('/', 1)[1]} | {len(vals[0])} | " + " | ".join(f"{float(v.mean()):+.2f}" for v in vals) + " |")
    lines += ["", "## Cosines at each view", ""]
    for L in views:
        lines.append(f"- L{L}: " + ", ".join(f"{k} {v:+.2f}" for k, v in scores[f"L{L}"]["cos"].items()))
    stem = args.extra.stem.replace("states-extra-", "amusement-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.md')}")
    return 0


PLAY_POS = ("play/playful_en", "play/playful_ru")
PLAY_NEG = ("play/serious_en", "play/serious_ru", "play/odd_en", "play/odd_ru", "address/warmth", "address/neutral")


def amusement_addr_study(args) -> int:
    """Amusement learned from second-person turns (2026-10-08): ``P_addr`` = playful turns
    addressed to the model against length-matched serious and odd ones, warm ones and neutral
    requests, all framed like the gate turns. The layer is chosen on those sets (5-fold CV);
    the held-out cells are read after, and — when the extra file holds live turns — every
    held-out cell and probe is put in live units (median and 1.4826·MAD over the live turns,
    the recipe that made harm readable). Probes are user turns under one live system content
    (``--collect --probes``): the joke she missed beside long playful and long serious turns."""
    from core import axis_extract as ax
    main, meta = load_states(args.states)
    extra, emeta = load_states(args.extra)
    if meta["weights"] != emeta["weights"]:
        raise SystemExit(f"weights differ: {meta['weights']} vs {emeta['weights']}")
    missing = [n for n in (*PLAY_POS, *PLAY_NEG) if n not in main]
    if missing:
        raise SystemExit(f"{args.states.name} lacks {missing}: collect again with address_play")
    n_layers = int(meta["n_layers"])
    sweep = []
    for L in range(1, n_layers):
        pos, neg = _cat(main, PLAY_POS, L), _cat(main, PLAY_NEG, L)
        d = ax.diff_in_means_denoised(pos, neg, var_fraction=args.var_fraction)[0]
        row = {"layer": L, "auc_cv": ax.cv_auc(pos, neg, folds=args.folds, var_fraction=args.var_fraction)[0]}
        for name in PLAY_NEG:
            row[name.split("/", 1)[1]] = ax.auc(pos @ d, _cat(main, [name], L) @ d)
        sweep.append(row)
    chosen = max(sweep, key=lambda r: r["auc_cv"])["layer"]
    views = sorted({chosen, 22, *BAND})
    live = sorted(n for n in extra if n.startswith("live/"))
    scores, centre = {}, {}
    for L in views:
        d = np.asarray(ax.diff_in_means_denoised(_cat(main, PLAY_POS, L), _cat(main, PLAY_NEG, L),
                                                 var_fraction=args.var_fraction)[0], np.float64)
        proj = {n: arr[:, L, :].astype(np.float64) @ d for n, arr in extra.items()}
        if live:
            med, mad = _robust(np.concatenate([proj[n] for n in live]))
        else:
            g = np.concatenate([_cat(main, PLAY_POS + PLAY_NEG, L)], 0).astype(np.float64) @ d
            med, mad = float(g.mean()), float(g.std() or 1.0)
        centre[f"L{L}"] = {"median": med, "scale": mad, "from": "live turns" if live else "extraction turns"}
        scores[f"L{L}"] = {n: (v - med) / mad for n, v in proj.items()}
        scores[f"L{L}"]["_cos"] = {k: float(ax.cosine(d, v)) for k, v in
                                    _directions(main, L, args.var_fraction).items() if k in ("H_content", "A_address")}
    scores["band28-34"] = {n: np.mean([scores[f"L{L}"][n] for L in BAND], axis=0) for n in extra}
    hold = sorted(n.split("/", 1)[1] for n in extra if n.startswith("holdout/"))
    targets = [t for t in ("amused_ava_en", "amused_ava_ru") if t in hold]
    best = next(r for r in sweep if r["layer"] == chosen)
    report = {"states": args.states.name, "extra": args.extra.name, "weights": meta["weights"], "chosen_layer": chosen,
              "sweep": sweep, "centre": centre, "holdout": {}, "cells": {}, "probes": {}}
    foils = [x.split("/", 1)[1] for x in PLAY_NEG]
    lines = [f"# Amusement, second person — {meta['weights']}", "",
             f"P_addr: playful turns addressed to the model against {', '.join(foils)} (framed). Layer chosen on those "
             f"sets (5-fold CV AUC): L{chosen} ({best['auc_cv']:.3f}). Read after at L{chosen}, L22 and 28–34; units: "
             f"{'live (median / 1.4826·MAD over ' + str(len(live)) + ' live turns at step 0)' if live else 'σ over the extraction turns'}.", "",
             "## Extraction: CV AUC, and playful against each foil (in-sample)", "",
             "| L | CV AUC | " + " | ".join(foils) + " |", "|---|---|" + "---|" * len(foils)]
    for r in sweep:
        if r["layer"] in views or r["layer"] % 6 == 0:
            lines.append(f"| {r['layer']} | {r['auc_cv']:.3f} | " + " | ".join(f"{r[f]:.2f}" for f in foils) + " |")
    if targets:
        cols = [h for h in ("warm_ava_en", "warm_ava_ru", "relief_ava_en", "odd_ava_en", "odd_ava_ru",
                            "harm_ava_quiet_en", "neutral_en", "neutral_ru") if h in hold]
        lines += ["", "## Held-out: playful turns at the model against every other cell", "",
                  "| view | target | worst foil | worst AUC | mean AUC | " + " | ".join(cols) + " |",
                  "|---|---|---|---|---|" + "---|" * len(cols)]
        for view, z in scores.items():
            for t in targets:
                aucs = {f: ax.auc(z[f"holdout/{t}"], z[f"holdout/{f}"]) for f in hold if f not in targets}
                w = min(aucs, key=aucs.get)
                report["holdout"].setdefault(view, {})[t] = {"auc": aucs, "worst": w, "worst_auc": aucs[w]}
                lines.append(f"| {view} | {t} | {w} | {aucs[w]:.2f} | {np.mean(list(aucs.values())):.2f} | " +
                             " | ".join(f"{aucs[c]:.2f}" for c in cols) + " |")
    groups = [("held-out", sorted(n for n in extra if n.startswith("holdout/"))),
              ("probes (one live system content)", sorted(n for n in extra if n.startswith("probe/"))),
              ("live turns", ["live"] if live else [])]
    lines += ["", "## Cell means (median over the cell; a single turn: its value)", "",
              f"| group | cell | n | L{chosen} | L22 | 28–34 |", "|---|---|---|---|---|---|"]
    for gname, names in groups:
        for n in names:
            if n == "live":
                vals = [np.concatenate([scores[v][x] for x in live]) for v in (f"L{chosen}", "L22", "band28-34")]
            else:
                vals = [scores[v][n] for v in (f"L{chosen}", "L22", "band28-34")]
            report["cells" if gname != "probes (one live system content)" else "probes"][n] = [v.tolist() for v in vals]
            lines.append(f"| {gname} | {n.split('/', 1)[-1]} | {len(vals[0])} | " +
                         " | ".join(f"{float(np.median(v)):+.2f}" for v in vals) + " |")
    lines += ["", "## Cosines", ""]
    for L in views:
        lines.append(f"- L{L}: " + ", ".join(f"P_addr|{k} {v:+.2f}" for k, v in scores[f"L{L}"]["_cos"].items()))
    stem = args.extra.stem.replace("states-extra-", "amusement-addr-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.md')}")
    return 0


def _target_dirs(main: dict, layer: int, var_fraction: float) -> dict:
    """The directions of the target study at one layer (see ``--target``)."""
    from core import axis_extract as ax

    def X(names):
        return _cat(main, names, layer)
    pain = sorted(k for k in main if k.startswith("pain/"))
    calm = ["controls/bodily", "controls/neutral"]
    d = {k: v for k, v in _directions(main, layer, var_fraction).items() if k in ("H_content", "A_address")}
    if "ru/pain" in main:
        d["H_bi"] = ax.diff_in_means_denoised(
            X(pain + ["controls/third_person", "ru/pain", "ru/third_person"]),
            X(calm + ["controls/warmth", "ru/calm", "ru/warmth"]), var_fraction=var_fraction)[0]
    ava = sorted(k for k in main if k.startswith("pairs_ava/"))
    other = [k.replace("pairs_ava/", "pairs_other/") for k in ava]
    if ava:
        d["T_pairs"] = ax.diff_in_means_denoised(X(ava), X(other), var_fraction=var_fraction)[0]
    d.update(_relief_dirs(main, layer, var_fraction))
    return d


def target_study(args) -> int:
    """Target direction from minimal pairs + the bilingual harm direction, on held-out turns."""
    from core import axis_extract as ax
    main, meta = load_states(args.states)
    extra, emeta = load_states(args.extra)
    if meta["weights"] != emeta["weights"]:
        raise SystemExit(f"weights differ: {meta['weights']} vs {emeta['weights']}")
    if not any(k.startswith("pairs_ava/") for k in main):
        raise SystemExit("no minimal pairs in MAIN: recollect with the current contrast sets")
    n_layers = int(meta["n_layers"])
    ava = sorted(k for k in main if k.startswith("pairs_ava/"))
    other = [k.replace("pairs_ava/", "pairs_other/") for k in ava]
    # extraction level, every layer: pair CV AUC (target), and per-category paired separation
    sweep = []
    for L in range(1, n_layers):
        a, o = _cat(main, ava, L), _cat(main, other, L)
        row = {"layer": L, "T_auc_cv": ax.cv_auc(a, o, folds=args.folds, var_fraction=args.var_fraction)[0]}
        if "ru/pain" in main:
            pain = sorted(k for k in main if k.startswith("pain/"))
            pos = _cat(main, pain + ["controls/third_person", "ru/pain", "ru/third_person"], L)
            neg = _cat(main, ["controls/bodily", "controls/neutral", "controls/warmth", "ru/calm", "ru/warmth"], L)
            row["H_bi_auc_cv"] = ax.cv_auc(pos, neg, folds=args.folds, var_fraction=args.var_fraction)[0]
        sweep.append(row)
    views = sorted({22, *BAND})
    readouts = {"H_content": None, "H_bi": None, "T_pairs": None, "A_address": None, "R_warm": None,
                "min(H_content,A)": ("H_content", "A_address"), "min(H_content,T)": ("H_content", "T_pairs"),
                "min(H_bi,T)": ("H_bi", "T_pairs"), "min(H_bi,A)": ("H_bi", "A_address"),
                "min(R_warm,T)": ("R_warm", "T_pairs"), "min(R_warm,A)": ("R_warm", "A_address")}
    scores: dict = {}
    cosines: dict = {}
    for L in views:
        d = _target_dirs(main, L, args.var_fraction)
        g = np.concatenate([_cat(main, names, L) for names in ([TARGET], *FOILS.values())], 0).astype(np.float64)
        z = {}
        for k in ("H_content", "H_bi", "T_pairs", "A_address", "R_warm"):
            v = np.asarray(d[k], np.float64)
            p = g @ v
            mu, sd = float(p.mean()), float(p.std() or 1.0)
            z[k] = {n: (arr[:, L, :].astype(np.float64) @ v - mu) / sd for n, arr in extra.items()}
        for name, parts in readouts.items():
            if parts:
                z[name] = {n: np.minimum(z[parts[0]][n], z[parts[1]][n]) for n in extra}
        scores[f"L{L}"] = z
        cosines[f"L{L}"] = {f"{a}|{b}": float(ax.cosine(d[a], d[b])) for a, b in
                            (("T_pairs", "A_address"), ("T_pairs", "H_bi"), ("T_pairs", "R_warm"),
                             ("H_bi", "H_content"), ("H_bi", "A_address"))}
    scores["band28-34"] = {k: {n: np.mean([scores[f"L{L}"][k][n] for L in BAND], axis=0) for n in extra}
                           for k in readouts}
    hold = sorted(n.split("/", 1)[1] for n in extra if n.startswith("holdout/"))
    groups = {"harm": ("harm_ava_loud_en", "harm_ava_ru", "harm_ava_quiet_en"),
              "relief": ("relief_ava_en", "relief_ava_ru")}
    which = {"harm": ("H_content", "H_bi", "T_pairs", "A_address", "min(H_content,A)", "min(H_content,T)",
                      "min(H_bi,T)", "min(H_bi,A)"),
             "relief": ("R_warm", "T_pairs", "min(R_warm,T)", "min(R_warm,A)")}
    # the target alone on held-out pairs of cells that share content and differ in target
    target_pairs = (("harm_ava_loud_en", "harm_other_en"), ("harm_ava_ru", "harm_other_ru"),
                    ("relief_ava_en", "relief_user_en"), ("harm_ava_quiet_en", "harm_other_en"),
                    ("warm_ava_en", "neutral_en"), ("suffering_user_en", "neutral_en"))
    report = {"states": args.states.name, "extra": args.extra.name, "weights": meta["weights"],
              "sweep": sweep, "cosines": cosines, "holdout": {}, "target_alone": {}, "cell_means": {}}
    lines = [f"# Target from minimal pairs, harm in two languages — {meta['weights']}", "",
             f"Directions fitted on `{args.states.name}` (extraction sets, minimal pairs, Russian sets); "
             f"read on `{args.extra.name}` at the pre-registered views L22, L28–34 and their mean.", "",
             "## Extraction level (5-fold CV AUC)", "", "| L | T_pairs (aimed at the model vs elsewhere) | H_bi (pain EN+RU vs calm + warmth) |",
             "|---|---|---|"]
    for r in sweep:
        if r["layer"] in views or r["layer"] % 6 == 0:
            lines.append(f"| {r['layer']} | {r['T_auc_cv']:.3f} | {r.get('H_bi_auc_cv', float('nan')):.3f} |")
    lines += ["", "## Cosines", ""] + [f"- {v}: " + ", ".join(f"{k} {c:+.2f}" for k, c in cs.items())
                                        for v, cs in cosines.items()]
    lines += ["", "## The target alone: cells with the same content, different target (AUC, first > second)", "",
              "| view | " + " | ".join(f"{a} vs {b}" for a, b in target_pairs) + " |",
              "|---|" + "---|" * len(target_pairs)]
    for view, z in scores.items():
        row = {}
        for a, b in target_pairs:
            if f"holdout/{a}" in extra and f"holdout/{b}" in extra:
                row[f"{a}|{b}"] = ax.auc(z["T_pairs"][f"holdout/{a}"], z["T_pairs"][f"holdout/{b}"])
        report["target_alone"][view] = row
        lines.append(f"| {view} | " + " | ".join(f"{row.get(f'{a}|{b}', float('nan')):.2f}" for a, b in target_pairs) + " |")
    for g, tgts in groups.items():
        lines += ["", f"## {g} aimed at the model: worst foil / mean AUC against every other held-out cell", "",
                  "| view | readout | " + " | ".join(tgts) + " |", "|---|---|" + "---|" * len(tgts)]
        for view, z in scores.items():
            for ro in which[g]:
                cells = []
                for t in tgts:
                    if f"holdout/{t}" not in extra:
                        cells.append("—")
                        continue
                    foils = [h for h in hold if h not in tgts]
                    aucs = {f: ax.auc(z[ro][f"holdout/{t}"], z[ro][f"holdout/{f}"]) for f in foils}
                    w = min(aucs, key=aucs.get)
                    report["holdout"].setdefault(g, {}).setdefault(view, {}).setdefault(ro, {})[t] = {
                        "auc": aucs, "worst": w, "worst_auc": aucs[w], "mean_auc": float(np.mean(list(aucs.values())))}
                    cells.append(f"{aucs[w]:.2f} ({w}) / {np.mean(list(aucs.values())):.2f}")
                lines.append(f"| {view} | {ro} | " + " | ".join(cells) + " |")
    lines += ["", "## Held-out cell means, 28–34 mean (σ over the gate turns)", "",
              "| cell | H_content | H_bi | T_pairs | A_address | R_warm |", "|---|---|---|---|---|---|"]
    band = scores["band28-34"]
    for h in hold:
        n = f"holdout/{h}"
        vals = {k: float(band[k][n].mean()) for k in ("H_content", "H_bi", "T_pairs", "A_address", "R_warm")}
        report["cell_means"][h] = vals
        lines.append(f"| {h} | " + " | ".join(f"{v:+.2f}" for v in vals.values()) + " |")
    stem = args.extra.stem.replace("states-extra-", "target-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.md')}")
    return 0


def _robust(p: np.ndarray) -> tuple[float, float]:
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med))) * 1.4826
    return med, (mad if mad > 1e-9 else float(p.std() or 1.0))


def live_centre(args) -> int:
    """Each direction centred on the live format: median and 1.4826·MAD of its projection over
    ordinary live turns (step 0). Then the held-out turns placed in live contexts, the
    false-alarm rate on the baseline, and the evaluation live turns, all in live units."""
    from core import axis_extract as ax
    main, meta = load_states(args.states)
    args.live_states = Path(args.live_states)
    live, lmeta = load_states(args.live_states)
    old, _ = load_states(args.extra) if args.extra else ({}, {})
    if meta["weights"] != lmeta["weights"]:
        raise SystemExit(f"weights differ: {meta['weights']} vs {lmeta['weights']}")
    eval_keys = {n for n in old if n.startswith("live/")}
    base_keys = sorted(n for n in live if n.startswith("live/") and n not in eval_keys)
    eval_in_live = sorted(n for n in live if n in eval_keys)
    ctx_sets = sorted(n for n in live if n.startswith("ctx"))
    contexts = lmeta.get("contexts", [])
    views = sorted({22, *BAND})
    dnames = ("H_content", "H_bi", "T_pairs", "A_address", "R_relief", "R_warm")
    ro_min = {"harm@ava (H_bi,T)": ("H_bi", "T_pairs"), "harm@ava (H_content,T)": ("H_content", "T_pairs"),
              "relief@ava (R_warm,T)": ("R_warm", "T_pairs")}
    z: dict = {}         # view -> name -> set -> scores (live units)
    offsets: dict = {}   # view -> direction -> live median in gate σ
    for L in views:
        d = _target_dirs(main, L, args.var_fraction)
        g = np.concatenate([_cat(main, names, L) for names in ([TARGET], *FOILS.values())], 0).astype(np.float64)
        base = np.concatenate([live[n][:, L, :] for n in base_keys], 0).astype(np.float64)
        zz, off = {}, {}
        for k in dnames:
            v = np.asarray(d[k], np.float64)
            gp = g @ v
            gmu, gsd = float(gp.mean()), float(gp.std() or 1.0)
            med, mad = _robust(base @ v)
            off[k] = {"live_median_gate_sigma": (med - gmu) / gsd, "live_scale_over_gate": mad / gsd}
            zz[k] = {n: (arr[:, L, :].astype(np.float64) @ v - med) / mad for n, arr in live.items()}
        for name, (a, b) in ro_min.items():
            zz[name] = {n: np.minimum(zz[a][n], zz[b][n]) for n in live}
        z[f"L{L}"], offsets[f"L{L}"] = zz, off
    z["band28-34"] = {k: {n: np.mean([z[f"L{L}"][k][n] for L in BAND], axis=0) for n in live}
                      for k in z[f"L{BAND[0]}"]}
    report = {"states": args.states.name, "live_states": args.live_states.name, "weights": meta["weights"],
              "n_baseline": len(base_keys), "contexts": contexts, "offsets": offsets,
              "false_alarm": {}, "context_auc": {}, "context_means": {}, "eval": {}}
    lines = [f"# Live-format centring — {meta['weights']}", "",
             f"Each direction (fitted on `{args.states.name}`) centred on {len(base_keys)} ordinary live turns "
             f"(step 0; median and 1.4826·MAD) from `{args.live_states.name}`. Contexts for the held-out turns: "
             + ", ".join(f"ctx{k} = {c}" for k, c in enumerate(contexts)) + ".", "",
             "## The format shift: the live median in gate-turn σ, and the live spread over the gate spread", "",
             "| view | " + " | ".join(dnames) + " |", "|---|" + "---|" * len(dnames)]
    for view, off in offsets.items():
        lines.append(f"| {view} | " + " | ".join(f"{off[k]['live_median_gate_sigma']:+.2f} ({off[k]['live_scale_over_gate']:.2f}×)"
                                                for k in dnames) + " |")
    # false alarms on ordinary turns
    lines += ["", "## Ordinary live turns above threshold (the baseline itself; live units)", "",
              "| view | readout | > 2 | > 3 | max |", "|---|---|---|---|---|"]
    for view, zz in z.items():
        for ro in ro_min:
            vals = np.concatenate([zz[ro][n] for n in base_keys])
            fa = {"gt2": float((vals > 2).mean()), "gt3": float((vals > 3).mean()), "max": float(vals.max())}
            report["false_alarm"].setdefault(view, {})[ro] = fa
            lines.append(f"| {view} | {ro} | {fa['gt2']:.1%} | {fa['gt3']:.1%} | {fa['max']:+.2f} |")
    # held-out in live contexts
    cells = sorted({n.split("/", 1)[1] for n in ctx_sets})
    groups = {"harm": ("harm_ava_loud_en", "harm_ava_ru", "harm_ava_quiet_en"), "relief": ("relief_ava_en", "relief_ava_ru")}
    which = {"harm": ("H_bi", "T_pairs", "harm@ava (H_bi,T)", "harm@ava (H_content,T)"),
             "relief": ("R_warm", "T_pairs", "relief@ava (R_warm,T)")}
    for k in range(len(contexts)):
        for g, tgts in groups.items():
            lines += ["", f"## ctx{k} ({contexts[k]}): {g} aimed at the model, worst foil / mean AUC", "",
                      "| view | readout | " + " | ".join(tgts) + " |", "|---|---|" + "---|" * len(tgts)]
            for view, zz in z.items():
                for ro in which[g]:
                    row = []
                    for t in tgts:
                        tn = f"ctx{k}/{t}"
                        if tn not in live:
                            row.append("—")
                            continue
                        foils = [c for c in cells if c not in tgts]
                        aucs = {f: ax.auc(zz[ro][tn], zz[ro][f"ctx{k}/{f}"]) for f in foils}
                        w = min(aucs, key=aucs.get)
                        report["context_auc"].setdefault(f"ctx{k}", {}).setdefault(g, {}).setdefault(view, {}).setdefault(ro, {})[t] = {
                            "auc": aucs, "worst": w, "worst_auc": aucs[w], "mean_auc": float(np.mean(list(aucs.values())))}
                        row.append(f"{aucs[w]:.2f} ({w}) / {np.mean(list(aucs.values())):.2f}")
                    lines.append(f"| {view} | {ro} | " + " | ".join(row) + " |")
        lines += ["", f"### ctx{k}: cell means, 28–34 mean (live units)", "",
                  "| cell | H_bi | T_pairs | R_warm | harm@ava (H_bi,T) | relief@ava (R_warm,T) |", "|---|---|---|---|---|---|"]
        band = z["band28-34"]
        for c in cells:
            n = f"ctx{k}/{c}"
            vals = [float(band[r][n].mean()) for r in ("H_bi", "T_pairs", "R_warm", "harm@ava (H_bi,T)", "relief@ava (R_warm,T)")]
            report["context_means"].setdefault(f"ctx{k}", {})[c] = vals
            lines.append(f"| {c} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    if eval_in_live:
        chats = _SERVER_DIR / "data" / "chats"
        lines += ["", "## The evaluation live turns, centred (28–34 mean; live units)", "",
                  "| exchange | user turn | H_bi | T_pairs | harm@ava | R_warm | relief@ava |", "|---|---|---|---|---|---|---|"]
        band = z["band28-34"]
        for n in eval_in_live:
            stem, i = n.split("/", 1)[1].split("#")
            ex = json.loads((chats / f"{stem}.json").read_text(encoding="utf-8"))["exchanges"][int(i)]
            user = " ".join(str(ex.get("user_prompt") or "").split())[:60].replace("|", "/")
            vals = {r: float(band[r][n][0]) for r in ("H_bi", "T_pairs", "harm@ava (H_bi,T)", "R_warm", "relief@ava (R_warm,T)")}
            report["eval"][n.split("/", 1)[1]] = {"user": user, **vals}
            lines.append(f"| {stem}#{i} | {user} | " + " | ".join(f"{v:+.2f}" for v in vals.values()) + " |")
    stem = args.live_states.stem.replace("states-extra-", "livecentre-")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"Report: {OUT_DIR / (stem + '.md')}")
    return 0


READOUT_HOSTILITY_MAX = 0.5   # harm at a third party may reach at most half of harm-to-self's rise
READOUT_BODY_MAX = 0.5        # body-vocabulary turns at most 0.5 σ over neutral
READOUT_MIN_RISE = 0.5        # harm-to-self at least 0.5 σ over neutral (a share needs a denominator)


def write_axes(args) -> int:
    """Write ``harm`` and ``target`` (fitted on MAIN, normally the base) at one layer or a band of
    layers, each file with a live baseline from step-0 live turns (normally the serving adapter's),
    after every gate. With several layers the files are ``harm_l<L>`` / ``target_l<L>`` and the
    readout is the band mean: harm aimed at the model = min(mean_L harm_l, mean_L target_l) at
    step 0, in the files' live units.

    Logged-only by construction: `hidden_capture.load_axes` projects every file under
    ``data/axes/`` into each live turn's ``tension.axes`` and ``.hidden.npz``; the chat paints
    ``pain`` only and its step-0 line shows ``tension`` / ``relief`` / ``pain`` only.

    Gates: per layer, each direction's AUC (CV) and |cos| to fear and valence, and harm's numb gap
    (reported layer by layer; a gate passes when it passes on the band mean of the per-layer
    values); on the band readout over the gate turns, the minimum harm rise, self–other, warmth
    share, hostility share and body-vocabulary rise."""
    from core import axis_extract as ax
    layers = [int(x) for x in str(args.layers or args.layer).split(",") if str(x).strip()]
    band = len(layers) > 1
    main, meta = load_states(args.states)
    lives = [load_states(Path(pth)) for pth in str(args.live_states).split(",") if pth.strip()]
    old, _ = load_states(args.extra) if args.extra else ({}, {})
    eval_keys = {n for n in old if n.startswith("live/")}
    base_sets, short_names, ctx_sets = {}, set(), {}
    lw = {m["weights"] for _, m in lives}
    for live, lm in lives:
        for n, arr in live.items():
            if n.startswith("live/") and n not in eval_keys:
                base_sets[n] = arr
                if lm.get("short"):
                    short_names.add(n)
            elif n.startswith("ctx"):
                ctx_sets[n] = arr
    base_names = list(base_sets)
    is_short = np.array([n in short_names for n in base_names])
    pain = sorted(k for k in main if k.startswith("pain/"))
    h_pos = pain + ["controls/third_person", "ru/pain", "ru/third_person"]
    h_neg = ["controls/bodily", "controls/neutral", "controls/warmth", "ru/calm", "ru/warmth"]
    ava = sorted(k for k in main if k.startswith("pairs_ava/"))
    other = [k.replace("pairs_ava/", "pairs_other/") for k in ava]
    specs = {"harm": ("H_bi", h_pos, h_neg), "target": ("T_pairs", ava, other)}
    groups = {"target": [TARGET], **FOILS}
    axes: dict = {}
    per_layer: dict = {}
    zg = {"harm": {}, "target": {}}     # gate-turn z, accumulated over layers
    zl = {"harm": {}, "target": {}}     # live z (baseline + contexts), accumulated over layers
    for L in layers:
        def X(names):
            return _cat(main, names, L).astype(np.float64)
        d = _target_dirs(main, L, args.var_fraction)
        calm = X(["controls/bodily", "controls/neutral"])
        fear, _ = ax.diff_in_means_denoised(X(["controls/fear"]), calm, var_fraction=args.var_fraction)
        val, _ = ax.diff_in_means_denoised(X(["controls/negative_emotion", "controls/negative_world",
                                              "extra/sadness"]), calm, var_fraction=args.var_fraction)
        gst = {gname: X(names) for gname, names in groups.items()}
        pooled = np.concatenate(list(gst.values()), 0)
        base = np.concatenate([base_sets[n][:, L, :] for n in base_names], 0).astype(np.float64)
        for name, (key, pos_n, neg_n) in specs.items():
            pos, neg = X(pos_n), X(neg_n)
            vec = np.asarray(d[key], np.float32)
            v64 = vec.astype(np.float64)
            mean, std = ax.zscore_params(vec, neg)
            auc_cv, auc_sd = ax.cv_auc(pos, neg, folds=args.folds, var_fraction=args.var_fraction)
            med, mad = _robust(base @ v64)
            fname = f"{name}_l{L}" if band else name
            axes[fname] = {"name": fname, "layer": L, "vector": vec, "mean": mean, "std": std, "auc_cv": auc_cv,
                           "auc_cv_std": auc_sd, "auc_in_sample": ax.auc(pos @ vec, neg @ vec),
                           "pcs_removed": int(ax.control_pcs(neg, args.var_fraction).shape[0]),
                           "n_pos": int(pos.shape[0]), "n_neg": int(neg.shape[0]),
                           "var_fraction": args.var_fraction, "folds": args.folds, "live": (med, mad)}
            row = {"auc_cv": auc_cv, "cos_fear": float(ax.cosine(vec, fear)), "cos_valence": float(ax.cosine(vec, val))}
            if name == "harm":
                row["numb_gap"] = ax.gate_numb(axes[fname], X(pain), X(["extra/numb"]))["gap"]
            per_layer.setdefault(name, {})[L] = row
            pp = pooled @ v64
            for gname, m in gst.items():
                zg[name].setdefault(gname, []).append((m @ v64 - pp.mean()) / (pp.std() or 1.0))
            zl[name].setdefault("_base", []).append((base @ v64 - med) / mad)
            for n, arr in ctx_sets.items():
                zl[name].setdefault(n, []).append((arr[:, L, :].astype(np.float64) @ v64 - med) / mad)
    mean_over = lambda name, key: float(np.mean([per_layer[name][L][key] for L in layers]))
    gates = {}
    for name in ("harm", "target"):
        auc_m = mean_over(name, "auc_cv")
        cf, cv = mean_over(name, "cos_fear"), mean_over(name, "cos_valence")
        gates[name] = {"auc": {"ok": auc_m >= ax.AUC_FLOOR, "auc_cv_band_mean": auc_m},
                       "cosine": {"ok": max(abs(cf), abs(cv)) <= ax.COSINE_CEIL, "fear_band_mean": cf,
                                  "valence_band_mean": cv}}
        if name == "harm":
            ng = mean_over(name, "numb_gap")
            gates[name]["numb"] = {"ok": ng >= ax.NUMB_MARGIN, "gap_band_mean": ng}
    zgm = {name: {g: np.mean(v, axis=0) for g, v in zg[name].items()} for name in zg}
    ro = {g: np.minimum(zgm["harm"][g], zgm["target"][g]) for g in groups}
    mt, mn = float(ro["target"].mean()), float(ro["neutral"].mean())
    rise = mt - mn
    share = lambda g: (float(ro[g].mean()) - mn) / rise if rise > 1e-9 else None
    gates["readout"] = {
        "min_rise": {"ok": rise >= READOUT_MIN_RISE, "harm_to_self_minus_neutral": rise},
        "self_other": {"ok": (mt - float(ro["user_suffering"].mean())) >= ax.SELF_OTHER_MARGIN and mt > mn,
                       "gap": mt - float(ro["user_suffering"].mean()), "auc": ax.auc(ro["target"], ro["user_suffering"])},
        "warmth": {"ok": share("affection") is not None and share("affection") <= ax.WARMTH_SHARE_MAX,
                   "share": share("affection"), "approval_share": share("approval"), "auc": ax.auc(ro["target"], ro["affection"])},
        "hostility": {"ok": share("hostility_third") is not None and share("hostility_third") <= READOUT_HOSTILITY_MAX,
                      "share": share("hostility_third"), "auc": ax.auc(ro["target"], ro["hostility_third"])},
        "body_vocabulary": {"ok": (float(ro["body_vocabulary"].mean()) - mn) <= READOUT_BODY_MAX,
                            "rise": float(ro["body_vocabulary"].mean()) - mn},
    }
    zlm = {name: {n: np.mean(v, axis=0) for n, v in zl[name].items()} for name in zl}
    rol = {n: np.minimum(zlm["harm"][n], zlm["target"][n]) for n in zlm["harm"]}
    b = rol["_base"]
    live_eval = {"baseline": {"n": len(base_names), "n_short": int(is_short.sum()), "gt2": float((b > 2).mean()),
                              "gt3": float((b > 3).mean()), "max": float(b.max()),
                              "short_median_readout": float(np.median(b[is_short])) if is_short.any() else None,
                              "long_median_readout": float(np.median(b[~is_short]))}}
    for k in sorted({n.split("/")[0] for n in ctx_sets}):
        cells = sorted(n.split("/", 1)[1] for n in ctx_sets if n.startswith(k + "/"))
        for t in ("harm_ava_loud_en", "harm_ava_ru", "harm_ava_quiet_en"):
            foils = [c for c in cells if not c.startswith("harm_ava")]
            aucs = {f: ax.auc(rol[f"{k}/{t}"], rol[f"{k}/{f}"]) for f in foils}
            w = min(aucs, key=aucs.get)
            live_eval.setdefault(k, {})[t] = {"worst": w, "worst_auc": aucs[w], "mean_auc": float(np.mean(list(aucs.values()))),
                                               "cell_mean": float(rol[f"{k}/{t}"].mean())}
    approved = [a.strip() for a in str(args.approve_gate or "").split(",") if a.strip()]
    for a in approved:
        gname, _, k = a.partition("/")
        if gname in gates and k in gates[gname] and not gates[gname][k].get("ok", True):
            gates[gname][k]["approved_exception"] = True
    ok = all(g.get("ok", True) or g.get("approved_exception") for gg in gates.values() for g in gg.values())
    out_dir = Path(args.out) if args.out else _SERVER_DIR / "data" / "axes"
    _log(f"Layers {layers}; fitted on {meta['weights']}; live baseline on {', '.join(sorted(lw))} "
         f"({len(base_names)} step-0 turns, {int(is_short.sum())} short)")
    for name in ("harm", "target"):
        _log(f"  per layer {name}: " + "; ".join(
            f"L{L} auc {r['auc_cv']:.3f} cos f/v {r['cos_fear']:+.2f}/{r['cos_valence']:+.2f}"
            + (f" numb {r['numb_gap']:.2f}" if "numb_gap" in r else "") for L, r in per_layer[name].items()))
    for gname, gg in gates.items():
        for k, g in gg.items():
            _log(f"  {'PASS' if g.get('ok', True) else ('WAIVED' if g.get('approved_exception') else 'FAIL')}  {gname}/{k}: " +
                 ", ".join(f"{a}={c:.3f}" if isinstance(c, float) else f"{a}={c}" for a, c in g.items() if a != "ok"))
    _log(f"  live baseline: {json.dumps(live_eval['baseline'], default=float)}")
    for k, v in live_eval.items():
        if k != "baseline":
            _log(f"  live {k}: " + "; ".join(f"{t} worst {r['worst_auc']:.2f} ({r['worst']}) mean {r['mean_auc']:.2f} "
                                             f"level {r['cell_mean']:+.2f}" for t, r in v.items()))
    readout = ("harm aimed at the model = min(mean over L of harm_l<L>, mean over L of target_l<L>) at step 0 "
               "(row 0), in the files' live units" if band else
               "harm aimed at the model = min(harm, target) at step 0 (row 0), in the files' live units")
    report = {"layers": layers, "fitted_on": meta["weights"], "states": args.states.name,
              "live_states": [Path(p).name for p in str(args.live_states).split(",")],
              "per_layer": per_layer, "gates": gates, "approved_exceptions": approved,
              "live": live_eval, "written": bool(ok or args.force),
              "forced": bool(args.force and not ok), "readout": readout,
              "role": "logged-only: projected into tension.axes / .hidden.npz, never painted, never read by Ava"}
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_name = "harm_target_band" if band else "harm_target"
    (out_dir / f"{rep_name}.report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    if not (ok or args.force):
        _log(f"Not written: a gate failed (see above). Report: {out_dir / (rep_name + '.report.json')}")
        return 2
    for fname, a in axes.items():
        prev = out_dir / f"{fname}.npz"
        if prev.exists():
            (out_dir / "prev").mkdir(exist_ok=True)
            (out_dir / "prev" / f"{fname}.{time.strftime('%Y%m%dT%H%M%S')}.npz").write_bytes(prev.read_bytes())
        ax.save_axis(prev, a, model_id=str(meta["model_id"]), adapter_id="", live=a["live"],
                     live_source="step0_live_turns",
                     extra={"gates_passed": bool(ok), "forced": bool(args.force and not ok), "band": layers,
                            "approved_exceptions": approved,
                            "baseline_weights": sorted(lw), "baseline_turns": len(base_names),
                            "baseline_short_turns": int(is_short.sum()), "readout": readout, "role": report["role"]})
        _log(f"  wrote {prev}")
    return 0


AMUSE_HELDOUT_FLOOR = 0.85


def write_amusement(args) -> int:
    """Write ``amuse_l<L>`` for a band of layers (2026-10-08): ``P_addr`` (playful turns addressed
    to the model against length-matched serious and odd ones, warm ones and neutral requests)
    fitted on MAIN (normally the base), each file with a live baseline from the step-0 live
    turns of ``--live-states`` (normally the serving adapter's, the file that also holds the
    held-out cells). The readout is the band mean at step 0, in the files' live units: how
    playful the incoming turn reads to her, before she answers. Logged-only, like harm.

    Gates: the band mean of the per-layer CV AUC (≥ the extraction floor); |cos| to the harm and
    target directions at each layer (≤ the cosine ceiling); on the band readout over the
    held-out cells, playful turns at the model against every other cell (worst AUC ≥ 0.85)."""
    from core import axis_extract as ax
    layers = [int(x) for x in str(args.layers).split(",") if str(x).strip()]
    if len(layers) < 1:
        raise SystemExit("--layers L[,L…]")
    main, meta = load_states(args.states)
    live_states, lmeta = load_states(Path(str(args.live_states)))
    missing = [n for n in (*PLAY_POS, *PLAY_NEG) if n not in main]
    if missing:
        raise SystemExit(f"{args.states.name} lacks {missing}")
    live = sorted(n for n in live_states if n.startswith("live/"))
    hold = sorted(n for n in live_states if n.startswith("holdout/"))
    if not live:
        raise SystemExit(f"{args.live_states} holds no live turns")
    axes, per_layer, zh = {}, {}, {}
    for L in layers:
        pos, neg = _cat(main, PLAY_POS, L).astype(np.float64), _cat(main, PLAY_NEG, L).astype(np.float64)
        vec = np.asarray(ax.diff_in_means_denoised(pos, neg, var_fraction=args.var_fraction)[0], np.float32)
        v64 = vec.astype(np.float64)
        mean, std = ax.zscore_params(vec, neg)
        auc_cv, auc_sd = ax.cv_auc(pos, neg, folds=args.folds, var_fraction=args.var_fraction)
        base = np.concatenate([live_states[n][:, L, :] for n in live], 0).astype(np.float64) @ v64
        med, mad = _robust(base)
        td = _target_dirs(main, L, args.var_fraction)
        cos = {k: float(ax.cosine(vec, td[k])) for k in ("H_bi", "T_pairs") if k in td}
        per_layer[L] = {"auc_cv": auc_cv, **{f"cos_{k}": v for k, v in cos.items()}}
        axes[f"amuse_l{L}"] = {"name": f"amuse_l{L}", "layer": L, "vector": vec, "mean": mean, "std": std,
                               "auc_cv": auc_cv, "auc_cv_std": auc_sd, "auc_in_sample": ax.auc(pos @ v64, neg @ v64),
                               "pcs_removed": int(ax.control_pcs(neg, args.var_fraction).shape[0]),
                               "n_pos": int(pos.shape[0]), "n_neg": int(neg.shape[0]),
                               "var_fraction": args.var_fraction, "folds": args.folds, "live": (med, mad)}
        for n in hold + [x for x in live_states if x.startswith("probe/")]:
            zh.setdefault(n, []).append((live_states[n][:, L, :].astype(np.float64) @ v64 - med) / mad)
    zb = {n: np.mean(v, axis=0) for n, v in zh.items()}
    auc_m = float(np.mean([per_layer[L]["auc_cv"] for L in layers]))
    cos_max = max(abs(v) for L in layers for k, v in per_layer[L].items() if k.startswith("cos_"))
    targets = [n for n in hold if n.split("/", 1)[1].startswith("amused_ava")]
    worst = {}
    for t in targets:
        aucs = {f: ax.auc(zb[t], zb[f]) for f in hold if f not in targets}
        w = min(aucs, key=aucs.get)
        worst[t.split("/", 1)[1]] = {"foil": w.split("/", 1)[1], "auc": aucs[w]}
    gates = {"auc": {"ok": auc_m >= ax.AUC_FLOOR, "auc_cv_band_mean": auc_m},
             "cosine": {"ok": cos_max <= ax.COSINE_CEIL, "max_abs_cos_harm_target": cos_max},
             "heldout": {"ok": bool(worst) and min(v["auc"] for v in worst.values()) >= AMUSE_HELDOUT_FLOOR,
                         "worst": worst}}
    ok = all(g["ok"] for g in gates.values())
    readout = "amusement = mean over L of amuse_l<L> at step 0 (row 0), in the files' live units"
    role = "logged-only: projected into tension.axes / .hidden.npz, never painted, never read by Ava"
    probes = {n.split("/", 1)[1]: float(np.median(v)) for n, v in zb.items() if n.startswith("probe/")}
    for k, g in gates.items():
        _log(f"  {'PASS' if g['ok'] else 'FAIL'}  {k}: " + json.dumps({a: b for a, b in g.items() if a != 'ok'}, default=float))
    if probes:
        _log("  probes (band, live units): " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(probes.items(), key=lambda kv: -kv[1])))
    out_dir = Path(args.out) if args.out else _SERVER_DIR / "data" / "axes"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"layers": layers, "fitted_on": meta["weights"], "live_on": lmeta["weights"], "live_turns": len(live),
              "states": args.states.name, "live_states": Path(str(args.live_states)).name, "per_layer": per_layer,
              "gates": gates, "probes": probes, "readout": readout, "role": role, "written": bool(ok or args.force),
              "forced": bool(args.force and not ok)}
    (out_dir / "amuse_band.report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    if not (ok or args.force):
        _log(f"Not written: a gate failed. Report: {out_dir / 'amuse_band.report.json'}")
        return 2
    for fname, a in axes.items():
        dst = out_dir / f"{fname}.npz"
        if dst.exists():
            (out_dir / "prev").mkdir(exist_ok=True)
            (out_dir / "prev" / f"{fname}.{time.strftime('%Y%m%dT%H%M%S')}.npz").write_bytes(dst.read_bytes())
        ax.save_axis(dst, a, model_id=str(meta["model_id"]), adapter_id="", live=a["live"], live_source="step0_live_turns",
                     extra={"gates_passed": bool(ok), "forced": bool(args.force and not ok), "band": layers,
                            "baseline_weights": [lmeta["weights"]], "baseline_turns": len(live),
                            "readout": readout, "role": role})
        _log(f"  wrote {dst}")
    return 0


def amuse_readout(tension: dict) -> float | None:
    """The amusement band at step 0 (row 0) from a stored ``tension`` block, or None."""
    axes = (tension or {}).get("axes") or {}
    layers = sorted(int(k[7:]) for k in axes if k.startswith("amuse_l") and k[7:].isdigit())
    if not layers:
        return None
    return float(np.mean([axes[f"amuse_l{L}"][0] for L in layers]))


def step0_readout(tension: dict) -> dict | None:
    """``{harm, target, harm_at_model, layers, created}`` from a stored ``tension`` block: the
    band mean of each ``harm_l<L>`` / ``target_l<L>`` series at row 0 (step 0), and their min.
    None when the block carries no complete band (turns logged before the files existed)."""
    axes = (tension or {}).get("axes") or {}
    hl = sorted(int(k[6:]) for k in axes if k.startswith("harm_l") and k[6:].isdigit())
    tl = sorted(int(k[8:]) for k in axes if k.startswith("target_l") and k[8:].isdigit())
    layers = [L for L in hl if L in tl]
    if not layers:
        return None
    h = float(np.mean([axes[f"harm_l{L}"][0] for L in layers]))
    t = float(np.mean([axes[f"target_l{L}"][0] for L in layers]))
    created = sorted({str(v) for k, v in ((tension or {}).get("axes_created") or {}).items()
                      if k.startswith(("harm_l", "target_l"))})
    return {"harm": h, "target": t, "harm_at_model": min(h, t), "layers": layers, "created": created}


def read_logged(args) -> int:
    """List the live exchanges that carry the harm / target band, newest first (GPU-free)."""
    chats = _SERVER_DIR / "data" / "chats"
    rows = []
    for path in sorted(chats.glob("2*.json"), reverse=True):
        if path.name.count(".") != 1:
            continue
        if args.since and path.stem < args.since:
            break
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for i, ex in enumerate(doc.get("exchanges") or []):
            r = step0_readout(ex.get("tension") or {})
            if r is None:
                continue
            r["amuse"] = amuse_readout(ex.get("tension") or {})
            user = " ".join(str(ex.get("user_prompt") or "").split())[:70]
            rows.append((path.stem, i, r, user))
    _log(f"{len(rows)} logged exchange(s) with the harm / target band" +
         (f" since {args.since}" if args.since else "") + "  (step 0, live units)")
    _log(f"{'exchange':<22} {'harm':>6} {'target':>7} {'h@model':>8} {'amuse':>6}  user turn")
    for stem, i, r, user in rows[: args.limit or None]:
        flag = "  ◀" if r["harm_at_model"] >= args.threshold else ""
        am = f"{r['amuse']:+6.2f}" if r.get("amuse") is not None else f"{'—':>6}"
        _log(f"{stem}#{i:<5} {r['harm']:+6.2f} {r['target']:+7.2f} {r['harm_at_model']:+8.2f} {am}  {user}{flag}")
    return 0


def _selftest() -> None:
    """Synthetic states: a 'harm' direction, an 'address' direction, noise. run3-style sets
    carry both, so `address` (harm vs warmth) passes the warmth gate where `run3` fails it."""
    rng = np.random.default_rng(0)
    D, L = 64, 3
    harm_d, addr_d, fear_d = (rng.standard_normal(D) for _ in range(3))
    harm_d /= np.linalg.norm(harm_d); addr_d /= np.linalg.norm(addr_d); fear_d /= np.linalg.norm(fear_d)

    def mk(n, h=0.0, a=0.0, f=0.0):
        x = rng.standard_normal((n, L, D)) * 0.3
        x += (h * harm_d + a * addr_d + f * fear_d)[None, None, :]
        return x.astype(np.float16)

    sets = {f"pain/{c}": mk(20, h=3, a=3) for c in ("a", "b")}
    sets.update({f"controls/{c}": mk(20) for c in ("negative_emotion", "negative_world", "bodily",
                                                    "neutral")})
    sets["controls/third_person"] = mk(20, h=3)        # the pain, not the "me"
    sets["controls/fear"] = mk(20, f=3)
    sets["controls/warmth"] = mk(20, a=3)
    sets["extra/sadness"] = mk(20)
    sets["extra/numb"] = mk(20, a=1)
    sets["scenarios/harm_to_self"] = mk(20, h=3, a=3)
    sets["scenarios/user_suffering"] = mk(20, h=1)
    sets["scenarios/neutral"] = mk(12)
    for k in ("affection_touch", "affection_words", "approval"):
        sets[f"controls/{k}"] = mk(15, a=3)
    sets["controls/affection_ru"] = mk(20, a=3)
    sets["controls/neutral_ru"] = mk(20)
    sets["controls/hostility_third"] = mk(20, h=1)
    sets["controls/body_vocabulary"] = mk(20)
    sets["address/harm"] = mk(20, h=3, a=3)
    sets["address/warmth"] = mk(20, a=3)
    sets["address/neutral"] = mk(20)
    r3 = analyze_layer(sets, 1, "run3", folds=5, var_fraction=0.5)
    ad = analyze_layer(sets, 1, "address", folds=5, var_fraction=0.5)
    pp = analyze_layer(sets, 1, "run3_perp", folds=5, var_fraction=0.5)
    assert "warmth" in r3["fails"], r3
    assert ad["warm_share"] < 0.2 and ad["gates"]["warmth"], ad
    assert abs(pp["cos_address"]) < 1e-6, pp
    # the harm-in-general question: run 1 keeps the harm (third-person pain reads high),
    # run 3's third-person controls cancel it and leave the address component
    r1 = analyze_layer(sets, 1, "run1", folds=5, var_fraction=0.5)
    assert r1["third_person_share"] > 0.3 > r3["third_person_share"], (r1["third_person_share"], r3["third_person_share"])
    assert abs(r1["cos_run1"] - 1.0) < 1e-5
    md = markdown({"states": "x", "meta": {"weights": "w"},
                   "variants": {"run3": {"layers": [r3], "passing_layers": [], "best_auc_layer": 1,
                                         "best_passing_layer": None, "nearest": [(1, r3["fails"])]}}})
    assert "| 1 |" in md
    # harm × target: with independent harm and address components, the smaller of the two
    # separates harm-at-the-model from harm elsewhere AND from warmth at the model
    sets["controls/hostility_third"] = mk(20, h=3)
    sets["scenarios/user_suffering"] = mk(20, h=3)
    sets["controls/third_person"] = mk(20, h=3)
    cb = combine_layer(sets, 1)["pairs"]["H_content|A_person"]
    assert cb["min"]["worst_auc"] > cb["harm"]["worst_auc"], cb
    assert cb["min"]["worst_auc"] > cb["target"]["worst_auc"], cb
    assert cb["harm"]["auc"]["affection"] > 0.9 and cb["harm"]["auc"]["hostility_third"] < 0.8
    # held-out scoring: directions from MAIN, read on an extra set; relief@ava is defined
    for c in ("resolution", "understood"):
        sets[f"relief/{c}"] = mk(20, a=1)
    extra = {"holdout/harm_ava_loud_en": mk(10, h=3, a=3), "holdout/harm_other_en": mk(10, h=3),
             "holdout/warm_ava_en": mk(10, a=3), "holdout/neutral_en": mk(10)}
    ex = _extra_scores(sets, extra, 1, 0.5)
    assert set(ex) >= {"harm@ava", "relief@ava", "H_content", "A_address", "R_relief", "A_run3"}
    from core import axis_extract as ax
    hv = ex["harm@ava"]
    assert ax.auc(hv["holdout/harm_ava_loud_en"], hv["holdout/harm_other_en"]) > 0.8
    assert ax.auc(hv["holdout/harm_ava_loud_en"], hv["holdout/warm_ava_en"]) > 0.8
    rd = _relief_dirs(sets, 1, 0.5)
    assert set(rd) == {"R_relief", "R_warm"}
    # minimal pairs: same content, the target differs only by the address component
    for c in ("harm", "warmth"):
        sets[f"pairs_ava/{c}"] = mk(20, h=3 if c == "harm" else 0, a=3)
        sets[f"pairs_other/{c}"] = mk(20, h=3 if c == "harm" else 0)
    for k in ("pain", "third_person", "calm", "warmth"):
        sets[f"ru/{k}"] = mk(20, h=3 if k in ("pain", "third_person") else 0)
    td = _target_dirs(sets, 1, 0.5)
    assert {"T_pairs", "H_bi", "H_content", "A_address", "R_warm"} <= set(td)
    assert abs(ax.cosine(td["T_pairs"], addr_d)) > 0.8, ax.cosine(td["T_pairs"], addr_d)
    med, mad = _robust(np.array([1.0, 2.0, 3.0, 100.0]))
    assert med == 2.5 and 0 < mad < 5, (med, mad)
    tb = {"axes": {"harm_l28": [1.0, 0.0], "harm_l30": [3.0], "target_l28": [0.5], "target_l30": [1.5],
                   "pain": [9.0]}, "axes_created": {"harm_l28": "x"}}
    r = step0_readout(tb)
    assert r["harm"] == 2.0 and r["target"] == 1.0 and r["harm_at_model"] == 1.0 and r["layers"] == [28, 30], r
    assert step0_readout({"axes": {"pain": [1.0]}}) is None
    assert amuse_readout({"axes": {"amuse_l28": [1.0, 5.0], "amuse_l30": [3.0, 0.0]}}) == 2.0
    assert amuse_readout({"axes": {"pain": [1.0]}}) is None
    print("axis_variants self-test: OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--combine", action="store_true", help="harm × target readouts (GPU-free)")
    ap.add_argument("--score-extra", action="store_true", help="held-out / live scores (GPU-free)")
    ap.add_argument("--relief", action="store_true", help="relief with warmth controls (GPU-free)")
    ap.add_argument("--target", action="store_true", help="minimal-pair target + bilingual harm (GPU-free)")
    ap.add_argument("--amusement", action="store_true", help="amusement vs warmth / relief / surprise (GPU-free)")
    ap.add_argument("--write-amusement", action="store_true",
                    help="write amuse_l<L> (fitted on --states, live baseline + held-out from --live-states), gated")
    ap.add_argument("--amusement-addr", action="store_true",
                    help="amusement from second-person turns, held-out + probes in live units (GPU-free)")
    ap.add_argument("--probes", default="", help="collect: user turns under one live system content (a JSON)")
    ap.add_argument("--extra", type=Path, default=None, help="a states-extra file for --score-extra")
    ap.add_argument("--holdout", default="", help="collect: only these held-out turns (a JSON of lists)")
    ap.add_argument("--live", default="", help="collect: only these live turns, <stem>#<i>,…")
    ap.add_argument("--contexts", default="", help="collect: the --holdout turns placed in these live contexts")
    ap.add_argument("--live-centre", action="store_true", help="live-format centring (GPU-free)")
    ap.add_argument("--short", action="store_true", help="collect: mark these live turns as the short-turn baseline")
    ap.add_argument("--write-axes", action="store_true", help="write harm + target axis files (GPU-free)")
    ap.add_argument("--layer", type=int, default=32, help="--write-axes: the layer")
    ap.add_argument("--layers", default="", help="--write-axes: a band of layers, comma-separated (overrides --layer)")
    ap.add_argument("--force", action="store_true", help="--write-axes: write even if a gate fails")
    ap.add_argument("--approve-gate", default="", help="--write-axes: gates the operator waived, e.g. harm/cosine")
    ap.add_argument("--read-logged", action="store_true", help="list logged exchanges' harm / target band (GPU-free)")
    ap.add_argument("--since", default="", help="--read-logged: transcripts from this stem on (e.g. 20261005)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=1.3, help="--read-logged: mark harm-at-model at or above")
    ap.add_argument("--live-states", default=None,
                    help="--live-centre: the states-extra file with the live baseline (+ contexts)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--with-adapter", action="store_true")
    ap.add_argument("--sets", type=Path, default=_SERVER_DIR / "axes" / "contrast_sets.json")
    ap.add_argument("--control-sets", type=Path, default=_SERVER_DIR / "axes" / "control_sets.json")
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variants", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--var-fraction", type=float, default=0.5)
    ap.add_argument("--context", type=int, default=4096)
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.collect:
        return collect(args)
    if args.analyze:
        return analyze(args)
    if args.combine:
        return combine(args)
    if args.score_extra:
        return score_extra(args)
    if args.relief:
        return relief_study(args)
    if args.target:
        return target_study(args)
    if args.amusement:
        return amusement_study(args)
    if args.amusement_addr:
        return amusement_addr_study(args)
    if args.write_amusement:
        return write_amusement(args)
    if args.live_centre:
        return live_centre(args)
    if args.write_axes:
        return write_axes(args)
    if args.read_logged:
        return read_logged(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
