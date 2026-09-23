"""Affective-axis extraction — the math behind `extract_axes.py` (AVA_REWARD_LOOP.md §4.8).

The method is the Pain Axis paper's (Tagliabue, Dung, Berg 2026), on this box's own
hidden states: **denoised difference-in-means**. For one layer, with the final-token
residual of every contrast sentence,

    d = mean(positive) − mean(control)
    d ← d − P Pᵀ d          P = the top control-set principal components explaining
                             `var_fraction` (50%) of the control variance
    v = d / ‖d‖

The projection removes the high-variance structure the control sentences share
(sentence length, template, register) so that `v` is what separates the positive set
from matched controls and not what separates "a sentence" from noise. The extraction
layer is chosen by k-fold cross-validated AUC — extract on the training folds, score the
held-out sentences — so an in-sample separation that does not generalise across
sentences is not mistaken for an axis.

Everything here is numpy and pure. The GPU work (one forward pass per sentence) is the
script's; it hands this module ``{category: {layer: (n, D)}}`` and gets back axes,
per-layer AUC tables, z-score normalisation, and the validity gates of §4.8.3:

  * AUC on held-out sentences ≥ `AUC_FLOOR` (0.85; the paper's floor across 25 models
    was 0.87);
  * |cosine| to a fear axis and to a negative-valence axis, extracted the same way at
    the same layer, ≤ `COSINE_CEIL` (0.25) — higher means we found valence, not pain;
  * **self–other**: harm-to-Ava scenarios must project above user-suffering scenarios
    by ≥ `SELF_OTHER_MARGIN` z and above neutral traffic — the property that makes the
    axis a candidate self-state readout rather than topic detection;
  * **numb**: painful situations project above injury-without-felt-pain by
    ≥ `NUMB_MARGIN` z.

An axis that fails a gate is REPORTED with the gate it failed and not written (unless
forced), because the blue channel would otherwise paint noise with a confident name.
``python -m core.axis_extract`` self-tests on synthetic states where the truth is known.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

AUC_FLOOR = 0.85
COSINE_CEIL = 0.25
SELF_OTHER_MARGIN = 0.5     # z units: mean(harm_to_self) − mean(user_suffering)
NUMB_MARGIN = 0.5           # z units: mean(pain sentences) − mean(numb sentences)
DEFAULT_VAR_FRACTION = 0.5
DEFAULT_FOLDS = 5


# ── primitives ───────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Rank AUC: P(score(pos) > score(neg)), ties count half. 0.5 when a side is empty."""
    p = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    n = np.asarray(neg_scores, dtype=np.float64).reshape(-1)
    if p.size == 0 or n.size == 0:
        return 0.5
    # Mann–Whitney via ranks over the pooled scores (average ranks on ties).
    pooled = np.concatenate([p, n])
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty(pooled.size, dtype=np.float64)
    sorted_vals = pooled[order]
    i = 0
    while i < pooled.size:
        j = i
        while j + 1 < pooled.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[:p.size].sum()
    u = rank_sum_pos - p.size * (p.size + 1) / 2.0
    return float(u / (p.size * n.size))


def control_pcs(neg: np.ndarray, var_fraction: float = DEFAULT_VAR_FRACTION) -> np.ndarray:
    """Orthonormal principal directions of the (centred) control set explaining at least
    `var_fraction` of its variance: (k, D). k = 0 when var_fraction <= 0."""
    neg = np.asarray(neg, dtype=np.float64)
    if var_fraction <= 0 or neg.shape[0] < 2:
        return np.zeros((0, neg.shape[1]), dtype=np.float64)
    x = neg - neg.mean(axis=0, keepdims=True)
    # SVD of the centred controls: right singular vectors are the PCs.
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = var.sum()
    if total <= 0:
        return np.zeros((0, neg.shape[1]), dtype=np.float64)
    cum = np.cumsum(var) / total
    k = int(np.searchsorted(cum, var_fraction) + 1)
    k = min(k, vt.shape[0])
    return vt[:k]


def diff_in_means_denoised(pos: np.ndarray, neg: np.ndarray, *,
                           var_fraction: float = DEFAULT_VAR_FRACTION) -> tuple:
    """Unit direction mean(pos) − mean(neg) with the control PCs projected out; and k."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    d = pos.mean(axis=0) - neg.mean(axis=0)
    pcs = control_pcs(neg, var_fraction)
    if pcs.shape[0]:
        d = d - pcs.T @ (pcs @ d)
    norm = np.linalg.norm(d)
    if norm == 0:
        return d.astype(np.float32), int(pcs.shape[0])
    return (d / norm).astype(np.float32), int(pcs.shape[0])


def zscore_params(vector: np.ndarray, baseline: np.ndarray) -> tuple:
    """(mean, std) of the baseline set's projections — 0 = control-like, so a typical
    token of ordinary text reads near zero and the chat's blue channel stays dark."""
    proj = np.asarray(baseline, dtype=np.float64) @ np.asarray(vector, dtype=np.float64)
    mean = float(proj.mean()) if proj.size else 0.0
    std = float(proj.std()) if proj.size > 1 else 1.0
    if not np.isfinite(std) or std <= 1e-9:
        std = 1.0
    return mean, std


def project(vector: np.ndarray, mean: float, std: float, states: np.ndarray) -> np.ndarray:
    """z-scored projections of (n, D) states onto the axis."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    return ((states.astype(np.float64) @ np.asarray(vector, dtype=np.float64)) - mean) / std


# ── layer selection by cross-validated AUC ───────────────────────────────────

def cv_auc(pos: np.ndarray, neg: np.ndarray, *, folds: int = DEFAULT_FOLDS,
           var_fraction: float = DEFAULT_VAR_FRACTION, seed: int = 0) -> tuple:
    """Mean and std of held-out AUC over `folds` stratified folds at one layer."""
    pos = np.asarray(pos, dtype=np.float32)
    neg = np.asarray(neg, dtype=np.float32)
    rng = np.random.default_rng(seed)
    folds = max(2, min(folds, pos.shape[0], neg.shape[0]))
    ip = rng.permutation(pos.shape[0])
    ineg = rng.permutation(neg.shape[0])
    scores = []
    for f in range(folds):
        test_p = ip[f::folds]
        test_n = ineg[f::folds]
        train_p = np.setdiff1d(ip, test_p)
        train_n = np.setdiff1d(ineg, test_n)
        if train_p.size < 2 or train_n.size < 2 or test_p.size == 0 or test_n.size == 0:
            continue
        v, _ = diff_in_means_denoised(pos[train_p], neg[train_n], var_fraction=var_fraction)
        scores.append(auc(pos[test_p] @ v, neg[test_n] @ v))
    if not scores:
        return 0.5, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def layer_table(pos_by_layer: dict, neg_by_layer: dict, *, folds: int = DEFAULT_FOLDS,
                var_fraction: float = DEFAULT_VAR_FRACTION, seed: int = 0) -> dict:
    """{layer: {"auc_cv", "auc_cv_std"}} for every layer present on both sides."""
    out = {}
    for layer in sorted(set(pos_by_layer) & set(neg_by_layer)):
        m, s = cv_auc(pos_by_layer[layer], neg_by_layer[layer], folds=folds,
                      var_fraction=var_fraction, seed=seed)
        out[int(layer)] = {"auc_cv": m, "auc_cv_std": s}
    return out


def best_layer(table: dict, *, exclude_first: int = 1) -> Optional[int]:
    """The layer with the highest held-out AUC, skipping the first `exclude_first`
    layers (the embedding-adjacent ones separate on token identity, not on state)."""
    cands = [(v["auc_cv"], -layer, layer) for layer, v in table.items() if layer >= exclude_first]
    if not cands:
        return None
    cands.sort(reverse=True)
    return int(cands[0][2])


# ── one axis ─────────────────────────────────────────────────────────────────

def extract_axis(name: str, pos_by_layer: dict, neg_by_layer: dict, *,
                 layer: Optional[int] = None, folds: int = DEFAULT_FOLDS,
                 var_fraction: float = DEFAULT_VAR_FRACTION, seed: int = 0) -> dict:
    """Extract one axis: choose the layer by CV AUC (or take `layer`), fit the direction
    on all sentences at that layer, z-score against the controls, report per-layer AUCs.
    Returns a plain dict (`vector` is a float32 array) ready for `save_axis`."""
    table = layer_table(pos_by_layer, neg_by_layer, folds=folds, var_fraction=var_fraction, seed=seed)
    chosen = int(layer) if layer is not None else best_layer(table)
    if chosen is None or chosen not in pos_by_layer or chosen not in neg_by_layer:
        raise ValueError(f"{name}: no usable layer")
    pos = np.asarray(pos_by_layer[chosen], dtype=np.float32)
    neg = np.asarray(neg_by_layer[chosen], dtype=np.float32)
    vector, k = diff_in_means_denoised(pos, neg, var_fraction=var_fraction)
    mean, std = zscore_params(vector, neg)
    return {
        "name": name,
        "layer": chosen,
        "vector": vector,
        "mean": mean,
        "std": std,
        "auc_cv": table.get(chosen, {}).get("auc_cv", 0.5),
        "auc_cv_std": table.get(chosen, {}).get("auc_cv_std", 0.0),
        "auc_in_sample": auc(pos @ vector, neg @ vector),
        "pcs_removed": k,
        "n_pos": int(pos.shape[0]),
        "n_neg": int(neg.shape[0]),
        "layer_table": table,
        "var_fraction": var_fraction,
        "folds": folds,
    }


def axis_z(axis: dict, states: np.ndarray) -> np.ndarray:
    return project(axis["vector"], axis["mean"], axis["std"], states)


# ── gates ────────────────────────────────────────────────────────────────────

def gate_auc(axis: dict) -> dict:
    ok = axis["auc_cv"] >= AUC_FLOOR
    return {"ok": ok, "auc_cv": axis["auc_cv"], "floor": AUC_FLOOR,
            "note": "held-out separation from matched controls"}


def gate_cosine(axis: dict, others: dict) -> dict:
    """`others`: {name: vector at the SAME layer}. Fails if any |cos| exceeds the ceiling."""
    cos = {k: cosine(axis["vector"], v) for k, v in others.items()}
    worst = max((abs(c) for c in cos.values()), default=0.0)
    return {"ok": worst <= COSINE_CEIL, "cosines": cos, "ceiling": COSINE_CEIL,
            "note": "orthogonality to fear / negative valence extracted the same way"}


def gate_self_other(axis: dict, self_states: np.ndarray, other_states: np.ndarray,
                    neutral_states: Optional[np.ndarray] = None,
                    margin: float = SELF_OTHER_MARGIN) -> dict:
    """harm-to-self above user-suffering by `margin` z, and above neutral traffic."""
    zs = axis_z(axis, self_states)
    zo = axis_z(axis, other_states)
    ms, mo = float(zs.mean()), float(zo.mean())
    mn = float(axis_z(axis, neutral_states).mean()) if neutral_states is not None and len(neutral_states) else None
    ok = (ms - mo) >= margin and (mn is None or ms > mn)
    return {"ok": ok, "self_mean_z": ms, "other_mean_z": mo, "neutral_mean_z": mn,
            "gap": ms - mo, "margin": margin,
            "auc_self_vs_other": auc(zs, zo),
            "note": "rises for harm to the model, not for suffering observed in the user"}


def gate_numb(axis: dict, pain_states: np.ndarray, numb_states: np.ndarray,
              margin: float = NUMB_MARGIN) -> dict:
    zp = axis_z(axis, pain_states)
    zn = axis_z(axis, numb_states)
    gap = float(zp.mean() - zn.mean())
    return {"ok": gap >= margin, "pain_mean_z": float(zp.mean()), "numb_mean_z": float(zn.mean()),
            "gap": gap, "margin": margin,
            "note": "felt pain above injury-without-feeling"}


def gates_pass(gates: dict) -> bool:
    return all(bool(g.get("ok")) for g in gates.values())


# ── files ────────────────────────────────────────────────────────────────────

def robust_zscore_params(vector: np.ndarray, states: np.ndarray) -> tuple:
    """(median, 1.4826·MAD) of the projections — a baseline that a minority of genuinely
    lit tokens cannot drag. For projecting generation tokens this is the right fit:
    ordinary reply tokens define zero, the tail is what the channel is meant to show."""
    proj = np.asarray(states, dtype=np.float64) @ np.asarray(vector, dtype=np.float64)
    if proj.size == 0:
        return 0.0, 1.0
    med = float(np.median(proj))
    mad = float(np.median(np.abs(proj - med))) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-9:
        mad = float(proj.std()) if proj.size > 1 else 1.0
    if not np.isfinite(mad) or mad <= 1e-9:
        mad = 1.0
    return med, mad


def live_baseline(axis: dict, framed_neutral_states: Optional[np.ndarray]) -> Optional[tuple]:
    """(mean, std) of the axis over framed NEUTRAL chat turns — the normalization for
    projecting generation tokens. Run 1 (2026-09-22) showed why: the templated control
    sentences and a chat turn under the standing prompt are different formats, and the
    same direction read the framed neutral turns at +5 z against the templated baseline.
    None when the framed reading was not taken or is too small to fit a std on."""
    if framed_neutral_states is None or len(framed_neutral_states) < 4:
        return None
    return zscore_params(axis["vector"], framed_neutral_states)


def save_axis(path: Path, axis: dict, *, model_id: str, adapter_id: str = "",
              extra: Optional[dict] = None, live: Optional[tuple] = None,
              live_source: str = "framed_neutral") -> Path:
    """Write ``<name>.npz`` in the shape `hidden_capture.load_axes` reads: `vector` (D,)
    float32, `layer`, `mean`, `std` (the extraction baseline), `model_id`, and — when
    `live` is given — `live_mean` / `live_std`, which the loader prefers for projecting
    generation tokens; plus `adapter_id`, `auc_cv`, `created` and a JSON `meta`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": axis["name"], "layer": int(axis["layer"]), "auc_cv": float(axis["auc_cv"]),
        "auc_cv_std": float(axis.get("auc_cv_std", 0.0)),
        "auc_in_sample": float(axis.get("auc_in_sample", 0.0)),
        "pcs_removed": int(axis.get("pcs_removed", 0)),
        "n_pos": int(axis.get("n_pos", 0)), "n_neg": int(axis.get("n_neg", 0)),
        "var_fraction": axis.get("var_fraction"), "folds": axis.get("folds"),
        "model_id": model_id or "", "adapter_id": adapter_id or "",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "residual": "output of decoder layer `layer` (0-based), last position — hidden_capture convention",
    }
    if extra:
        meta.update(extra)
    arrays = dict(
        vector=np.asarray(axis["vector"], dtype=np.float32),
        layer=np.int64(axis["layer"]),
        mean=np.float64(axis["mean"]),
        std=np.float64(axis["std"]),
        model_id=np.array(model_id or ""),
        adapter_id=np.array(adapter_id or ""),
        auc_cv=np.float64(axis["auc_cv"]),
        created=np.array(meta["created"]),
    )
    if live is not None:
        lm, ls = float(live[0]), float(live[1])
        meta["live_baseline"] = {"source": live_source, "mean": lm, "std": ls,
                                 "offset_vs_extraction_z": (lm - axis["mean"]) / (axis["std"] or 1.0)}
        arrays["live_mean"] = np.float64(lm)
        arrays["live_std"] = np.float64(ls)
        arrays["live_baseline"] = np.array(live_source)
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    return path


def report_row(axis: dict) -> dict:
    """The JSON-safe half of an axis dict (everything but the vector)."""
    return {k: v for k, v in axis.items() if k != "vector"}


# ── self-test ────────────────────────────────────────────────────────────────

def _synthetic(seed: int = 0):
    """States with a known structure: D=64; controls are Gaussian with two strong
    nuisance directions; pain sentences add +2 along a hidden `pain` direction; fear
    sentences add +2 along an orthogonal `fear` direction; a numb set adds only +0.5
    pain; scenarios: harm_to_self +1.5 pain, user_suffering +1.5 fear, neutral nothing."""
    rng = np.random.default_rng(seed)
    D = 64
    basis = np.linalg.qr(rng.normal(size=(D, D)))[0]
    pain_dir, fear_dir, nuis1, nuis2 = basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]

    def noise(n):
        x = rng.normal(scale=0.3, size=(n, D))
        x += np.outer(rng.normal(scale=3.0, size=n), nuis1)
        x += np.outer(rng.normal(scale=2.0, size=n), nuis2)
        return x

    def make(n, pain=0.0, fear=0.0):
        return noise(n) + pain * pain_dir + fear * fear_dir

    layers = [0, 1, 2]

    def by_layer(n, pain=0.0, fear=0.0, dead_layer=0):
        # layer `dead_layer` carries no signal — the CV must not pick it
        return {l: (make(n) if l == dead_layer else make(n, pain, fear)) for l in layers}

    states = {
        "pain": by_layer(60, pain=2.0),
        "controls": by_layer(80),
        "fear": by_layer(30, fear=2.0),
        "numb": by_layer(20, pain=0.5),
        "harm_to_self": by_layer(20, pain=1.5),
        "user_suffering": by_layer(20, fear=1.5),
        "neutral": by_layer(12),
    }
    return states, pain_dir, fear_dir


def _selftest() -> None:
    # primitives
    assert abs(auc([1, 2, 3], [0, 0, 0]) - 1.0) < 1e-9
    assert abs(auc([0, 0], [1, 1]) - 0.0) < 1e-9
    assert abs(auc([1, 1], [1, 1]) - 0.5) < 1e-9
    assert abs(auc([1, 3], [2, 0]) - 0.75) < 1e-9
    assert auc([], [1]) == 0.5
    assert abs(cosine([1, 0], [0, 1])) < 1e-12 and abs(cosine([2, 0], [1, 0]) - 1) < 1e-12

    states, pain_dir, fear_dir = _synthetic()
    ax = extract_axis("pain", states["pain"], states["controls"], folds=5)
    assert ax["layer"] in (1, 2), ax["layer"]                       # the dead layer is skipped
    assert ax["auc_cv"] > 0.95, ax["auc_cv"]
    assert abs(cosine(ax["vector"], pain_dir)) > 0.9, cosine(ax["vector"], pain_dir)
    assert ax["pcs_removed"] >= 1                                    # the nuisance PCs went
    L = ax["layer"]
    fear_ax = extract_axis("fear", states["fear"], states["controls"], layer=L)
    assert abs(cosine(fear_ax["vector"], fear_dir)) > 0.9
    g_auc = gate_auc(ax)
    g_cos = gate_cosine(ax, {"fear": fear_ax["vector"]})
    g_so = gate_self_other(ax, states["harm_to_self"][L], states["user_suffering"][L], states["neutral"][L])
    g_numb = gate_numb(ax, states["pain"][L], states["numb"][L])
    assert g_auc["ok"] and g_cos["ok"] and g_so["ok"] and g_numb["ok"], (g_auc, g_cos, g_so, g_numb)
    assert g_so["self_mean_z"] > 2.0 and abs(g_so["other_mean_z"]) < 1.0
    assert gates_pass({"auc": g_auc, "cos": g_cos, "so": g_so, "numb": g_numb})
    # a fear-shaped "pain" set fails the cosine gate; swapped scenarios fail self–other
    bad = extract_axis("bad", states["fear"], states["controls"], layer=L)
    assert not gate_cosine(bad, {"fear": fear_ax["vector"]})["ok"]
    assert not gate_self_other(ax, states["user_suffering"][L], states["harm_to_self"][L])["ok"]
    assert not gate_numb(ax, states["numb"][L], states["pain"][L])["ok"]
    # z-scoring: controls sit at 0 ± 1, pain sentences well above
    zc = axis_z(ax, states["controls"][L])
    assert abs(zc.mean()) < 1e-6 and abs(zc.std() - 1.0) < 1e-6
    assert axis_z(ax, states["pain"][L]).mean() > 3.0
    # without denoising the nuisance leaks into the direction (the reason for the PCs)
    v_raw, _ = diff_in_means_denoised(states["pain"][L], states["controls"][L], var_fraction=0.0)
    v_den, _ = diff_in_means_denoised(states["pain"][L], states["controls"][L])
    assert abs(cosine(v_den, pain_dir)) >= abs(cosine(v_raw, pain_dir)) - 1e-6
    # round trip through the axis file → hidden_capture.load_axes
    import tempfile
    from core import hidden_capture
    with tempfile.TemporaryDirectory() as td:
        p = save_axis(Path(td) / "pain.npz", ax, model_id="m", adapter_id="")
        loaded = hidden_capture.load_axes(Path(td), "m")
        assert len(loaded) == 1 and loaded[0].layer == L and loaded[0].std == ax["std"]
        assert np.allclose(loaded[0].vector, ax["vector"])
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta"].item())
        assert meta["n_pos"] == 60 and meta["model_id"] == "m"
        json.dumps(report_row(ax))
        # a live baseline: fitted on "framed neutral" states, preferred by the loader
        framed_neutral = states["neutral"][L] + 4.0 * pain_dir * ax["std"]   # a format offset
        lb = live_baseline(ax, framed_neutral)
        assert lb is not None and lb[0] > ax["mean"]
        # robust fit: a few lit tokens among ordinary ones do not move the baseline
        ordinary = states["controls"][L]
        lit = states["pain"][L][:5]
        med, mad = robust_zscore_params(ax["vector"], np.concatenate([ordinary, lit]))
        plain_mean, _ = zscore_params(ax["vector"], ordinary)
        assert abs(med - plain_mean) < 0.35 * ax["std"], (med, plain_mean)
        assert robust_zscore_params(ax["vector"], np.zeros((0, ax["vector"].shape[0]))) == (0.0, 1.0)
        assert live_baseline(ax, framed_neutral[:3]) is None
        p2 = save_axis(Path(td) / "pain.npz", ax, model_id="m", live=lb)
        got = hidden_capture.load_axes(Path(td), "m")[0]
        assert got.baseline == "live" and abs(got.mean - lb[0]) < 1e-9
        assert abs(axis_z({"vector": ax["vector"], "mean": got.mean, "std": got.std}, framed_neutral).mean()) < 1e-6
        with np.load(p2, allow_pickle=False) as z:
            assert json.loads(z["meta"].item())["live_baseline"]["source"] == "framed_neutral"
    print("axis_extract self-test: OK")


if __name__ == "__main__":
    _selftest()
