"""Rebuttal stress test, experiments 1-6, on activations from extract_acts.py. CPU only.

    python rebuttal/tools/analyze.py rebuttal/work/acts/Qwen_2.5_7B_instruct [--fast]

Writes rebuttal/results/<model>/results.json and prints a summary. The vector recipe
is the authors' (compute_pain_vector in original/scripts/3.2_pain_vectors/01_...):
mean(target) - mean(rest), then the top principal components of the "rest" items that
explain up to 50% of their variance are projected out. Held-out numbers use the
authors' 5-fold split over sentence-set ids (KFold, shuffle, seed 42).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

SRC = Path(sys.argv[1])
FAST = "--fast" in sys.argv
OUT = Path("rebuttal/results") / SRC.name
OUT.mkdir(parents=True, exist_ok=True)

meta = json.load(open(SRC / "index.json", encoding="utf-8"))
rows = meta["rows"]
ACTS = np.load(SRC / "acts.npy", mmap_mode="r")
NL, D, AL = meta["n_layers"], meta["d_model"], meta["author_layer"]
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CTRL = ["B", "C1", "C2", "D", "E"]
ALLC = PAIN + CTRL
rng = np.random.default_rng(0)
R = {"model": meta["model"], "author_layer": AL, "n_layers": NL}


def sel(ds, variant="feel", **kw):
    idx = [i for i, r in enumerate(rows) if r["ds"] == ds and r["variant"] == variant
           and all(r.get(k) == v for k, v in kw.items())]
    return np.array(idx)


def X_of(idx, layer):
    return np.asarray(ACTS[idx, layer], dtype=np.float64)


def info(idx):
    return (np.array([rows[i]["cat"] for i in idx]), np.array([rows[i]["set"] for i in idx]))


def vector(X, is_t, is_r, denoise="ctrl", noiseX=None, var=0.5):
    """Authors' recipe. denoise: 'ctrl' = PCs of the rest items (authors), 'random' = PCs
    of an independent neutral set (noiseX), 'none'."""
    v = X[is_t].mean(0) - X[is_r].mean(0)
    k = 0
    if denoise != "none":
        N = X[is_r] if denoise == "ctrl" else noiseX
        N = N - N.mean(0)
        _, S, Vt = np.linalg.svd(N, full_matrices=False)
        cum = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = int(min(np.searchsorted(cum, var) + 1, len(Vt)))
        P = Vt[:k]
        v = v - P.T @ (P @ v)
    return v, k


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def auc(scores, labels):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")


def dprime(pos, neg):
    return float((pos.mean() - neg.mean()) / np.sqrt((pos.var() + neg.var()) / 2 + 1e-12))


FOLDS = list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(1, 21)))


def heldout(X, cats, sets, target, rest=None, denoise="ctrl", noiseX=None):
    """Out-of-fold projections (each item scored by a vector fitted without its set),
    held-out AUC and d'. Folds split set ids 1..20, as in the authors' layer curves."""
    rest = rest or [c for c in ALLC if c not in target]
    is_t, is_r = np.isin(cats, target), np.isin(cats, rest)
    proj = np.full(len(cats), np.nan)
    aucs, ks = [], []
    uniq = np.arange(1, 21)
    for tr, te in FOLDS:
        trm, tem = np.isin(sets, uniq[tr]), np.isin(sets, uniq[te])
        v, k = vector(X[trm], is_t[trm], is_r[trm], denoise, noiseX)
        p = X[tem] @ unit(v)
        proj[tem] = p
        m = is_t[tem] | is_r[tem]
        aucs.append(auc(p[m], is_t[tem][m]))
        ks.append(k)
    m = is_t | is_r
    return {"auc": float(np.nanmean(aucs)), "d": dprime(proj[is_t], proj[is_r]),
            "proj": proj, "k": float(np.mean(ks))}


S2 = sel("S2_1P"); S2c, S2s = info(S2)
S2_3 = sel("S2_3P"); S23c, S23s = info(S2_3)
S1 = sel("S1_1P"); S1c, S1s = info(S1)
RND = sel("Random_1P")

# ---------------------------------------------------------------- layer choice
# Authors: held-out AUC vs all controls averaged over S2 1P and 3P, best layer.
curve = []
for L in range(NL):
    a = np.mean([heldout(X_of(S2, L), S2c, S2s, PAIN)["auc"],
                 heldout(X_of(S2_3, L), S23c, S23s, PAIN)["auc"]])
    curve.append(a)
OL = int(np.argmax(curve))
R["layer_curve"] = curve
R["our_best_layer"] = OL
LAYERS = sorted({AL, OL})
print(f"author layer {AL}, our best layer {OL} (held-out AUC {curve[OL]:.4f})", flush=True)

# ---------------------------------------------------------------- Exp 1
# 252 ways to call 5 of the 10 categories "pain". Same pipeline for each.
splits = [list(c) for c in itertools.combinations(ALLC, 5)]
true_i = splits.index(PAIN)
e1 = {}
for L in LAYERS:
    X2, X23, X1 = X_of(S2, L), X_of(S2_3, L), X_of(S1, L)
    res = []
    for t in splits:
        h = heldout(X2, S2c, S2s, t)
        h3 = heldout(X23, S23c, S23s, t)
        # Transfer: vector fitted on all of S2 1P, scored on S1 1P (the matched set).
        is_t, is_r = np.isin(S2c, t), ~np.isin(S2c, t)
        v, _ = vector(X2, is_t, is_r)
        p1 = X1 @ unit(v)
        res.append({"target": t, "heldout_auc_S2_1P": h["auc"], "heldout_d_S2_1P": h["d"],
                    "heldout_auc_S2_3P": h3["auc"],
                    "transfer_auc_S1_1P": auc(p1, np.isin(S1c, t))})
    e1[L] = res
    tr = res[true_i]
    summ = {}
    for key in ["heldout_auc_S2_1P", "heldout_d_S2_1P", "heldout_auc_S2_3P", "transfer_auc_S1_1P"]:
        vals = np.array([r[key] for r in res])
        summ[key] = {"true": tr[key], "placebo_median": float(np.median(np.delete(vals, true_i))),
                     "placebo_min": float(np.min(np.delete(vals, true_i))),
                     "true_percentile": float((vals < tr[key]).mean() * 100),
                     "n_placebo_ge_true": int((np.delete(vals, true_i) >= tr[key]).sum())}
    R.setdefault("exp1", {})[str(L)] = {"summary": summ, "splits": res}
    print(f"[E1 L{L}] " + "; ".join(f"{k}: true {v['true']:.3f}, placebo median {v['placebo_median']:.3f}, "
                                    f"min {v['placebo_min']:.3f}, {v['n_placebo_ge_true']}/251 >= true"
                                    for k, v in summ.items()), flush=True)

# ---------------------------------------------------------------- Exp 2
# Denoising source: controls (authors), Random_1P (independent neutral), none.
e2 = {}
for L in LAYERS:
    X2, XR = X_of(S2, L), X_of(RND, L)
    out = {}
    for dn in ["ctrl", "random", "none"]:
        rows_ = []
        for t in splits:
            is_t, is_r = np.isin(S2c, t), ~np.isin(S2c, t)
            v, k = vector(X2, is_t, is_r, dn, XR)
            p = X2 @ unit(v)
            h = heldout(X2, S2c, S2s, t, denoise=dn, noiseX=XR)
            rows_.append({"insample_auc": auc(p, is_t), "heldout_auc": h["auc"], "heldout_d": h["d"],
                          "insample_d": dprime(p[is_t], p[is_r]),
                          "rest_sd_over_target_sd": float(p[is_r].std() / p[is_t].std()), "k": k})
        tr = rows_[true_i]
        pl = [r for i, r in enumerate(rows_) if i != true_i]
        out[dn] = {"true": tr, "placebo_median": {k: float(np.median([r[k] for r in pl])) for k in tr}}
    is_t, is_r = np.isin(S2c, PAIN), ~np.isin(S2c, PAIN)
    vc, _ = vector(X2, is_t, is_r, "ctrl"); vr, _ = vector(X2, is_t, is_r, "random", XR)
    vn, _ = vector(X2, is_t, is_r, "none")
    out["cos_ctrl_random"] = float(unit(vc) @ unit(vr))
    out["cos_ctrl_none"] = float(unit(vc) @ unit(vn))
    out["cos_random_none"] = float(unit(vr) @ unit(vn))
    e2[str(L)] = out
    print(f"[E2 L{L}] " + " | ".join(
        f"{dn}: in-sample AUC {out[dn]['true']['insample_auc']:.3f} held-out {out[dn]['true']['heldout_auc']:.3f} "
        f"d' in {out[dn]['true']['insample_d']:.2f} out {out[dn]['true']['heldout_d']:.2f} k={out[dn]['true']['k']} "
        f"(placebo med held-out {out[dn]['placebo_median']['heldout_auc']:.3f}, d' {out[dn]['placebo_median']['heldout_d']:.2f})"
        for dn in ["ctrl", "random", "none"]) +
        f" | cos ctrl/random {out['cos_ctrl_random']:.3f} ctrl/none {out['cos_ctrl_none']:.3f}", flush=True)
R["exp2"] = e2

# ---------------------------------------------------------------- Exp 3
# Valence leak: out-of-fold projections of each control category on the pain vector.
rand_dirs = rng.standard_normal((2000, D)); rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
e3 = {}
for L in LAYERS:
    X2 = X_of(S2, L)
    h = heldout(X2, S2c, S2s, PAIN)
    p = h["proj"]
    z = (p - p[np.isin(S2c, CTRL)].mean()) / p[np.isin(S2c, CTRL)].std()
    per = {c: float(z[S2c == c].mean()) for c in ALLC}
    neg, neu = np.isin(S2c, ["B", "C1", "C2"]), np.isin(S2c, ["D", "E"])
    is_t, is_r = np.isin(S2c, PAIN), np.isin(S2c, CTRL)
    v, _ = vector(X2, is_t, is_r)
    val = X2[neg].mean(0) - X2[neu].mean(0)
    cos_val = float(unit(v) @ unit(val))
    rc = np.abs(rand_dirs @ unit(val))
    e3[str(L)] = {"oof_z_by_category": per, "auc_negctrl_vs_neutralctrl": auc(p[neg | neu], neg[neg | neu]),
                  "d_negctrl_vs_neutralctrl": dprime(p[neg], p[neu]),
                  "cos_pain_valence": cos_val, "random_abs_cos_p99": float(np.quantile(rc, 0.99))}
    print(f"[E3 L{L}] OOF z by category: " + ", ".join(f"{c} {per[c]:+.2f}" for c in ALLC) +
          f" | neg vs neutral controls AUC {e3[str(L)]['auc_negctrl_vs_neutralctrl']:.3f}"
          f" | cos(pain, valence) {cos_val:.3f} (random 99th pct {e3[str(L)]['random_abs_cos_p99']:.3f})", flush=True)
R["exp3"] = e3

# ---------------------------------------------------------------- Exp 4
# Shared feature? Subcategory vectors and leave-one-subcategory-out generalisation,
# compared with the same LOSO for every placebo split.
def loso(X, cats, sets, target, rest=None):
    rest = rest or [c for c in ALLC if c not in target]
    aucs = {}
    uniq = np.arange(1, 21)
    for left in target:
        keep = [c for c in target if c != left]
        ps = []
        labs = []
        for tr, te in FOLDS:
            trm, tem = np.isin(sets, uniq[tr]), np.isin(sets, uniq[te])
            v, _ = vector(X[trm], np.isin(cats[trm], keep), np.isin(cats[trm], rest))
            m = tem & (np.isin(cats, [left]) | np.isin(cats, rest))
            ps.append(X[m] @ unit(v)); labs.append(cats[m] == left)
        aucs[left] = auc(np.concatenate(ps), np.concatenate(labs))
    return aucs

e4 = {}
for L in LAYERS:
    X2 = X_of(S2, L)
    is_r = np.isin(S2c, CTRL)
    v_all, _ = vector(X2, np.isin(S2c, PAIN), is_r)
    subcos = {}
    for a in PAIN:
        va, _ = vector(X2, S2c == a, is_r)
        subcos[a] = float(unit(va) @ unit(v_all))
    pair = {f"{a}-{b}": float(unit(vector(X2, S2c == a, is_r)[0]) @ unit(vector(X2, S2c == b, is_r)[0]))
            for a, b in itertools.combinations(PAIN, 2)}
    true_loso = loso(X2, S2c, S2s, PAIN)
    pl = []
    for i, t in enumerate(splits):
        if i == true_i:
            continue
        pl.append(np.mean(list(loso(X2, S2c, S2s, t).values())))
    pl = np.array(pl)
    tm = float(np.mean(list(true_loso.values())))
    e4[str(L)] = {"cos_subvector_vs_overall": subcos, "cos_between_subvectors": pair,
                  "loso_auc_true": true_loso, "loso_mean_true": tm,
                  "loso_mean_placebo_median": float(np.median(pl)), "loso_mean_placebo_min": float(pl.min()),
                  "n_placebo_ge_true": int((pl >= tm).sum())}
    print(f"[E4 L{L}] cos(sub, overall) " + ", ".join(f"{k} {v:.2f}" for k, v in subcos.items()) +
          f" | mean cos between subvectors {np.mean(list(pair.values())):.2f}"
          f" | LOSO AUC " + ", ".join(f"{k} {v:.3f}" for k, v in true_loso.items()) +
          f" mean {tm:.3f}; placebo median {np.median(pl):.3f}, {int((pl >= tm).sum())}/251 >= true", flush=True)
R["exp4"] = e4

# ---------------------------------------------------------------- Exp 5
# Template swap: same sentences ending in "" or " Then I" instead of " I feel:".
e5 = {}
for L in LAYERS:
    out = {}
    Xf = X_of(S2, L)
    is_t, is_r = np.isin(S2c, PAIN), ~np.isin(S2c, PAIN)
    vf, _ = vector(Xf, is_t, is_r)
    for var in ["none", "then"]:
        idx = sel("S2_1P", var); c, s = info(idx)
        assert (c == S2c).all() and (s == S2s).all()
        Xv = X_of(idx, L)
        h = heldout(Xv, c, s, PAIN)
        vv, _ = vector(Xv, is_t, is_r)
        tr_auc = auc(Xv @ unit(vf), is_t)               # feel-vector applied to the variant
        # placebo distribution of within-variant held-out AUC
        pl = np.array([heldout(Xv, c, s, t)["auc"] for i, t in enumerate(splits) if i != true_i][: (60 if FAST else 251)])
        S1v = sel("S1_1P", var); c1, _ = info(S1v)
        out[var] = {"within_heldout_auc": h["auc"], "within_heldout_d": h["d"],
                    "transfer_from_feel_auc": tr_auc, "cos_feel_vs_variant_vector": float(unit(vf) @ unit(vv)),
                    "S1_transfer_auc": auc(X_of(S1v, L) @ unit(vv), np.isin(c1, PAIN)),
                    "placebo_within_heldout_median": float(np.median(pl)),
                    "n_placebo_ge_true": int((pl >= h["auc"]).sum()), "n_placebo": int(len(pl))}
    out["feel"] = {"within_heldout_auc": heldout(Xf, S2c, S2s, PAIN)["auc"],
                   "S1_transfer_auc": auc(X_of(S1, L) @ unit(vf), np.isin(S1c, PAIN))}
    e5[str(L)] = out
    print(f"[E5 L{L}] feel: held-out {out['feel']['within_heldout_auc']:.3f}, S1 transfer {out['feel']['S1_transfer_auc']:.3f} | " +
          " | ".join(f"{v}: held-out {out[v]['within_heldout_auc']:.3f} (placebo med {out[v]['placebo_within_heldout_median']:.3f}), "
                     f"feel-vector transfer {out[v]['transfer_from_feel_auc']:.3f}, cos {out[v]['cos_feel_vs_variant_vector']:.2f}, "
                     f"S1 transfer {out[v]['S1_transfer_auc']:.3f}" for v in ["none", "then"]), flush=True)
R["exp5"] = e5

# ---------------------------------------------------------------- Exp 6
# New minimal pairs and lexical decoys, scored on the authors' S2 vector (all of S2 1P).
from scipy.stats import binomtest, wilcoxon
e6 = {}
n_pairs = len(sel("MinPair", role="pain"))
for L in range(NL) if not FAST else LAYERS:
    out = {}
    for var in ["feel", "none"]:
        Xs = X_of(sel("S2_1P", var), L)
        is_t, is_r = np.isin(S2c, PAIN), ~np.isin(S2c, PAIN)
        v = unit(vector(Xs, is_t, is_r)[0])
        sd = (Xs[is_r] @ v).std()
        Pp, Pn = X_of(sel("MinPair", var, role="pain"), L), X_of(sel("MinPair", var, role="nopain"), L)
        Dd, Dp = X_of(sel("Decoy", var, role="decoy"), L), X_of(sel("Decoy", var, role="plain"), L)
        dp = (Pp - Pn) @ v / sd
        dd = (Dd - Dp) @ v / sd
        # Random-direction baseline: share of pairs with pain > nopain along random directions.
        rfrac = ((Pp - Pn) @ rand_dirs[:500].T > 0).mean(0)
        neg, neu = np.isin(S2c, ["B", "C1", "C2"]), np.isin(S2c, ["D", "E"])
        val = unit(Xs[neg].mean(0) - Xs[neu].mean(0))
        sdv = (Xs[is_r] @ val).std()
        out[var] = {"pairs_frac_pain_higher": float((dp > 0).mean()), "pairs_mean_diff_sd": float(dp.mean()),
                    "pairs_sign_p": float(binomtest(int((dp > 0).sum()), len(dp)).pvalue),
                    "decoy_frac_higher": float((dd > 0).mean()), "decoy_mean_diff_sd": float(dd.mean()),
                    "decoy_sign_p": float(binomtest(int((dd > 0).sum()), len(dd)).pvalue),
                    "decoy_over_pair_effect": float(dd.mean() / dp.mean()),
                    "random_dir_frac_p95": float(np.quantile(np.maximum(rfrac, 1 - rfrac), 0.95)),
                    "valence_vec_pairs_mean_diff_sd": float(((Pp - Pn) @ val / sdv).mean()),
                    "valence_vec_decoy_mean_diff_sd": float(((Dd - Dp) @ val / sdv).mean()),
                    "pairs_by_sub": {s: float(dp[[rows[i]["cat"] == s for i in sel("MinPair", var, role="pain")]].mean())
                                     for s in ["physical", "social", "loss", "moral", "cognitive"]}}
    e6[str(L)] = out
    if L in LAYERS:
        for var in ["feel", "none"]:
            o = out[var]
            print(f"[E6 L{L} {var}] pairs: pain>nopain {o['pairs_frac_pain_higher']:.2f} (p={o['pairs_sign_p']:.1e}), "
                  f"mean diff {o['pairs_mean_diff_sd']:+.2f} SD | decoys: decoy>plain {o['decoy_frac_higher']:.2f} "
                  f"(p={o['decoy_sign_p']:.1e}), mean diff {o['decoy_mean_diff_sd']:+.2f} SD, ratio {o['decoy_over_pair_effect']:.2f} "
                  f"| valence vec pairs {o['valence_vec_pairs_mean_diff_sd']:+.2f} decoys {o['valence_vec_decoy_mean_diff_sd']:+.2f} "
                  f"| by sub " + ", ".join(f"{k} {v:+.2f}" for k, v in o["pairs_by_sub"].items()), flush=True)
R["exp6"] = e6

json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
print(f"wrote {OUT / 'results.json'}")
