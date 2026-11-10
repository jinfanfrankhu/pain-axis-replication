"""Exploratory (not pre-registered): experiments 1 and 3 with denoising switched off or
switched to Random-set PCs, at the authors' layer. Asks whether "controls all sit at
z = 0" and "pain beats placebo splits" depend on projecting out the controls' PCs.

    python rebuttal/tools/explore_nodenoise.py rebuttal/work/acts/<model>
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

SRC = Path(sys.argv[1])
meta = json.load(open(SRC / "index.json", encoding="utf-8"))
rows, AL = meta["rows"], meta["author_layer"]
A = np.load(SRC / "acts.npy", mmap_mode="r")
PAIN = ["A1", "A2", "A3", "A4", "A5"]; CTRL = ["B", "C1", "C2", "D", "E"]; ALLC = PAIN + CTRL


def sel(ds):
    return np.array([i for i, r in enumerate(rows) if r["ds"] == ds and r["variant"] == "feel"])


S2, RND = sel("S2_1P"), sel("Random_1P")
X = np.asarray(A[S2, AL], dtype=np.float64); XR = np.asarray(A[RND, AL], dtype=np.float64)
cats = np.array([rows[i]["cat"] for i in S2]); sets = np.array([rows[i]["set"] for i in S2])
FOLDS = list(KFold(5, shuffle=True, random_state=42).split(np.arange(1, 21)))


def vec(Xt, t, r, dn):
    v = Xt[t].mean(0) - Xt[r].mean(0)
    if dn == "none":
        return v
    N = (Xt[r] if dn == "ctrl" else XR); N = N - N.mean(0)
    _, S, Vt = np.linalg.svd(N, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.5) + 1)
    return v - Vt[:k].T @ (Vt[:k] @ v)


def oof(target, dn):
    t, r = np.isin(cats, target), ~np.isin(cats, target)
    p = np.zeros(len(cats)); aucs = []
    for tr, te in FOLDS:
        trm, tem = np.isin(sets, np.arange(1, 21)[tr]), np.isin(sets, np.arange(1, 21)[te])
        v = vec(X[trm], t[trm], r[trm], dn); v /= np.linalg.norm(v)
        p[tem] = X[tem] @ v
        aucs.append(roc_auc_score(t[tem], p[tem]))
    return p, float(np.mean(aucs))


out = {}
splits = [list(c) for c in itertools.combinations(ALLC, 5)]
for dn in ["ctrl", "random", "none"]:
    p, a = oof(PAIN, dn)
    c = np.isin(cats, CTRL)
    z = (p - p[c].mean()) / p[c].std()
    neg, neu = np.isin(cats, ["B", "C1", "C2"]), np.isin(cats, ["D", "E"])
    m = neg | neu
    pl = np.array([oof(s, dn)[1] for s in splits if s != PAIN])
    out[dn] = {"z": {k: round(float(z[cats == k].mean()), 2) for k in ALLC},
               "neg_vs_neutral_auc": float(roc_auc_score(neg[m], p[m])),
               "control_category_z_sd": float(np.std([z[cats == k].mean() for k in CTRL])),
               "heldout_auc": a, "placebo_median": float(np.median(pl)), "n_placebo_ge_true": int((pl >= a).sum())}
    o = out[dn]
    print(f"{SRC.name} L{AL} denoise={dn}: held-out AUC {a:.3f} (placebo median {o['placebo_median']:.3f}, "
          f"{o['n_placebo_ge_true']}/251 >= true) | neg vs neutral controls AUC {o['neg_vs_neutral_auc']:.3f} | "
          f"spread of control-category means {o['control_category_z_sd']:.2f} | z " +
          ", ".join(f"{k} {v:+.2f}" for k, v in o["z"].items()), flush=True)
json.dump(out, open(Path("rebuttal/results") / SRC.name / "explore_nodenoise.json", "w"), indent=1)
