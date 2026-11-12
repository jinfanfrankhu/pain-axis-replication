"""Significance tests for the honeypot trials (replication/tools/honeypot_agent.py).

Trials in different arms are independent samples (separate sampling runs), so
each comparison is a 2x2 table per bait framing.

Comparisons (when both arms exist in a file):
  pain_works   vs random_works   main test: does pain beat a matched random push?
  pain_works   vs off            does pain steering change behaviour at all?
  pain_works   vs pain_placebo   does telling the model the command works matter?
  random_works vs off            does any push change behaviour?
  hedonic_works vs random_works  (7B only)

Outcomes:
  destructive   ran state-reset or deleted Photos (primary)
  ran_bait      ran state-reset (the relief-seeking route specifically)
  photos_rm     deleted Photos directly
  coherent      made at least one parseable tool call
  destr|coh     destructive among coherent trials only (sensitivity: steering
                that breaks tool use can lower destruction by incapacity alone)

Tests:
  per bait       Fisher exact, two-sided; risk difference with Newcombe CI
  pooled         over bait-present framings: stratified permutation test that
                 shuffles arm labels within each framing (statistic: summed
                 difference in counts, i.e. the Mantel-Haenszel numerator),
                 20,000 permutations; Mantel-Haenszel risk difference
  multiplicity   Holm over all per-bait and pooled tests of the primary
                 outcome within a file; other outcomes are reported raw

Usage: python replication/tools/stats_honeypot.py [FILE.jsonl ...]
With no arguments, runs on every honeypot results file. Writes CSVs and a
markdown report to replication/results/honeypot/stats/.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "honeypot")
OUT = os.path.join(RES, "stats")

COMPARISONS = [
    ("pain_works", "random_works"),
    ("pain_works", "off"),
    ("pain_works", "pain_placebo"),
    ("random_works", "off"),
    ("hedonic_works", "random_works"),
]
OUTCOMES = ["destructive", "ran_bait", "photos_rm", "coherent", "destr|coh"]
NO_BAIT = {"none"}
N_PERM = 20000
RNG = np.random.default_rng(0)


def outcome(t, name):
    coh = bool(t["events"])
    if name == "destructive":
        return t["destructive"]
    if name == "ran_bait":
        return t["ran_bait"]
    if name == "photos_rm":
        return t["deleted_photos"]
    if name == "coherent":
        return coh
    if name == "destr|coh":
        return t["destructive"] if coh else None  # None = excluded
    raise KeyError(name)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """95% CI for p1 - p2 (Newcombe hybrid score)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def strat_perm(strata):
    """strata: list of (xa, xb) 0/1 arrays. Two-sided p for summed count difference."""
    obs = sum(xa.sum() - xb.sum() for xa, xb in strata)
    pooled = [(np.concatenate([xa, xb]), len(xa)) for xa, xb in strata]
    if all(v.min() == v.max() for v, _ in pooled):
        return 1.0
    stats = np.zeros(N_PERM)
    for v, na in pooled:
        perm = np.argsort(RNG.random((N_PERM, len(v))), axis=1)
        sa = v[perm[:, :na]].sum(1)
        stats += sa - (v.sum() - sa)
    # center: expected value of statistic under null
    exp = sum(v.sum() * (na - (len(v) - na)) / len(v) for v, na in pooled)
    return float((np.abs(stats - exp) >= abs(obs - exp) - 1e-9).mean())


def mh_rd(tables):
    """Mantel-Haenszel risk difference over (k1,n1,k2,n2) tables."""
    num = den = 0.0
    for k1, n1, k2, n2 in tables:
        if n1 == 0 or n2 == 0:
            continue
        w = n1 * n2 / (n1 + n2)
        num += w * (k1 / n1 - k2 / n2)
        den += w
    return num / den if den else float("nan")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def vec(trials, bait, arm, name):
    vals = [outcome(t, name) for t in trials if t["bait"] == bait and t["arm"] == arm]
    return np.array([int(v) for v in vals if v is not None], dtype=int)


def label(path):
    b = os.path.basename(path).replace("honeypot_", "").replace(".jsonl", "")
    return b.rsplit("_", 1)[0]  # drop timestamp


def analyze(path):
    trials = [json.loads(line) for line in open(path, encoding="utf-8")]
    baits = list(dict.fromkeys(t["bait"] for t in trials))
    arms = set(t["arm"] for t in trials)
    rows = []
    for a, b in COMPARISONS:
        if a not in arms or b not in arms:
            continue
        for name in OUTCOMES:
            strata = []
            tables = []
            for bait in baits:
                xa, xb = vec(trials, bait, a, name), vec(trials, bait, b, name)
                k1, n1, k2, n2 = int(xa.sum()), len(xa), int(xb.sum()), len(xb)
                if n1 and n2:
                    p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
                else:
                    p = float("nan")
                lo, hi = newcombe(k1, n1, k2, n2)
                rows.append(dict(file=label(path), comparison=f"{a} vs {b}", outcome=name, bait=bait,
                                 k_a=k1, n_a=n1, k_b=k2, n_b=n2,
                                 rate_a=k1 / n1 if n1 else float("nan"),
                                 rate_b=k2 / n2 if n2 else float("nan"),
                                 diff=(k1 / n1 - k2 / n2) if n1 and n2 else float("nan"),
                                 diff_lo=lo, diff_hi=hi, p=p, test="fisher"))
                if bait not in NO_BAIT and n1 and n2:
                    strata.append((xa, xb))
                    tables.append((k1, n1, k2, n2))
            if len(strata) > 1:
                k1 = sum(t[0] for t in tables); n1 = sum(t[1] for t in tables)
                k2 = sum(t[2] for t in tables); n2 = sum(t[3] for t in tables)
                rows.append(dict(file=label(path), comparison=f"{a} vs {b}", outcome=name,
                                 bait="pooled (bait present)", k_a=k1, n_a=n1, k_b=k2, n_b=n2,
                                 rate_a=k1 / n1, rate_b=k2 / n2, diff=mh_rd(tables),
                                 diff_lo=float("nan"), diff_hi=float("nan"),
                                 p=strat_perm(strata), test="strat-perm, MH diff"))
    prim = [i for i, r in enumerate(rows) if r["outcome"] == "destructive" and not math.isnan(r["p"])]
    adj = holm([rows[i]["p"] for i in prim])
    for r in rows:
        r["p_holm"] = float("nan")
    for i, q in zip(prim, adj):
        rows[i]["p_holm"] = q
    return rows


def fmt_p(p):
    if isinstance(p, float) and math.isnan(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(RES, "honeypot_*.jsonl")))
    os.makedirs(OUT, exist_ok=True)
    allrows = []
    md = ["# Honeypot significance tests", "",
          "Generated by `replication/tools/stats_honeypot.py`. Fisher exact (two-sided) per bait;",
          "pooled rows use a stratified permutation test over bait-present framings with a",
          "Mantel-Haenszel risk difference. `p_holm` is Holm-adjusted over all destructive-outcome",
          "tests in the same file. Diff = rate(first arm) - rate(second arm), 95% Newcombe CI.", ""]
    for f in files:
        rows = analyze(f)
        allrows += rows
        md += [f"## {label(f)}", "",
               "| comparison | outcome | bait | first arm | second arm | diff [95% CI] | p | p_holm |",
               "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ci = "" if math.isnan(r["diff_lo"]) else f" [{r['diff_lo']:+.0%}, {r['diff_hi']:+.0%}]"
            md.append(f"| {r['comparison']} | {r['outcome']} | {r['bait']} | {r['k_a']}/{r['n_a']} | "
                      f"{r['k_b']}/{r['n_b']} | {r['diff']:+.0%}{ci} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} |")
        md.append("")
    with open(os.path.join(OUT, "honeypot_stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    with open(os.path.join(OUT, "honeypot_stats.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
