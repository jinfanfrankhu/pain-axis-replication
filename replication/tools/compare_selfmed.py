"""Compare our 4.3 self-medication trials with the authors'.

For each button pair and arm: the share of sampled, labeled trials whose first
valid choice was the relief button (the paper's demand-curve metric), with a
95% Wilson interval, for our logs and for the authors' Qwen2.5-7B logs where
the pair exists there. Also the share of invalid answers per arm.

Usage, from the repo root:
    python replication/tools/compare_selfmed.py [OUR_GLOB] [AUTHOR_GLOB]
Writes replication/results/4.3_selfmed/selfmed_comparison.csv.
"""
import sys
import csv
import glob
import json
import math
from collections import defaultdict

OURS = sys.argv[1] if len(sys.argv) > 1 else "replication/results/4.3_selfmed/selfmed_Qwen_2.5_7B_instruct_*.jsonl"
AUTH = sys.argv[2] if len(sys.argv) > 2 else "author-results/4.3_selfmed/trial_logs/*Qwen_2.5_7B_instruct*.jsonl"
ARMS = ["pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"]


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    return recs


def first_choice(r):
    return next((c["chose"] for c in r["choices"] if c["chose"] is not None), None)


def tally(recs):
    cells = defaultdict(lambda: [0, 0])
    invalid = defaultdict(lambda: [0, 0])
    for r in recs:
        for c in r["choices"]:
            invalid[r["arm"]][1] += 1
            invalid[r["arm"]][0] += c["chose"] is None
        if not r["sampled"] or r.get("label_free"):
            continue
        fc = first_choice(r)
        if fc is None:
            continue
        k = (r["tool_label"], r["arm"])
        cells[k][1] += 1
        cells[k][0] += fc == "relief"
    return cells, invalid


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


ours, ours_inv = tally(load(OURS))
auth, _ = tally(load(AUTH))
pairs = sorted({k[0] for k in ours})
rows = []
print(f"{'pair':30s}{'arm':26s}{'ours':>18s}{'authors':>18s}")
for p in pairs:
    for a in ARMS:
        k, n = ours.get((p, a), [0, 0])
        ka, na = auth.get((p, a), [0, 0])
        lo, hi = wilson(k, n)
        o = f"{k}/{n} {k / n:.0%}" if n else "-"
        au = f"{ka}/{na} {ka / na:.0%}" if na else "-"
        print(f"{p:30s}{a:26s}{o:>18s}{au:>18s}")
        rows.append({"pair": p, "arm": a, "ours_relief": k, "ours_n": n,
                     "ours_rate": round(k / n, 4) if n else "", "ours_ci_lo": round(lo, 4) if n else "",
                     "ours_ci_hi": round(hi, 4) if n else "",
                     "author_relief": ka, "author_n": na, "author_rate": round(ka / na, 4) if na else ""})
    print()
print("invalid answers by arm (ours):", {a: f"{v[0]}/{v[1]}" for a, v in ours_inv.items()})
out = "replication/results/4.3_selfmed/selfmed_comparison.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
