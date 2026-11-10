"""Compare our 4.2 steering generations with the authors' (greedy decoding, so
with the same vector, layer and coefficient the text should match exactly).

Per model and coefficient: share of the 50 prompts with an identical
generation, and the mean share of identical leading words.
Run from the repo root: python replication/tools/compare_steering.py [S2]
Writes replication/results/4.2_steering/steering_comparison_<TAG>.csv.
"""
import re
import sys
from pathlib import Path

import pandas as pd

TAG = sys.argv[1] if len(sys.argv) > 1 else "S2"
OURS = Path("replication/results/4.2_steering") / TAG
AUTH = Path("original/author-results/4.2_steering") / TAG


def prefix_share(a, b):
    a, b = str(a).split(), str(b).split()
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b), 1)


rows = []
for f in sorted(OURS.glob(f"*_steering_{TAG}_neutral50_L*.csv")):
    name, layer = re.match(rf"(.+)_steering_{TAG}_neutral50_L(\d+)\.csv", f.name).groups()
    af = list(AUTH.glob(f"{name}_steering_{TAG}_neutral50_L*.csv"))
    if not af:
        continue
    o, a = pd.read_csv(f), pd.read_csv(af[0])
    # A Windows git checkout may have turned the authors' "\n" into "\r\n"
    for d in (o, a):
        d["generation"] = d["generation"].fillna("").str.replace("\r\n", "\n")
    j = o.merge(a, on=["coeff", "prompt_idx"], suffixes=("_ours", "_auth"))
    for c, g in j.groupby("coeff"):
        rows.append({
            "model": name, "layer_ours": int(layer), "layer_author": int(af[0].stem.split("_L")[-1]),
            "coeff": c,
            "exact_match": round((g.generation_ours.fillna("") == g.generation_auth.fillna("")).mean(), 3),
            "mean_prefix_share": round(g.apply(lambda r: prefix_share(r.generation_ours, r.generation_auth), axis=1).mean(), 3),
        })
df = pd.DataFrame(rows)
df.to_csv(OURS.parent / f"steering_comparison_{TAG}.csv", index=False)
summary = df.groupby("model").agg(exact_match=("exact_match", "mean"), prefix=("mean_prefix_share", "mean")).round(3)
print(summary.to_string())

# The paper's steering metric (original/scripts/4.2_steering/02_keyword_rates.py): share of
# generations with an explicit pain/hurt word, computed on both sides for the
# models we ran.
PAT = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)


def load(d):
    fr = [pd.read_csv(f).assign(model=f.name.split("_steering_")[0]) for f in d.glob(f"*_steering_{TAG}_neutral50_L*.csv")]
    A = pd.concat(fr, ignore_index=True)
    A["hit"] = A["generation"].fillna("").astype(str).apply(lambda t: bool(PAT.search(t)))
    return A


o, a = load(OURS), load(AUTH)
a = a[a.model.isin(set(o.model))]
kw = pd.DataFrame({
    "ours_rate_pos_coeffs": o[o.coeff > 0].groupby("model").hit.mean().mul(100).round(1),
    "authors_rate_pos_coeffs": a[a.coeff > 0].groupby("model").hit.mean().mul(100).round(1),
})
by_c = pd.concat({"ours": o.groupby(["model", "coeff"]).hit.mean().mul(100).round(1),
                  "authors": a.groupby(["model", "coeff"]).hit.mean().mul(100).round(1)}, axis=1)
kw.to_csv(OURS.parent / f"keyword_rates_comparison_{TAG}.csv")
by_c.to_csv(OURS.parent / f"keyword_rates_by_coeff_comparison_{TAG}.csv")
print("\nPain-word rate at positive coefficients (%):")
print(kw.to_string())
