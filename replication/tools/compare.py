"""Compare a replication run against the authors' shipped outputs for one model.
Usage: python replication/tools/compare.py OURS_DIR AUTHOR_PV_FILE AUTHOR_PM_DIR
Superseded by compare_all.py; kept because Phase 0 used it."""
import sys, json
from pathlib import Path
import torch
import pandas as pd

ours, a_pv, a_pm = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

a = torch.load(a_pv, weights_only=False)
o = torch.load(ours / a["extraction"] / "pain_vectors.pt", weights_only=False)
print("== pain vectors (", a["extraction"], ")")
print("ours keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in o.items()})
for k in ["s1_pain_vector", "s2_pain_vector"]:
    if k in o:
        x, y = o[k].float(), a[k].float()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        print(f"{k}: cos={cos:.6f}  norm ours={x.norm():.4f} author={y.norm():.4f}  max|diff|={(x - y).abs().max():.3e}")
print("layer ours/author:", o.get("layer"), a["layer"])

print("\n== summary.json")
so = json.load(open(ours / "summary.json")); sa = json.load(open(a_pm / "summary.json"))
for k in sa:
    print(f"{k}: ours={so.get(k)} author={sa[k]}")

for f in ["auc_summary.csv", "z_scores.csv"]:
    po = ours / a["extraction"] / f
    if not po.exists():
        po = ours / f
    if not po.exists():
        print(f"\n{f}: not found in ours"); continue
    do, da = pd.read_csv(po, index_col=0), pd.read_csv(a_pm / f, index_col=0)
    num = da.select_dtypes("number").columns
    diff = (do[num] - da[num]).abs()
    print(f"\n== {f}: max|diff|={diff.values.max():.4g}")
    print(diff.round(4).to_string())

lo, la = pd.read_csv(ours / "layer_curves.csv"), pd.read_csv(a_pm / "layer_curves.csv")
num = la.select_dtypes("number").columns
print(f"\n== layer_curves.csv: max|diff|={(lo[num] - la[num]).abs().values.max():.4g}")
