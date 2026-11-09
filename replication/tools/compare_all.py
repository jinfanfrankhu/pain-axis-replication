"""Compare every replicated model in replication/results/3.2_pain_vectors against author-results.

For each model: layers chosen (ours vs authors), cosine of our S1/S2 final-token
vectors with the authors' when recomputed at the authors' layer (needs
activations.pt), human pain z, max S2 AUC difference, and the held-out AUC of
both layers on each side's layer curve. Run from the repo root:
    python replication/tools/compare_all.py [OURS_ROOT] [AUTHOR_ROOT]
Writes OURS_ROOT/replication_comparison.csv.
"""
import sys
import json
import pathlib
import importlib.util

import numpy as np
import pandas as pd
import torch

OURS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "replication/results/3.2_pain_vectors")
AUTH = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "author-results/3.2_pain_vectors")

spec = importlib.util.spec_from_file_location(
    "extract", "scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.LOG_FILE = pathlib.Path("/tmp/compare_all.log")


def cos(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def heldout(curves, ext, layer):
    g = curves[curves.extraction == ext].groupby("layer")["auc_vs_all_controls"].mean()
    return round(float(g.loc[layer]), 4)


rows = []
for d in sorted(p for p in OURS.iterdir() if (p / "summary.json").exists()):
    name = d.name
    if not (AUTH / "per_model" / name / "summary.json").exists():
        print(f"{name}: no author results, skipped")
        continue
    so = json.load(open(d / "summary.json"))
    sa = json.load(open(AUTH / "per_model" / name / "summary.json"))
    a = torch.load(AUTH / "pain_vectors" / name / "pain_vectors.pt", weights_only=False)
    a_layer = int(a["layer"])
    row = {
        "model": name,
        "layer_final_ours": so["best_layer_final_token"], "layer_final_author": sa["best_layer_final_token"],
        "layer_mean_ours": so["best_layer_mean"], "layer_mean_author": sa["best_layer_mean"],
        "human_pain_z_ours": round(so["human_pain_z"], 4), "human_pain_z_author": round(sa["human_pain_z"], 4),
    }

    lo, la = pd.read_csv(d / "layer_curves.csv"), pd.read_csv(AUTH / "per_model" / name / "layer_curves.csv")
    lf_o, lf_a = row["layer_final_ours"], row["layer_final_author"]
    row.update({
        "heldout_ours_at_our_layer": heldout(lo, "final_token", lf_o),
        "heldout_ours_at_author_layer": heldout(lo, "final_token", lf_a),
        "heldout_author_at_author_layer": heldout(la, "final_token", lf_a),
        "heldout_author_at_our_layer": heldout(la, "final_token", lf_o),
        "layer_curve_max_absdiff": round(float((lo.select_dtypes("number") - la.select_dtypes("number")).abs().values.max()), 4),
    })

    ao = pd.read_csv(d / "final_token" / "auc_summary.csv", index_col=0)
    aa = pd.read_csv(AUTH / "per_model" / name / "auc_summary.csv", index_col=0)
    if lf_o == lf_a:
        row["s2_auc_max_absdiff"] = round(float((ao - aa).abs().values.max()), 4)

    acts_file = d / "activations.pt"
    if acts_file.exists():
        s = torch.load(acts_file, weights_only=False)
        A, M = s["activations"]["final_token"], s["metadata"]
        s2 = m.compute_pain_vector(A["S2_1P"][a_layer], M["S2_1P"]["categories"])
        s1 = m.compute_pain_vector(A["S1_1P"][a_layer], M["S1_1P"]["categories"])
        row["cos_s2_at_author_layer"] = round(cos(s2, a["s2_pain_vector"].numpy()), 4)
        row["cos_s1_at_author_layer"] = round(cos(s1, a["s1_pain_vector"].numpy()), 4)
        row["norm_ratio_s2"] = round(float(np.linalg.norm(s2) / a["s2_pain_vector"].norm()), 3)
    rows.append(row)
    print(row, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OURS / "replication_comparison.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(df.to_string(index=False))
