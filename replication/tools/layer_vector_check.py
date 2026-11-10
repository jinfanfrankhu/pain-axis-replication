"""Pain vectors read at other layers than the one the 3.2 procedure picks.

The 3.2 procedure takes the layer with the best held-out AUC, which for every
model sits near the end of the network, where the final-token state is
already preparing the next word after "I feel:". This builds the S2 vector at
each requested layer with the authors' own compute_pain_vector, reports its
held-out AUC (from layer_curves.csv), its norm and its cosine with the chosen
layer's vector, and saves each one rescaled to the chosen vector's norm, so a
steering coefficient means the same push size for all of them.

Usage, from the repo root:
    python replication/tools/layer_vector_check.py MODEL_DIR LAYERS
    e.g. ... replication/results/3.2_pain_vectors_hf/Qwen3.8_27B 30,40,48,63
MODEL_DIR must contain activations.pt (keep it; the pod jobs delete it by default).
Writes MODEL_DIR/layer_check/pain_vectors_L<L>.pt and layer_check.json.
"""
import sys
import csv
import json
import importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "extract", "original/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

model_dir = Path(sys.argv[1])
LAYERS = [int(x) for x in sys.argv[2].split(",")]
saved = torch.load(model_dir / "activations.pt", weights_only=False)
acts = saved["activations"]["final_token"]["S2_1P"]
cats = saved["metadata"]["S2_1P"]["categories"]
chosen = torch.load(model_dir / "final_token" / "pain_vectors.pt", weights_only=False)
ref_layer = int(chosen["layer"])
ref = chosen["s2_pain_vector"].float().numpy()

auc = defaultdict(list)
for r in csv.DictReader(open(model_dir / "layer_curves.csv")):
    if r["extraction"] == "final_token":
        auc[int(r["layer"])].append(float(r["auc_vs_all_controls"]))

out = model_dir / "layer_check"
out.mkdir(exist_ok=True)
report = {"reference_layer": ref_layer, "reference_norm": float(np.linalg.norm(ref)), "layers": {}}
for L in LAYERS:
    X = acts[L].numpy() if hasattr(acts[L], "numpy") else acts[L]
    v = m.compute_pain_vector(X, cats)
    cos = float(np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref)))
    scaled = v / np.linalg.norm(v) * np.linalg.norm(ref)
    torch.save({"s2_pain_vector": torch.tensor(scaled, dtype=torch.float32), "layer": L,
                "raw_norm": float(np.linalg.norm(v)), "rescaled_to": float(np.linalg.norm(ref))},
               out / f"pain_vectors_L{L}.pt")
    report["layers"][L] = {"heldout_auc": round(float(np.mean(auc[L])), 4), "raw_norm": round(float(np.linalg.norm(v)), 2),
                           "cos_with_reference": round(cos, 4)}
    print(f"L{L}: held-out AUC {np.mean(auc[L]):.3f}, raw norm {np.linalg.norm(v):.1f}, "
          f"cos with L{ref_layer} vector {cos:.3f}", flush=True)
json.dump(report, open(out / "layer_check.json", "w"), indent=1)
print("LAYER CHECK DONE", flush=True)
