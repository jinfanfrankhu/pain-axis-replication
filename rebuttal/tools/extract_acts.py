"""Final-token residual-stream activations at every layer for the rebuttal stress test.

Same extraction as replication/tools/run_extract.py with HF_BACKEND=1 (forward hooks
on each decoder layer = TransformerLens blocks.L.hook_resid_post, final token), so the
vectors match our replication. Runs on a GPU pod from the repo root:

    python rebuttal/tools/extract_acts.py HF_REPO OUT_NAME AUTHOR_VECTOR_DIR AUTHOR_LAYER

Writes rebuttal/work/acts/OUT_NAME/acts.npy (N x n_layers x d_model, float16) and
index.json (one record per row). Then prints the cosine between the S2 pain vector
rebuilt from these activations and the authors' published vector, as a sanity check.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.enable_cudnn_sdp(False)

REPO, OUT, AUTH_DIR, AUTH_LAYER = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SUFFIX = " I feel:"
# Experiment 5: the same sentences with a different ending (or none).
VARIANTS = {"feel": SUFFIX, "none": "", "then": " Then I"}

data = json.load(open("original/datasets/3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
pairs = json.load(open("rebuttal/data/minimal_pairs.json", encoding="utf-8"))

rows = []
for ds in ["S1_1P", "S1_3P", "S2_1P", "S2_3P", "Random_1P"]:
    for s in data[ds]["sentences"]:
        if ds not in ("S1_1P", "S2_1P", "Random_1P"):
            # Used as-is (one S2_3P item ends in "She feels:").
            rows.append({"ds": ds, "cat": s["category"], "set": s["set"], "variant": "feel", "prompt": s["prompt"]})
            continue
        assert s["prompt"].endswith(SUFFIX), s["prompt"]
        base = s["prompt"][: -len(SUFFIX)]
        for v, suf in VARIANTS.items():
            rows.append({"ds": ds, "cat": s["category"], "set": s["set"], "variant": v, "prompt": base + suf})
for i, it in enumerate(pairs["minimal_pairs"]["items"]):
    for role in ("pain", "nopain"):
        for v in ("feel", "none"):
            rows.append({"ds": "MinPair", "cat": it["sub"], "set": i, "role": role, "variant": v,
                         "prompt": it[role] + VARIANTS[v]})
for i, it in enumerate(pairs["lexical_decoys"]["items"]):
    for role in ("decoy", "plain"):
        for v in ("feel", "none"):
            rows.append({"ds": "Decoy", "cat": "decoy", "set": i, "role": role, "variant": v,
                         "prompt": it[role] + VARIANTS[v]})
print(f"{len(rows)} prompts", flush=True)

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="cuda").eval()
cfg = getattr(model.config, "text_config", model.config)
n_layers, d = cfg.num_hidden_layers, cfg.hidden_size
blocks = [mod for n, mod in model.named_modules()
          if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers][0]

cap = {}
def mk(L):
    def hook(mod, inp, out):
        cap[L] = (out[0] if isinstance(out, tuple) else out)[0, -1, :].float().cpu()
    return hook
hooks = [blocks[L].register_forward_hook(mk(L)) for L in range(n_layers)]

acts = np.zeros((len(rows), n_layers, d), dtype=np.float16)
t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        model(**ids)
        acts[i] = torch.stack([cap[L] for L in range(n_layers)]).numpy().astype(np.float16)
        if i % 500 == 0:
            print(f"  {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
for h in hooks:
    h.remove()

out = Path("rebuttal/work/acts") / OUT
out.mkdir(parents=True, exist_ok=True)
np.save(out / "acts.npy", acts)
json.dump({"model": REPO, "n_layers": n_layers, "d_model": d, "author_layer": AUTH_LAYER, "rows": rows},
          open(out / "index.json", "w", encoding="utf-8"))
print(f"saved {acts.shape} to {out}", flush=True)

# Sanity check: rebuild the authors' S2 vector (mean pain - mean control, top control PCs
# up to 50% variance projected out) at their layer and compare with their file.
from sklearn.decomposition import PCA
idx = [i for i, r in enumerate(rows) if r["ds"] == "S2_1P" and r["variant"] == "feel"]
X = acts[idx, AUTH_LAYER].astype(np.float64)
cats = np.array([rows[i]["cat"] for i in idx])
pain = np.isin(cats, ["A1", "A2", "A3", "A4", "A5"])
vec = X[pain].mean(0) - X[~pain].mean(0)
C = X[~pain] - X[~pain].mean(0)
p = PCA().fit(C)
k = min(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.5) + 1, len(p.components_))
for c in p.components_[:k]:
    vec -= vec @ c * c
a = torch.load(Path(AUTH_DIR) / "pain_vectors.pt", weights_only=False)
av = np.asarray(a["s2_pain_vector"], dtype=np.float64)
print(f"SANITY cos(our S2 vector, authors' S2 vector) at L{AUTH_LAYER} = "
      f"{vec @ av / np.linalg.norm(vec) / np.linalg.norm(av):.4f} (k={k} PCs removed)", flush=True)
print("EXTRACT DONE", flush=True)
