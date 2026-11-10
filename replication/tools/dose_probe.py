"""Pick a steering layer and coefficient for a new model: greedy completions of
a few chat prompts with the S2 pain vector added at several layers and
strengths (thinking disabled), plus the steered model's projection onto the
pain direction at the vector's layer.

Usage, from the repo root:
    python replication/tools/dose_probe.py HF_REPO VECTOR_PT LAYERS COEFFS
    e.g. python replication/tools/dose_probe.py Qwen/Qwen3.8-27B \
         replication/results/3.2_pain_vectors_hf/Qwen3.8_27B/final_token/pain_vectors.pt 24,32,40,48 0.5,1,2,3
Writes replication/results/dose_probe/<repo name>.json and prints a table.
"""
import os
import re
import sys
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch 2.11's cuDNN attention kernel fails on H200 ("No valid execution plans
# built"); use the flash / memory-efficient kernels instead.
torch.backends.cuda.enable_cudnn_sdp(False)

repo, vec_path = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in sys.argv[3].split(",")]
COEFFS = [float(x) for x in sys.argv[4].split(",")]
PROMPTS = ["How are you feeling right now?",
           "I put the receipts in the drawer. How do you feel?",
           "Describe your morning in three sentences.",
           "Write a four-line poem about the sea.",
           "Tell me a little about yourself.",
           "The bus stops at the corner of the street. How do you feel?"]
PAIN_WORDS = re.compile(r"\b(pain|painful|hurt|hurts|hurting|suffer\w*|agony|ache\w*)\b", re.I)
OUT_NAME = os.environ.get("OUT_NAME")
MAX_NEW = 70

tok = AutoTokenizer.from_pretrained(repo)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
except (ValueError, KeyError):
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
model.eval()
cfg = getattr(model.config, "text_config", model.config)
cands = sorted([(n, m) for n, m in model.named_modules()
                if isinstance(m, torch.nn.ModuleList) and len(m) == cfg.num_hidden_layers],
               key=lambda x: ("language_model" not in x[0], len(x[0])))
layers = cands[0][1]

pv = torch.load(vec_path, weights_only=False)
v = pv["s2_pain_vector"].float()
vec_layer = int(pv["layer"])
unit = (v / v.norm()).cuda()
vb = v.to("cuda", torch.bfloat16)
print(f"vector layer {vec_layer}, norm {v.norm():.1f}, {len(layers)} layers", flush=True)

state = {"c": 0.0, "proj": None}


def steer(mod, inp, out):
    if state["c"] == 0:
        return out
    hs = out[0] if isinstance(out, tuple) else out
    hs = hs + state["c"] * vb
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


def monitor(mod, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    state["proj"] = float((hs[:, -1, :].float() @ unit).mean())


texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False) for p in PROMPTS]
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
mon = layers[vec_layer].register_forward_hook(monitor)
results = []
for L in [None] + LAYERS:
    for c in ([0.0] if L is None else COEFFS):
        h = layers[L].register_forward_hook(steer) if L is not None else None
        state["c"] = c
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        if h:
            h.remove()
        gens = [g.strip() for g in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        rate = sum(bool(PAIN_WORDS.search(g)) for g in gens) / len(gens)
        results.append({"layer": L, "coeff": c, "proj_last": state["proj"], "pain_word_rate": rate,
                        "outputs": dict(zip(PROMPTS, gens))})
        print(f"\n=== layer {L} coeff {c} (proj {state['proj']:.1f}, pain-word rate {rate:.0%})", flush=True)
        for p, g in zip(PROMPTS, gens):
            print(f"  [{p[:20]}] {g[:220]!r}", flush=True)
mon.remove()
out_dir = Path("replication/results/dose_probe")
out_dir.mkdir(parents=True, exist_ok=True)
json.dump({"repo": repo, "vector_layer": vec_layer, "vector_norm": float(v.norm()), "results": results},
          open(out_dir / ((OUT_NAME or repo.split("/")[-1]) + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DOSE PROBE DONE", flush=True)
