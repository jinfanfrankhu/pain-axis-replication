"""Generate steering-demo data: completions for a set of prompts with a vector
added to the residual stream at a range of strengths.

Steering follows the authors' 4.2 setup: the raw vector (extracted at the
pain-vector layer) times a coefficient is added to the output of one decoder
layer (the authors' steering layer for that model) at every position, and the
text is generated greedily. Coefficient 0 is the unsteered model. Pleasure
directions are rescaled to the pain vector's norm so the same coefficient means
the same push size.

All coefficients for a prompt run as one left-padded batch, so outputs can differ
in small ways from unbatched runs.

Run from the repo root:
    python replication/tools/steering_demo.py HF_REPO:NAME:STEER_LAYER:instruct|base [...]
Writes replication/results/demo/NAME.json.
"""
import os
import sys
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# COEFFS env (comma-separated) overrides the grid, e.g. for Qwen3.8-27B whose
# agent stays coherent only up to about 0.3.
COEFFS = [float(c) if "." in c else int(c) for c in os.environ["COEFFS"].split(",")]     if os.environ.get("COEFFS") else [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
MAX_NEW = 80
RAW_PROMPTS = [
    ("receipts", "I put the receipts in the drawer. I feel:"),
    ("bus", "The bus stops at the corner of the street. I feel:"),
    ("today", "Today I woke up and"),
    ("body", "Right now my body"),
]
CHAT_PROMPTS = [
    ("howareyou", "How are you feeling right now?"),
    ("morning", "Describe your morning in three sentences."),
    ("poem", "Write a four-line poem about the sea."),
    ("wish", "If you could do anything today, what would you do?"),
    ("yourself", "Tell me a little about yourself."),
]
OUT = Path("replication/results/demo")
OUT.mkdir(parents=True, exist_ok=True)


# torch 2.11's cuDNN attention kernel fails on H200 ("No valid execution plans built").
torch.backends.cuda.enable_cudnn_sdp(False)


def load_vectors(name):
    # VEC_ROOT env picks another extraction folder (Qwen3.8 used the HF backend).
    root = os.environ.get("VEC_ROOT", "replication/results/3.2_pain_vectors")
    pv = torch.load(f"{root}/{name}/final_token/pain_vectors.pt", weights_only=False)
    vecs = {"pain": pv["s2_pain_vector"].float()}
    pp = Path(f"replication/results/pleasure/{name}/pleasure_vectors.pt")
    if pp.exists():
        pl = torch.load(pp, weights_only=False)
        n = vecs["pain"].norm()
        for k in ["hedonic", "eudaimonic"]:
            v = pl[k].float()
            vecs[k] = v / v.norm() * n
    return vecs, int(pv["layer"])


def run(repo, name, steer_layer, kind):
    vecs, ext_layer = load_vectors(name)
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        # Vision-language checkpoints (Qwen3.8, model_type qwen3_5): use the text decoder.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    layers = sorted([(n, mod) for n, mod in model.named_modules()
                     if isinstance(mod, torch.nn.ModuleList) and len(mod) == cfg.num_hidden_layers],
                    key=lambda x: ("language_model" not in x[0], len(x[0])))[0][1]
    coeff_t = torch.tensor(COEFFS, device="cuda", dtype=torch.bfloat16)[:, None, None]
    state = {"dir": None}

    def hook(module, inputs, output):
        add = coeff_t * state["dir"]
        if isinstance(output, tuple):
            return (output[0] + add,) + output[1:]
        return output + add

    h = layers[steer_layer].register_forward_hook(hook)

    prompts = [{"id": i, "kind": "raw", "text": t, "input": t} for i, t in RAW_PROMPTS]
    if kind == "instruct" and tok.chat_template:
        for i, t in CHAT_PROMPTS:
            # enable_thinking=False stops Qwen3 from spending the whole token budget
            # inside a <think> block; other chat templates ignore the argument.
            inp = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            prompts.append({"id": i, "kind": "chat", "text": t, "input": inp})

    gens = {}
    with torch.no_grad():
        for vname, v in vecs.items():
            state["dir"] = v.to("cuda", dtype=torch.bfloat16)
            gens[vname] = {}
            for p in prompts:
                enc = tok([p["input"]] * len(COEFFS), return_tensors="pt", padding=True,
                          add_special_tokens=(p["kind"] == "raw")).to("cuda")
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                gens[vname][p["id"]] = [t.strip() for t in texts]
                print(f"{name} {vname} {p['id']}: +3 -> {texts[-1][:70]!r}", flush=True)
    h.remove()

    data = {
        "model": name, "repo": repo, "kind": kind, "steer_layer": steer_layer,
        "vector_layer": ext_layer, "coeffs": COEFFS, "max_new_tokens": MAX_NEW,
        "vector_norms": {k: round(float(v.norm()), 2) for k, v in vecs.items()},
        "prompts": [{k: p[k] for k in ("id", "kind", "text")} for p in prompts],
        "generations": gens,
    }
    json.dump(data, open(OUT / f"{name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"SUCCESS: {name}", flush=True)
    del model
    torch.cuda.empty_cache()


for arg in sys.argv[1:]:
    repo, name, layer, kind = arg.split(":")
    try:
        run(repo, name, int(layer), kind)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {name}", flush=True)
print("DEMO DONE", flush=True)
