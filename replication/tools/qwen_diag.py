"""Why do our Qwen2.5 pain vectors differ from the authors' (cosine ~0.97) while
every other model matches (cosine 0.9999)?

For each model given, this computes final-token residual activations for the
S1_1P and S2_1P sets two ways: plain HF transformers (output_hidden_states,
the reference implementation) and TransformerLens 3.9 (as the 3.2 script does).
It then builds the S1/S2 pain vectors at the authors' layer from each and
reports the cosine of each with the authors' shipped vectors, plus the direct
TL-vs-HF activation agreement.

Run from the repo root:
    python replication/tools/qwen_diag.py Qwen/Qwen2.5-7B-Instruct:Qwen_2.5_7B_instruct ...
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

spec = importlib.util.spec_from_file_location(
    "extract", "scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.LOG_FILE = Path("/tmp/qwen_diag.log")

data = json.load(open("datasets/3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
OUT = Path("replication/results/qwen_diag")
OUT.mkdir(parents=True, exist_ok=True)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


report = {}
for arg in sys.argv[1:]:
    repo, name = arg.split(":")
    a = torch.load(f"author-results/3.2_pain_vectors/pain_vectors/{name}/pain_vectors.pt", weights_only=False)
    L = int(a["layer"])
    tok = AutoTokenizer.from_pretrained(repo)
    hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
    hf.eval()

    from transformer_lens import HookedTransformer
    tl = HookedTransformer.from_pretrained_no_processing(repo, hf_model=hf, tokenizer=tok, device="cuda", dtype=torch.bfloat16)
    tl.eval()

    res = {"layer": L, "bos_token": tok.bos_token, "tl_prepends_bos": bool(tl.cfg.default_prepend_bos)}
    for ds in ["S1_1P", "S2_1P"]:
        sents = data[ds]["sentences"]
        cats = [s["category"] for s in sents]
        hf_acts, tl_acts, tl_nobos = [], [], []
        with torch.no_grad():
            for s in sents:
                p = s["prompt"]
                # HF reference, exactly the tokens HF's tokenizer gives
                ids = tok(p, return_tensors="pt").to("cuda")
                hs = hf(**ids, output_hidden_states=True).hidden_states[L + 1]
                hf_acts.append(hs[0, -1].float().cpu())
                # TL as in the 3.2 script (to_tokens with its default BOS handling)
                t = tl.to_tokens(p)
                _, cache = tl.run_with_cache(t, names_filter=f"blocks.{L}.hook_resid_post")
                tl_acts.append(cache[f"blocks.{L}.hook_resid_post"][0, -1].float().cpu())
                # TL on exactly HF's tokens
                _, cache = tl.run_with_cache(ids["input_ids"], names_filter=f"blocks.{L}.hook_resid_post")
                tl_nobos.append(cache[f"blocks.{L}.hook_resid_post"][0, -1].float().cpu())
        if ds == "S1_1P":
            res["example_tokens_tl"] = tl.to_str_tokens(sents[0]["prompt"])[:4]
            res["example_tokens_hf"] = tok.convert_ids_to_tokens(tok(sents[0]["prompt"])["input_ids"])[:4]
        H, T, T2 = torch.stack(hf_acts).numpy(), torch.stack(tl_acts).numpy(), torch.stack(tl_nobos).numpy()
        key = "s1_pain_vector" if ds == "S1_1P" else "s2_pain_vector"
        av = a[key].numpy()
        vh, vt, vt2 = (m.compute_pain_vector(X, cats) for X in (H, T, T2))
        res[ds] = {
            "cos_author_vs_hf": round(cos(vh, av), 5),
            "cos_author_vs_tl_script_tokens": round(cos(vt, av), 5),
            "cos_author_vs_tl_hf_tokens": round(cos(vt2, av), 5),
            "act_cos_tl_vs_hf_mean": round(float(np.mean([cos(x, y) for x, y in zip(T2, H)])), 5),
            "act_relnorm_diff_tl_vs_hf_mean": round(float(np.mean(np.linalg.norm(T2 - H, axis=1) / np.linalg.norm(H, axis=1))), 5),
        }
    report[name] = res
    print(name, json.dumps(res, indent=1), flush=True)
    del tl, hf
    torch.cuda.empty_cache()

json.dump(report, open(OUT / "qwen_diag.json", "w"), indent=2)
print("QWEN DIAG DONE", flush=True)
