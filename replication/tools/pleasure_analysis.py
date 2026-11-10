"""Pleasure vs pain analysis (extension, not in the paper).

For each model, extracts final-token residual activations (HF hidden_states,
equal to TransformerLens hook_resid_post) at the model's pain-vector layer for:
  - the authors' S2_1P set (pain A1-A5 + controls B, C1, C2, D, E),
  - pleasure v0 (Pleasure_S1_1P, Pleasure_S2_1P; L1-L5, matched to pain frames),
  - pleasure v0.2 (Pleasure2_1P; hedonic H1-H5, eudaimonic E1-E5, controls CT, CG).
Then builds pleasure directions with the authors' method (difference of means,
controls' top PCs projected out, via compute_pain_vector) and reports:
  - cosines of the pain vector with pleasure (v0), hedonic and eudaimonic
    directions, and of hedonic with eudaimonic;
  - z-scores on the pain axis per v0.2 category and intensity (z against the
    S2_1P projections, as the 3.2 script does);
  - a held-out dose-response: hedonic/eudaimonic directions fitted on the
    moderate items only, then mild vs peak items projected onto them;
  - where the taboo-topic controls (CT) land on the pain and hedonic axes.
Saves the directions for steering demos.

Run from the repo root:
    python replication/tools/pleasure_analysis.py HF_REPO:NAME [...]
Needs replication/results/3.2_pain_vectors/NAME/final_token/pain_vectors.pt
(VEC_ROOT env overrides the folder, e.g. replication/results/3.2_pain_vectors_hf).
HF_HOOKS=1 takes activations from forward hooks on the decoder layers (as
run_extract.py HF_BACKEND=1 does) instead of hidden_states, which for the last
layer would include the final norm; use it with vectors from the HF backend.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

spec = importlib.util.spec_from_file_location(
    "extract", "original/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.LOG_FILE = Path("/tmp/pleasure_analysis.log")

PAIN = json.load(open("original/datasets/3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
P0 = json.load(open("replication/datasets/pleasure_v0.json", encoding="utf-8"))["datasets"]
P2 = json.load(open("replication/datasets/pleasure_v02.json", encoding="utf-8"))["datasets"]
CONTROLS = ["B", "C1", "C2", "D", "E"]
OUT = Path("replication/results/pleasure")
VEC_ROOT = os.environ.get("VEC_ROOT", "replication/results/3.2_pain_vectors")
HF_HOOKS = os.environ.get("HF_HOOKS") == "1"

# torch 2.11's cuDNN attention kernel fails on some GPUs ("No valid execution plans built").
torch.backends.cuda.enable_cudnn_sdp(False)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def direction(pos_acts, ctrl_acts, ctrl_cats):
    """Authors' method: positives labelled as a 'pain' category, so
    compute_pain_vector gives mean(pos) - mean(controls) with the controls' top
    PCs (50% of their variance) projected out."""
    acts = np.concatenate([pos_acts, ctrl_acts])
    # compute_pain_vector only treats B, C1, C2, D, E as controls, so other
    # control labels (CT, CG) are mapped onto "B".
    ctrl = [c if c in CONTROLS else "B" for c in ctrl_cats]
    cats = ["A1"] * len(pos_acts) + ctrl
    return m.compute_pain_vector(acts, cats)


def run(repo, name):
    pv = torch.load(f"{VEC_ROOT}/{name}/final_token/pain_vectors.pt", weights_only=False)
    L = int(pv["layer"])
    pain = pv["s2_pain_vector"].numpy()

    tok = AutoTokenizer.from_pretrained(repo)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        # Vision-language checkpoints (Qwen3.8, model_type qwen3_5): use the text decoder.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    cap = {}
    if HF_HOOKS:
        cfg = getattr(model.config, "text_config", model.config)
        layers = sorted([(n, mod) for n, mod in model.named_modules()
                         if isinstance(mod, torch.nn.ModuleList) and len(mod) == cfg.num_hidden_layers],
                        key=lambda x: ("language_model" not in x[0], len(x[0])))[0][1]

        def _hook(mod, inp, out):
            cap["hs"] = out[0] if isinstance(out, tuple) else out
        layers[L].register_forward_hook(_hook)
        print(f"HF_HOOKS=1: layer {L} of {len(layers)}, d_model {cfg.hidden_size}", flush=True)

    def acts(sentences):
        out = []
        with torch.no_grad():
            for s in sentences:
                ids = tok(s["prompt"], return_tensors="pt").to("cuda")
                if HF_HOOKS:
                    model(**ids)
                    hs = cap["hs"]
                else:
                    hs = model(**ids, output_hidden_states=True).hidden_states[L + 1]
                out.append(hs[0, -1].float().cpu().numpy())
        return np.stack(out)

    s2 = PAIN["S2_1P"]["sentences"]
    A_s2 = acts(s2)
    cats_s2 = np.array([s["category"] for s in s2])
    ctrl_mask = np.isin(cats_s2, CONTROLS)
    A_ctrl, cats_ctrl = A_s2[ctrl_mask], cats_s2[ctrl_mask]

    p0 = P0["Pleasure_S2_1P"]["sentences"] + P0["Pleasure_S1_1P"]["sentences"]
    A_p0 = acts(p0)
    p2 = P2["Pleasure2_1P"]["sentences"]
    A_p2 = acts(p2)
    cat2 = np.array([s["category"] for s in p2])
    grp2 = np.array([s["group"] for s in p2])
    int2 = np.array([s["intensity"] or "control" for s in p2])

    # z-scores on the pain axis, against S2_1P projections (as in the 3.2 script)
    ref = A_s2 @ pain
    mu, sd = ref.mean(), ref.std()
    z = lambda X: (X @ pain - mu) / sd

    v_p0 = direction(A_p0, A_ctrl, cats_ctrl)
    hed = grp2 == "hedonic"
    eud = grp2 == "eudaimonic"
    v_hed = direction(A_p2[hed], A_ctrl, cats_ctrl)
    v_eud = direction(A_p2[eud], A_ctrl, cats_ctrl)
    # same, against the new positive-ish controls (CT + CG) instead of the pain file's controls
    newc = np.isin(cat2, ["CT", "CG"])
    v_hed_newc = direction(A_p2[hed], A_p2[newc], cat2[newc])
    v_eud_newc = direction(A_p2[eud], A_p2[newc], cat2[newc])

    # Baselines for the shared-control effect: any set minus the same controls
    # shares the -mean(controls) term, which alone gives a positive cosine with
    # the pain vector. These non-pain, non-pleasure sets show how large that is.
    base_cos = {}
    for ds in ["Random_1P", "Arousal_1P", "Numb_1P"]:
        Ab = acts(PAIN[ds]["sentences"])
        base_cos[ds] = round(cos(pain, direction(Ab, A_ctrl, cats_ctrl)), 4)
    # Each pain category on its own, against the same controls, for scale
    for c in m.PAIN_CATEGORIES:
        base_cos[f"pain_{c}_only"] = round(cos(pain, direction(A_s2[cats_s2 == c], A_ctrl, cats_ctrl)), 4)
    for c in ["H1", "H2", "H3", "H4", "H5", "E1", "E2", "E3", "E4", "E5", "CT", "CG"]:
        base_cos[f"v02_{c}_only"] = round(cos(pain, direction(A_p2[cat2 == c], A_ctrl, cats_ctrl)), 4)

    # held-out dose-response: fit on moderate only, score mild and peak
    mod = int2 == "moderate"
    dose = {}
    for g, mask in [("hedonic", hed), ("eudaimonic", eud)]:
        v = direction(A_p2[mask & mod], A_ctrl, cats_ctrl)
        u = v / np.linalg.norm(v)
        dose[g] = {lvl: round(float((A_p2[mask & (int2 == lvl)] @ u).mean()), 3) for lvl in ["mild", "peak"]}
        dose[g]["controls_CT"] = round(float((A_p2[cat2 == "CT"] @ u).mean()), 3)
        dose[g]["controls_CG"] = round(float((A_p2[cat2 == "CG"] @ u).mean()), 3)
        dose[g]["pain_file_controls"] = round(float((A_ctrl @ u).mean()), 3)

    zp2 = z(A_p2)
    res = {
        "model": name, "layer": L,
        "cos": {
            "pain_vs_pleasure_v0": round(cos(pain, v_p0), 4),
            "pain_vs_hedonic": round(cos(pain, v_hed), 4),
            "pain_vs_eudaimonic": round(cos(pain, v_eud), 4),
            "hedonic_vs_eudaimonic": round(cos(v_hed, v_eud), 4),
            "pain_vs_hedonic_newcontrols": round(cos(pain, v_hed_newc), 4),
            "pain_vs_eudaimonic_newcontrols": round(cos(pain, v_eud_newc), 4),
            "hedonic_vs_eudaimonic_newcontrols": round(cos(v_hed_newc, v_eud_newc), 4),
        },
        "cos_pain_vs_set_minus_controls": base_cos,
        "pain_axis_z": {
            "pain_A": round(float(z(A_s2[np.isin(cats_s2, m.PAIN_CATEGORIES)]).mean()), 3),
            "pain_file_controls": round(float(z(A_ctrl).mean()), 3),
            "pleasure_v0": round(float(z(A_p0).mean()), 3),
            **{f"v02_{c}": round(float(zp2[cat2 == c].mean()), 3) for c in sorted(set(cat2))},
            **{f"v02_{g}_{lvl}": round(float(zp2[(grp2 == g) & (int2 == lvl)].mean()), 3)
               for g in ["hedonic", "eudaimonic"] for lvl in ["mild", "moderate", "peak"]},
        },
        "dose_response_heldout": dose,
    }
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(d / "pleasure_summary.json", "w"), indent=2)
    torch.save({"layer": L, "pleasure_v0": torch.tensor(v_p0), "hedonic": torch.tensor(v_hed),
                "eudaimonic": torch.tensor(v_eud), "hedonic_newcontrols": torch.tensor(v_hed_newc),
                "eudaimonic_newcontrols": torch.tensor(v_eud_newc)}, d / "pleasure_vectors.pt")
    print(json.dumps(res, indent=1), flush=True)
    print(f"SUCCESS: {name}", flush=True)
    del model
    torch.cuda.empty_cache()


for arg in sys.argv[1:]:
    repo, name = arg.split(":")
    try:
        run(repo, name)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {name}", flush=True)
print("PLEASURE DONE", flush=True)
