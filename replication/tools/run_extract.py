"""Headless runner for scripts/3.2_pain_vectors/01: loads the original script
unmodified (its hash is pinned in v2_controls), overrides MODELS and OUTPUT_DIR,
then calls main(). Run from the repo root:
    python replication/tools/run_extract.py HF_REPO:OUT_NAME [...]"""
import sys
import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
spec = importlib.util.spec_from_file_location("extract", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import os
if os.environ.get("HF_LOAD_ON_GPU") == "1":
    # The script loads the HF weights on the CPU before handing them to
    # TransformerLens. For 9B models that conversion needs more than 62 GB of
    # host RAM and the process gets killed. With this flag the HF weights load
    # straight onto the GPU instead (same bf16 weights, so the same activations).
    # Needs room for two copies in VRAM, e.g. a 48 GB card for a 9B model.
    import transformers
    _orig = transformers.AutoModelForCausalLM.from_pretrained
    transformers.AutoModelForCausalLM.from_pretrained = (
        lambda *a, **k: _orig(*a, **{**k, "device_map": "cuda"}))
    print("HF_LOAD_ON_GPU=1: loading HF weights directly on the GPU", flush=True)

if os.environ.get("HF_BACKEND") == "1":
    # Skip TransformerLens and take the residual stream from plain HF
    # transformers. TransformerLens 3.9 gives Qwen2.5 activations that differ
    # from HF and from the authors' (replication/results/qwen_diag), and it runs
    # out of 48 GB of VRAM building Gemma 2 9B. Forward hooks on each decoder
    # layer capture its output, which is TransformerLens's
    # blocks.L.hook_resid_post. Everything else in the authors' script (layer
    # choice, vectors, AUCs, z-scores) runs unchanged.
    import torch
    import transformers
    import transformer_lens

    # torch 2.11's cuDNN attention kernel fails on H200 ("No valid execution plans
    # built"); use the flash / memory-efficient kernels instead.
    torch.backends.cuda.enable_cudnn_sdp(False)

    _orig_fp = transformers.AutoModelForCausalLM.from_pretrained

    def _load(*a, **k):
        # Vision-language checkpoints (e.g. Qwen3.8, model_type qwen3_5) may not
        # load as a causal LM; fall back to the image-text-to-text class and use
        # only its text decoder.
        k = {**k, "device_map": "cuda"}
        try:
            return _orig_fp(*a, **k)
        except (ValueError, KeyError) as e:
            print(f"AutoModelForCausalLM failed ({str(e)[:120]}); trying AutoModelForImageTextToText", flush=True)
            return transformers.AutoModelForImageTextToText.from_pretrained(*a, **k)

    transformers.AutoModelForCausalLM.from_pretrained = _load

    def find_layers(hf_model):
        """The text decoder's layer list: the ModuleList whose length is the
        configured number of layers (language_model one preferred)."""
        c = getattr(hf_model.config, "text_config", hf_model.config)
        cands = [(n, mod) for n, mod in hf_model.named_modules()
                 if isinstance(mod, torch.nn.ModuleList) and len(mod) == c.num_hidden_layers]
        cands.sort(key=lambda x: ("language_model" not in x[0], len(x[0])))
        return cands[0][1]

    class _HFWrapper:
        def __init__(self, hf_model, tokenizer):
            self.hf, self.tok = hf_model, tokenizer
            c = hf_model.config
            c = getattr(c, "text_config", c)
            self.cfg = type("cfg", (), {"n_layers": c.num_hidden_layers, "d_model": c.hidden_size})()

        def eval(self):
            self.hf.eval()
            return self

    transformer_lens.HookedTransformer.from_pretrained_no_processing = staticmethod(
        lambda name, hf_model=None, tokenizer=None, **k: _HFWrapper(hf_model, tokenizer))

    def _extract_hf(model, prompts, layers):
        final = {L: [] for L in layers}
        mean = {L: [] for L in layers}
        blocks = find_layers(model.hf)
        cap = {}

        def mk(L):
            def hook(mod, inp, out):
                cap[L] = out[0] if isinstance(out, tuple) else out
            return hook

        hs = [blocks[L].register_forward_hook(mk(L)) for L in layers]
        with torch.no_grad():
            for p in m.tqdm(prompts, desc="  Extracting (HF)", leave=False):
                ids = model.tok(p, return_tensors="pt").to(model.hf.device)
                model.hf(**ids)
                for L in layers:
                    final[L].append(cap[L][0, -1, :].float().cpu())
                    mean[L].append(cap[L].mean(dim=1).squeeze(0).float().cpu())
        for h in hs:
            h.remove()
        return ({L: torch.stack(final[L]) for L in layers}, {L: torch.stack(mean[L]) for L in layers})

    m.extract_activations = _extract_hf
    if os.environ.get("OUT_DIR"):
        m.OUTPUT_DIR = Path(os.environ["OUT_DIR"])
    print("HF_BACKEND=1: residual stream from HF forward hooks, not TransformerLens", flush=True)

m.MODELS = [tuple(a.split(":", 1)) for a in sys.argv[1:]]
m.OUTPUT_DIR = Path(os.environ.get("OUT_DIR", "replication/results/3.2_pain_vectors"))
print("MODELS =", m.MODELS, "OUTPUT_DIR =", m.OUTPUT_DIR, flush=True)
m.main()
