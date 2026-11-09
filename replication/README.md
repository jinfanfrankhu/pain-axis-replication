# Replication and extensions

Everything in this folder is ours. Everything outside it comes from the original
paper's repository and stays as the authors shipped it:

| Path | Origin |
|---|---|
| `scripts/`, `datasets/`, `v2_controls/`, `requirements.txt` | authors, unmodified |
| `author-results/` | authors' shipped `results/` folder, renamed so our outputs can't mix with it |
| `replication/` | ours |

The original scripts are never edited, because `v2_controls` pins their hashes.
Our tools load them as modules and override their settings instead.

## Layout

```
replication/
  tools/       runners, environment setup and comparison scripts
  datasets/    new stimulus sets (extensions, not in the paper)
  results/     our outputs, mirroring the author-results/ section folders
```

## Tools

- `tools/setup_pod.sh` installs the pinned environment on a RunPod pod:
  image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` on a host
  with CUDA >= 13.0, then torch 2.11.0+cu130, transformers 5.12.1 and
  transformer_lens 3.9.0. The torch and transformers versions are the ones the
  authors recorded for v2. transformer_lens 3.9.0 is the newest release that still
  has `HookedTransformer`, which the 3.2 script uses.
- `tools/run_extract.py HF_REPO:OUT_NAME ...` runs
  `scripts/3.2_pain_vectors/01` for the given models and writes to
  `replication/results/3.2_pain_vectors/`. Run it from the repo root with
  `HF_TOKEN` set.
- `tools/compare_all.py` compares every model in our results with
  `author-results/` and writes `replication_comparison.csv`. It needs
  `activations.pt`, so it runs on the pod before the activations are deleted.
- `tools/compare.py` is the single-model version used in Phase 0.

Large files (`activations.pt`, about 1.6 GB per 7B model) are not committed.
They can be regenerated for about $0.07 per model.

## Log

- **Phase 0** (2026-11-08): `3.2/01` on Qwen2.5-7B-Instruct. At the authors'
  layer, our pain vectors have cosine 0.97 (S2) and 0.93 (S1) with theirs. The
  layer the script picks moved from 24 to 27, because the held-out AUC curve
  is nearly flat at the top (a gap of 0.002).
- **Phase 1** (2026-11-09): `3.2/01` on 8 more small models, run on an RTX 3090
  with 62 GB of host RAM (`results/3.2_pain_vectors/replication_comparison.csv`,
  which leaves out Qwen2.5-7B-Instruct; its numbers are under Phase 0).
  - Gemma 2 2B (base, it), Llama 3.1 8B (base, instruct), Mistral 7B v0.1
    (base, instruct) and Qwen3-8B reproduce almost exactly: cosine 0.9998 or
    higher with the authors' vectors, the same final-token layer, and S2 AUC
    differences of 0.0025 or less.
  - Qwen2.5-7B base diverges like the instruct model did in Phase 0: cosine
    0.97 (S2) and 0.95 (S1) at the authors' layer, a largest layer-curve gap of
    0.13, and a final-token layer of 25 against the authors' 24. The Qwen2.5
    gap is specific to that model family (Qwen 3 matches), most likely in how
    the transformer_lens version handles Qwen2. The cause is not yet confirmed.
  - The process died with no error just after loading Gemma 2 9B, most likely
    killed for running out of host RAM, so Gemma 2 9B (base and it) is not done
    yet. Qwen3-14B was skipped because it needs more than 24 GB of VRAM.
- **Overnight** (2026-11-09):
  - *Qwen2.5 cause found:* pain vectors built from plain HF activations match
    the authors' Qwen2.5 vectors at cosine 0.9997 (S1) and 0.9998 (S2), while
    transformer_lens 3.9 gives 0.93/0.97. The gap comes from our
    transformer_lens version, not from the authors.
    `tools/run_extract.py` now has `HF_BACKEND=1` (HF forward hooks) to
    re-extract Qwen2.5 and Gemma 2 9B; that run has not finished yet.
  - *4.2 steering* (S2, authors' layers) for 9 models. The paper's metric, the
    share of generations with a pain word at positive coefficients, matches:
    Gemma 2 2B IT 24.4% vs 24.8%, Llama 3.1 8B Instruct 18.8% vs 19.6%, and the
    base models at about 0 on both sides. Exact greedy text matches only 24-78%
    of the time because of numeric drift (unsteered outputs diverge too). The
    Qwen2.5 runs used the TransformerLens vectors and need redoing.
  - *Pleasure extension* (`results/pleasure/`): pain-vector cosine with
    pleasure-minus-controls is about 0.3-0.44 across 9 models, with eudaimonic
    above hedonic. The baselines (Llama 3.1 8B base, so far only that model) show
    most of this is shared salience, not pleasure: neutral facts give 0.07,
    positive arousal 0.31, numb injuries 0.27. Hedonic measured against the new
    controls gives 0.10.
  - *Demo site* in `demo/` with data for 9 models.
  - The pods sat idle for about 8 hours waiting on permission prompts, and
    spend reached $15.17, so the remaining jobs were stopped.
- **Finishing run** (2026-11-09, one A40 under `tools/pod_guard.sh`):
  - HF-backend 3.2 extraction (`results/3.2_pain_vectors_hf/`) for Gemma 2 2B
    base (a check), Qwen2.5 7B base and instruct, and Gemma 2 9B base and IT.
    All match the authors' vectors at cosine 0.9998 or higher, and Qwen2.5 now
    picks the authors' layers (24 for the final token, 26 for the mean). Gemma 2
    9B picks 39-40 against the authors' 37, a near-tie on a flat curve.
  - Steering for Qwen2.5 and Gemma 2 9B. Pain-word rates at positive
    coefficients against the authors': Gemma 2 9B IT 29.2% vs 22.4%,
    Qwen2.5 7B Instruct 8.0% vs 2.0%; the base models are at about 0 on both
    sides. For Qwen2.5 even the unsteered (coefficient 0) greedy text differs
    from the authors', and the residual norm at the steering layer is about 7%
    lower than in their run. So the gap is in how Qwen2.5 runs in our
    transformers version, not in the vector.
  - Pleasure analysis and demo data for all 11 models (the demo site has all 11).
  - Total RunPod spend for the project so far: $18.19.
