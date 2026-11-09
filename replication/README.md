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
