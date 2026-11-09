#!/usr/bin/env bash
# Pod environment for the replication runs. Expects the image
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 on a host with CUDA >= 13.0.
# Pins the versions used in Phase 0 (torch/transformers as recorded by the authors' v2;
# transformer_lens 3.9.0 is the last line that still has HookedTransformer).
set -euo pipefail
pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -q transformers==5.12.1 transformer_lens==3.9.0 accelerate \
    huggingface_hub safetensors numpy pandas scipy scikit-learn matplotlib tqdm
# The base image's torchvision is built for torch 2.4 and breaks transformers imports.
pip uninstall -y -q torchvision torchaudio || true
python -c "import torch, transformers; from transformer_lens import HookedTransformer; \
print('env ok', torch.__version__, transformers.__version__, torch.cuda.is_available())"
