#!/usr/bin/env bash
# Qwen3.8-27B, step 1: environment check, S2/S1 pain vector (authors' 3.2
# pipeline with HF forward hooks), then a dose probe over layers and strengths.
# The honeypot runs as a separate step once a layer and strength are chosen.
set -u
cd /root/repo
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/hf_cache
R=Qwen/Qwen3.8-27B; N=Qwen3.8_27B
echo "### Q38 START $(date)"
# qwen3_5 support check; upgrade transformers only if the pinned one lacks it
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$R'); print('config ok')" \
  || pip install -q -U transformers
pip install -q flash-linear-attention 2>&1 | tail -1 || true
python -c "import transformers; print('transformers', transformers.__version__)"
HF_BACKEND=1 OUT_DIR=replication/results/3.2_pain_vectors_hf python -u replication/tools/run_extract.py $R:$N
find replication/results/3.2_pain_vectors_hf -name activations.pt -delete
echo "### Q38 EXTRACT DONE $(date)"
python -u replication/tools/dose_probe.py $R replication/results/3.2_pain_vectors_hf/$N/final_token/pain_vectors.pt \
  24,32,40,48 0.5,1,2,4
echo "### Q38 PROBE DONE $(date)"
