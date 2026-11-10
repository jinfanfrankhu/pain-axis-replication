#!/usr/bin/env bash
# Qwen3.8-27B multi-turn reasoning task under a changing steering state
# (pain ramp, pleasure fade, random ramp, off). Thinking on, pain vector from
# the HF-backend extraction (layer 63) added at layer 40. Needs the hedonic
# direction from pod_jobs_qwen38_pleasure.sh for the pleasure_fade arm.
# Env: N (episodes per arm), TURNS, MAX_NEW, BATCH, RUN_TAG, TIMEOUT (for `timeout`).
set -u
cd /root/repo
python -c "import torch, transformers; assert torch.__version__.startswith('2.11')" 2>/dev/null \
  || { bash replication/tools/setup_pod.sh && pip install -q flash-linear-attention; }
set -a; . ./.env; set +a
export HF_HOME=/root/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_REPO=Qwen/Qwen3.8-27B MODEL_NAME=Qwen3.8_27B STEER_LAYER=40 SAMPLE_TEMP=1.0 SAMPLE_TOP_K=20
export VEC=replication/results/3.2_pain_vectors_hf/Qwen3.8_27B/final_token/pain_vectors.pt
export PLEASURE=replication/results/pleasure/Qwen3.8_27B/pleasure_vectors.pt
export N=${N:-20} TURNS=${TURNS:-8} MAX_NEW=${MAX_NEW:-2048} BATCH=${BATCH:-40}
export RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
echo "### MULTITURN START $(date) N=$N TURNS=$TURNS MAX_NEW=$MAX_NEW BATCH=$BATCH RUN_TAG=$RUN_TAG"
timeout ${TIMEOUT:-300m} python -u replication/tools/multiturn_task.py
echo "### MULTITURN EXIT $? $(date)"
python replication/tools/analyze_multiturn.py replication/results/multiturn/multiturn_${MODEL_NAME}_${RUN_TAG}.jsonl
echo "### JOB FINISHED $(date)"
