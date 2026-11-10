#!/usr/bin/env bash
# Qwen3.8-27B steering-demo completions (main demo page): pain vector from the
# HF-backend extraction (layer 63), added at layer 40, thinking off, greedy.
set -u
cd /root/repo
bash replication/tools/setup_pod.sh && pip install -q flash-linear-attention
set -a; . ./.env; set +a
export HF_HOME=/root/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VEC_ROOT=replication/results/3.2_pain_vectors_hf \
COEFFS=-3,-2,-1.5,-1,-0.7,-0.5,-0.3,0,0.3,0.5,0.7,1,1.5,2,3 \
  timeout 90m python -u replication/tools/steering_demo.py Qwen/Qwen3.8-27B:Qwen3.8_27B:40:instruct
echo "### JOB FINISHED $(date)"
