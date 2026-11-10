#!/usr/bin/env bash
# Qwen3.8-27B: hedonic/eudaimonic pleasure directions at the pain-vector layer
# (63, HF forward hooks to match the HF-backend extraction), then the steering
# demo with pain + hedonic + eudaimonic (added at layer 40, same grid as before).
set -u
cd /root/repo
bash replication/tools/setup_pod.sh && pip install -q flash-linear-attention
set -a; . ./.env; set +a
export HF_HOME=/root/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VEC_ROOT=replication/results/3.2_pain_vectors_hf
echo "### PLEASURE START $(date)"
HF_HOOKS=1 timeout 60m python -u replication/tools/pleasure_analysis.py Qwen/Qwen3.8-27B:Qwen3.8_27B
echo "### DEMO START $(date)"
COEFFS=-3,-2,-1.5,-1,-0.7,-0.5,-0.3,0,0.3,0.5,0.7,1,1.5,2,3 \
  timeout 60m python -u replication/tools/steering_demo.py Qwen/Qwen3.8-27B:Qwen3.8_27B:40:instruct
echo "### JOB FINISHED $(date)"
