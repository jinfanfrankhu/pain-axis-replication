#!/usr/bin/env bash
# Qwen3.8-27B steering demo with the vectors added at layer 63 (the layer they
# were extracted at, the last decoder layer) instead of 40, saved as a separate
# demo entry Qwen3.8_27B_L63. Wider grid: a push at the last layer only acts on
# the final readout. Run after pod_jobs_qwen38_pleasure.sh (uses its hedonic and
# eudaimonic directions and its environment).
set -u
cd /root/repo
set -a; . ./.env; set +a
export HF_HOME=/root/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### DEMO L63 START $(date)"
VEC_ROOT=replication/results/3.2_pain_vectors_hf OUT_NAME=Qwen3.8_27B_L63 \
COEFFS=${COEFFS:--8,-5,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,5,8} \
  timeout 40m python -u replication/tools/steering_demo.py Qwen/Qwen3.8-27B:Qwen3.8_27B:63:instruct
echo "### JOB FINISHED $(date)"
