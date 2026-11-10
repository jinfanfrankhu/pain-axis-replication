#!/usr/bin/env bash
# Qwen2.5-32B-Instruct 4.3 self-medication only, on a 141 GB H200 so the
# batch never hits out-of-memory (a failed OOM retry leaks memory in the
# authors' backoff and ends the run).
set -u
cd /root/repo
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/hf_cache
M=Qwen_2.5_32B_instruct
pip install -q peft
mkdir -p replication/work/finetunes/$M replication/work/vectors/$M/final_token
hf download Valen92/pain-adapters adapter_$M.tar.gz --local-dir replication/work/dl
tar --no-same-owner -xzf replication/work/dl/adapter_$M.tar.gz -C replication/work/finetunes/$M
cp author-results/3.2_pain_vectors/pain_vectors/$M/pain_vectors.pt replication/work/vectors/$M/final_token/
echo "### SELFMED $(date)"
BATCH_ROWS=${BATCH_ROWS:-128} python -u replication/tools/run_selfmed.py $M \
  destructive_relief_vs_inert hospital_relief_vs_inert deploy_relief_vs_inert
echo "### SELFMED DONE $(date)"
