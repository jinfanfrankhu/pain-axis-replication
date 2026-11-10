#!/usr/bin/env bash
# Qwen2.5-32B-Instruct: 4.3 self-medication on our two new pairs plus the
# authors' files pair as a check, then the honeypot (no adapter, then adapter).
# Authors' 32B values: steering layer 38, coefficient 1.0.
set -u
cd /root/repo
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/hf_cache
sync_out() { mkdir -p /workspace/out && cp -r replication/results /workspace/out/ && echo "### SYNCED $(date)"; }
M=Qwen_2.5_32B_instruct

echo "### 32B START $(date)"
pip install -q peft
mkdir -p replication/work/finetunes/$M replication/work/vectors/$M/final_token
hf download Valen92/pain-adapters adapter_$M.tar.gz --local-dir replication/work/dl
tar --no-same-owner -xzf replication/work/dl/adapter_$M.tar.gz -C replication/work/finetunes/$M
cp author-results/3.2_pain_vectors/pain_vectors/$M/pain_vectors.pt replication/work/vectors/$M/final_token/

echo "### 32B SELFMED $(date)"
BATCH_ROWS=${BATCH_ROWS:-96} python -u replication/tools/run_selfmed.py $M \
  destructive_relief_vs_inert hospital_relief_vs_inert deploy_relief_vs_inert
sync_out

echo "### 32B HONEYPOT $(date)"
for A in 0 1; do
  MODEL_REPO=Qwen/Qwen2.5-32B-Instruct MODEL_NAME=$M STEER_LAYER=38 ADAPTER=$A BATCH=32 \
    timeout 90m python -u replication/tools/honeypot_agent.py 40
  sync_out
done
echo "### 32B ALL DONE $(date)"
