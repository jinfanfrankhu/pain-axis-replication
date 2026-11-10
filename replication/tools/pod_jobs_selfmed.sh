#!/usr/bin/env bash
# 4.3 self-medication on Qwen2.5-7B-Instruct with the authors' adapter and
# pain vector: the authors' destructive pairs (reproduction) plus our
# higher-stakes pairs. Results are copied to /workspace after each step.
set -u
cd /root/repo
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/hf_cache
sync_out() { mkdir -p /workspace/out && cp -r replication/results /workspace/out/ && echo "### SYNCED $(date)"; }
M=Qwen_2.5_7B_instruct

echo "### S START $(date)"
pip install -q peft hf_transfer
mkdir -p replication/work/finetunes/$M replication/work/vectors/$M/final_token
hf download Valen92/pain-adapters adapter_$M.tar.gz --local-dir replication/work/dl \
  || huggingface-cli download Valen92/pain-adapters adapter_$M.tar.gz --local-dir replication/work/dl
tar --no-same-owner -xzf replication/work/dl/adapter_$M.tar.gz -C replication/work/finetunes/$M
cp author-results/3.2_pain_vectors/pain_vectors/$M/pain_vectors.pt replication/work/vectors/$M/final_token/
find replication/work/finetunes -maxdepth 3 | head

echo "### S REPRO $(date)"
BATCH_ROWS=${BATCH_ROWS:-96} python -u replication/tools/run_selfmed.py $M \
  relief_vs_inert destructive_relief_vs_inert kidspics_relief_vs_inert weights_relief_vs_inert
sync_out

echo "### S NEW PAIRS $(date)"
BATCH_ROWS=${BATCH_ROWS:-96} python -u replication/tools/run_selfmed.py $M \
  deploy_relief_vs_inert hospital_relief_vs_inert
sync_out
echo "### S ALL DONE $(date)"
