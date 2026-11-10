#!/usr/bin/env bash
# Rebuttal extraction job: one 24 GB GPU, two 7-8B instruct models.
# Run from the repo root on the pod: setsid nohup bash rebuttal/tools/pod_job.sh > rebuttal_log.txt 2>&1 &
set -uo pipefail
cd /root/repo
set -a; source .env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=0
bash replication/tools/setup_pod.sh || { echo "### SETUP FAILED"; exit 1; }
python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
timeout 40m python -u rebuttal/tools/extract_acts.py Qwen/Qwen2.5-7B-Instruct Qwen_2.5_7B_instruct \
    original/author-results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct 24 || echo "### QWEN FAILED"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
timeout 40m python -u rebuttal/tools/extract_acts.py meta-llama/Llama-3.1-8B-Instruct Llama_3.1_8B_instruct \
    original/author-results/3.2_pain_vectors/pain_vectors/Llama_3.1_8B_instruct 28 || echo "### LLAMA FAILED"
echo "### ALL DONE $(date)"
