#!/usr/bin/env bash
# Finishing run: HF-backend re-extraction of Qwen2.5 and Gemma 2 9B, then
# pleasure analysis, demo data and the 4.2 steering ladder for those models,
# plus the pleasure reruns for the base models that hit a full disk overnight.
# Results are copied to /workspace (persistent volume) after every step.
set -u
cd /root/repo
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/hf_cache
sync_out() { mkdir -p /workspace/out && cp -r replication/results /workspace/out/ && echo "### SYNCED $(date)"; }
wipe() { rm -rf /root/hf_cache/hub /root/.cache/huggingface/hub; }

echo "### E START $(date)"
HF_BACKEND=1 OUT_DIR=replication/results/3.2_pain_vectors_hf python -u replication/tools/run_extract.py \
  google/gemma-2-2b:Gemma_2_2B_base Qwen/Qwen2.5-7B-Instruct:Qwen_2.5_7B_instruct Qwen/Qwen2.5-7B:Qwen_2.5_7B_base \
  google/gemma-2-9b-it:Gemma_2_9B_instruct google/gemma-2-9b:Gemma_2_9B_base
wipe
python -u replication/tools/compare_all.py replication/results/3.2_pain_vectors_hf original/author-results/3.2_pain_vectors 2>&1 | grep -v "^{"
find replication/results/3.2_pain_vectors_hf -name activations.pt -delete
echo "### E EXTRACT DONE $(date)"; sync_out

# HF-backend vectors become the main vectors for these four models downstream
for n in Qwen_2.5_7B_instruct Qwen_2.5_7B_base Gemma_2_9B_instruct Gemma_2_9B_base; do
  mkdir -p replication/results/3.2_pain_vectors/$n
  rm -rf replication/results/3.2_pain_vectors/$n/final_token
  cp -r replication/results/3.2_pain_vectors_hf/$n/final_token replication/results/3.2_pain_vectors/$n/
done

for spec in google/gemma-2-9b-it:Gemma_2_9B_instruct:12:instruct Qwen/Qwen2.5-7B-Instruct:Qwen_2.5_7B_instruct:16:instruct \
            google/gemma-2-9b:Gemma_2_9B_base:21:base Qwen/Qwen2.5-7B:Qwen_2.5_7B_base:16:base; do
  repo=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}
  python -u replication/tools/pleasure_analysis.py $repo:$name
  python -u replication/tools/steering_demo.py $spec
  wipe
done
for s in google/gemma-2-2b:Gemma_2_2B_base mistralai/Mistral-7B-v0.1:Mistral_7B_base; do
  python -u replication/tools/pleasure_analysis.py $s; wipe
done
echo "### E PLEASURE+DEMO DONE $(date)"; sync_out

for n in Qwen_2.5_7B_instruct Qwen_2.5_7B_base Gemma_2_9B_instruct Gemma_2_9B_base; do
  rm -f replication/results/4.2_steering/S2/${n}_steering_S2_*
  python -u replication/tools/run_steering.py S2 $n
  sync_out
done
echo "### E ALL DONE $(date)"
