#!/usr/bin/env bash
# Run a job script with a hard time limit and copy its results to the
# persistent /workspace volume when it ends, even if it failed or timed out.
# The pod itself is stopped from outside (RunPod MCP delete-pod or the console),
# so this script never touches credentials.
# Usage: pod_guard.sh JOB_SCRIPT HOURS
JOB=$1; HOURS=$2
cd /root/repo
timeout "${HOURS}h" bash "$JOB"
echo "### GUARD job ended (exit $?) $(date)"
mkdir -p /workspace/out && cp -r replication/results /workspace/out/
echo "### GUARD results on /workspace, pod can be stopped $(date)"
