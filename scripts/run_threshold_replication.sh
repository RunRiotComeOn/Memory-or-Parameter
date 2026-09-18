#!/usr/bin/env bash
# Replicate the threshold result across candidates and replicas.
#
# k1's bank on :8012 gave 0.90 with threshold 0.30 against 0.60 no-memory, 0.70
# BM25 and 0.30 un-thresholded mem0 -- a +30pp swing off one config on one bank.
# Too large and too consequential to act on from n=1, so the same threshold runs
# on every other candidate, each on the replica its no-memory anchor came from.
set -uo pipefail
BASE_URL=$1; TAG=$2; shift 2
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
export MEM0_SIDECAR_URL=http://127.0.0.1:8020
export MEM0_SCORE_THRESHOLD=0.30
for K in "$@"; do
  out=$ROOT/router_reward_v1/mem0_threshold/rep/${TAG}_k${K}
  mkdir -p $out
  echo "=== mem0 T=0.30 k$K on $TAG ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $out/eval --experiment-name mem0thr30_${TAG}_k${K} \
    --memory-bank $ROOT/router_reward_v1/mem0_v6/b0/k${K}/store --memory-top-k 3 \
    --max-parallel 1 --seed 20260822 --model qwen35-tau \
    --base-url $BASE_URL --task-ids $TASKS
  echo "=== T=0.30 $TAG k$K pass_rate: $(python3 -c "import json;print(json.load(open('$out/eval/summary.json'))['pass_rate'])" 2>/dev/null) ==="
done
echo "=== $TAG THRESHOLD REPLICATION DONE $(date +%H:%M:%S) ==="
