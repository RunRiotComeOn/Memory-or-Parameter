#!/usr/bin/env bash
# Third arm of the memory-backend A/B, on batch 0's same 10 tasks:
#   v4 (destructive Jaccard refine) | v5 (non-destructive merge) | v6 (mem0)
# Waits for the v5 eval to finish first: both drive the same serial :8012
# replica, so running them at once only halves throughput for both.
set -uo pipefail
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
export MEM0_SIDECAR_URL=http://127.0.0.1:8020

while tmux has-session -t v5_dedup_eval 2>/dev/null; do sleep 60; done
echo "=== v5 eval finished, starting mem0 arm $(date +%H:%M:%S) ==="
for k in 0 2 4; do
  d=$ROOT/router_reward_v1/mem0_v6/b0/k$k
  echo "=== evaluating mem0 k$k ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $d/eval_self --experiment-name mem0v6_b0_k${k}_self \
    --memory-bank $d/store --memory-top-k 3 \
    --max-parallel 1 --seed 20260822 --model qwen35-tau \
    --base-url http://127.0.0.1:8012/v1 --task-ids $TASKS
  echo "=== mem0 k$k pass_rate: $(python3 -c "import json;print(json.load(open('$d/eval_self/summary.json'))['pass_rate'])" 2>/dev/null) ==="
done
echo "=== MEM0 ARM DONE $(date +%H:%M:%S) ==="
