#!/usr/bin/env bash
# A/B the non-destructive refine fix on batch 0's candidates.
#   k0, k2 -- banks changed by the fix (v4 pass 0.40, 0.40)
#   k4     -- bank unchanged; a determinism control that must reproduce v4's 0.70
set -uo pipefail
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
for k in 0 2 4; do
  d=$ROOT/router_reward_v1/dedup_fix_v5/b0/k$k
  echo "=== evaluating k$k  ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $d/eval_self --experiment-name v5dedup_b0_k${k}_self \
    --memory-bank $d/banks/memory_appworld.json --memory-top-k 3 \
    --max-parallel 1 --seed 20260822 --model qwen35-tau \
    --base-url http://127.0.0.1:8012/v1 --task-ids $TASKS
  echo "=== k$k pass_rate: $(python3 -c "import json;print(json.load(open('$d/eval_self/summary.json'))['pass_rate'])" 2>/dev/null) ==="
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
