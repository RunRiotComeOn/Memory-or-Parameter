#!/usr/bin/env bash
# Does a similarity floor fix mem0's deficit?
#
# On :8012, same bank (k1), same tasks: no-memory 0.60, BM25 0.70, mem0 with no
# threshold 0.30 while injecting 3.0 entries/task against BM25's 1.7. 24 of
# mem0's 30 retrieved slots scored below 0.40. Thresholds picked from that
# observed distribution: 0.30 -> ~1.1 entries/task, 0.50 -> ~0.6.
set -uo pipefail
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
export MEM0_SIDECAR_URL=http://127.0.0.1:8020
for T in 0.30 0.50; do
  out=$ROOT/router_reward_v1/mem0_threshold/t${T}
  mkdir -p $out
  export MEM0_SCORE_THRESHOLD=$T
  echo "=== mem0 top_k=3 threshold=$T ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $out/eval --experiment-name mem0thr_${T} \
    --memory-bank $ROOT/router_reward_v1/mem0_v6/b0/k1/store --memory-top-k 3 \
    --max-parallel 1 --seed 20260822 --model qwen35-tau \
    --base-url http://127.0.0.1:8012/v1 --task-ids $TASKS
  echo "=== threshold=$T pass_rate: $(python3 -c "import json;print(json.load(open('$out/eval/summary.json'))['pass_rate'])" 2>/dev/null) ==="
done
echo "=== THRESHOLD SWEEP DONE $(date +%H:%M:%S) ==="
