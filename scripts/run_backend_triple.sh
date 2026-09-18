#!/usr/bin/env bash
# Run all three memory backends for one candidate ON ONE REPLICA.
#
# Why per-candidate affinity: the k4 control showed replicas are not
# output-equivalent (a byte-identical bank scored 0.70 on TP=2 and 0.60 on
# PP=3). Keeping a candidate's three arms on a single replica makes the
# within-candidate deltas (v5-v4, mem0-v4) clean; those deltas can then be
# pooled across candidates even though candidates sat on different replicas.
#
# usage: run_backend_triple.sh <k> <base_url> <tag>
set -uo pipefail
K=$1; BASE_URL=$2; TAG=$3
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
export MEM0_SIDECAR_URL=http://127.0.0.1:8020

run () {  # arm_name bank_path
  local arm=$1 bank=$2
  local out=$ROOT/router_reward_v1/backend_ab/$TAG/k$K/$arm
  mkdir -p $out
  echo "=== k$K $arm  ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $out/eval --experiment-name ab_${TAG}_k${K}_${arm} \
    --memory-bank "$bank" --memory-top-k 3 --max-parallel 1 --seed 20260822 \
    --model qwen35-tau --base-url $BASE_URL --task-ids $TASKS
  echo "=== k$K $arm pass_rate: $(python3 -c "import json;print(json.load(open('$out/eval/summary.json'))['pass_rate'])" 2>/dev/null) ==="
}

run v4   $ROOT/router_reward_v1/cheap_train_v4/iter1/b0/k$K/banks/memory_appworld.json
run v5   $ROOT/router_reward_v1/dedup_fix_v5/b0/k$K/banks/memory_appworld.json
run mem0 $ROOT/router_reward_v1/mem0_v6/b0/k$K/store
echo "=== k$K TRIPLE DONE $(date +%H:%M:%S) ==="
