#!/usr/bin/env bash
# Same-server baseline for the memory-backend comparison.
#
# The k4 control proved the PP=3 replica (:8012) is not output-equivalent to the
# TP=2 replicas (:8010/:8011) that produced the recorded v4 numbers: with a
# byte-identical bank, 229360a_1 flipped pass->fail, costing 0.10. So the
# recorded v4 pass rates cannot be used as the baseline for v5/mem0 runs done on
# :8012. This re-runs v4's OWN banks on :8012 so all three arms share a server
# and only the memory backend differs.
set -uo pipefail
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src

while tmux has-session -t mem0_eval 2>/dev/null; do sleep 60; done
echo "=== mem0 arm finished, starting v4-on-:8012 baselines $(date +%H:%M:%S) ==="
for k in 0 2; do
  d=$ROOT/router_reward_v1/v4_baseline_pp/b0/k$k
  mkdir -p $d
  echo "=== evaluating v4 bank k$k on :8012 ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $d/eval_self --experiment-name v4base_pp_b0_k${k}_self \
    --memory-bank $ROOT/router_reward_v1/cheap_train_v4/iter1/b0/k$k/banks/memory_appworld.json \
    --memory-top-k 3 --max-parallel 1 --seed 20260822 --model qwen35-tau \
    --base-url http://127.0.0.1:8012/v1 --task-ids $TASKS
  echo "=== v4-on-PP k$k pass_rate: $(python3 -c "import json;print(json.load(open('$d/eval_self/summary.json'))['pass_rate'])" 2>/dev/null) ==="
done
echo "=== V4 BASELINE ON PP DONE $(date +%H:%M:%S) ==="
