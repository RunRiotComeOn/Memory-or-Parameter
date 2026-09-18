#!/usr/bin/env bash
# The missing anchor: no-memory on the SAME replica as everything else.
#
# Every "-20pp vs baseline" claim so far compared against 0.80 from
# appworld_experiment/base_train_v2, recorded in a different run on a different
# server. Replicas are not output-equivalent (k4/k1/k6 controls all showed
# this), so that baseline cannot anchor :8012 numbers. Omitting --memory-bank
# makes retrieved_block return an empty block, i.e. a true no-memory run.
set -uo pipefail
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
while tmux has-session -t topk_sweep 2>/dev/null; do sleep 60; done
out=$ROOT/router_reward_v1/nomem_control/pp
mkdir -p $out
echo "=== no-memory control on :8012 ($(date +%H:%M:%S)) ==="
/nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
  --split train --output $out/eval --experiment-name nomem_control_pp \
  --max-parallel 1 --seed 20260822 --model qwen35-tau \
  --base-url http://127.0.0.1:8012/v1 --task-ids $TASKS
echo "=== NO-MEMORY CONTROL pass_rate: $(python3 -c "import json;print(json.load(open('$out/eval/summary.json'))['pass_rate'])" 2>/dev/null) ==="
