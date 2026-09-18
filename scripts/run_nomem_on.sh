#!/usr/bin/env bash
# No-memory anchor for one replica. Each replica needs its own: the candidates
# measured on it can only be judged against a no-memory run from the same
# replica, since replicas are not output-equivalent.
set -uo pipefail
BASE_URL=$1; TAG=$2
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
out=$ROOT/router_reward_v1/nomem_control/$TAG
mkdir -p $out
echo "=== no-memory control $TAG ($(date +%H:%M:%S)) ==="
/nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
  --split train --output $out/eval --experiment-name nomem_$TAG \
  --max-parallel 1 --seed 20260822 --model qwen35-tau \
  --base-url $BASE_URL --task-ids $TASKS
echo "=== NO-MEMORY $TAG pass_rate: $(python3 -c "import json;print(json.load(open('$out/eval/summary.json'))['pass_rate'])" 2>/dev/null) ==="
