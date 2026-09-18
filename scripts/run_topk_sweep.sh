#!/usr/bin/env bash
# Does injecting MORE retrieved memory help or hurt?
#
# top_k=3 was inherited from tau2-bench and never swept here. Two opposing
# priors: top-5 is the common RAG default, but k1 showed the opposite locally --
# same memory content, BM25 injecting ~1.7 entries/task scored 8/10 while mem0
# filling all 3 slots scored 6/10. A sweep on a FIXED bank and a FIXED replica
# settles it: the only variable is how many entries get injected.
#
# usage: run_topk_sweep.sh <base_url> <tag>
set -uo pipefail
BASE_URL=$1; TAG=$2
ROOT=/nas04/yixuh/memory
TASKS="07b42fd_1 07b42fd_2 07b42fd_3 229360a_1 229360a_2 229360a_3 22cc237_1 22cc237_2 22cc237_3 27e1026_1"
export APPWORLD_ROOT=/nas04/yixuh/appworld_root
export PYTHONPATH=$ROOT/src
export MEM0_SIDECAR_URL=http://127.0.0.1:8020

sweep () {  # arm bank_path k
  local arm=$1 bank=$2 k=$3
  local out=$ROOT/router_reward_v1/topk_sweep/$TAG/${arm}_top${k}
  mkdir -p $out
  echo "=== $arm top_k=$k ($(date +%H:%M:%S)) ==="
  /nas04/yixuh/appworld_venv/bin/python -u $ROOT/scripts/run_appworld_rollout.py \
    --split train --output $out/eval --experiment-name sweep_${TAG}_${arm}_top${k} \
    --memory-bank "$bank" --memory-top-k $k --max-parallel 1 --seed 20260822 \
    --model qwen35-tau --base-url $BASE_URL --task-ids $TASKS
  echo "=== $arm top_k=$k pass_rate: $(python3 -c "import json;print(json.load(open('$out/eval/summary.json'))['pass_rate'])" 2>/dev/null) ==="
}

# k1: the candidate where the retrieval-volume effect showed up. Its BM25 bank
# scored 8/10 at top_k=3 (== the no-memory baseline), so both directions are
# visible from here.
BM25=$ROOT/router_reward_v1/cheap_train_v4/iter1/b0/k1/banks/memory_appworld.json
MEM0=$ROOT/router_reward_v1/mem0_v6/b0/k1/store
# top_k=3 is re-measured here rather than reused from replica A: replicas are
# not output-equivalent, so a sweep that borrowed its reference point from a
# different replica would confound "more memory" with "different replica" --
# the exact error the k4 control caught earlier.
for k in 1 3 5 10; do sweep bm25 $BM25 $k; done
for k in 1 3 5;    do sweep mem0 $MEM0 $k; done
echo "=== TOPK SWEEP DONE $(date +%H:%M:%S) ==="
