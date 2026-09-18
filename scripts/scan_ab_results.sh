#!/usr/bin/env bash
# Emit one line per completed eval, found by scanning for summary.json rather
# than by tailing logs -- tmux pipe-pane proved unreliable, and a monitor that
# depends on log capture goes silent exactly when capture breaks.
R=/nas04/yixuh/memory/router_reward_v1
for f in $(find $R/backend_ab $R/mem0_v6 $R/dedup_fix_v5 $R/v4_baseline_pp -name summary.json 2>/dev/null | sort); do
  pr=$(python3 -c "import json;print(json.load(open('$f'))['pass_rate'])" 2>/dev/null)
  echo "$(echo $f | sed "s|$R/||; s|/eval.*||") pass_rate=$pr"
done
