#!/usr/bin/env bash
# Drive the tau2 arm of the memory/SFT ablation grid, one sub-domain at a time.
#
# tau2 differs from the other three benchmarks in two ways that shape this
# script:
#
#   1. It is THREE sub-domains (airline/retail/telecom), each with its own
#      policy document and tool set, and `run_tau2_router_llm_probe.py`
#      deliberately builds one bank per domain -- airline's refund policy has
#      no business being retrieved for a telecom task. So this is three
#      independent grids, not one.
#   2. The user side is a live Gemini simulator, so every episode costs an
#      external API call on top of the local task agent. Domains run
#      sequentially rather than in parallel to keep that rate predictable.
#
# Stage 1 only: base rollouts (train pool + test baseline) for each domain.
# The build arms and evals follow once these pass rates are known, because
# airline's 30-task train pool may be too thin to be worth the rest.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src
PY=third_party/tau2-bench/.venv/bin/python
R=scripts/run_tau2_rollout.py
URL="${TAU2_BASE_URL:-http://127.0.0.1:8032/v1}"
OUT=tau2_experiment

for dom in retail telecom airline; do
  echo "### [$dom] train pool"
  $PY -u $R --domain "$dom" --split train --output "$OUT/${dom}_base_train" \
      --max-parallel 4 --max-steps 200 --seed 20260822 \
      --model qwen35-tau --base-url "$URL"
  echo "### [$dom] test baseline"
  $PY -u $R --domain "$dom" --split test --output "$OUT/${dom}_baseline_test" \
      --max-parallel 4 --max-steps 200 --seed 20260822 \
      --model qwen35-tau --base-url "$URL"
done
echo "### TAU2 STAGE 1 (baselines) DONE"
