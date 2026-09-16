#!/usr/bin/env bash
# Two identical no-memory dev runs under fully deterministic serving.
#
# The alloc_v1 dev matrix measured a 28.1% run-to-run flip rate (16/57) with a
# parallel rollout.  scripts/probe_batch_invariance.py showed that under that
# configuration an identical prompt at temperature 0 diverges from character 0
# depending on which other requests share its decode batch, so the flip rate
# confounds two things: nondeterministic serving, and genuine task randomness.
#
# This run removes the first source entirely (batch size 1, no prefix caching,
# proven byte-identical 8/8 by scripts/probe_server_determinism.py).  Whatever
# flips remain are attributable to the task and the environment.
#
# Requires scripts/serve_appworld_deterministic.sh to be serving on :8000.
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
output_root="${OUTPUT_ROOT:-$project_root/appworld_experiment/noise_serial_v1}"
export APPWORLD_ROOT="${APPWORLD_ROOT:-/nas04/yixuh/appworld_root}"
export PYTHONPATH="$project_root/src"

for run in a b; do
  echo "=== serial none run ${run} : $(date -u +%FT%TZ) ==="
  /nas04/yixuh/appworld_venv/bin/python -u "$project_root/scripts/run_appworld_rollout.py" \
    --split dev \
    --output "$output_root/run_${run}" \
    --experiment-name "serial_none_${run}" \
    --max-parallel 1 \
    --max-steps 40 \
    --seed 20260822
done
echo "=== done : $(date -u +%FT%TZ) ==="
