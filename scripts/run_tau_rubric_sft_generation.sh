#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
exec >>"$project_root/runtime_logs/tau_rubric_sft_gen.log" 2>&1

for rubric_id in r0_faithful r1_causal_minimal r2_state_transition r3_robust; do
  echo "START:$rubric_id"
  PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
    "$project_root/scripts/run_tau_sft_data_writer_inference.py" \
    --inputs "$project_root/tau_experiment/writer_rubric_smoke_v1/prepared/sft_inputs_${rubric_id}.jsonl" \
    --output "$project_root/tau_experiment/writer_rubric_smoke_v1/sft_candidates/$rubric_id" \
    --model qwen35-tau \
    --max-tokens 8192 \
    --max-parallel 2 \
    --timeout 1200 \
    --seed 20260820
done
echo "EXIT:0"
