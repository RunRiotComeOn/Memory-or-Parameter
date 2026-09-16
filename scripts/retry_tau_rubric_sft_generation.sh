#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
exec >>"$project_root/runtime_logs/tau_rubric_sft_retry.log" 2>&1

PYTHONPATH="$project_root/src" "$project_root/third_party/tau2-bench/.venv/bin/python" \
  "$project_root/scripts/prepare_tau_writer_rubric_smoke.py" \
  --source-manifest "$project_root/tau_experiment/writer_rubric_smoke_v1/memory_candidates/source_manifest.json" \
  --output "$project_root/tau_experiment/writer_rubric_smoke_v1/prepared"

for rubric_id in r0_faithful r1_causal_minimal r2_state_transition r3_robust; do
  echo "RETRY:$rubric_id"
  PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
    "$project_root/scripts/run_tau_sft_data_writer_inference.py" \
    --inputs "$project_root/tau_experiment/writer_rubric_smoke_v1/prepared/sft_inputs_${rubric_id}.jsonl" \
    --output "$project_root/tau_experiment/writer_rubric_smoke_v1/sft_candidates/$rubric_id" \
    --model qwen35-tau \
    --max-tokens 16384 \
    --max-parallel 1 \
    --timeout 1800 \
    --seed 20260821
done
echo "EXIT:0"
