#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
experiment="$project_root/tau_experiment/writer_rubric_smoke_v1"
manifest="$experiment/prepared/smoke_manifest.json"
exec >>"$project_root/runtime_logs/tau_rubric_resume.log" 2>&1

while pgrep -f 'bash scripts/run_tau_writer_rubric_smoke.sh' >/dev/null; do
  sleep 20
done

memory_ok=0
for attempt in 1 2 3; do
  PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
    "$project_root/scripts/prepare_tau_memory_replay_retry.py" \
    --experiment "$experiment/memory_candidates" \
    --attempt "$attempt"
  if PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
    "$project_root/scripts/run_tau_memory_writer_replay.py" \
    --experiment "$experiment/memory_candidates" \
    --max-parallel-runs 2; then
    memory_ok=1
    break
  fi
done
if [[ "$memory_ok" != 1 ]]; then
  echo "memory replay failed after three fresh-seed attempts"
  exit 1
fi

for rubric_id in r0_faithful r1_causal_minimal r2_state_transition r3_robust; do
  for domain in airline retail telecom; do
    echo "starting SFT guided replay rubric=$rubric_id domain=$domain"
    PYTHONPATH="$project_root/src" "$project_root/third_party/tau2-bench/.venv/bin/python" \
      "$project_root/scripts/run_tau_sft_data_guided_replay.py" \
      --manifest "$manifest" \
      --candidates "$experiment/sft_candidates/$rubric_id" \
      --output "$experiment/sft_replays/$rubric_id" \
      --domain "$domain" \
      --split-section writer_generation \
      --run-tag "rubric_${rubric_id}_v1" \
      --max-concurrency 2 \
      --seed 20260820 \
      --timeout 2400
  done
done

PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
  "$project_root/scripts/summarize_tau_writer_rubric_smoke.py" \
  --experiment "$experiment" \
  --output "$experiment/summary.json"
echo "EXIT:0"
