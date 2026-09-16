#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
experiment="$project_root/tau_experiment/writer_rubric_smoke_v1"
runtime="$project_root/runtime_logs"
manifest="$experiment/prepared/smoke_manifest.json"

while ! grep -q '^EXIT:0$' "$runtime/tau_rubric_memory_gen.log" 2>/dev/null; do
  if grep -q '^EXIT:[^0]' "$runtime/tau_rubric_memory_gen.log" 2>/dev/null; then
    echo "memory generation failed"
    exit 1
  fi
  sleep 20
done
while ! grep -q '^EXIT:0$' "$runtime/tau_rubric_sft_gen.log" 2>/dev/null; do
  if grep -q '^EXIT:[^0]' "$runtime/tau_rubric_sft_gen.log" 2>/dev/null; then
    echo "SFT generation failed"
    exit 1
  fi
  sleep 20
done

echo "candidate generation complete"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
  "$project_root/scripts/prepare_tau_rubric_memory_replay.py" \
  --experiment "$experiment/memory_candidates" \
  --output "$experiment/memory_candidates/replay_manifest_smoke.json" \
  --timeout 2400

cp "$experiment/memory_candidates/replay_manifest.json" \
  "$experiment/memory_candidates/replay_manifest_full.json"
cp "$experiment/memory_candidates/replay_manifest_smoke.json" \
  "$experiment/memory_candidates/replay_manifest.json"

echo "starting memory paired replay"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
  "$project_root/scripts/run_tau_memory_writer_replay.py" \
  --experiment "$experiment/memory_candidates" \
  --max-parallel-runs 3

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
