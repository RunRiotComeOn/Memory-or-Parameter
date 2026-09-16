#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="${1:-$project_root/tau_experiment/memory_writer_utility_v2}"
generation_session="${TAU_MEMORY_WRITER_GENERATION_SESSION:-tau_memory_writer_generate}"
log_dir="$project_root/runtime_logs/tau_memory_writer_utility_v1"

mkdir -p "$log_dir"
printf '%s waiting_for:candidate_generation\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while [[ ! -s "$experiment/replay_manifest.json" ]]; do
  if ! tmux has-session -t "$generation_session" 2>/dev/null; then
    echo "Candidate generation exited without replay_manifest.json" >&2
    exit 1
  fi
  sleep 15
done
printf '%s candidate_generation:complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$project_root"
PYTHONPATH=src .venv/bin/python scripts/run_tau_memory_writer_replay.py \
  --experiment "$experiment" \
  --max-parallel-runs 6
printf '%s utility_pipeline:complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
