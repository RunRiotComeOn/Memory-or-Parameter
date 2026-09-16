#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
retention="$project_root/tau_experiment/memory_writer_sft_cumulative_20260817"
replay="$project_root/tau_experiment/memory_writer_cumulative_replay_20260817"
build_session="tau_sft_writer_cumulative_build"
merged_server_session="tau_sft_writer_merged_server"
base_server_session="tau_sft_writer_replay_base_server"
log="$project_root/runtime_logs/tau_sft_writer_cumulative_pipeline.log"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$log"
}

stage "waiting_for:cumulative_memory"
while [[ ! -s "$retention/summary.json" ]]; do
  if ! tmux has-session -t "$build_session" 2>/dev/null; then
    echo "Cumulative writer build exited without summary.json" >&2
    exit 1
  fi
  sleep 30
done
stage "cumulative_memory:complete"

writer_errors="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["memory_writer"]["errors"])' "$retention/summary.json")"
if [[ "$writer_errors" != 0 ]]; then
  stage "cumulative_memory:warning writer_errors=$writer_errors"
fi

tmux kill-session -t "$merged_server_session" 2>/dev/null || true
for _ in $(seq 1 90); do
  if ! curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "Merged writer server did not stop" >&2
  exit 1
fi
stage "merged_writer_server:stopped"

tmux new-session -d -s "$base_server_session" \
  "cd '$project_root' && ./scripts/serve_tau_base_with_lora.sh > runtime_logs/tau_sft_writer_replay_base_server.log 2>&1"
server_ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null \
    | grep -q 'qwen35-tau'; then
    server_ready=true
    break
  fi
  if ! tmux has-session -t "$base_server_session" 2>/dev/null; then
    echo "Base replay server exited during startup" >&2
    tail -n 120 "$project_root/runtime_logs/tau_sft_writer_replay_base_server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$server_ready" != true ]]; then
  echo "Base replay server did not become ready" >&2
  exit 1
fi
stage "base_replay_server:ready"

cd "$project_root"
PYTHONPATH=src .venv/bin/python scripts/run_tau_writer_cumulative_replay.py \
  --output "$replay" \
  --base-writer-retention "$project_root/tau_experiment/v1/retention" \
  --sft-writer-retention "$retention" \
  --max-parallel-runs 3
stage "replay:complete"
