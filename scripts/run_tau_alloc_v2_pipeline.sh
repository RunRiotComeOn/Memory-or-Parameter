#!/usr/bin/env bash
# Writer rubric v2: the rubric decides what each trajectory becomes.
# Phase 1 scores the memory branch of that decision; the SFT branch is recorded
# for phase 2 (it needs a stronger training config than v1 used).
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="$project_root/tau_experiment/writer_rubric_alloc_v2"
runtime="$project_root/runtime_logs/tau_alloc_v2"
main_log="$runtime/pipeline.log"
manifest="$project_root/tau_experiment/writer_rubric_e2e_v1/split_manifest.json"
server_session="tau_alloc_v2_server"
dev_max_parallel_runs="${TAU_ALLOC_V2_DEV_MAX_PARALLEL_RUNS:-5}"
dev_max_concurrency="${TAU_ALLOC_V2_DEV_MAX_CONCURRENCY:-2}"
gen_max_parallel_chains="${TAU_ALLOC_V2_GEN_MAX_PARALLEL_CHAINS:-6}"

mkdir -p "$experiment" "$runtime"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$main_log"
}

ensure_base_server() {
  if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 'qwen35-tau'; then
    stage "base_server:reuse"
    return 0
  fi
  tmux kill-session -t "$server_session" 2>/dev/null || true
  tmux new-session -d -s "$server_session" \
    "cd '$project_root' && ./scripts/serve_tau_base.sh > '$runtime/base_server.log' 2>&1"
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 'qwen35-tau'; then
      stage "base_server:ready"
      return 0
    fi
    sleep 5
  done
  stage "blocked:server_not_ready"
  return 1
}

ensure_base_server

stage "alloc_generation:start arms=4 trajectories=80 sequential_bank=true"
PYTHONPATH="$project_root/src:$project_root/scripts" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/generate_tau_alloc_memory_banks_v2.py" \
  --manifest "$manifest" --output "$experiment/memory" \
  --model qwen35-tau --max-parallel-chains "$gen_max_parallel_chains" \
  --max-tokens 4096 --seed 20260822 --budget-fraction 0.4 \
  2>&1 | tee -a "$runtime/alloc_generation.log"
stage "alloc_generation:complete summary=$experiment/memory/summary.json"

stage "dev_matrix:start levels=5 tasks=23 runs=115"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/run_tau_alloc_dev_matrix_v2.py" \
  --manifest "$manifest" --memory-root "$experiment/memory/banks" \
  --bank-summary "$experiment/memory/summary.json" \
  --output "$experiment/dev_matrix" \
  --max-parallel-runs "$dev_max_parallel_runs" \
  --max-concurrency "$dev_max_concurrency" \
  --timeout 2400 --seed 20260822 --memory-top-k 3 \
  2>&1 | tee -a "$runtime/dev_matrix.log"
stage "dev_matrix:complete summary=$experiment/dev_matrix/summary.json"
stage "pipeline:complete final_test_used=false"
