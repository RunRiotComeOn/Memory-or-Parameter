#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="$project_root/tau_experiment/writer_rubric_e2e_v1"
runtime="$project_root/runtime_logs/tau_rubric_e2e"
watch_log="$runtime/watchdog.log"
pipeline_session="tau_rubric_e2e_v1"
server_session="tau_rubric_e2e_server"
summary="$experiment/dev_matrix/summary.json"
consecutive_server_misses=0

mkdir -p "$runtime"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$watch_log"
}

progress() {
  "$project_root/.venv/bin/python" - "$experiment/dev_matrix/run_manifest.json" \
    "$project_root/third_party/tau2-bench/data/simulations" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
simulations_root = Path(sys.argv[2])
if not manifest_path.exists():
    print("progress=manifest_missing")
    raise SystemExit

runs = json.loads(manifest_path.read_text(encoding="utf-8"))
valid = 0
infra = 0
complete_runs = 0
total = 0
for run in runs:
    wanted = set(map(str, run["task_ids"]))
    total += len(wanted)
    result_path = simulations_root / run["save_name"] / "results.json"
    rewards = {}
    if result_path.exists():
        try:
            simulations = json.loads(result_path.read_text(encoding="utf-8")).get("simulations", [])
        except (OSError, json.JSONDecodeError):
            simulations = []
        for simulation in simulations:
            task_id = str(simulation.get("task_id"))
            if task_id not in wanted:
                continue
            if simulation.get("termination_reason") == "infrastructure_error":
                infra += 1
                continue
            reward_info = simulation.get("reward_info")
            reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
            if isinstance(reward, (int, float)):
                rewards[task_id] = float(reward)
    valid += len(rewards)
    complete_runs += len(rewards) == len(wanted)
print(f"progress={valid}/{total} complete_domain_runs={complete_runs}/{len(runs)} infra={infra}")
PY
}

launch_pipeline() {
  tmux kill-session -t "$pipeline_session" 2>/dev/null || true
  tmux new-session -d -s "$pipeline_session" \
    "cd '$project_root' && env CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 TAU_TENSOR_PARALLEL_SIZE=2 TAU_MAX_MODEL_LEN=32768 TAU_GPU_MEMORY_UTILIZATION=0.95 TAU_RUBRIC_E2E_DEV_MAX_PARALLEL_RUNS=10 TAU_RUBRIC_E2E_DEV_MAX_CONCURRENCY=2 TAU_RUBRIC_E2E_RESUME_FROM_TRAINING=1 bash scripts/run_tau_rubric_e2e_pipeline.sh >> runtime_logs/tau_rubric_e2e/resume_dev.log 2>&1"
  log "action=pipeline_restarted"
}

log "watchdog=start"
while true; do
  log "$(progress)"
  if [[ -s "$summary" ]]; then
    log "watchdog=complete summary=$summary"
    exit 0
  fi

  pipeline_alive=0
  server_healthy=0
  tmux has-session -t "$pipeline_session" 2>/dev/null && pipeline_alive=1
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && server_healthy=1

  if (( server_healthy == 1 )); then
    consecutive_server_misses=0
  else
    consecutive_server_misses=$((consecutive_server_misses + 1))
  fi
  log "health=pipeline_${pipeline_alive}_server_${server_healthy} server_misses=$consecutive_server_misses"

  if (( pipeline_alive == 0 || consecutive_server_misses >= 3 )); then
    tmux kill-session -t "$server_session" 2>/dev/null || true
    launch_pipeline
    consecutive_server_misses=0
  fi
  sleep 60
done
