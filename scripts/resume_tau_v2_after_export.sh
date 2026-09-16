#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment_dir="$project_root/tau_experiment/v2"
retention_dir="$project_root/tau_experiment/v1/retention"
export_dir="$project_root/training/tau_qwen35_verl_export_v2"
adapter="$export_dir/lora_adapter/adapter_model.safetensors"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$experiment_dir/stages.log"
}

if [[ ! -s "$adapter" ]]; then
  echo "Missing or empty v2 adapter: $adapter" >&2
  exit 1
fi
if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "Port 8000 is already serving another model" >&2
  exit 1
fi

stage "adapter_export:complete"
stage "lora_server:start"
tmux new-session -d -s tau_qwen35_sft_lora \
  "cd '$project_root' && TAU_LORA_PATH='$export_dir/lora_adapter' ./scripts/serve_tau_base_with_lora.sh > runtime_logs/tau_pipeline_v2/lora_server.log 2>&1"

server_ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null \
    | grep -q 'qwen35-tau-sft'; then
    server_ready=true
    break
  fi
  if ! tmux has-session -t tau_qwen35_sft_lora 2>/dev/null; then
    echo "v2 LoRA server exited during startup" >&2
    tail -n 120 "$project_root/runtime_logs/tau_pipeline_v2/lora_server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$server_ready" != true ]]; then
  echo "v2 LoRA server did not become ready" >&2
  exit 1
fi
stage "lora_server:ready"

declare -A totals=( [airline]=20 [retail]=40 [telecom]=40 )
for domain in airline retail telecom; do
  session="tau_replay_v2_${domain}"
  result="$project_root/third_party/tau2-bench/data/simulations/qwen35_tau_sft_memory_${domain}_test_v2/results.json"
  if [[ -e "$result" ]]; then
    echo "Refusing to overwrite existing v2 replay: $result" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" \
    "cd '$project_root' && TAU2_AGENT_LLM='openai/qwen35-tau-sft' TAU2_AGENT_MEMORY_PATH='$retention_dir/memory_${domain}.json' ./scripts/run_tau_bench.sh '$domain' 'qwen35_tau_sft_memory_${domain}_test_v2' --task-split-name test --num-trials 1 --max-concurrency 4 --max-steps 200 --timeout 1200 --max-retries 2 --retry-delay 2 --seed 300 --verbose-logs --llm-log-mode latest --auto-resume --log-level INFO > 'runtime_logs/tau_pipeline_v2/replay_${domain}.log' 2>&1"
done
stage "replay:start"

while true; do
  all_done=true
  for domain in airline retail telecom; do
    result="$project_root/third_party/tau2-bench/data/simulations/qwen35_tau_sft_memory_${domain}_test_v2/results.json"
    completed=0
    infrastructure_errors=0
    if [[ -f "$result" ]]; then
      read -r completed infrastructure_errors < <(python -c 'import json,sys; s=json.load(open(sys.argv[1]))["simulations"]; print(len(s),sum(x.get("termination_reason")=="infrastructure_error" for x in s))' "$result")
    fi
    printf '%s %s=%s/%s infrastructure_errors=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$domain" "$completed" "${totals[$domain]}" "$infrastructure_errors" | tee -a "$experiment_dir/replay_progress.log"
    if (( infrastructure_errors > 0 )); then
      echo "$domain replay has infrastructure errors" >&2
      exit 1
    fi
    if (( completed < totals[$domain] )); then
      all_done=false
      if ! tmux has-session -t "tau_replay_v2_${domain}" 2>/dev/null; then
        echo "$domain replay exited before completion" >&2
        exit 1
      fi
    fi
  done
  if [[ "$all_done" == true ]]; then
    break
  fi
  sleep 60
done

stage "replay:complete"
python "$project_root/scripts/summarize_tau_replay.py" --replay-version v2 \
  > "$project_root/runtime_logs/tau_pipeline_v2/replay_comparison.log"
stage "pipeline:complete"
