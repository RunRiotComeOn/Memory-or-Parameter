#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_experiment="$project_root/tau_experiment/v2"
experiment_dir="$project_root/tau_experiment/v3"
dataset_path="$source_experiment/tau_train.parquet"
retention_dir="$project_root/tau_experiment/v1/retention"
train_dir="$project_root/training/tau_qwen35_verl_lora_v3_3epoch"
export_dir="$project_root/training/tau_qwen35_verl_export_v3_3epoch"
log_dir="$project_root/runtime_logs/tau_pipeline_v3"

mkdir -p "$experiment_dir" "$log_dir"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$experiment_dir/stages.log"
}

stage "waiting_for:v2_pipeline_complete"
while ! grep -qxE '.* pipeline:complete' "$source_experiment/stages.log" 2>/dev/null; do
  if ! tmux has-session -t tau_v2_resume 2>/dev/null; then
    echo "v2 replay controller exited before pipeline:complete" >&2
    exit 1
  fi
  sleep 60
done
stage "waiting_for:v2_pipeline_complete:complete"

if [[ ! -f "$dataset_path" ]]; then
  echo "Missing token-exact v2 dataset: $dataset_path" >&2
  exit 1
fi
if [[ -e "$train_dir" || -e "$export_dir" ]]; then
  echo "Refusing to overwrite an existing tau v3 training/export directory" >&2
  exit 1
fi

stage "v2_lora_server:stop"
tmux kill-session -t tau_qwen35_sft_lora 2>/dev/null || true
for _ in $(seq 1 90); do
  if ! curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "v2 LoRA server did not stop" >&2
  exit 1
fi

# Start from the same untouched base model.  EPOCHS is the only training
# difference from v2, making the 1-epoch and 3-epoch replays comparable.
stage "training:start epochs=3"
MAX_LENGTH=45056 EPOCHS=3 LEARNING_RATE=1e-5 \
  "$project_root/scripts/train_tau_verl.sh" "$dataset_path" "$train_dir"
stage "training:complete epochs=3"

checkpoint_dir="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$checkpoint_dir" ]]; then
  echo "No VERL checkpoint found" >&2
  exit 1
fi

stage "adapter_export:start"
source "$project_root/scripts/activate_verl.sh"
set +e
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$checkpoint_dir" \
  --target_dir "$export_dir" \
  --use_cpu_initialization
export_status=$?
set -e
if [[ ! -s "$export_dir/lora_adapter/adapter_model.safetensors" ]]; then
  echo "Adapter export failed with status $export_status and produced no adapter" >&2
  exit 1
fi
stage "adapter_export:complete"

stage "lora_server:start"
tmux new-session -d -s tau_qwen35_sft_lora \
  "cd '$project_root' && TAU_LORA_PATH='$export_dir/lora_adapter' ./scripts/serve_tau_base_with_lora.sh > runtime_logs/tau_pipeline_v3/lora_server.log 2>&1"
server_ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null \
    | grep -q 'qwen35-tau-sft'; then
    server_ready=true
    break
  fi
  if ! tmux has-session -t tau_qwen35_sft_lora 2>/dev/null; then
    echo "v3 LoRA server exited during startup" >&2
    tail -n 120 "$log_dir/lora_server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$server_ready" != true ]]; then
  echo "v3 LoRA server did not become ready" >&2
  exit 1
fi
stage "lora_server:ready"

declare -A totals=( [airline]=20 [retail]=40 [telecom]=40 )
for domain in airline retail telecom; do
  session="tau_replay_v3_${domain}"
  result="$project_root/third_party/tau2-bench/data/simulations/qwen35_tau_sft_memory_${domain}_test_v3/results.json"
  if [[ -e "$result" ]]; then
    echo "Refusing to overwrite existing v3 replay: $result" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" \
    "cd '$project_root' && TAU2_AGENT_LLM='openai/qwen35-tau-sft' TAU2_AGENT_MEMORY_PATH='$retention_dir/memory_${domain}.json' ./scripts/run_tau_bench.sh '$domain' 'qwen35_tau_sft_memory_${domain}_test_v3' --task-split-name test --num-trials 1 --max-concurrency 4 --max-steps 200 --timeout 1200 --max-retries 2 --retry-delay 2 --seed 300 --verbose-logs --llm-log-mode latest --auto-resume --log-level INFO > 'runtime_logs/tau_pipeline_v3/replay_${domain}.log' 2>&1"
done
stage "replay:start"

while true; do
  all_done=true
  for domain in airline retail telecom; do
    result="$project_root/third_party/tau2-bench/data/simulations/qwen35_tau_sft_memory_${domain}_test_v3/results.json"
    completed=0
    scored=0
    infrastructure_errors=0
    if [[ -f "$result" ]]; then
      read -r completed scored infrastructure_errors < <(python -c 'import json,sys; s=json.load(open(sys.argv[1]))["simulations"]; print(len(s),sum(isinstance(x.get("reward_info"),dict) and x["reward_info"].get("reward") is not None for x in s),sum(x.get("termination_reason")=="infrastructure_error" for x in s))' "$result")
    fi
    printf '%s %s=%s/%s scored=%s infrastructure_errors=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$domain" "$completed" "${totals[$domain]}" "$scored" "$infrastructure_errors" | tee -a "$experiment_dir/replay_progress.log"
    if (( infrastructure_errors > 0 )); then
      echo "$domain replay has infrastructure errors" >&2
      exit 1
    fi
    if (( completed < totals[$domain] )); then
      all_done=false
      if ! tmux has-session -t "tau_replay_v3_${domain}" 2>/dev/null; then
        echo "$domain replay exited before completion" >&2
        exit 1
      fi
    elif (( scored != totals[$domain] )); then
      echo "$domain replay completed with missing scores" >&2
      exit 1
    fi
  done
  if [[ "$all_done" == true ]]; then
    break
  fi
  sleep 60
done

stage "replay:complete"
python "$project_root/scripts/summarize_tau_replay.py" --replay-version v3 \
  > "$log_dir/replay_comparison.log"
stage "pipeline:complete"
