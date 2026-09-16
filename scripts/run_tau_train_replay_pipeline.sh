#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment_dir="$project_root/tau_experiment/v1"
retention_dir="$experiment_dir/retention"
dataset_path="$experiment_dir/tau_train.parquet"
train_dir="$project_root/training/tau_qwen35_verl_lora_v1"
export_dir="$project_root/training/tau_qwen35_verl_export_v1"
base_model="/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"

mkdir -p "$experiment_dir" "$project_root/runtime_logs/tau_pipeline"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$experiment_dir/stages.log"
}

stage "retention:start"
PYTHONPATH="$project_root/src" \
  "$project_root/third_party/tau2-bench/.venv/bin/python" \
  "$project_root/scripts/run_tau_retention.py" \
  --output "$retention_dir" \
  --results-root "$project_root/third_party/tau2-bench/data/simulations" \
  --seed 300 \
  --max-tokens 4096
stage "retention:complete"

stage "dataset:start"
"$project_root/.verl-venv/bin/python" \
  "$project_root/scripts/prepare_tau_sft_data.py" \
  "$retention_dir/sft_examples.jsonl" \
  "$dataset_path" \
  --model "$base_model" \
  --max-length 45056
stage "dataset:complete"

stage "base_server:stop"
tmux kill-session -t tau_qwen35_base 2>/dev/null || true
for _ in $(seq 1 60); do
  if ! curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "Base server did not stop" >&2
  exit 1
fi

stage "training:start"
MAX_LENGTH=45056 EPOCHS=1 LEARNING_RATE=1e-5 \
  "$project_root/scripts/train_tau_verl.sh" "$dataset_path" "$train_dir"
stage "training:complete"

checkpoint_dir="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$checkpoint_dir" ]]; then
  echo "No VERL checkpoint found" >&2
  exit 1
fi

stage "adapter_export:start"
set +e
source "$project_root/scripts/activate_verl.sh"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$checkpoint_dir" \
  --target_dir "$export_dir" \
  --use_cpu_initialization
export_status=$?
set -e
if [[ ! -f "$export_dir/lora_adapter/adapter_model.safetensors" ]]; then
  echo "Adapter export failed with status $export_status" >&2
  exit 1
fi
stage "adapter_export:complete"

stage "lora_server:start"
tmux kill-session -t tau_qwen35_sft_lora 2>/dev/null || true
tmux new-session -d -s tau_qwen35_sft_lora \
  "cd '$project_root' && TAU_LORA_PATH='$export_dir/lora_adapter' ./scripts/serve_tau_base_with_lora.sh > runtime_logs/tau_pipeline/lora_server.log 2>&1"
server_ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null \
    | grep -q 'qwen35-tau-sft'; then
    server_ready=true
    break
  fi
  if ! tmux has-session -t tau_qwen35_sft_lora 2>/dev/null; then
    echo "LoRA server exited during startup" >&2
    tail -n 120 "$project_root/runtime_logs/tau_pipeline/lora_server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$server_ready" != true ]]; then
  echo "LoRA server did not become ready" >&2
  exit 1
fi

curl -fsS --max-time 120 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer EMPTY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-tau-sft","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  > "$experiment_dir/lora_smoke_response.json"
stage "lora_server:ready"

declare -A totals=( [airline]=20 [retail]=40 [telecom]=40 )
for domain in airline retail telecom; do
  session="tau_replay_${domain}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$project_root' && TAU2_AGENT_LLM='openai/qwen35-tau-sft' TAU2_AGENT_MEMORY_PATH='$retention_dir/memory_${domain}.json' ./scripts/run_tau_bench.sh '$domain' 'qwen35_tau_sft_memory_${domain}_test_v1' --task-split-name test --num-trials 1 --max-concurrency 4 --max-steps 200 --timeout 1200 --max-retries 2 --retry-delay 2 --seed 300 --verbose-logs --llm-log-mode latest --auto-resume --log-level INFO > 'runtime_logs/tau_pipeline/replay_${domain}.log' 2>&1"
done
stage "replay:start"

while true; do
  all_done=true
  for domain in airline retail telecom; do
    result="$project_root/third_party/tau2-bench/data/simulations/qwen35_tau_sft_memory_${domain}_test_v1/results.json"
    completed=0
    if [[ -f "$result" ]]; then
      completed="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["simulations"]))' "$result")"
    fi
    printf '%s %s=%s/%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$domain" "$completed" "${totals[$domain]}" | tee -a "$experiment_dir/replay_progress.log"
    if (( completed < totals[$domain] )); then
      all_done=false
      if ! tmux has-session -t "tau_replay_${domain}" 2>/dev/null; then
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
python "$project_root/scripts/summarize_tau_replay.py" \
  > "$project_root/runtime_logs/tau_pipeline/replay_comparison.log"
stage "pipeline:complete"
