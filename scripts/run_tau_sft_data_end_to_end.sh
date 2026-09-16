#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$project_root/tau_experiment/tau_sft_data_writer_v1/split_manifest.json"
teacher_dir="$project_root/tau_experiment/tau_sft_data_writer_v1/codex_teacher_full"
teacher_replay_dir="$project_root/tau_experiment/tau_sft_data_writer_v1/guided_teacher_full"
writer_data_dir="$project_root/training/tau_sft_data_writer_v1"
writer_train_dir="$project_root/training/tau_sft_data_writer_v1_lora"
writer_export_dir="$project_root/training/tau_sft_data_writer_v1_export"
writer_merged_dir="$project_root/training/tau_sft_data_writer_v1_merged"
writer_predictions="$project_root/tau_experiment/tau_sft_data_writer_v1/writer_predictions"
generation_replay_dir="$project_root/tau_experiment/tau_sft_data_writer_v1/guided_generation"
agent_jsonl="$project_root/training/tau_agent_from_sftdata_writer_v1.jsonl"
agent_parquet="$project_root/training/tau_agent_from_sftdata_writer_v1.parquet"
agent_train_dir="$project_root/training/tau_agent_from_sftdata_writer_v1_lora"
agent_export_dir="$project_root/training/tau_agent_from_sftdata_writer_v1_export"
final_eval="$project_root/tau_experiment/tau_agent_sftdata_four_arm_v1"
model_path="/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
main_log="$project_root/runtime_logs/tau_sft_data_end_to_end.log"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$main_log"
}

wait_server() {
  local expected="$1"
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$expected"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_server() {
  local session="$1"
  tmux kill-session -t "$session" 2>/dev/null || true
  for _ in $(seq 1 120); do
    if ! curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_guided_domains() {
  local candidates="$1"
  local output="$2"
  local section="$3"
  local tag="$4"
  local pids=()
  for domain in airline retail telecom; do
    PYTHONPATH="$project_root/src" \
      "$project_root/third_party/tau2-bench/.venv/bin/python" -u \
      "$project_root/scripts/run_tau_sft_data_guided_replay.py" \
      --manifest "$manifest" \
      --candidates "$candidates" \
      --output "$output" \
      --domain "$domain" \
      --split-section "$section" \
      --run-tag "$tag" \
      --max-concurrency 2 \
      > "$project_root/runtime_logs/tau_sftdata_guided_${tag}_${domain}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" != 0 ]]; then
    return 1
  fi
}

stage "waiting:codex_teacher"
while [[ ! -s "$teacher_dir/summary.json" ]]; do
  if ! tmux has-session -t tau_sftdata_codex_teacher 2>/dev/null; then
    stage "codex_teacher:session_missing_retry"
    PYTHONPATH="$project_root/src" \
      "$project_root/third_party/tau2-bench/.venv/bin/python" -u \
      "$project_root/scripts/run_codex_tau_sft_data_teacher.py" \
      --manifest "$manifest" --output "$teacher_dir" --workers 3 \
      --model gpt-5.6-sol --reasoning-effort high \
      >> "$project_root/runtime_logs/tau_sftdata_codex_teacher.log" 2>&1
  fi
  sleep 30
done

teacher_errors="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["errors"])' "$teacher_dir/summary.json")"
if [[ "$teacher_errors" != 0 ]]; then
  stage "codex_teacher:retry errors=$teacher_errors"
  PYTHONPATH="$project_root/src" \
    "$project_root/third_party/tau2-bench/.venv/bin/python" -u \
    "$project_root/scripts/run_codex_tau_sft_data_teacher.py" \
    --manifest "$manifest" --output "$teacher_dir" --workers 3 \
    --model gpt-5.6-sol --reasoning-effort high \
    >> "$project_root/runtime_logs/tau_sftdata_codex_teacher.log" 2>&1
fi
teacher_errors="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["errors"])' "$teacher_dir/summary.json")"
if [[ "$teacher_errors" != 0 ]]; then
  stage "blocked:codex_teacher errors=$teacher_errors"
  exit 1
fi
stage "codex_teacher:complete"

if [[ -s "$writer_data_dir/train.parquet" && -s "$writer_data_dir/summary.json" ]]; then
  stage "teacher_guided_replay:skip_existing_writer_dataset"
else
  if ! curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen35-tau; then
    tmux new-session -d -s tau_sftdata_base_server \
      "cd '$project_root' && ./scripts/serve_tau_base.sh > runtime_logs/tau_sftdata_base_server.log 2>&1"
    wait_server qwen35-tau
  fi
  stage "teacher_guided_replay:start"
  run_guided_domains "$teacher_dir" "$teacher_replay_dir" teacher_writer teacher_v1
  stage "teacher_guided_replay:complete"
fi

if [[ -s "$writer_data_dir/train.parquet" && -s "$writer_data_dir/summary.json" ]]; then
  stage "writer_dataset:existing_complete"
else
  if [[ -e "$writer_data_dir" ]]; then
    stage "blocked:writer_data_dir_incomplete $writer_data_dir"
    exit 1
  fi
  PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
    "$project_root/scripts/prepare_tau_sft_data_writer_sft.py" \
    --manifest "$manifest" \
    --candidates "$teacher_dir" \
    --replay-tag teacher_v1 \
    --output "$writer_data_dir" \
    --model "$model_path" \
    --max-length 45056 \
    2>&1 | tee -a "$main_log"
  stage "writer_dataset:complete"
fi

stop_server tau_codex56_replay_server || true
stop_server tau_sftdata_base_server || true
if [[ -s "$writer_export_dir/lora_adapter/adapter_model.safetensors" ]]; then
  stage "writer_training_and_export:existing_complete"
else
  writer_checkpoint="$(find "$writer_train_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1)"
  if [[ -n "$writer_checkpoint" ]]; then
    stage "writer_training:existing_checkpoint $writer_checkpoint"
  else
    if [[ -e "$writer_train_dir" || -e "$writer_export_dir" ]]; then
      stage "blocked:writer_training_output_incomplete"
      exit 1
    fi
    stage "writer_training:start"
    EPOCHS=1 LEARNING_RATE=1e-5 MAX_LENGTH=45056 \
      "$project_root/scripts/train_tau_sft_data_writer_verl.sh" \
      "$writer_data_dir/train.parquet" "$writer_train_dir" \
      2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_writer_train.log"
    stage "writer_training:complete"
    writer_checkpoint="$(find "$writer_train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
  fi
  if [[ -z "$writer_checkpoint" ]]; then
    stage "blocked:no_writer_checkpoint"
    exit 1
  fi
  source "$project_root/scripts/activate_verl.sh"
  set +e
  python -m verl.model_merger merge \
    --backend fsdp --local_dir "$writer_checkpoint" \
    --target_dir "$writer_export_dir" --use_cpu_initialization \
    2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_writer_export.log"
  writer_export_status=${PIPESTATUS[0]}
  set -e
  test -s "$writer_export_dir/lora_adapter/adapter_model.safetensors"
  stage "writer_export:complete status=$writer_export_status"
fi

if [[ -s "$writer_merged_dir/model.safetensors.index.json" ]]; then
  stage "writer_merge:existing_complete"
else
  if [[ -e "$writer_merged_dir" ]]; then
    stage "blocked:writer_merged_dir_incomplete $writer_merged_dir"
    exit 1
  fi
  stage "writer_merge:start"
  "$project_root/.verl-venv/bin/python" "$project_root/scripts/merge_qwen_lora.py" \
    "$model_path" "$writer_export_dir/lora_adapter" "$writer_merged_dir" \
    2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_writer_merge.log"
  test -s "$writer_merged_dir/model.safetensors.index.json"
  stage "writer_merge:complete"
fi

tmux new-session -d -s tau_sftdata_writer_server \
  "cd '$project_root' && TAU_SFT_DATA_WRITER_MERGED_PATH='$writer_merged_dir' ./scripts/serve_tau_sft_data_writer_merged.sh > runtime_logs/tau_sftdata_writer_server.log 2>&1"
wait_server qwen35-tau-sftdata-writer
stage "writer_inference:start"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/run_tau_sft_data_writer_inference.py" \
  --inputs "$project_root/tau_experiment/tau_sft_data_writer_v1/writer_generation_inputs.jsonl" \
  --output "$writer_predictions" \
  --model qwen35-tau-sftdata-writer \
  --max-tokens 16384 --max-parallel 3 \
  2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_writer_inference.log"
stage "writer_inference:complete"

stop_server tau_sftdata_writer_server
tmux new-session -d -s tau_sftdata_base_server \
  "cd '$project_root' && ./scripts/serve_tau_base.sh > runtime_logs/tau_sftdata_base_server.log 2>&1"
wait_server qwen35-tau
stage "writer_generation_replay:start"
run_guided_domains "$writer_predictions" "$generation_replay_dir" writer_generation generation_v1
stage "writer_generation_replay:complete"

PYTHONPATH="$project_root/third_party/tau2-bench/src" \
  "$project_root/third_party/tau2-bench/.venv/bin/python" \
  "$project_root/scripts/collect_tau_agent_sft_from_writer_replays.py" \
  --manifest "$manifest" --replay-tag generation_v1 --output "$agent_jsonl" \
  2>&1 | tee -a "$main_log"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
  "$project_root/scripts/prepare_tau_sft_data.py" \
  "$agent_jsonl" "$agent_parquet" --model "$model_path" --max-length 45056 \
  2>&1 | tee -a "$main_log"
stage "agent_dataset:complete"

stop_server tau_sftdata_base_server
if [[ -e "$agent_train_dir" || -e "$agent_export_dir" ]]; then
  stage "blocked:agent_training_output_exists"
  exit 1
fi
stage "agent_training:start"
EPOCHS=1 LEARNING_RATE=1e-5 MAX_LENGTH=45056 \
  "$project_root/scripts/train_tau_verl.sh" "$agent_parquet" "$agent_train_dir" \
  2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_agent_train.log"
stage "agent_training:complete"

agent_checkpoint="$(find "$agent_train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$agent_checkpoint" ]]; then
  stage "blocked:no_agent_checkpoint"
  exit 1
fi
source "$project_root/scripts/activate_verl.sh"
set +e
python -m verl.model_merger merge \
  --backend fsdp --local_dir "$agent_checkpoint" \
  --target_dir "$agent_export_dir" --use_cpu_initialization \
  2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_agent_export.log"
agent_export_status=${PIPESTATUS[0]}
set -e
test -s "$agent_export_dir/lora_adapter/adapter_model.safetensors"
stage "agent_export:complete status=$agent_export_status"

tmux new-session -d -s tau_sftdata_agent_server \
  "cd '$project_root' && TAU_AGENT_SFT_LORA_PATH='$agent_export_dir/lora_adapter' ./scripts/serve_tau_agent_sft_lora.sh > runtime_logs/tau_sftdata_agent_server.log 2>&1"
wait_server qwen35-tau-agent-sft
stage "four_arm_eval:start"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/run_tau_agent_sft_four_arm_eval.py" \
  --output "$final_eval" \
  --memory-dir "$project_root/tau_experiment/codex56_writer_sft_cumulative_20260818" \
  --max-parallel-runs 3 --max-concurrency 2 --run-tag v1 \
  2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_four_arm_eval.log"
stage "four_arm_eval:complete summary=$final_eval/summary.json"
stop_server tau_sftdata_agent_server || true
stage "pipeline:complete"
