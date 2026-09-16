#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="$project_root/tau_experiment/writer_rubric_e2e_v1"
runtime="$project_root/runtime_logs/tau_rubric_e2e"
main_log="$runtime/pipeline.log"
source_split="$project_root/tau_experiment/tau_sft_data_writer_v1/split_manifest.json"
manifest="$experiment/split_manifest.json"
model_path="/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
server_session="tau_rubric_e2e_server"
rubrics=(r0_faithful r1_causal_minimal r2_state_transition r3_robust)
resume_from_training="${TAU_RUBRIC_E2E_RESUME_FROM_TRAINING:-0}"
dev_max_parallel_runs="${TAU_RUBRIC_E2E_DEV_MAX_PARALLEL_RUNS:-5}"
dev_max_concurrency="${TAU_RUBRIC_E2E_DEV_MAX_CONCURRENCY:-2}"
server_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
server_tensor_parallel_size="${TAU_TENSOR_PARALLEL_SIZE:-4}"
server_max_model_len="${TAU_MAX_MODEL_LEN:-65536}"
server_gpu_memory_utilization="${TAU_GPU_MEMORY_UTILIZATION:-0.88}"

mkdir -p "$experiment" "$runtime"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$main_log"
}

wait_server() {
  local wanted="$1"
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$wanted"; then
      return 0
    fi
    sleep 5
  done
  stage "blocked:server_not_ready wanted=$wanted"
  return 1
}

stop_server() {
  tmux kill-session -t "$server_session" 2>/dev/null || true
  local pids
  pids="$(lsof -ti tcp:8000 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
  fi
  for _ in $(seq 1 60); do
    if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  stage "blocked:port_8000_did_not_stop"
  return 1
}

ensure_base_server() {
  if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 'qwen35-tau'; then
    stage "base_server:reuse"
    return 0
  fi
  stop_server
  tmux new-session -d -s "$server_session" \
    "cd '$project_root' && ./scripts/serve_tau_base.sh > '$runtime/base_server.log' 2>&1"
  wait_server qwen35-tau
  stage "base_server:ready"
}

if [[ "$resume_from_training" == "1" ]]; then
  test -s "$experiment/matched_sft/matching_audit.json"
  for rubric_id in "${rubrics[@]}"; do
    test -s "$experiment/matched_sft/$rubric_id/train.parquet"
  done
  stage "pretraining_stages:resume-skip matched_common_dataset=ready"
else
stage "split:start"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
  "$project_root/scripts/prepare_tau_rubric_e2e_split.py" \
  --source-manifest "$source_split" --output "$manifest" --seed 20260822 \
  2>&1 | tee -a "$main_log"

PYTHONPATH="$project_root/src:$project_root/third_party/tau2-bench/src" \
  "$project_root/third_party/tau2-bench/.venv/bin/python" \
  "$project_root/scripts/prepare_tau_writer_rubric_smoke.py" \
  --source-manifest "$experiment/train_source_manifest.json" \
  --output "$experiment/prepared" \
  2>&1 | tee -a "$main_log"
stage "split:complete train=80 dev=23 final_test=100_unused"

ensure_base_server

for rubric_id in "${rubrics[@]}"; do
  stage "sft_candidate_generation:start rubric=$rubric_id"
  PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" -u \
    "$project_root/scripts/run_tau_sft_data_writer_inference.py" \
    --inputs "$experiment/prepared/sft_inputs_${rubric_id}.jsonl" \
    --output "$experiment/sft_candidates/$rubric_id" \
    --model qwen35-tau --max-tokens 16384 --max-parallel 3 \
    --timeout 1200 --seed 20260822 \
    2>&1 | tee -a "$runtime/sft_candidates_${rubric_id}.log"
  stage "sft_candidate_generation:complete rubric=$rubric_id"
done

stage "memory_generation:start"
PYTHONPATH="$project_root/src:$project_root/scripts" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/generate_tau_rubric_memory_banks.py" \
  --manifest "$manifest" --output "$experiment/memory" \
  --model qwen35-tau --max-parallel 3 --max-tokens 2048 --seed 20260822 \
  2>&1 | tee -a "$runtime/memory_generation.log"
stage "memory_generation:complete"

for rubric_id in "${rubrics[@]}"; do
  for domain in airline retail telecom; do
    stage "guided_replay:start rubric=$rubric_id domain=$domain"
    PYTHONPATH="$project_root/src:$project_root/third_party/tau2-bench/src" \
      "$project_root/third_party/tau2-bench/.venv/bin/python" -u \
      "$project_root/scripts/run_tau_sft_data_guided_replay.py" \
      --manifest "$experiment/prepared/smoke_manifest.json" \
      --candidates "$experiment/sft_candidates/$rubric_id" \
      --output "$experiment/guided_replays/$rubric_id" \
      --domain "$domain" --split-section writer_generation \
      --run-tag "rubric_e2e_${rubric_id}_v1" \
      --max-concurrency 2 --seed 20260822 --timeout 2400 \
      2>&1 | tee -a "$runtime/guided_${rubric_id}_${domain}.log"
    stage "guided_replay:complete rubric=$rubric_id domain=$domain"
  done
  mkdir -p "$experiment/collected_sft/$rubric_id"
  PYTHONPATH="$project_root/src:$project_root/third_party/tau2-bench/src" \
    "$project_root/third_party/tau2-bench/.venv/bin/python" \
    "$project_root/scripts/collect_tau_agent_sft_from_writer_replays.py" \
    --manifest "$experiment/prepared/smoke_manifest.json" \
    --replay-tag "rubric_e2e_${rubric_id}_v1" \
    --output "$experiment/collected_sft/$rubric_id/all.jsonl" \
    2>&1 | tee -a "$main_log"
done

stage "sft_matching:start"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
  "$project_root/scripts/match_tau_rubric_agent_sft.py" \
  --input-root "$experiment/collected_sft" \
  --output-root "$experiment/matched_sft" \
  --manifest "$manifest" --minimum-common 12 \
  2>&1 | tee -a "$main_log"

for rubric_id in "${rubrics[@]}"; do
  PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" \
    "$project_root/scripts/prepare_tau_sft_data.py" \
    "$experiment/matched_sft/$rubric_id/matched.jsonl" \
    "$experiment/matched_sft/$rubric_id/train.parquet" \
    --model "$model_path" --max-length 45056 \
    2>&1 | tee -a "$main_log"
done
stage "sft_matching:complete"
fi

stop_server
stage "training:start"
for rubric_id in "${rubrics[@]}"; do
  train_dir="$experiment/training/$rubric_id/verl"
  export_dir="$experiment/training/$rubric_id/export"
  adapter="$export_dir/lora_adapter/adapter_model.safetensors"
  if [[ -s "$adapter" ]]; then
    stage "training:resume-skip rubric=$rubric_id"
    continue
  fi
  checkpoint="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "$checkpoint" ]]; then
    if [[ -e "$train_dir" ]]; then
      stage "blocked:incomplete_training_dir rubric=$rubric_id path=$train_dir"
      exit 1
    fi
    stage "training:fit rubric=$rubric_id epochs=1 lr=1e-5"
    EPOCHS=1 LEARNING_RATE=1e-5 MAX_LENGTH=45056 \
      "$project_root/scripts/train_tau_verl.sh" \
      "$experiment/matched_sft/$rubric_id/train.parquet" "$train_dir" \
      2>&1 | tee -a "$runtime/train_${rubric_id}.log"
    checkpoint="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
  fi
  if [[ -z "$checkpoint" ]]; then
    stage "blocked:no_checkpoint rubric=$rubric_id"
    exit 1
  fi
  stage "training:export rubric=$rubric_id checkpoint=$checkpoint"
  source "$project_root/scripts/activate_verl.sh"
  set +e
  python -m verl.model_merger merge \
    --backend fsdp --local_dir "$checkpoint" --target_dir "$export_dir" \
    --use_cpu_initialization \
    2>&1 | tee -a "$runtime/export_${rubric_id}.log"
  export_status=${PIPESTATUS[0]}
  set -e
  test -s "$adapter"
  stage "training:complete rubric=$rubric_id adapter_ready=true merger_status=$export_status"
done

stop_server
tmux new-session -d -s "$server_session" \
  "cd '$project_root' && CUDA_VISIBLE_DEVICES='$server_cuda_visible_devices' TAU_TENSOR_PARALLEL_SIZE='$server_tensor_parallel_size' TAU_MAX_MODEL_LEN='$server_max_model_len' TAU_GPU_MEMORY_UTILIZATION='$server_gpu_memory_utilization' TAU_RUBRIC_E2E_ROOT='$experiment' ./scripts/serve_tau_rubric_agent_loras.sh > '$runtime/lora_server.log' 2>&1"
wait_server qwen35-agent-r3_robust
stage "lora_server:ready raw_base_plus_four_adapters"

stage "dev_matrix:start cells=25 tasks=23 runs=575"
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/run_tau_rubric_e2e_dev_matrix.py" \
  --manifest "$manifest" --memory-root "$experiment/memory/banks" \
  --output "$experiment/dev_matrix" --max-parallel-runs "$dev_max_parallel_runs" \
  --max-concurrency "$dev_max_concurrency" --timeout 2400 --seed 20260822 --memory-top-k 3 \
  2>&1 | tee -a "$runtime/dev_matrix.log"
stage "dev_matrix:complete summary=$experiment/dev_matrix/summary.json"
stop_server
stage "pipeline:complete final_test_used=false"
