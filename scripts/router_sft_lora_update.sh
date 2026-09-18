#!/usr/bin/env bash
# Retrain the task-agent LoRA adapter from the FULL accumulated SFT pool
# (DESIGN.md section 15), merge it into a standalone checkpoint, and reload
# BOTH deterministic replicas onto the merged weights.
#
# Why merge-and-reload-both, not just load the adapter as an extra vLLM
# LoRA module on one replica: det_server_a/b are treated as INTERCHANGEABLE
# for GRPO comparisons (candidates are split across them by k-parity purely
# for wall-clock parallelism, see DESIGN.md section 11/13). If only one
# replica served the fine-tuned agent, which replica a candidate happens to
# land on would silently decide whether its self-eval sees the trained agent
# at all -- exactly the "cross-replica non-equivalence" failure mode section
# 13 spent a whole investigation on. Merging into one checkpoint and pointing
# both replicas at it keeps them identical, at the cost of a CPU-only merge
# step (no GPU needed, ~lightweight relative to the GPU time already spent).
#
# Always retrains the LoRA from the BASE model on the whole pool, not
# incrementally on top of the previous adapter -- simpler, avoids compounding
# drift from repeated small updates. Pool sizes stay small (multiples of 8).
set -euo pipefail

pool_path="${1:?usage: router_sft_lora_update.sh <pool.jsonl> <work_dir>}"
work_dir="${2:?usage: router_sft_lora_update.sh <pool.jsonl> <work_dir>}"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
base_model_path="/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
adapter_dir="$work_dir/lora_adapter"
merged_dir="$work_dir/merged"

# All machine-specific (tmux session names, ports, GPU indices) -- override
# these when det_server_a/b aren't at this machine's default ports/GPUs (e.g.
# a shared box where 8000/8001 are already taken by someone else).
server_a_name="${SERVER_A_NAME:-det_server_a}"
server_b_name="${SERVER_B_NAME:-det_server_b}"
server_a_port="${SERVER_A_PORT:-8000}"
server_b_port="${SERVER_B_PORT:-8001}"
server_a_gpus="${SERVER_A_GPUS:-0,1}"
server_b_gpus="${SERVER_B_GPUS:-2,3}"
train_gpus="${TRAIN_GPUS:-$server_b_gpus}"
train_nproc="${TRAIN_NPROC:-2}"

for name in "$server_a_name" "$server_b_name"; do
  if ! tmux has-session -t "$name" 2>/dev/null; then
    echo "$name tmux session not found -- refusing to guess GPU state, aborting" >&2
    exit 1
  fi
done

echo "[router_sft_lora_update] pausing $server_b_name to free GPUs $train_gpus for training"
tmux kill-session -t "$server_b_name"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export USE_HF=1
export CUDA_VISIBLE_DEVICES="$train_gpus"
export NPROC_PER_NODE="$train_nproc"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/router-sft-lora-triton-cache}"

echo "[router_sft_lora_update] training LoRA on $(wc -l < "$pool_path") pooled examples"
rm -rf "$adapter_dir"
"$project_root/.train-venv/bin/swift" sft \
  --model Qwen/Qwen3.5-35B-A3B \
  --tuner_type lora \
  --dataset "$pool_path" \
  --load_from_cache_file false \
  --add_non_thinking_prefix true \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --experts_impl grouped_mm \
  --router_aux_loss_coef 1e-3 \
  --gradient_accumulation_steps 1 \
  --output_dir "$adapter_dir" \
  --save_total_limit 1 \
  --logging_steps 1 \
  --max_length 4096 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --split_dataset_ratio 0 \
  --report_to none \
  --deepspeed zero3

checkpoint_dir=$(find "$adapter_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [[ -z "$checkpoint_dir" ]]; then
  echo "[router_sft_lora_update] training produced no checkpoint -- det_server_b stays down, fix manually" >&2
  exit 1
fi

echo "[router_sft_lora_update] pausing $server_a_name; merging LoRA into a standalone checkpoint (CPU, no GPU needed)"
tmux kill-session -t "$server_a_name"
rm -rf "$merged_dir"
"$project_root/.venv/bin/python" "$project_root/scripts/merge_qwen_lora.py" \
  "$base_model_path" "$checkpoint_dir" "$merged_dir"

echo "[router_sft_lora_update] reloading $server_a_name (GPUs $server_a_gpus, port $server_a_port) and $server_b_name (GPUs $server_b_gpus, port $server_b_port) on the merged checkpoint"
tmux new-session -d -s "$server_a_name" \
  "CUDA_VISIBLE_DEVICES=$server_a_gpus MODEL_PATH=$merged_dir PORT=$server_a_port TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 TRITON_CACHE_DIR=/tmp/appworld-det-server-cache-a \
   $project_root/scripts/serve_appworld_deterministic.sh 2>&1 | tee -a $project_root/appworld_experiment/${server_a_name}.log"
tmux new-session -d -s "$server_b_name" \
  "CUDA_VISIBLE_DEVICES=$server_b_gpus MODEL_PATH=$merged_dir PORT=$server_b_port TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 TRITON_CACHE_DIR=/tmp/appworld-det-server-cache-b \
   $project_root/scripts/serve_appworld_deterministic.sh 2>&1 | tee -a $project_root/appworld_experiment/${server_b_name}.log"

echo "[router_sft_lora_update] waiting for both replicas to come back up..."
for port in "$server_a_port" "$server_b_port"; do
  for _ in $(seq 1 60); do
    if curl -s -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "[router_sft_lora_update] port $port ready"
      break
    fi
    sleep 10
  done
done
echo "[router_sft_lora_update] done. Both replicas now serve the merged, fine-tuned checkpoint as 'qwen35-tau' -- no caller-side changes needed."
