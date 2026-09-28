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
# Single GPU, QLoRA (4-bit bnb quantized base): a real run against a
# 21-example ALFWorld pool OOM'd FIVE separate ways on plain/offloaded zero3
# across 2 AND 4 GPUs (see git history of this file for the numbers) --
# every attempt showed ~46-47GiB/GPU in use no matter the GPU count, offload
# target, or --experts_impl, meaning deepspeed zero3 was never actually
# shrinking this MoE model's per-GPU peak the way it would for a dense one.
# Quantizing the frozen base to int4 shrinks it to ~17.5GiB -- comfortably
# inside ONE 48GiB GPU with LoRA activations and optimizer state on top, no
# cross-GPU sharding involved at all, sidestepping whatever zero3+MoE
# interaction was actually at fault. This also means only ONE replica's GPU
# is needed, so the OTHER replica can keep serving through the whole
# training+merge window -- override TRAIN_GPUS/TRAIN_NPROC/TRAIN_QUANT_BITS=0
# to go back to a multi-GPU bf16 run if a future model/box handles zero3
# fine.
if [[ -n "${TRAIN_GPUS:-}" ]]; then
  train_gpus="$TRAIN_GPUS"
else
  train_gpus="${server_b_gpus%%,*}"  # first GPU of server_b's pair, e.g. "2"
fi
train_nproc="${TRAIN_NPROC:-1}"
train_quant_bits="${TRAIN_QUANT_BITS:-4}"

# Only pause the replica whose GPU(s) training will actually use -- QLoRA on
# one GPU no longer needs both down. Falls back to pausing both if
# TRAIN_GPUS was overridden to span both servers' ranges.
gpus_overlap() {
  local IFS=,
  local -a set_a=($1) set_b=($2)
  local a b
  for a in "${set_a[@]}"; do
    for b in "${set_b[@]}"; do
      [[ "$a" == "$b" ]] && return 0
    done
  done
  return 1
}
servers_to_pause=()
gpus_overlap "$train_gpus" "$server_a_gpus" && servers_to_pause+=("$server_a_name")
gpus_overlap "$train_gpus" "$server_b_gpus" && servers_to_pause+=("$server_b_name")
for name in "${servers_to_pause[@]}"; do
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[router_sft_lora_update] pausing $name to free its GPU(s) for training"
    tmux kill-session -t "$name"
  else
    echo "[router_sft_lora_update] $name already not running, nothing to pause"
  fi
done

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export USE_HF=1
export CUDA_VISIBLE_DEVICES="$train_gpus"
export NPROC_PER_NODE="$train_nproc"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/router-sft-lora-triton-cache}"

echo "[router_sft_lora_update] training LoRA on $(wc -l < "$pool_path") pooled examples"
rm -rf "$adapter_dir"
# QLoRA (4-bit bnb-quantized frozen base), single GPU, no deepspeed: a real
# 21-example ALFWorld pool OOM'd FIVE separate ways under plain/CPU-offloaded
# deepspeed zero3 across both 2 and 4 GPUs, with gradient_checkpointing,
# max_length trimmed to 3072, and both grouped_mm and eager expert impls --
# every attempt showed ~46-47GiB/GPU in use regardless of GPU count, offload
# target, or expert implementation, meaning zero3 was never actually
# shrinking this MoE model's per-GPU peak the way it would for a dense one
# (all five configs are preserved in this file's git history along with the
# specific numbers, in case zero3 is worth revisiting for a future model).
# Quantizing the frozen base to int4 shrinks it to ~17.5GiB, comfortably
# inside ONE 48GiB GPU with LoRA activations/optimizer state on top -- no
# cross-GPU sharding at all, sidestepping whatever the zero3+MoE interaction
# was. `--quant_bits 4` uses bitsandbytes nf4 (already installed, 0.49.1).
quant_args=()
if [[ "$train_quant_bits" != "0" ]]; then
  quant_args=(--quant_method bnb --quant_bits "$train_quant_bits")
fi
"$project_root/.train-venv/bin/swift" sft \
  --model Qwen/Qwen3.5-35B-A3B \
  --tuner_type lora \
  --dataset "$pool_path" \
  --load_from_cache_file false \
  --add_non_thinking_prefix true \
  --torch_dtype bfloat16 \
  "${quant_args[@]}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --experts_impl eager \
  --router_aux_loss_coef 1e-3 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --output_dir "$adapter_dir" \
  --save_total_limit 1 \
  --logging_steps 1 \
  --max_length 3072 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --split_dataset_ratio 0 \
  --report_to none

checkpoint_dir=$(find "$adapter_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [[ -z "$checkpoint_dir" ]]; then
  echo "[router_sft_lora_update] training produced no checkpoint -- paused replica(s) (${servers_to_pause[*]:-none}) stay down, fix manually" >&2
  exit 1
fi

echo "[router_sft_lora_update] merging LoRA into a standalone checkpoint (CPU, no GPU needed; both replicas already paused)"
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
