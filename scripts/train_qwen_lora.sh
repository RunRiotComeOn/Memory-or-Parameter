#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export USE_HF=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/trajectory-memory-training-triton-cache}"

dataset_path="${1:-/nas04/yixuh/memory/training/replay_sft.jsonl}"
output_dir="${2:-/nas04/yixuh/memory/training/qwen35_replay_lora}"

exec "$(dirname "$0")/../.train-venv/bin/swift" sft \
  --model Qwen/Qwen3.5-35B-A3B \
  --tuner_type lora \
  --dataset "$dataset_path" \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --experts_impl grouped_mm \
  --router_aux_loss_coef 1e-3 \
  --gradient_accumulation_steps 1 \
  --output_dir "$output_dir" \
  --save_steps 24 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --max_length 512 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --split_dataset_ratio 0 \
  --report_to none \
  --deepspeed zero3
