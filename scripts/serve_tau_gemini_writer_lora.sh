#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
model_path="${TAU_MEMORY_WRITER_MODEL_PATH:-$project_root/training/tau_memory_writer_gemini_v1_merged_retry1}"
served_name="${TAU_MEMORY_WRITER_SERVED_NAME:-qwen35-tau-writer-gemini-v1}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tau-gemini-writer-lora-triton-cache}"

if [[ ! -s "$model_path/model.safetensors.index.json" ]]; then
  echo "Merged memory-writer model not found: $model_path" >&2
  exit 1
fi

exec "$project_root/.venv/bin/vllm" serve "$model_path" \
  --served-model-name "$served_name" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --language-model-only
