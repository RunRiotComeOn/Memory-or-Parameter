#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
model_path="${TAU_SFT_DATA_WRITER_MERGED_PATH:-$project_root/training/tau_sft_data_writer_v1_merged}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tau-sft-data-writer-merged-cache}"

if [[ ! -f "$model_path/model.safetensors.index.json" ]]; then
  echo "Merged SFT-data-writer model not found: $model_path" >&2
  exit 1
fi

exec "$project_root/.venv/bin/vllm" serve "$model_path" \
  --served-model-name qwen35-tau-sftdata-writer \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --language-model-only
