#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
adapter_path="${TAU_SFT_DATA_WRITER_LORA_PATH:-$project_root/training/tau_sft_data_writer_v1_export/lora_adapter}"
model_path="${MODEL_PATH:-/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tau-sft-data-writer-server-cache}"

if [[ ! -f "$adapter_path/adapter_model.safetensors" ]]; then
  echo "SFT-data-writer LoRA not found: $adapter_path" >&2
  exit 1
fi

exec "$project_root/.venv/bin/vllm" serve "$model_path" \
  --served-model-name qwen35-tau \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 1 \
  --lora-modules "qwen35-tau-sftdata-writer=$adapter_path"
