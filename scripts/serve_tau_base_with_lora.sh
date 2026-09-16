#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
adapter_path="${TAU_LORA_PATH:-$project_root/training/tau_qwen35_verl_export_v1/lora_adapter}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tau-qwen35-lora-triton-cache}"

if [[ ! -f "$adapter_path/adapter_model.safetensors" ]]; then
  echo "LoRA adapter not found: $adapter_path" >&2
  exit 1
fi

exec "$project_root/.venv/bin/vllm" serve Qwen/Qwen3.5-35B-A3B \
  --served-model-name qwen35-tau \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 1 \
  --lora-modules "qwen35-tau-sft=$adapter_path"
