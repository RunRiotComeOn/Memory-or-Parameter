#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/trajectory-memory-triton-cache}"

exec "$(dirname "$0")/../.venv/bin/vllm" serve Qwen/Qwen3.5-35B-A3B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --language-model-only
