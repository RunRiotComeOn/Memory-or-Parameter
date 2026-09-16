#!/usr/bin/env bash
# Deterministic serving that still batches.
#
# scripts/serve_appworld_deterministic.sh gets reproducibility by serialising
# (--max-num-seqs 1), which measured 14 tok/s and would put the two-run noise
# experiment at roughly 28 GPU-hours.  vLLM's batch-invariant kernels aim at the
# same property without giving up continuous batching: the result for a sequence
# no longer depends on which other sequences share its batch.
#
# Verify with scripts/probe_batch_invariance.py before trusting it.
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
model_path="${MODEL_PATH:-/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/appworld-bi-server-cache}"
export VLLM_BATCH_INVARIANT=1

exec "$project_root/.venv/bin/vllm" serve "$model_path" \
  --served-model-name qwen35-tau \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only
