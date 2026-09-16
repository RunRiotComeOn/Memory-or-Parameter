#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
experiment_root="${TAU_RUBRIC_E2E_ROOT:-$project_root/tau_experiment/writer_rubric_e2e_v1}"
model_path="${MODEL_PATH:-/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}"
tensor_parallel_size="${TAU_TENSOR_PARALLEL_SIZE:-4}"
max_model_len="${TAU_MAX_MODEL_LEN:-65536}"
gpu_memory_utilization="${TAU_GPU_MEMORY_UTILIZATION:-0.88}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tau-rubric-e2e-server-cache}"

lora_args=()
for rubric_id in r0_faithful r1_causal_minimal r2_state_transition r3_robust; do
  adapter="$experiment_root/training/$rubric_id/export/lora_adapter"
  if [[ ! -s "$adapter/adapter_model.safetensors" ]]; then
    echo "missing rubric adapter: $adapter" >&2
    exit 1
  fi
  lora_args+=("qwen35-agent-$rubric_id=$adapter")
done

exec "$project_root/.venv/bin/vllm" serve "$model_path" \
  --served-model-name qwen35-tau \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size "$tensor_parallel_size" \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --enable-prefix-caching \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 4 \
  --lora-modules "${lora_args[@]}"
