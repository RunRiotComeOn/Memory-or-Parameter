#!/usr/bin/env bash
# Same model/server as serve_tau_base.sh, configured so that decoding is
# reproducible run-to-run.
#
# Why: the alloc_v1 dev matrix ran rollouts through a ProcessPoolExecutor, so
# concurrent requests landed in different continuous-batching batches on every
# run.  vLLM's kernels are not batch-invariant, so identical prompts at
# temperature 0 could take different argmax branches, and a 41-step rollout
# amplifies a single flipped token into a different episode.  That is a
# candidate explanation for the 28.1% run-to-run flip rate.
#
#   --max-num-seqs 1          one sequence in flight: batch composition is fixed
#   --no-enable-prefix-caching  no reuse of KV computed under a different batch
#   --enforce-eager           no CUDA-graph capture keyed on batch size
#
# Clients must ALSO be serial (run_appworld_rollout.py --max-parallel 1).
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
model_path="${MODEL_PATH:-/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}"

export HF_HOME="${HF_HOME:-/nas04/yixuh/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/appworld-det-server-cache}"
# Set VLLM_BATCH_INVARIANT=1 to additionally force batch-invariant kernels.
# Only needed if the probe shows the server is still nondeterministic at
# batch size 1 (MoE expert GEMMs can accumulate with atomics).

# PORT / TENSOR_PARALLEL_SIZE / GPU_MEMORY_UTILIZATION / MAX_MODEL_LEN are
# overridable so several replicas can run side by side (DESIGN.md section 11:
# determinism comes from --max-num-seqs 1 within a replica, so independent
# replicas parallelize without affecting it). Defaults are the original
# single-replica TP=4 settings.
exec "$project_root/.venv/bin/vllm" serve "$model_path" \
  --served-model-name qwen35-tau \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
  --enforce-eager \
  --max-num-seqs 1 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only
