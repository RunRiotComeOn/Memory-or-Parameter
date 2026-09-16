#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_verl.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/trajectory-memory-verl-triton-cache}"
export TOKENIZERS_PARALLELISM=true
export VERL_LOGGING_LEVEL=INFO
export VERL_SFT_LOGGING_LEVEL=INFO

model_path="${MODEL_PATH:-/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}"
dataset_path="${1:-${project_root}/training/replay_sft.parquet}"
output_dir="${2:-${project_root}/training/qwen35_replay_verl_lora}"
max_length="${MAX_LENGTH:-20480}"
learning_rate="${LEARNING_RATE:-2e-5}"
epochs="${EPOCHS:-3}"
train_batch_size="${TRAIN_BATCH_SIZE:-4}"
save_lora_only="${SAVE_LORA_ONLY:-true}"
activation_offload="${ACTIVATION_OFFLOAD:-false}"
sp_size="${SP_SIZE:-1}"
use_remove_padding="${USE_REMOVE_PADDING:-false}"

exec "${project_root}/.verl-venv/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  -m verl.trainer.sft_trainer \
  engine=fsdp \
  optim=fsdp \
  engine.strategy=fsdp2 \
  engine.fsdp_size=4 \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.reshard_after_forward=true \
  engine.offload_policy=false \
  engine.ulysses_sequence_parallel_size="${sp_size}" \
  engine.use_torch_compile=false \
  model.path="${model_path}" \
  model.trust_remote_code=true \
  model.use_remove_padding="${use_remove_padding}" \
  model.enable_gradient_checkpointing=true \
  model.enable_activation_offload="${activation_offload}" \
  model.lora_rank=8 \
  model.lora_alpha=32 \
  'model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj,down_proj,gate_proj,up_proj,shared_expert_gate]' \
  data.train_files="${dataset_path}" \
  data.val_files=null \
  data.messages_key=messages \
  data.train_batch_size="${train_batch_size}" \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  data.pad_mode=no_padding \
  data.max_length="${max_length}" \
  data.max_token_len_per_gpu="${max_length}" \
  data.enable_thinking_default=false \
  data.custom_cls.path="${project_root}/src/trajectory_memory_lab/verl_dataset.py" \
  data.custom_cls.name=Qwen35ActionSFTDataset \
  data.num_workers=0 \
  data.ignore_input_ids_mismatch=true \
  optim.lr="${learning_rate}" \
  optim.lr_warmup_steps_ratio=0.05 \
  optim.weight_decay=0.0 \
  optim.lr_scheduler_type=constant \
  checkpoint.save_contents='["model","extra"]' \
  +checkpoint.save_lora_only="${save_lora_only}" \
  trainer.default_local_dir="${output_dir}" \
  trainer.project_name=trajectory-memory-replay \
  trainer.experiment_name=qwen35-a3b-lora \
  trainer.total_epochs="${epochs}" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.logger='["console"]' \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=4
