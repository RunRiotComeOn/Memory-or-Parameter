#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="$project_root/training/tau_memory_writer_gemini_v1_compact3"
train_dir="$project_root/training/tau_memory_writer_gemini_v1_lora_retry1"
export_dir="$project_root/training/tau_memory_writer_gemini_v1_export_retry1"
log="$project_root/runtime_logs/tau_gemini_writer_train_retry1.log"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$log"
}

if [[ ! -s "$dataset_dir/train.parquet" ]]; then
  echo "Missing compact training dataset: $dataset_dir/train.parquet" >&2
  exit 1
fi
if [[ -e "$train_dir" || -e "$export_dir" ]]; then
  echo "Refusing to overwrite an existing retry training/export directory" >&2
  exit 1
fi

stage "training:start dataset=compact3 epochs=1 max_length=13000"
EPOCHS=1 LEARNING_RATE=1e-5 MAX_LENGTH=13000 \
  "$project_root/scripts/train_tau_memory_writer_verl.sh" \
  "$dataset_dir/train.parquet" "$train_dir" 2>&1 | tee -a "$log"
stage "training:complete"

checkpoint_dir="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$checkpoint_dir" ]]; then
  echo "No VERL checkpoint found" >&2
  exit 1
fi

stage "adapter_export:start checkpoint=$checkpoint_dir"
source "$project_root/scripts/activate_verl.sh"
set +e
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$checkpoint_dir" \
  --target_dir "$export_dir" \
  --use_cpu_initialization 2>&1 | tee -a "$log"
export_status=${PIPESTATUS[0]}
set -e
if [[ ! -s "$export_dir/lora_adapter/adapter_model.safetensors" ]]; then
  echo "Adapter export failed with status $export_status" >&2
  exit 1
fi
stage "adapter_export:complete status=$export_status adapter=$export_dir/lora_adapter"
