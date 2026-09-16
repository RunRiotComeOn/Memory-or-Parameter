#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="$project_root/training/tau_memory_writer_gemini_v2_balanced/train.parquet"
train_dir="$project_root/training/tau_memory_writer_gemini_v2_balanced_lora"
export_dir="$project_root/training/tau_memory_writer_gemini_v2_balanced_export"
log="$project_root/runtime_logs/tau_gemini_writer_v2_train.log"

stage() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$log"; }

[[ -s "$dataset" ]] || { echo "Missing dataset: $dataset" >&2; exit 1; }
[[ ! -e "$train_dir" && ! -e "$export_dir" ]] || {
  echo "Refusing to overwrite v2 train/export output" >&2
  exit 1
}

stage "training:start version=v2-balanced epochs=1 max_length=13000"
EPOCHS=1 LEARNING_RATE=1e-5 MAX_LENGTH=13000 \
  "$project_root/scripts/train_tau_memory_writer_verl.sh" "$dataset" "$train_dir" \
  2>&1 | tee -a "$log"
stage "training:complete"

checkpoint="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
[[ -n "$checkpoint" ]] || { echo "No checkpoint found" >&2; exit 1; }
source "$project_root/scripts/activate_verl.sh"
stage "adapter_export:start checkpoint=$checkpoint"
set +e
python -m verl.model_merger merge --backend fsdp --local_dir "$checkpoint" \
  --target_dir "$export_dir" --use_cpu_initialization 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
[[ -s "$export_dir/lora_adapter/adapter_model.safetensors" ]] || {
  echo "Adapter export failed with status $status" >&2
  exit 1
}
stage "adapter_export:complete status=$status adapter=$export_dir/lora_adapter"
