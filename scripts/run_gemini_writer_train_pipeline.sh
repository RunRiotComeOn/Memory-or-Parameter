#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
teacher_dir="$project_root/tau_experiment/gemini_memory_teacher_full_20260817"
dataset_dir="$project_root/training/tau_memory_writer_gemini_v1"
train_dir="$project_root/training/tau_memory_writer_gemini_v1_lora"
export_dir="$project_root/training/tau_memory_writer_gemini_v1_export"
teacher_session="tau_gemini_memory_teacher_full"
log="$project_root/runtime_logs/tau_gemini_writer_train_pipeline.log"

stage() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$log"
}

stage "waiting_for:teacher"
while [[ ! -s "$teacher_dir/summary.json" ]]; do
  if ! tmux has-session -t "$teacher_session" 2>/dev/null; then
    echo "Gemini teacher exited without summary.json" >&2
    exit 1
  fi
  sleep 30
done

teacher_errors="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["errors"])' "$teacher_dir/summary.json")"
if [[ "$teacher_errors" != 0 ]]; then
  echo "Teacher completed with $teacher_errors errors; refusing to train" >&2
  exit 1
fi
stage "teacher:complete"

if [[ -e "$dataset_dir" || -e "$train_dir" || -e "$export_dir" ]]; then
  echo "Refusing to overwrite an existing Gemini writer dataset/training/export directory" >&2
  exit 1
fi

cd "$project_root"
PYTHONPATH=src .venv/bin/python scripts/prepare_tau_memory_writer_sft.py \
  --retention-root "$teacher_dir" \
  --output "$dataset_dir"
stage "dataset:prepared"

read -r train_examples refine_replace total_examples < <(
  python - "$dataset_dir/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
labels = x["label_classes"]
print(x["train_examples"], labels.get("refine", 0) + labels.get("replace", 0), x["examples"])
PY
)
if (( train_examples < 20 )); then
  echo "Only $train_examples train examples; refusing to train" >&2
  exit 1
fi
if (( refine_replace < 3 )); then
  echo "Only $refine_replace refine/replace examples; refusing add-only training" >&2
  exit 1
fi
stage "dataset:quality_gate_pass train=$train_examples total=$total_examples refine_replace=$refine_replace"

stage "waiting_for:gpus"
while true; do
  busy="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 2000 {count++} END {print count+0}')"
  if [[ "$busy" == 0 ]]; then
    break
  fi
  sleep 60
done

stage "training:start epochs=1"
EPOCHS=1 LEARNING_RATE=1e-5 MAX_LENGTH=13000 \
  "$project_root/scripts/train_tau_memory_writer_verl.sh" \
  "$dataset_dir/train.parquet" "$train_dir"
stage "training:complete"

checkpoint_dir="$(find "$train_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$checkpoint_dir" ]]; then
  echo "No VERL checkpoint found" >&2
  exit 1
fi

stage "adapter_export:start"
source "$project_root/scripts/activate_verl.sh"
set +e
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$checkpoint_dir" \
  --target_dir "$export_dir" \
  --use_cpu_initialization
export_status=$?
set -e
if [[ ! -s "$export_dir/lora_adapter/adapter_model.safetensors" ]]; then
  echo "Adapter export failed with status $export_status" >&2
  exit 1
fi
stage "adapter_export:complete status=$export_status"

