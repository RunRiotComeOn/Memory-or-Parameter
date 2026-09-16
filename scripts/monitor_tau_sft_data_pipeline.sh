#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

while true; do
  printf '%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if tmux has-session -t tau_sftdata_pipeline 2>/dev/null; then
    printf 'pipeline=alive '
  else
    printf 'pipeline=stopped '
  fi
  "$project_root/.venv/bin/python" - <<'PY'
import glob
import json
from pathlib import Path

paths = glob.glob(
    "tau_experiment/tau_sft_data_writer_v1/writer_predictions/tasks/*/candidate.json"
)
records = []
for path in paths:
    try:
        records.append(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        pass
print(
    f"writer_records={len(records)} "
    f"writer_ready={sum(x.get('status') == 'prediction_ready' for x in records)} "
    f"writer_errors={sum(x.get('status') == 'error' for x in records)}",
    end=" ",
)
PY
  if [[ -s runtime_logs/tau_sft_data_end_to_end.log ]]; then
    tail -n 1 runtime_logs/tau_sft_data_end_to_end.log | tr '\n' ' '
  fi
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | paste -sd ';' - | sed 's/^/gpu=/'
  if ! tmux has-session -t tau_sftdata_pipeline 2>/dev/null; then
    break
  fi
  sleep 60
done
