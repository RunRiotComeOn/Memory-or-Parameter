#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

printf '%s four_arm_eval:resume_with_local_nl_judge\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | tee -a runtime_logs/tau_sft_data_end_to_end.log
PYTHONPATH="$project_root/src" "$project_root/.venv/bin/python" -u \
  "$project_root/scripts/run_tau_agent_sft_four_arm_eval.py" \
  --output "$project_root/tau_experiment/tau_agent_sftdata_four_arm_v1" \
  --memory-dir "$project_root/tau_experiment/codex56_writer_sft_cumulative_20260818" \
  --max-parallel-runs 3 --max-concurrency 2 --run-tag v1 \
  2>&1 | tee -a "$project_root/runtime_logs/tau_sftdata_four_arm_eval.log"
printf '%s four_arm_eval:complete summary=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$project_root/tau_experiment/tau_agent_sftdata_four_arm_v1/summary.json" \
  | tee -a runtime_logs/tau_sft_data_end_to_end.log
tmux kill-session -t tau_sftdata_agent_server 2>/dev/null || true
