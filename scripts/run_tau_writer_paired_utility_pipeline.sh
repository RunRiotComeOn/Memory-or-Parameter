#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="${1:-$project_root/tau_experiment/memory_writer_paired_utility_20260816}"

cd "$project_root"
PYTHONPATH=src .venv/bin/python scripts/run_tau_writer_paired_utility.py \
  --experiment "$experiment" \
  --max-parallel-runs 3 \
  --max-sources 1

PYTHONPATH=src .venv/bin/python scripts/run_tau_writer_paired_utility.py \
  --experiment "$experiment" \
  --max-parallel-runs 4
