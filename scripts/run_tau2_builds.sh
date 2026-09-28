#!/usr/bin/env bash
# One tau2 sub-domain's three build arms, in sequence, against one replica.
#
# Usage: run_tau2_builds.sh <domain> <base-url>
#
# The three arms are the same grid every other benchmark in this repo uses
# (see alfworld_summary.md's "消融配置方式"): the router deciding, every task
# forced to memory, every task forced to sft. They run sequentially because
# they share one replica; domains run in parallel because each has its own.
#
# `--sft-writer none` on the force_memory arm skips ~74 teacher calls that
# could never be used -- that arm's route is always `memory`.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src
dom="${1:?usage: run_tau2_builds.sh <domain> <base-url>}"
url="${2:?usage: run_tau2_builds.sh <domain> <base-url>}"
PY=.venv/bin/python
P=scripts/run_tau2_router_llm_probe.py
TR="tau2_experiment/${dom}_base_train"

echo "### [$dom] router probe"
$PY -u $P --domain "$dom" --output "tau2_experiment/${dom}_router_probe" \
    --train-rollout "$TR" --router-mode llm --sft-writer teacher --base-url "$url"
echo "### [$dom] force_memory"
$PY -u $P --domain "$dom" --output "tau2_experiment/${dom}_force_memory" \
    --train-rollout "$TR" --router-mode force_memory --sft-writer none --base-url "$url"
echo "### [$dom] force_sft"
$PY -u $P --domain "$dom" --output "tau2_experiment/${dom}_force_sft" \
    --train-rollout "$TR" --router-mode force_sft --sft-writer teacher --base-url "$url"
echo "### [$dom] BUILDS DONE"
