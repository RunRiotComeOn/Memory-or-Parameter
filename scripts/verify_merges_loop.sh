#!/usr/bin/env bash
# The driver deletes each merged checkpoint right after its phase-D evals, so
# verification has only that window. Poll for new merges until the sweep ends.
set -uo pipefail
cd /nas04/yixuh/memory
while tmux has-session -t gem_all 2>/dev/null; do
  bash scripts/verify_gemini_merges.sh >/dev/null 2>&1
  sleep 60
done
echo "[$(date +%F' '%T)] gem_all ended; merge-verify loop exiting" >> gemini_merge_verify.log
