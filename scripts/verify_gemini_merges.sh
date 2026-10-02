#!/usr/bin/env bash
# Verify every merged checkpoint the gemini sweep produces, as each appears.
#
# NOTE ON PROCESS CHECKS: use `ps -eo pid,cmd | grep '[m]erge_peft_lora.py'`,
# never `pgrep -f merge_peft_lora.py`. The -f form matches THIS script's own
# command line when the pattern appears in it, so a wait loop built on it never
# exits. That has now cost this project three incidents (twice with pkill -f,
# once with pgrep -f). The bracket trick makes the pattern not match itself.
set -uo pipefail
cd /nas04/yixuh/memory
BASE=/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
LOGF=gemini_merge_verify.log
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOGF"; }

for b in babyai scienceworld sqlgym textcraft webshop; do
  merged="/nas04/yixuh/gem_${b}_merged"
  adapter="${b}_experiment/router_gemini_v1_lora/adapter_fullmodel"
  marker="merge_verified/${b}"   # outside $merged: the driver rm -rf's it after phase D
  mkdir -p merge_verified
  [ -f "$marker" ] && continue
  [ -f "$merged/model.safetensors.index.json" ] || continue
  [ -f "$adapter/adapter_config.json" ] || continue
  # Skip while a merge is still writing into it.
  ps -eo cmd | grep -q "[m]erge_peft_lora.py.*${b}_merged" && { log "$b: merge still running, skipping"; continue; }
  log "$b: verifying $merged"
  if .venv/bin/python scripts/verify_merged_model.py "$BASE" "$merged" "$adapter" 2>&1 | tee -a "$LOGF"; then
    touch "$marker"; log "$b: VERIFIED"
  else
    log "$b: FAILED VERIFICATION -- its sft/both cells measure the wrong model"
  fi
done
