#!/usr/bin/env bash
# Free the sweep's own vLLM replica when a benchmark moves from phase A to the
# LoRA step.
#
# Why this is needed: run_gemini_router_all_benchmarks.sh leases a GPU pair for
# LoRA (phase B) WITHOUT first releasing the pair its own serve replica holds.
# That was harmless while a second pair was free. It no longer is -- 0,1 and 6,7
# belong to boyuzhu and 2,3 to haskari, so the only pair this run can use is the
# one its own server is sitting on. Left alone, every benchmark would wait the
# full 7200s and then ABORT at phase B.
#
# So: watch the driver log, and the moment a benchmark finishes phase A -- either
# "bank build done" or, on a relaunch, "skip bank build (done)" -- kill the serve
# session. The pending gpu_wait_for_pair poll (60s) then finds the
# pair free and trains on it. Phase D re-leases through serve_leased, whose
# health check on the dead port fails and falls through to a fresh lease.
#
# Only ever kills gemall_* sessions -- never another user's work, never the
# driver, never the env servers.
set -uo pipefail
cd /nas04/yixuh/memory
LOG=gemini_all_benchmarks.log
STATE=/tmp/free_gpu_for_lora.handled
: > "$STATE"
log() { echo "[$(date +%F' '%T)] $*" | tee -a free_gpu_for_lora.log; }

log "watching $LOG for phase-A completions"
while tmux has-session -t gem_all 2>/dev/null; do
  # Which benchmarks have reported a finished bank build?
  while read -r b; do
    grep -qx "$b" "$STATE" && continue
    # Only act if an SFT pool exists -- a memory-only arm skips LoRA entirely
    # and still needs its server for phase D.
    arm=$(ls -d ${b}_experiment/router_gemini_v1 2>/dev/null | head -1)
    if [ -n "$arm" ] && [ -s "$arm/sft_pool.jsonl" ]; then
      sess=$(tmux ls -F '#S' 2>/dev/null | grep '^gemall_' | head -1)
      if [ -n "$sess" ]; then
        log "$b: bank build done, sft_pool=$(wc -l < "$arm/sft_pool.jsonl") -- killing $sess to free its GPUs for LoRA"
        tmux kill-session -t "$sess" 2>/dev/null && log "$b: $sess killed"
      else
        log "$b: bank build done but no gemall_* session to free"
      fi
    else
      log "$b: bank build done, sft pool empty -- leaving the server up (no LoRA)"
    fi
    echo "$b" >> "$STATE"
  done < <(grep -oE '^\[[^]]*\] [a-z0-9]+: (bank build done|skip bank build)' "$LOG" 2>/dev/null \
           | sed -E 's/.*\] ([a-z0-9]+): (bank build done|skip bank build)/\1/')
  sleep 20
done
log "gem_all ended; watcher exiting"
