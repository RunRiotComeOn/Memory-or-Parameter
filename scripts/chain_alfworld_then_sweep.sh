#!/usr/bin/env bash
# Wait for the ALFWorld gemini grid to finish, then start the five-benchmark
# gemini-router sweep.
#
# Why chain rather than launch both now: both drivers lease GPUs, and the
# ALFWorld grid is pinned to GPU 0,1 and will not yield. Running them at once
# would have them contend for the same cards. Chaining also means the sweep
# inherits a quiet machine for its first bank build.
#
# The sweep starts whether ALFWorld SUCCEEDS or FAILS -- it does not depend on
# ALFWorld's artifacts, and a failed ALFWorld should not silently cancel five
# benchmarks' worth of queued work. Which case happened is logged.
#
# Runs itself in the foreground of a tmux session, so it survives this
# session ending.
set -uo pipefail
cd /nas04/yixuh/memory

LOG=chain_alfworld_then_sweep.log
ALF_LOG=alfworld_experiment/gemini_grid.log
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }

log "waiting for the ALFWorld gemini grid (tmux: gem_grid)"
while :; do
  if grep -q "ALFWORLD GEMINI GRID COMPLETE" "$ALF_LOG" 2>/dev/null; then
    log "ALFWorld grid COMPLETE"
    break
  fi
  if ! tmux has-session -t gem_grid 2>/dev/null; then
    log "ALFWorld grid session ended WITHOUT the completion marker"
    log "  last line: $(tail -1 "$ALF_LOG" 2>/dev/null | cut -c1-200)"
    log "  evals finished: $(ls alfworld_experiment/gem_*/summary.json 2>/dev/null | wc -l)/6"
    log "  starting the sweep anyway -- it does not depend on ALFWorld's artifacts"
    break
  fi
  sleep 60
done

# Give the ALFWorld replica a moment to release its cards before the sweep
# starts leasing.
sleep 30
log "GPU state before the sweep: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' ' )"

if tmux has-session -t gem_all 2>/dev/null; then
  log "a gem_all session already exists; not starting a second one"
  exit 0
fi
log "starting the five-benchmark sweep (tmux: gem_all)"
tmux new-session -d -s gem_all 'bash scripts/run_gemini_router_all_benchmarks.sh'
sleep 10
tmux has-session -t gem_all 2>/dev/null && log "gem_all started" || log "ABORT: gem_all failed to start"
log "chain done"
