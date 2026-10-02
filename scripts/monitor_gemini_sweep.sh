#!/usr/bin/env bash
# Event source for the sweep monitor. Each stdout line becomes a notification.
#
# Emits: driver-log phase lines, the gpu_free watcher's actions, a 15-minute
# heartbeat whose numbers are read off disk (never remembered), and a terminal
# event on either sweep completion or the driver session disappearing.
set -uo pipefail
cd /nas04/yixuh/memory
LOG=gemini_all_benchmarks.log
WLOG=free_gpu_for_lora.log
seen=$(wc -l < "$LOG" 2>/dev/null || echo 0)
wseen=$(wc -l < "$WLOG" 2>/dev/null || echo 0)
i=0
while true; do
  n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen+1)) "$LOG" \
      | grep -E --line-buffered "##########|ABORT|FAIL|Traceback|rror|serve:|bank|LoRA|merg|eval|skip|COMPLETE" \
      | cut -c1-240 || true
    tail -n +$((seen+1)) "$LOG" | grep -q "GEMINI ROUTER SWEEP COMPLETE" && {
      echo "SWEEP COMPLETE"; tail -15 "$LOG"; exit 0; }
    seen=$n
  fi
  w=$(wc -l < "$WLOG" 2>/dev/null || echo 0)
  if [ "$w" -gt "$wseen" ]; then
    tail -n +$((wseen+1)) "$WLOG" | sed 's/^/[gpu_free] /' | cut -c1-240 || true
    wseen=$w
  fi
  if ! tmux has-session -t gem_all 2>/dev/null; then
    echo "ALERT: gem_all session gone. last log lines:"; tail -3 "$LOG" | cut -c1-240; exit 1
  fi
  i=$((i+1))
  if [ $((i % 30)) -eq 0 ]; then
    hb=""
    for b in babyai scienceworld sqlgym textcraft webshop; do
      pa="${b}_experiment/gemini_phaseA.log"
      p=$(grep -oE '\[[0-9]+/[0-9]+\]' "$pa" 2>/dev/null | tail -1)
      ev=$(ls -d ${b}_experiment/gem_*/summary.json 2>/dev/null | wc -l)
      [ -n "$p$ev" ] && hb="$hb ${b}=${p:-none}/${ev}ev"
    done
    echo "progress:$hb  gpu=[$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,)]  ($(date +%T))"
  fi
  sleep 30
done
