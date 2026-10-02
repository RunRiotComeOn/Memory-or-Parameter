#!/usr/bin/env bash
# Shared GPU-picking helpers for the multi-benchmark router runs.
#
# Why this exists: the earlier grid scripts hard-coded GPU pairs and ports, and
# `serve` killed whatever held a port. That was safe while this user owned all
# eight cards. It stopped being safe once other runs appeared on 4-7, and twice
# in one session a script was within one phase of evicting someone else's work.
#
# The rule here is: never take a card that is in use, never kill anything to
# make room, and wait rather than fail when nothing is free. GPU 2 and 3 are
# permanently excluded -- they belong to a different user (haskari).
#
# Usage:
#   source scripts/gpu_lease.sh
#   pair=$(gpu_wait_for_pair 3600) || exit 1     # "0,1" | "4,5" | "6,7"
#   port=$(gpu_port_for_pair "$pair")            # 8030 | 8031 | 8032

GPU_FORBIDDEN_PAIRS="2,3"
GPU_CANDIDATE_PAIRS="${GPU_CANDIDATE_PAIRS:-0,1 4,5 6,7}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-2000}"

gpu_pair_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | paste -sd+ | bc
}

# Echo the first candidate pair whose combined usage is under the threshold.
# Empty output means nothing is free right now.
gpu_find_free_pair() {
  local pair used
  for pair in $GPU_CANDIDATE_PAIRS; do
    case " $GPU_FORBIDDEN_PAIRS " in *" $pair "*) continue;; esac
    used=$(gpu_pair_used_mib "$pair")
    [ -n "${used:-}" ] && [ "$used" -lt "$GPU_FREE_THRESHOLD_MIB" ] && { echo "$pair"; return 0; }
  done
  return 1
}

# Wait up to $1 seconds for a free pair. Prints the pair, or nothing and
# returns 1 on timeout. Waiting beats failing: these runs take hours and a
# neighbour's job usually finishes long before the deadline.
gpu_wait_for_pair() {
  local deadline=$(( $(date +%s) + ${1:-3600} )) pair
  while :; do
    pair=$(gpu_find_free_pair) && { echo "$pair"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && return 1
    sleep 60
  done
}

# One port per pair, so two phases that happen to pick different pairs cannot
# collide on a port and start killing each other's servers.
gpu_port_for_pair() {
  case "$1" in
    0,1) echo 8030;;
    4,5) echo 8031;;
    6,7) echo 8032;;
    *) echo "" ; return 1;;
  esac
}

# Refuse to serve on a pair this run does not hold. Call before any serve.
gpu_assert_allowed() {
  case " $GPU_FORBIDDEN_PAIRS " in
    *" $1 "*) echo "GPU pair '$1' belongs to another user; refusing" >&2; return 1;;
  esac
  return 0
}
