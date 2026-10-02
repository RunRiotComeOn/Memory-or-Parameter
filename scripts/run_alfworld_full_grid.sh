#!/usr/bin/env bash
# End-to-end driver for the ALFWorld memory/SFT ablation grid
# (`alfworld_summary.md`), for any backbone in model_profiles.py:
#
#   MODEL_PROFILE=gemma4     scripts/run_alfworld_full_grid.sh
#   MODEL_PROFILE=glm47flash scripts/run_alfworld_full_grid.sh
#
# The Qwen3.5 grid in alfworld_summary.md was assembled step by step (v4
# router, force_memory_v1, force_sft_v1, ...); this runs the same design in
# one pass so a second backbone gets the same six configurations:
#   0. baselines          no-memory base pool (train200) + valid_unseen57
#   A. three build arms   router (llm) / force_memory / force_sft, on the pool
#   B. two LoRA trainings router-pool, force_sft-pool
#   C. two merges         re-keyed, merged, verified
#   D. ten evaluations    5 configurations x {train200, unseen57}
#
# Same constraints as the summary's section "消融配置方式": one base pool
# shared by every arm, artifacts frozen before any replay, every eval on the
# same task ids (train200_ids.txt / unseen57_ids.txt -- the exact tasks the
# Qwen grid used), same seed, same serving config. Outputs go to
# alfworld_experiment/<profile tag>/ (backbone_out), so nothing here can
# collide with, or be skipped because of, the Qwen artifacts.
#
# Every step is SKIPPED if its output already exists, so this can be re-run
# after any interruption. Resource layout matches the other full-grid scripts
# (GPU 2,3 belong to another user and are never touched):
#   replica A 0,1:8030  B 4,5:8031  C 6,7:8032; LoRA on 6,7; merges on CPU.
#
# NOTE: per-task determinism of ALFWorld rollouts was verified for Qwen only
# (alfworld_summary.md conclusion 3). Re-check it for a new backbone before
# comparing across replicas: rerun baseline_valid_unseen_v1 on another port.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src

source scripts/lib_backbone.sh   # BASE, SERVED_NAME; MODEL_PROFILE selects the backbone
PY=.venv/bin/python
ROLL_PY=/nas04/yixuh/alfworld_venv310/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=$(backbone_out alfworld_experiment); mkdir -p "$OUT"; export OUT
TRAIN_IDS=alfworld_experiment/train200_ids.txt
UNSEEN_IDS=alfworld_experiment/unseen57_ids.txt
export ALFWORLD_DATA=/nas04/yixuh/alfworld_data
KEEP_MERGED="${KEEP_MERGED:-0}"
SEED=20260822
MAX_STEPS=40

log() { echo "[$(date +%H:%M:%S)] [$MODEL_PROFILE] $*"; }

serve() {
  local sess=$1 gpus=$2 port=$3 model=$4
  local current
  current=$(curl -s -m 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null \
            | $PY -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null || true)
  if [[ "$current" == "$model" ]]; then log "serve: $port already on $(basename "$model")"; return 0; fi
  # Only this script's own session is ever stopped (scripts/gpu_lease.sh's
  # rule: never evict someone else's run to make room). If the port is still
  # held after that, it belongs to another run: abort instead of killing it.
  tmux kill-session -t "$sess" 2>/dev/null
  for _ in $(seq 90); do curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null || break; sleep 2; done
  local holder
  holder=$(ss -lptnH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [[ -n "${holder:-}" ]]; then
    log "ABORT: port $port is held by pid $holder, not by $sess; not evicting it"; exit 1
  fi
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus" | paste -sd+ | bc)
  if [[ "${used:-99999}" -gt 2000 ]]; then
    log "ABORT: GPU $gpus already hold ${used}MiB (another run?); not starting $sess there"; exit 1
  fi
  sleep 5
  log "serve: $port <- $(basename "$model") on GPU $gpus"
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpus TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 PORT=$port \
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-alf-$port \
     scripts/serve_appworld_deterministic.sh 2>&1 | tee -a $OUT/server_$port.log"
  local waited=0
  until curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200; do
    sleep 30; waited=$((waited+30))
    if [[ $waited -gt 3600 ]]; then log "ABORT: $port did not come up in 60min"; exit 1; fi
    tmux has-session -t "$sess" 2>/dev/null || { log "ABORT: $sess died while loading"; exit 1; }
  done
  current=$(curl -s -m 5 "http://127.0.0.1:$port/v1/models" | $PY -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["root"])')
  [[ "$current" == "$model" ]] || { log "ABORT: $port serves $current, expected $model"; exit 1; }
  log "serve: $port ready on $(basename "$model")"
}

# rollout <dir> <split> <ids-file> <url> [memory-bank]
# A summary with a null pass_rate (every task errored) is redone, not skipped.
rollout() {
  local dir=$1 split=$2 ids=$3 url=$4 bank=${5:-}
  if [[ -f "$dir/summary.json" ]]; then
    if $PY -c "import json,sys;sys.exit(0 if json.load(open('$dir/summary.json')).get('pass_rate') is not None else 1)"; then
      log "skip $(basename "$dir") (done)"; return 0
    fi
    log "redoing $(basename "$dir") (null pass_rate)"; rm -rf "$dir"
  fi
  local args=(--split "$split" --output "$dir" --experiment-name "alf_${MODEL_PROFILE}_$(basename "$dir")"
              --task-ids $(cat "$ids")
              --max-parallel 4 --max-steps "$MAX_STEPS" --seed "$SEED"
              --model "$SERVED_NAME" --base-url "$url")
  if [[ -n "$bank" ]]; then
    # A missing bank does not stop the rollout: every task errors and a
    # summary is still written with pass_rate null. Fail here instead.
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $dir: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "rollout $(basename "$dir") ($split${bank:+, bank $(basename "$(dirname "$(dirname "$bank")")")})"
  $ROLL_PY -u scripts/run_alfworld_rollout.py "${args[@]}"
}

# --- phase 0: baselines -----------------------------------------------------
phase_0() {
  log "=== PHASE 0: no-memory baselines (train200 pool + unseen57) ==="
  serve alf_srv_a 0,1 8030 "$BASE"
  serve alf_srv_b 4,5 8031 "$BASE"
  rollout "$OUT/base_train_v2"            train        "$TRAIN_IDS"  http://127.0.0.1:8030/v1 > "$OUT/phase0_train.log" 2>&1 &
  local p1=$!
  rollout "$OUT/baseline_valid_unseen_v1" valid_unseen "$UNSEEN_IDS" http://127.0.0.1:8031/v1 > "$OUT/phase0_unseen.log" 2>&1 &
  local p2=$!
  local rc=0
  wait $p1 || rc=1; wait $p2 || rc=1
  [[ -f "$OUT/base_train_v2/summary.json" && -f "$OUT/baseline_valid_unseen_v1/summary.json" && "$rc" == 0 ]] \
    || { log "ABORT: baselines failed (see $OUT/phase0_*.log)"; exit 1; }
  log "=== PHASE 0 done ==="
}

# --- phase A: builds --------------------------------------------------------
build_arm() {
  local arm=$1 url=$2 mode=$3 writer=$4
  local dir="$OUT/$arm"
  [[ -f "$dir/summary.json" ]] && { log "skip build $arm (done)"; return 0; }
  log "build $arm (mode=$mode writer=$writer)"
  $PY -u scripts/run_alfworld_router_llm_probe.py --model "$SERVED_NAME" --output "$dir" \
      --train-rollout "$OUT/base_train_v2" --router-mode "$mode" \
      --sft-writer "$writer" --base-url "$url" \
    || { log "ABORT: build $arm failed"; return 1; }
  [[ -f "$dir/summary.json" ]] || { log "ABORT: build $arm wrote no summary"; return 1; }
}

phase_a() {
  log "=== PHASE A: three build arms, one per replica ==="
  serve alf_srv_a 0,1 8030 "$BASE"
  serve alf_srv_b 4,5 8031 "$BASE"
  serve alf_srv_c 6,7 8032 "$BASE"
  local pids=()
  local arms=("router_probe 8030 llm teacher"
              "force_memory 8031 force_memory none"
              "force_sft    8032 force_sft teacher")
  local spec
  for spec in "${arms[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    build_arm "$1" "http://127.0.0.1:$2/v1" "$3" "$4" > "$OUT/phaseA_$1.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  log "=== PHASE A done (rc=$rc) ==="
  [[ "$rc" == "0" ]] || { log "ABORT: phase A had failures (see $OUT/phaseA_*.log)"; exit 1; }
}

# --- phase B/C: LoRA + merge -----------------------------------------------
train_and_merge() {
  local arm=$1                              # router_probe | force_sft
  local pool="$OUT/$arm/sft_pool.jsonl"
  local adapter="$OUT/${arm}_lora/adapter"
  local merged; merged="$(backbone_merged alf_${arm}_merged)"
  [[ -s "$pool" ]] || { log "skip $arm: empty or missing sft pool"; return 0; }
  local n; n=$(wc -l < "$pool")
  if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    log "train LoRA $arm ($n examples)"
    CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/nas04/yixuh/hf_cache \
      $ROUTER_PY -u scripts/train_agent_sft_lora_peft.py --base-model "$BASE" \
        --pool "$pool" --output "$adapter" 2>&1 | tee -a "$OUT/${arm}_lora.log"
  else log "skip LoRA $arm (adapter exists)"; fi
  [[ -f "$adapter/adapter_model.safetensors" ]] || { log "ABORT: no adapter for $arm"; exit 1; }
  if [[ ! -f "${adapter}_fullmodel/adapter_model.safetensors" ]]; then
    $ROUTER_PY scripts/rekey_lora_to_full_model.py "$adapter" "${adapter}_fullmodel" --base-model "$BASE" \
      2>&1 | tee -a "$OUT/${arm}_lora.log"
  fi
  if [[ ! -f "$merged/config.json" ]]; then
    log "merge $arm -> $merged"
    HF_HOME=/nas04/yixuh/hf_cache $ROUTER_PY -u scripts/merge_peft_lora.py \
      "$BASE" "${adapter}_fullmodel" "$merged" 2>&1 | tee -a "$OUT/${arm}_merge.log"
    $ROUTER_PY scripts/verify_merged_model.py "$BASE" "$merged" "${adapter}_fullmodel" 2>&1 \
      | tee -a "$OUT/${arm}_merge.log" || { log "ABORT: merged $arm failed verification"; exit 1; }
  else log "skip merge $arm (exists)"; fi
}

phase_bc() {
  log "=== PHASE B/C: two LoRAs + merges ==="
  # Stop every session holding GPU 6,7, not just this script's: a leftover
  # replica once made a LoRA silently CPU-offload (lm_head on meta).
  for sess in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    case "$sess" in *srv_c|*server_c|*_c) tmux kill-session -t "$sess" 2>/dev/null;; esac
  done
  sleep 10
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | paste -sd+ | bc)
  if [[ "${used:-99999}" -gt 2000 ]]; then
    log "ABORT: GPU 6,7 still hold ${used}MiB; LoRA would silently CPU-offload"; exit 1
  fi
  train_and_merge router_probe
  train_and_merge force_sft
  log "=== PHASE B/C done ==="
}

# --- phase D: evaluations ---------------------------------------------------
phase_d() {
  log "=== PHASE D: ten evaluations (5 configs x train200/unseen57) ==="
  local RBANK="$OUT/router_probe/banks/memory_alfworld.json"
  local FBANK="$OUT/force_memory/banks/memory_alfworld.json"
  serve alf_srv_a 0,1 8030 "$BASE"

  ( for line in "train $TRAIN_IDS train200" "valid_unseen $UNSEEN_IDS unseen57"; do
      set -- $line
      rollout "$OUT/eval_$3_routermem" "$1" "$2" http://127.0.0.1:8030/v1 "$RBANK"
      rollout "$OUT/eval_$3_forcemem"  "$1" "$2" http://127.0.0.1:8030/v1 "$FBANK"
    done ) > "$OUT/phaseD_base.log" 2>&1 &
  local pid_base=$!

  ( m="$(backbone_merged alf_router_probe_merged)"
    if [[ -f "$m/config.json" ]]; then
      serve alf_srv_b 4,5 8031 "$m"
      for line in "train $TRAIN_IDS train200" "valid_unseen $UNSEEN_IDS unseen57"; do
        set -- $line
        rollout "$OUT/eval_$3_routersftonly" "$1" "$2" http://127.0.0.1:8031/v1
        rollout "$OUT/eval_$3_routerboth"    "$1" "$2" http://127.0.0.1:8031/v1 "$RBANK"
      done
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    else log "skip router sft/both evals (no merged model -- empty router sft pool?)"; fi
  ) > "$OUT/phaseD_router.log" 2>&1 &
  local pid_router=$!

  ( m="$(backbone_merged alf_force_sft_merged)"
    if [[ -f "$m/config.json" ]]; then
      serve alf_srv_c 6,7 8032 "$m"
      for line in "train $TRAIN_IDS train200" "valid_unseen $UNSEEN_IDS unseen57"; do
        set -- $line
        rollout "$OUT/eval_$3_forcesft" "$1" "$2" http://127.0.0.1:8032/v1
      done
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    else log "skip force_sft evals (no merged model)"; fi
  ) > "$OUT/phaseD_forcesft.log" 2>&1 &
  local pid_fsft=$!

  local rc=0
  for p in $pid_base $pid_router $pid_fsft; do wait "$p" || rc=1; done
  log "=== PHASE D done (rc=$rc) ==="
}

report() {
  log "=== RESULTS ($MODEL_PROFILE) ==="
  $PY - <<'PYR'
import json, os
OUT = os.environ["OUT"]
def load(d):
    p = os.path.join(OUT, d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None
def per_task(d):
    root = os.path.join(OUT, d, "trajectories")
    if not os.path.isdir(root): return {}
    out = {}
    for f in os.listdir(root):
        r = json.load(open(os.path.join(root, f)))
        out[r["task_id"]] = bool((r.get("trajectory") or {}).get("success"))
    return out
rows = [("baseline", None), ("force_mem", "forcemem"), ("router mem", "routermem"),
        ("router SFT", "routersftonly"), ("router both", "routerboth"), ("force_sft", "forcesft")]
for line, base_dir in (("unseen57", "baseline_valid_unseen_v1"), ("train200", "base_train_v2")):
    print(f"\n--- alfworld {line} ---")
    base = per_task(base_dir)
    for name, key in rows:
        d = base_dir if key is None else f"eval_{line}_{key}"
        x = load(d)
        if x is None: print(f"  {name:12s} (missing)"); continue
        tail = ""
        if key is not None and base:
            cur = per_task(d)
            up = sum(1 for t in cur if cur[t] and not base.get(t, False))
            down = sum(1 for t in cur if base.get(t, False) and not cur[t])
            tail = f"  vs baseline: {up} up {down} down (net {up-down:+d})"
        print(f"  {name:12s} {x['success']:3d}/{x['tasks']:3d}  pass={x['pass_rate']:.4f}  errors={x['errors']}{tail}")
PYR
}

phase_0
phase_a
phase_bc
phase_d
report
log "=== ALFWORLD FULL GRID COMPLETE ==="
