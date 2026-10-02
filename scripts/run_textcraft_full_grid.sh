#!/usr/bin/env bash
# End-to-end driver for the TextCraft arm of the memory/SFT ablation grid.
#
# Runs EVERYTHING that is left, in order:
#   A. three build arms   (router / force_memory / force_sft), one per replica
#   B. two LoRA trainings (router-pool, force_sft-pool)
#   C. two merges         (each LoRA re-keyed and merged into a full model)
#   D. ten evaluations    (5 configurations x 2 held-out lines)
#
# TextCraft is the first benchmark here with TWO held-out lines, because its
# difficulty axis is an explicit parameter (recipe-tree depth):
#   - `test` (80, depth 1-2): same-distribution, and near the ceiling
#     (baseline ~0.95). Low resolution for gains, but the most sensitive
#     place to detect DAMAGE -- BabyAI's force_memory lost 5 of 80 tasks.
#   - `deep` (127, depth 3-4): goals no pool ever sees, where intermediates
#     must themselves be built from intermediates. This is the suite's first
#     compositional generalization line and carries the resolution.
#
# Every step is SKIPPED if its output already exists, so this can be re-run
# after any interruption without redoing finished work.
#
# Resource choreography (GPU 2,3 belong to another user and are never touched):
#   - Phase A: three replicas on the BASE model, one per build arm.
#   - Phase B/C: replica C is stopped to free GPU 6,7 for LoRA; merges on CPU.
#   - Phase D: replica A stays on BASE for the base-model configs; replicas B
#     and C serve the two merged models.
#
# Invariants this script enforces, each because it was violated earlier in
# this project:
#   1. `serve` verifies the loaded model IS the requested one and stops the
#      actual port holder, not just the session name this script would use.
#   2. Phase A aborts on a failed arm. The BabyAI run "completed" with every
#      downstream step skipped, because each skip is conditioned on an
#      artifact a failed arm never wrote -- nothing errors, it just does
#      nothing.
#   3. A memory-bank path that does not exist aborts rather than producing a
#      summary full of errored tasks with a null pass_rate.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src

source scripts/lib_backbone.sh   # BASE, SERVED_NAME; MODEL_PROFILE selects the backbone
PY=.venv/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=$(backbone_out textcraft_experiment); mkdir -p "$OUT"; export OUT
DOM=textcraft
ENV_URL=http://127.0.0.1:36002   # AgentGym textcraft env server (tmux: textcraft_env)
EVAL_STEPS=40                    # measured, not guessed: see textcraft_agent.run_task
KEEP_MERGED="${KEEP_MERGED:-0}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

serve() {
  local sess=$1 gpus=$2 port=$3 model=$4
  local current
  current=$(curl -s -m 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null \
            | $PY -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null || true)
  if [[ "$current" == "$model" ]]; then log "serve: $port already on $(basename "$model")"; return 0; fi
  tmux kill-session -t "$sess" 2>/dev/null
  local holder
  holder=$(ss -lptnH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [[ -n "${holder:-}" ]]; then
    log "serve: port $port held by pid $holder; stopping it"
    kill "$holder" 2>/dev/null
  fi
  for _ in $(seq 90); do
    curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null || break
    sleep 2
  done
  if curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null; then
    log "ABORT: $port still serving after 180s"; exit 1
  fi
  sleep 5
  log "serve: $port <- $(basename "$model") on GPU $gpus"
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpus TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 PORT=$port \
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-tc-$port \
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

# --- phase A: builds -------------------------------------------------------
build_arm() {
  local arm=$1 url=$2 mode=$3 writer=$4
  local dir="$OUT/${DOM}_${arm}"
  [[ -f "$dir/summary.json" ]] && { log "skip build $arm (done)"; return 0; }
  log "build $arm (mode=$mode writer=$writer)"
  $PY -u scripts/run_textcraft_router_llm_probe.py --model "$SERVED_NAME" --output "$dir" \
      --train-rollout "$OUT/base_train_v1" --router-mode "$mode" \
      --sft-writer "$writer" --base-url "$url" --env-url "$ENV_URL" \
    || { log "ABORT: build $arm failed"; return 1; }
  [[ -f "$dir/summary.json" ]] || { log "ABORT: build $arm wrote no summary"; return 1; }
}

phase_a() {
  log "=== PHASE A: three build arms, one per replica ==="
  [[ -f "$OUT/base_train_v1/summary.json" ]] || { log "ABORT: no base_train_v1; run the baselines first"; exit 1; }
  serve tc_srv_a 0,1 8030 "$BASE"
  serve tc_srv_b 4,5 8031 "$BASE"
  serve tc_srv_c 6,7 8032 "$BASE"
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
  if [[ "$rc" != "0" ]]; then
    log "ABORT: phase A had failures; not continuing (see $OUT/phaseA_*.log)"
    exit 1
  fi
}

# --- phase B/C: LoRA + merge ----------------------------------------------
train_and_merge() {
  local arm=$1                              # router_probe | force_sft
  local tag="${DOM}_${arm}"
  local pool="$OUT/${tag}/sft_pool.jsonl"
  local adapter="$OUT/${tag}_lora/adapter"
  local merged="$(backbone_merged tc_${tag}_merged)"
  [[ -s "$pool" ]] || { log "skip $tag: empty or missing sft pool"; return 1; }
  local n; n=$(wc -l < "$pool")
  if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    log "train LoRA $tag ($n examples)"
    CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/nas04/yixuh/hf_cache \
      $ROUTER_PY -u scripts/train_agent_sft_lora_peft.py \
        --pool "$pool" --output "$adapter" 2>&1 | tee -a "$OUT/${tag}_lora.log"
  else log "skip LoRA $tag (adapter exists)"; fi
  [[ -f "$adapter/adapter_model.safetensors" ]] || { log "FAIL: no adapter for $tag"; return 1; }
  if [[ ! -f "$adapter"_fullmodel/adapter_model.safetensors ]]; then
    $ROUTER_PY scripts/rekey_lora_to_full_model.py "$adapter" "${adapter}_fullmodel" --base-model "$BASE" 2>&1 | tee -a "$OUT/${tag}_lora.log"
  fi
  if [[ ! -f "$merged/config.json" ]]; then
    log "merge $tag -> $merged"
    HF_HOME=/nas04/yixuh/hf_cache $ROUTER_PY -u scripts/merge_peft_lora.py \
      "$BASE" "${adapter}_fullmodel" "$merged" 2>&1 | tee -a "$OUT/${tag}_merge.log"
    for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
      cp -L "$BASE/$f" "$merged/$f" 2>/dev/null
    done
    $ROUTER_PY scripts/verify_merged_model.py "$BASE" "$merged" "${adapter}_fullmodel" 2>&1 \
      | tee -a "$OUT/${tag}_merge.log" || { log "ABORT: merged $tag failed verification"; exit 1; }
  else log "skip merge $tag (exists)"; fi
}

phase_bc() {
  log "=== PHASE B/C: two LoRAs + merges ==="
  # Stop every session holding GPU 6,7, not just the name this script uses:
  # a leftover replica from an earlier benchmark silently CPU-offloaded six
  # tau2 LoRAs, which "loaded" in 3s with lm_head on the meta device.
  for sess in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    case "$sess" in *srv_c|*server_c|*_c) tmux kill-session -t "$sess" 2>/dev/null;; esac
  done
  sleep 10
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | paste -sd+ | bc)
  if [[ "${used:-99999}" -gt 2000 ]]; then
    log "ABORT: GPU 6,7 still hold ${used}MiB; LoRA would silently CPU-offload"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tail -5
    exit 1
  fi
  log "GPU 6,7 free (${used}MiB) -- starting LoRA training"
  train_and_merge router_probe
  train_and_merge force_sft
  log "=== PHASE B/C done ==="
}

# --- phase D: evaluations --------------------------------------------------
# eval_one <name> <split> <url> [memory-bank]
eval_one() {
  local name=$1 split=$2 url=$3 bank=${4:-}
  local dir="$OUT/eval_${split}_${name}"
  if [[ -f "$dir/summary.json" ]]; then
    if $PY -c "import json,sys;sys.exit(0 if json.load(open('$dir/summary.json')).get('pass_rate') is not None else 1)"; then
      log "skip eval $split/$name (done)"; return 0
    fi
    log "redoing eval $split/$name (previous run has a null pass_rate)"; rm -rf "$dir"
  fi
  local args=(--split "$split" --output "$dir" --experiment-name "tc_${split}_${name}"
              --env-url "$ENV_URL"
              --max-parallel 6 --max-steps "$EVAL_STEPS" --seed 20260822
              --model "$SERVED_NAME" --base-url "$url")
  if [[ -n "$bank" ]]; then
    # A wrong bank path does not stop the rollout: every task fails with
    # FileNotFoundError and a summary is still written, pass_rate null, which
    # the skip check above would read as "done". Fail here instead.
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $split/$name: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "eval $split/$name"
  $PY -u scripts/run_textcraft_rollout.py "${args[@]}"
}

phase_d() {
  log "=== PHASE D: ten evaluations (5 configs x test/deep) ==="
  local RBANK="$OUT/${DOM}_router_probe/banks/memory_${DOM}.json"
  local FBANK="$OUT/${DOM}_force_memory/banks/memory_${DOM}.json"
  serve tc_srv_a 0,1 8030 "$BASE"

  ( for split in test deep; do
      eval_one routermem "$split" http://127.0.0.1:8030/v1 "$RBANK"
      eval_one forcemem  "$split" http://127.0.0.1:8030/v1 "$FBANK"
    done ) > "$OUT/phaseD_base.log" 2>&1 &
  local pid_base=$!

  ( m="$(backbone_merged tc_${DOM}_router_probe_merged)"
    if [[ -f "$m/config.json" ]]; then
      serve tc_srv_b 4,5 8031 "$m"
      for split in test deep; do
        eval_one routersftonly "$split" http://127.0.0.1:8031/v1
        eval_one routerboth    "$split" http://127.0.0.1:8031/v1 "$RBANK"
      done
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    else log "skip router evals (no merged model)"; fi
  ) > "$OUT/phaseD_router.log" 2>&1 &
  local pid_router=$!

  ( m="$(backbone_merged tc_${DOM}_force_sft_merged)"
    if [[ -f "$m/config.json" ]]; then
      serve tc_srv_c 6,7 8032 "$m"
      for split in test deep; do
        eval_one forcesft "$split" http://127.0.0.1:8032/v1
      done
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    else log "skip force_sft eval (no merged model)"; fi
  ) > "$OUT/phaseD_forcesft.log" 2>&1 &
  local pid_fsft=$!

  local rc=0
  for p in $pid_base $pid_router $pid_fsft; do wait "$p" || rc=1; done
  log "=== PHASE D done (rc=$rc) ==="
}

report() {
  log "=== RESULTS ==="
  $PY - <<'PYR'
import json, os
OUT = os.environ["OUT"]
rows = [("baseline", None), ("router mem", "routermem"), ("force_mem", "forcemem"),
        ("router SFT", "routersftonly"), ("router both", "routerboth"), ("force_sft", "forcesft")]
for split, base_dir, n, label in (("test", "baseline_test80", 80, "same-distribution, depth 1-2"),
                                  ("deep", "baseline_deep127", 127, "compositional, depth 3-4")):
    print(f"\n--- textcraft {split} ({n} tasks: {label}) ---")
    for name, key in rows:
        path = os.path.join(OUT, base_dir if key is None else f"eval_{split}_{key}", "summary.json")
        if not os.path.exists(path):
            print(f"  {name:14s} (missing)"); continue
        x = json.load(open(path))
        by = x.get("by_depth") or {}
        tail = "  " + " ".join(f"d{d}={v['success']}/{v['tasks']}" for d, v in sorted(by.items()))
        print(f"  {name:14s} {x['success']:3d}/{x['tasks']:3d}  pass={x['pass_rate']:.4f}"
              f"  score={x['mean_score']:.4f}  errors={x['errors']}{tail}")
PYR
}

phase_a
phase_bc
phase_d
report
log "=== TEXTCRAFT FULL GRID COMPLETE ==="
