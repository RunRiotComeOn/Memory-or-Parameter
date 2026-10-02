#!/usr/bin/env bash
# End-to-end driver for the tau2 arm of the memory/SFT ablation grid.
#
# Runs EVERYTHING that is left, in order, for all three sub-domains:
#   A. nine build arms      (3 domains x router / force_memory / force_sft)
#   B. six LoRA trainings   (3 domains x router-pool / force_sft-pool)
#   C. six merges           (each LoRA re-keyed and merged into a full model)
#   D. eighteen evaluations (3 domains x 6 configurations)
#
# Every step is SKIPPED if its output already exists, so this can be started
# while phase A is already running from an earlier launch, and re-run after
# any interruption without redoing finished work. That is the point: no
# second script should ever be needed to pick up where this left off.
#
# Resource choreography (GPU 2,3 belong to another user and are never touched):
#   - Phase A: three replicas on the BASE model, one per domain, in parallel.
#   - Phase B/C: replica C is stopped to free GPU 6,7 for LoRA training;
#     merges run on CPU.
#   - Phase D: replica A stays on BASE and runs every base-model config;
#     replicas B and C serve the per-domain merged models and run the rest.
#
# Two invariants this script enforces because both were violated by hand
# earlier in this project:
#   1. `serve` verifies the model actually loaded is the one asked for, and
#      aborts rather than producing data against the wrong weights. A baseline
#      accidentally measured on an SFT model is silent, plausible, and wrong.
#   2. Merged models are deleted as soon as the last eval that needs them is
#      done -- six of them at 66GB each is 396GB, and nothing downstream reads
#      them again.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src

source scripts/lib_backbone.sh   # BASE, SERVED_NAME; MODEL_PROFILE selects the backbone
# Gemma 4's template renders a tool-calling turn differently in history than
# at generation, so its tau2 SFT pools cannot be trained exactly
# (train_agent_sft_lora_peft._spliced_turn refuses). Stop before phase A
# spends hours on builds whose LoRA step would then fail.
[[ "$MODEL_PROFILE" == "gemma4" ]] && { echo "ABORT: tau2 SFT arms are not supported on gemma4"; exit 1; }
PY=.venv/bin/python
TAU2_PY=third_party/tau2-bench/.venv/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=$(backbone_out tau2_experiment); mkdir -p "$OUT"; export OUT
DOMAINS=(retail telecom airline)
KEEP_MERGED="${KEEP_MERGED:-0}"   # set to 1 to keep the 66GB merged models

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- serving ---------------------------------------------------------------
# serve <session> <gpus> <port> <model_path> -- idempotent, and verifies the
# loaded model is the requested one before returning.
serve() {
  local sess=$1 gpus=$2 port=$3 model=$4
  local current
  current=$(curl -s -m 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null \
            | $PY -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null || true)
  if [[ "$current" == "$model" ]]; then log "serve: $port already on $(basename "$model")"; return 0; fi
  # Stop whatever actually holds this port, not just the session name this
  # script would have used. Sessions from earlier benchmarks linger under
  # their own names (tau2_srv_b was still serving 8031 here), so killing
  # "$sess" alone leaves the old vLLM running and the identity check below
  # then aborts on a server that was never replaced. This is the same
  # wrong-session-name bug that silently CPU-offloaded six LoRA trainings
  # on the tau2 run; fixing it only for GPUs was not enough.
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
    log "ABORT: $port still serving after 180s; refusing to start a second server on it"
    exit 1
  fi
  sleep 5
  log "serve: $port <- $(basename "$model") on GPU $gpus"
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpus TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 PORT=$port \
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-t2-$port \
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
  local dom=$1 arm=$2 url=$3 mode=$4 writer=$5
  local dir="$OUT/${dom}_${arm}"
  [[ -f "$dir/summary.json" ]] && { log "skip build $dom/$arm (done)"; return 0; }
  log "build $dom/$arm (mode=$mode writer=$writer)"
  $PY -u scripts/run_tau2_router_llm_probe.py --model "$SERVED_NAME" --domain "$dom" --output "$dir" \
      --train-rollout "$OUT/${dom}_base_train" --router-mode "$mode" \
      --sft-writer "$writer" --base-url "$url"
}

phase_a() {
  log "=== PHASE A: nine build arms ==="
  serve tau2_srv_a 0,1 8030 "$BASE"
  serve tau2_srv_b 4,5 8031 "$BASE"
  serve tau2_srv_c 6,7 8032 "$BASE"
  local pids=()
  local i=0
  for dom in "${DOMAINS[@]}"; do
    local port=$((8030 + i)); i=$((i+1))
    (
      build_arm "$dom" router_probe  "http://127.0.0.1:$port/v1" llm          teacher
      build_arm "$dom" force_memory  "http://127.0.0.1:$port/v1" force_memory none
      build_arm "$dom" force_sft     "http://127.0.0.1:$port/v1" force_sft    teacher
    ) > "$OUT/phaseA_${dom}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  log "=== PHASE A done (rc=$rc) ==="
}

# --- phase B/C: LoRA + merge ----------------------------------------------
train_and_merge() {
  local dom=$1 arm=$2                       # arm: router_probe | force_sft
  local tag="${dom}_${arm}"
  local pool="$OUT/${tag}/sft_pool.jsonl"
  local adapter="$OUT/${tag}_lora/adapter"
  local merged="$(backbone_merged t2_${tag}_merged)"
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
  log "=== PHASE B/C: six LoRAs + merges ==="
  # Free GPU 6,7 for LoRA training. Killing one known session name is not
  # enough: whichever session happens to serve port 8032 may have been
  # started by an earlier benchmark under a different name (it was
  # `ws_server_c`, left over from WebShop, the first time this ran -- the
  # six LoRAs then all "loaded" in 3s with lm_head on the meta device,
  # because accelerate silently offloaded to CPU, and every one of them
  # crashed in the first forward). So stop every session holding those GPUs
  # and verify the memory is actually released before training.
  for sess in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    case "$sess" in *srv_c|*server_c|*_c) tmux kill-session -t "$sess" 2>/dev/null;; esac
  done
  tmux kill-session -t tau2_srv_c 2>/dev/null
  sleep 10
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | paste -sd+ | bc)
  if [[ "${used:-99999}" -gt 2000 ]]; then
    log "ABORT: GPU 6,7 still hold ${used}MiB after stopping their sessions; LoRA would silently CPU-offload"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tail -5
    exit 1
  fi
  log "GPU 6,7 free (${used}MiB) -- starting LoRA training"
  for dom in "${DOMAINS[@]}"; do
    train_and_merge "$dom" router_probe
    train_and_merge "$dom" force_sft
  done
  log "=== PHASE B/C done ==="
}

# --- phase D: evaluations --------------------------------------------------
# eval_one <name> <domain> <url> [memory-bank]
eval_one() {
  local name=$1 dom=$2 url=$3 bank=${4:-}
  local dir="$OUT/eval_${dom}_${name}"
  if [[ -f "$dir/summary.json" ]]; then
    if $PY -c "import json,sys;sys.exit(0 if json.load(open('$dir/summary.json')).get('pass_rate') is not None else 1)"; then
      log "skip eval $dom/$name (done)"; return 0
    fi
    log "redoing eval $dom/$name (previous run has a null pass_rate)"; rm -rf "$dir"
  fi
  local args=(--domain "$dom" --split test --output "$dir"
              --max-parallel 4 --max-steps 200 --seed 20260822
              --model "$SERVED_NAME" --base-url "$url")
  if [[ -n "$bank" ]]; then
    # A wrong bank path does not stop the rollout: every task fails with
    # FileNotFoundError and the run still writes a summary, with pass_rate
    # null and 40 "errors". That looks like a finished eval to the skip
    # check, so the hole stays. Fail here instead. (The group name is
    # `tau2_<domain>`, so the file is memory_tau2_retail.json, not
    # memory_retail.json -- which is exactly how this was discovered.)
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $dom/$name: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "eval $dom/$name"
  $TAU2_PY -u scripts/run_tau2_rollout.py "${args[@]}"
}

phase_d() {
  log "=== PHASE D: eighteen evaluations ==="
  serve tau2_srv_a 0,1 8030 "$BASE"

  # base-model configs for every domain, on replica A
  ( for dom in "${DOMAINS[@]}"; do
      eval_one routermem  "$dom" http://127.0.0.1:8030/v1 "$OUT/${dom}_router_probe/banks/memory_tau2_${dom}.json"
      eval_one forcemem   "$dom" http://127.0.0.1:8030/v1 "$OUT/${dom}_force_memory/banks/memory_tau2_${dom}.json"
    done ) > "$OUT/phaseD_base.log" 2>&1 &
  local pid_base=$!

  # per-domain merged models, on replicas B (router) and C (force_sft)
  ( for dom in "${DOMAINS[@]}"; do
      m="$(backbone_merged t2_${dom}_router_probe_merged)"
      [[ -f "$m/config.json" ]] || { log "skip $dom router evals (no merged model)"; continue; }
      serve tau2_srv_b 4,5 8031 "$m"
      eval_one routersftonly "$dom" http://127.0.0.1:8031/v1
      eval_one routerboth    "$dom" http://127.0.0.1:8031/v1 "$OUT/${dom}_router_probe/banks/memory_tau2_${dom}.json"
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    done ) > "$OUT/phaseD_router.log" 2>&1 &
  local pid_router=$!

  ( for dom in "${DOMAINS[@]}"; do
      m="$(backbone_merged t2_${dom}_force_sft_merged)"
      [[ -f "$m/config.json" ]] || { log "skip $dom force_sft eval (no merged model)"; continue; }
      serve tau2_srv_c 6,7 8032 "$m"
      eval_one forcesft "$dom" http://127.0.0.1:8032/v1
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    done ) > "$OUT/phaseD_forcesft.log" 2>&1 &
  local pid_fsft=$!

  local rc=0
  for p in $pid_base $pid_router $pid_fsft; do wait "$p" || rc=1; done
  log "=== PHASE D done (rc=$rc) ==="
}

# --- report ----------------------------------------------------------------
report() {
  log "=== RESULTS ==="
  $PY - <<'PYR'
import json, glob, os
OUT = os.environ["OUT"]
rows = [("baseline", "{d}_baseline_test"), ("router mem", "eval_{d}_routermem"),
        ("force_mem", "eval_{d}_forcemem"), ("router SFT", "eval_{d}_routersftonly"),
        ("router both", "eval_{d}_routerboth"), ("force_sft", "eval_{d}_forcesft")]
for dom in ("retail", "telecom", "airline"):
    print(f"\n--- {dom} (test split) ---")
    for label, pat in rows:
        p = os.path.join(OUT, pat.format(d=dom), "summary.json")
        if not os.path.exists(p):
            print(f"  {label:14s} (missing)"); continue
        s = json.load(open(p))
        print(f"  {label:14s} {s['success']:3d}/{s['tasks']:3d}  pass={s['pass_rate']:.4f}  errors={s['errors']}")
PYR
}

phase_a
phase_bc
phase_d
report
log "=== TAU2 FULL GRID COMPLETE ==="
