#!/usr/bin/env bash
# End-to-end driver for the SQLGym arm of the memory/SFT ablation grid.
#
# Runs EVERYTHING that is left, in order:
#   A. three build arms   (router / force_memory / force_sft), one per replica
#   B. two LoRA trainings (router-pool, force_sft-pool)
#   C. two merges         (each LoRA re-keyed and merged into a full model)
#   D. ten evaluations    (5 configurations x 2 held-out lines)
#
# Two held-out lines, differing in whether the SCHEMA was seen during pool
# construction:
#   - `test` (80): more BIRD train questions over the SAME 69 databases the
#     pool was built from -- the same-distribution line.
#   - `xdb` (149): BIRD dev, whose 11 databases appear nowhere in train.
#     Cross-schema, and the line that decides whether SFT gains are
#     transferable clause patterns or memorized schemas.
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

BASE=/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
PY=.venv/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=sqlgym_experiment
DOM=sqlgym
BIRD=/nas04/yixuh/bird           # fixed dataset on disk; no env server for this domain
ROLL_PY=/nas04/yixuh/sqlgym_venv/bin/python   # the rollout opens SQLite itself
EVAL_STEPS=15                    # measured, not guessed: see sqlgym_agent.run_task
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
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-sq-$port \
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
  $PY -u scripts/run_sqlgym_router_llm_probe.py --output "$dir" \
      --train-rollout "$OUT/base_train_v1" --router-mode "$mode" \
      --sft-writer "$writer" --base-url "$url" --bird-path "$BIRD" \
    || { log "ABORT: build $arm failed"; return 1; }
  [[ -f "$dir/summary.json" ]] || { log "ABORT: build $arm wrote no summary"; return 1; }
}

phase_a() {
  log "=== PHASE A: three build arms, one per replica ==="
  [[ -f "$OUT/base_train_v1/summary.json" ]] || { log "ABORT: no base_train_v1; run the baselines first"; exit 1; }
  serve sq_srv_a 0,1 8030 "$BASE"
  serve sq_srv_b 4,5 8031 "$BASE"
  serve sq_srv_c 6,7 8032 "$BASE"
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
  local merged="/nas04/yixuh/sq_${tag}_merged"
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
    $ROUTER_PY scripts/rekey_lora_to_full_model.py "$adapter" "${adapter}_fullmodel" 2>&1 | tee -a "$OUT/${tag}_lora.log"
  fi
  if [[ ! -f "$merged/config.json" ]]; then
    log "merge $tag -> $merged"
    HF_HOME=/nas04/yixuh/hf_cache $ROUTER_PY -u scripts/merge_peft_lora.py \
      "$BASE" "${adapter}_fullmodel" "$merged" 2>&1 | tee -a "$OUT/${tag}_merge.log"
    for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
      cp -L "$BASE/$f" "$merged/$f" 2>/dev/null
    done
    $ROUTER_PY - "$BASE" "$merged" <<'PYV' 2>&1 | tee -a "$OUT/${tag}_merge.log"
import json, os, sys
from safetensors.torch import load_file
B, M = sys.argv[1], sys.argv[2]
bi = json.load(open(B + "/model.safetensors.index.json"))["weight_map"]
mi = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
for k, tag in (("model.language_model.layers.0.linear_attn.out_proj.weight", "target"),
               ("model.visual.blocks.0.attn.qkv.weight", "untouched")):
    a = load_file(os.path.join(B, bi[k]))[k].float()
    b = load_file(os.path.join(M, mi[k]))[k].float()
    print(f"  [{tag}] rel={float((a-b).norm()/a.norm()):.2e}")
PYV
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
  local args=(--split "$split" --output "$dir" --experiment-name "sq_${split}_${name}"
              --bird-path "$BIRD"
              --max-parallel 6 --max-steps "$EVAL_STEPS" --seed 20260822
              --model qwen35-tau --base-url "$url")
  if [[ -n "$bank" ]]; then
    # A wrong bank path does not stop the rollout: every task fails with
    # FileNotFoundError and a summary is still written, pass_rate null, which
    # the skip check above would read as "done". Fail here instead.
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $split/$name: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "eval $split/$name"
  $ROLL_PY -u scripts/run_sqlgym_rollout.py "${args[@]}"
}

phase_d() {
  log "=== PHASE D: ten evaluations (5 configs x test/xdb) ==="
  local RBANK="$OUT/${DOM}_router_probe/banks/memory_${DOM}.json"
  local FBANK="$OUT/${DOM}_force_memory/banks/memory_${DOM}.json"
  serve sq_srv_a 0,1 8030 "$BASE"

  ( for split in test xdb; do
      eval_one routermem "$split" http://127.0.0.1:8030/v1 "$RBANK"
      eval_one forcemem  "$split" http://127.0.0.1:8030/v1 "$FBANK"
    done ) > "$OUT/phaseD_base.log" 2>&1 &
  local pid_base=$!

  ( m="/nas04/yixuh/sq_${DOM}_router_probe_merged"
    if [[ -f "$m/config.json" ]]; then
      serve sq_srv_b 4,5 8031 "$m"
      for split in test xdb; do
        eval_one routersftonly "$split" http://127.0.0.1:8031/v1
        eval_one routerboth    "$split" http://127.0.0.1:8031/v1 "$RBANK"
      done
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    else log "skip router evals (no merged model)"; fi
  ) > "$OUT/phaseD_router.log" 2>&1 &
  local pid_router=$!

  ( m="/nas04/yixuh/sq_${DOM}_force_sft_merged"
    if [[ -f "$m/config.json" ]]; then
      serve sq_srv_c 6,7 8032 "$m"
      for split in test xdb; do
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
OUT = "sqlgym_experiment"
rows = [("baseline", None), ("router mem", "routermem"), ("force_mem", "forcemem"),
        ("router SFT", "routersftonly"), ("router both", "routerboth"), ("force_sft", "forcesft")]
for split, base_dir, n, label in (("test", "baseline_test80", 80, "same schemas, unseen questions"),
                                  ("xdb", "baseline_xdb149", 149, "unseen schemas (BIRD dev)")):
    print(f"\n--- sqlgym {split} ({n} tasks: {label}) ---")
    for name, key in rows:
        path = os.path.join(OUT, base_dir if key is None else f"eval_{split}_{key}", "summary.json")
        if not os.path.exists(path):
            print(f"  {name:14s} (missing)"); continue
        x = json.load(open(path))
        by = x.get("by_difficulty") or {}
        tail = "  " + " ".join(f"{d[:4]}={v['success']}/{v['tasks']}" for d, v in by.items())
        print(f"  {name:14s} {x['success']:3d}/{x['tasks']:3d}  pass={x['pass_rate']:.4f}"
              f"  score={x['mean_score']:.4f}  errors={x['errors']}{tail}")
PYR
}

phase_a
phase_bc
phase_d
report
log "=== SQLGYM FULL GRID COMPLETE ==="
