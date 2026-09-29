#!/usr/bin/env bash
# End-to-end driver for the BabyAI arm of the memory/SFT ablation grid.
#
# Runs EVERYTHING that is left, in order. BabyAI is a single domain, so the
# parallelism that tau2 got from its three sub-domains comes here from the
# three build arms instead: one arm per serving replica.
#   0. the held-out baseline rollout (80 test tasks), if not already there
#   A. three build arms     (router / force_memory / force_sft), one per replica
#   B. two LoRA trainings   (router-pool, force_sft-pool)
#   C. two merges           (each LoRA re-keyed and merged into a full model)
#   D. five evaluations     (the other five cells of the six-point grid)
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

BASE=/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
PY=.venv/bin/python
BENCH_PY=.venv/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=babyai_experiment
DOM=babyai
ENV_URL=http://127.0.0.1:36001   # AgentGym babyai env server (tmux: babyai_env)
# The held-out set is every task with layout seed 20-21, i.e. all forty levels
# twice over. The earlier `baseline_test57` covered only 57 of those 80 for no
# recorded reason, so phase 0 re-measures the baseline on the full split and
# every arm is scored on exactly the same 80 tasks.
EVAL_STEPS=15                    # agent turns; must equal the baseline's
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
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-ba-$port \
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

# --- phase 0: held-out baseline -------------------------------------------
phase_0() {
  local dir="$OUT/baseline_test80"
  [[ -f "$dir/summary.json" ]] && { log "skip baseline (done)"; return 0; }
  log "=== PHASE 0: baseline on the full 80-task test split ==="
  serve ba_srv_a 0,1 8030 "$BASE"
  $BENCH_PY -u scripts/run_babyai_rollout.py --split test --output "$dir" \
      --experiment-name babyai_baseline_test80 --env-url "$ENV_URL" \
      --max-parallel 4 --max-steps "$EVAL_STEPS" --seed 20260822 \
      --model qwen35-tau --base-url http://127.0.0.1:8030/v1 \
      > "$OUT/baseline_test80.log" 2>&1 \
    || { log "ABORT: baseline rollout failed (see $OUT/baseline_test80.log)"; exit 1; }
  log "=== PHASE 0 done ==="
}

# --- phase A: builds -------------------------------------------------------
build_arm() {
  local dom=$1 arm=$2 url=$3 mode=$4 writer=$5
  local dir="$OUT/${dom}_${arm}"
  [[ -f "$dir/summary.json" ]] && { log "skip build $dom/$arm (done)"; return 0; }
  log "build $dom/$arm (mode=$mode writer=$writer)"
  # No --domain: BabyAI is a single domain and its probe script has no such
  # flag (tau2's does, and the leftover argument silently failed all three
  # arms here with an argparse usage error).
  $PY -u scripts/run_babyai_router_llm_probe.py --output "$dir" \
      --train-rollout "$OUT/base_train_v1" --router-mode "$mode" \
      --sft-writer "$writer" --base-url "$url" --env-url "$ENV_URL" || {
    log "ABORT: build $dom/$arm failed"; return 1; }
  [[ -f "$dir/summary.json" ]] || { log "ABORT: build $dom/$arm wrote no summary"; return 1; }
}

phase_a() {
  log "=== PHASE A: three build arms, one per replica ==="
  serve ba_srv_a 0,1 8030 "$BASE"
  serve ba_srv_b 4,5 8031 "$BASE"
  serve ba_srv_c 6,7 8032 "$BASE"
  local pids=()
  # tau2 parallelised by sub-domain and ran its three arms sequentially on one
  # replica. There is only one domain here, so the arms take that place: each
  # gets its own replica and all three run at once. They are independent --
  # different router mode, different output dir, same read-only train rollout.
  local arms=("router_probe 8030 llm teacher"
              "force_memory 8031 force_memory none"
              "force_sft    8032 force_sft teacher")
  local spec
  for spec in "${arms[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    build_arm "$DOM" "$1" "http://127.0.0.1:$2/v1" "$3" "$4" > "$OUT/phaseA_$1.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  log "=== PHASE A done (rc=$rc) ==="
  if [[ "$rc" != "0" ]]; then
    # Continuing past a failed build arm is what produced a "COMPLETE" run
    # with every LoRA skipped and every eval missing: the downstream skips
    # are all conditioned on artifacts that a failed arm never wrote, so
    # nothing errors, it just does nothing. Stop here instead.
    log "ABORT: phase A had failures; not continuing to B/C/D (see $OUT/phaseA_*.log)"
    exit 1
  fi
}

# --- phase B/C: LoRA + merge ----------------------------------------------
train_and_merge() {
  local dom=$1 arm=$2                       # arm: router_probe | force_sft
  local tag="${dom}_${arm}"
  local pool="$OUT/${tag}/sft_pool.jsonl"
  local adapter="$OUT/${tag}_lora/adapter"
  local merged="/nas04/yixuh/ba_${tag}_merged"
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
  tmux kill-session -t ba_srv_c 2>/dev/null
  sleep 10
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | paste -sd+ | bc)
  if [[ "${used:-99999}" -gt 2000 ]]; then
    log "ABORT: GPU 6,7 still hold ${used}MiB after stopping their sessions; LoRA would silently CPU-offload"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tail -5
    exit 1
  fi
  log "GPU 6,7 free (${used}MiB) -- starting LoRA training"
  train_and_merge "$DOM" router_probe
  train_and_merge "$DOM" force_sft
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
  # run_babyai_rollout.py has no --domain (BabyAI is one domain) and REQUIRES
  # --experiment-name; the tau2 argument list left here failed all five evals
  # instantly with an argparse error. --max-steps must stay equal to the
  # baseline's: it counts agent turns, two thirds of the baseline's failures
  # are `max_steps`, and raising it for one arm alone would hand that arm the
  # difference.
  local args=(--split test --output "$dir" --experiment-name "babyai_${name}"
              --env-url "$ENV_URL"
              --max-parallel 4 --max-steps "$EVAL_STEPS" --seed 20260822
              --model qwen35-tau --base-url "$url")
  if [[ -n "$bank" ]]; then
    # A wrong bank path does not stop the rollout: every task fails with
    # FileNotFoundError and the run still writes a summary, with pass_rate
    # null and 40 "errors". That looks like a finished eval to the skip
    # check, so the hole stays. Fail here instead. (The group name is
    # `babyai`, so the file is memory_babyai.json. On tau2 the equivalent
    # mistake silently produced nine evals of 40 errored tasks each.)
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $dom/$name: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "eval $dom/$name"
  $BENCH_PY -u scripts/run_babyai_rollout.py "${args[@]}"
}

phase_d() {
  log "=== PHASE D: five evaluations ==="
  serve ba_srv_a 0,1 8030 "$BASE"

  # base-model configs for every domain, on replica A
  ( for dom in "$DOM"; do
      eval_one routermem  "$dom" http://127.0.0.1:8030/v1 "$OUT/${dom}_router_probe/banks/memory_babyai.json"
      eval_one forcemem   "$dom" http://127.0.0.1:8030/v1 "$OUT/${dom}_force_memory/banks/memory_babyai.json"
    done ) > "$OUT/phaseD_base.log" 2>&1 &
  local pid_base=$!

  # per-domain merged models, on replicas B (router) and C (force_sft)
  ( for dom in "$DOM"; do
      m="/nas04/yixuh/ba_${dom}_router_probe_merged"
      [[ -f "$m/config.json" ]] || { log "skip $dom router evals (no merged model)"; continue; }
      serve ba_srv_b 4,5 8031 "$m"
      eval_one routersftonly "$dom" http://127.0.0.1:8031/v1
      eval_one routerboth    "$dom" http://127.0.0.1:8031/v1 "$OUT/${dom}_router_probe/banks/memory_babyai.json"
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $m"; rm -rf "$m"; }
    done ) > "$OUT/phaseD_router.log" 2>&1 &
  local pid_router=$!

  ( for dom in "$DOM"; do
      m="/nas04/yixuh/ba_${dom}_force_sft_merged"
      [[ -f "$m/config.json" ]] || { log "skip $dom force_sft eval (no merged model)"; continue; }
      serve ba_srv_c 6,7 8032 "$m"
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
import json, os
OUT = "babyai_experiment"
rows = [("baseline", "baseline_test80"), ("router mem", "eval_babyai_routermem"),
        ("force_mem", "eval_babyai_forcemem"), ("router SFT", "eval_babyai_routersftonly"),
        ("router both", "eval_babyai_routerboth"), ("force_sft", "eval_babyai_forcesft")]
print("\n--- babyai (test split, 80 tasks: layout seeds 20-21, all 40 levels x2) ---")
for label, d in rows:
    path = os.path.join(OUT, d, "summary.json")
    if not os.path.exists(path):
        print(f"  {label:14s} (missing)"); continue
    x = json.load(open(path))
    ms = x.get("mean_score")
    tail = f"  score={ms:.4f}" if isinstance(ms, (int, float)) else ""
    print(f"  {label:14s} {x['success']:3d}/{x['tasks']:3d}  pass={x['pass_rate']:.4f}{tail}  errors={x['errors']}")
PYR
}

phase_0
phase_a
phase_bc
phase_d
report
log "=== BABYAI FULL GRID COMPLETE ==="
