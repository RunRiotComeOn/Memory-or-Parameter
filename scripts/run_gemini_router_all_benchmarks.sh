#!/usr/bin/env bash
# Run the gemini-flash router arm on five benchmarks, in sequence.
#
#   babyai -> scienceworld -> sqlgym -> textcraft -> webshop
#
# ALFWorld is NOT here: it has its own driver (run_alfworld_gemini_grid.sh)
# because its artifact names predate the later naming convention.
#
# WHAT EACH BENCHMARK GETS
#
# Only the router arm depends on which router is used, so for each benchmark
# this builds the gemini bank + SFT pool, trains and merges its LoRA, and runs
# the evals. Baselines and the force_memory / force_sft arms are reused: those
# take branches in `router_bank_builder` that never call the router at all, so
# swapping the router cannot change them.
#
# GPU DISCIPLINE
#
# Cards are leased, never seized. `gpu_lease.sh` picks a pair whose memory is
# free, waits if none is, and refuses GPU 2,3 outright (another user's). No
# step kills a session or a port holder to make room. This matters because
# earlier versions of these scripts hard-coded pairs and killed port holders,
# and came within one phase of evicting a neighbour's job twice in one session.
#
# RESUMABILITY
#
# Every step is skipped when its output exists, so this can be stopped and
# restarted. The one exception is a partial bank build: that is stateful (the
# bank evolves task by task) so a half-finished arm directory is removed and
# rebuilt rather than resumed, which would otherwise produce a bank no clean
# run could reproduce.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src
source scripts/gpu_lease.sh

BASE=/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
PY=.venv/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
KEEP_MERGED="${KEEP_MERGED:-0}"
ONLY="${ONLY:-}"          # e.g. ONLY="sqlgym textcraft" to run a subset
LOG=gemini_all_benchmarks.log

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }

# benchmark | experiment dir | train pool | rollout interpreter | eval lines
#   eval line format: name:split[:ids-file]   (ids-file optional)
bench_spec() {
  case "$1" in
    babyai)       echo "babyai_experiment|base_train_v1|.venv/bin/python|test:test";;
    scienceworld) echo "scienceworld_experiment|base_train_v1|/nas04/yixuh/scienceworld_venv/bin/python|test:test";;
    sqlgym)       echo "sqlgym_experiment|base_train_v1|/nas04/yixuh/sqlgym_venv/bin/python|test:test xdb:xdb";;
    textcraft)    echo "textcraft_experiment|base_train_v1|.venv/bin/python|test:test deep:deep";;
    webshop)      echo "webshop_experiment|base_train_v1|.venv/bin/python|test:test";;
    *) return 1;;
  esac
}

# --- serving, leased ---------------------------------------------------------
SERVE_SESS=""; SERVE_PORT=""; SERVE_PAIR=""
serve_leased() {   # serve_leased <model-path> <label>
  local model=$1 label=$2 pair port current
  # Already serving this model somewhere this script started? Reuse it.
  if [[ -n "$SERVE_PORT" ]]; then
    current=$(curl -s -m 5 "http://127.0.0.1:$SERVE_PORT/v1/models" 2>/dev/null \
              | $PY -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null || true)
    [[ "$current" == "$model" ]] && { log "serve: reusing :$SERVE_PORT ($label)"; return 0; }
  fi
  pair=$(gpu_wait_for_pair 7200) || { log "ABORT: no free GPU pair in 2h"; return 1; }
  gpu_assert_allowed "$pair" || return 1
  port=$(gpu_port_for_pair "$pair")
  local sess="gemall_${pair//,/}"
  # Only ever stop a session this script itself started.
  [[ -n "$SERVE_SESS" ]] && tmux kill-session -t "$SERVE_SESS" 2>/dev/null
  sleep 5
  log "serve: :$port <- $label on GPU $pair"
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$pair TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 PORT=$port \
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-gemall-$port \
     scripts/serve_appworld_deterministic.sh 2>&1 | tee -a gemini_all_serve_$port.log"
  local waited=0
  until curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200; do
    sleep 30; waited=$((waited+30))
    [[ $waited -gt 3600 ]] && { log "ABORT: :$port did not come up in 60min"; return 1; }
    tmux has-session -t "$sess" 2>/dev/null || { log "ABORT: $sess died while loading"; return 1; }
  done
  current=$(curl -s -m 5 "http://127.0.0.1:$port/v1/models" | $PY -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["root"])')
  [[ "$current" == "$model" ]] || { log "ABORT: :$port serves $current, expected $model"; return 1; }
  SERVE_SESS="$sess"; SERVE_PORT="$port"; SERVE_PAIR="$pair"
  log "serve: :$port ready ($label)"
}

# --- one benchmark -----------------------------------------------------------
run_bench() {
  local b=$1 spec exp pool_dir roll_py lines
  spec=$(bench_spec "$b") || { log "unknown benchmark $b"; return 1; }
  IFS='|' read -r exp pool_dir roll_py lines <<<"$spec"
  local arm="$exp/router_gemini_v1"
  local bank="$arm/banks/memory_${b}.json"
  local adapter="$exp/router_gemini_v1_lora/adapter"
  local merged="/nas04/yixuh/gem_${b}_merged"

  log "########## $b ##########"
  [[ -f "$exp/$pool_dir/summary.json" ]] || { log "SKIP $b: no $pool_dir (base pool missing)"; return 0; }

  # ---- phase A: bank + sft pool
  if [[ -f "$arm/summary.json" ]]; then
    log "$b: skip bank build (done)"
  else
    [[ -d "$arm" ]] && { log "$b: removing partial bank build (stateful, cannot resume)"; rm -rf "$arm"; }
    serve_leased "$BASE" "base" || return 1
    log "$b: building gemini bank over $pool_dir"
    $PY -u "scripts/run_${b}_router_llm_probe.py" \
        --output "$arm" --train-rollout "$exp/$pool_dir" \
        --router-mode gemini --sft-writer teacher \
        --base-url "http://127.0.0.1:$SERVE_PORT/v1" \
      > "$exp/gemini_phaseA.log" 2>&1 \
      || { log "$b: ABORT bank build (see $exp/gemini_phaseA.log)"; return 1; }
    [[ -f "$arm/summary.json" ]] || { log "$b: ABORT bank build wrote no summary"; return 1; }
    log "$b: bank build done -- $(grep -aoE 'route_counts=\{[^}]*\}|active_entries=[0-9]+' "$exp/gemini_phaseA.log" | tail -2 | tr '\n' ' ')"
  fi

  # ---- phase B/C: LoRA + merge (only if the router produced SFT data)
  local have_sft=0
  if [[ -s "$arm/sft_pool.jsonl" ]]; then
    have_sft=1
    local n; n=$(wc -l < "$arm/sft_pool.jsonl")
    if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
      local tpair; tpair=$(gpu_wait_for_pair 7200) || { log "$b: ABORT no free GPU for LoRA"; return 1; }
      log "$b: training LoRA ($n examples) on GPU $tpair"
      CUDA_VISIBLE_DEVICES=$tpair HF_HOME=/nas04/yixuh/hf_cache \
        $ROUTER_PY -u scripts/train_agent_sft_lora_peft.py --pool "$arm/sft_pool.jsonl" --output "$adapter" \
          >> "$exp/gemini_lora.log" 2>&1 \
        || { log "$b: ABORT LoRA training"; return 1; }
    else log "$b: skip LoRA (adapter exists)"; fi
    [[ -f "${adapter}_fullmodel/adapter_model.safetensors" ]] || \
      $ROUTER_PY scripts/rekey_lora_to_full_model.py "$adapter" "${adapter}_fullmodel" >> "$exp/gemini_lora.log" 2>&1
    if [[ ! -f "$merged/config.json" ]]; then
      log "$b: merging -> $merged"
      HF_HOME=/nas04/yixuh/hf_cache $ROUTER_PY -u scripts/merge_peft_lora.py \
        "$BASE" "${adapter}_fullmodel" "$merged" >> "$exp/gemini_merge.log" 2>&1 \
        || { log "$b: ABORT merge"; return 1; }
      for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
        cp -L "$BASE/$f" "$merged/$f" 2>/dev/null
      done
    else log "$b: skip merge (exists)"; fi
  else
    log "$b: sft pool empty -- memory-only arm (no LoRA, no sft/both evals)"
  fi

  # ---- phase D: evals
  # Base-model configs first so the one leased replica serves BASE once.
  serve_leased "$BASE" "base" || return 1
  local line name split
  for line in $lines; do
    name=${line%%:*}; split=${line##*:}
    eval_one "$b" "$exp" "memonly_$name" "$split" "$bank" || return 1
  done
  if [[ "$have_sft" == "1" && -f "$merged/config.json" ]]; then
    serve_leased "$merged" "$b merged" || return 1
    for line in $lines; do
      name=${line%%:*}; split=${line##*:}
      eval_one "$b" "$exp" "sftonly_$name" "$split" "" || return 1
      eval_one "$b" "$exp" "both_$name"    "$split" "$bank" || return 1
    done
    [[ "$KEEP_MERGED" == "1" ]] || { log "$b: rm $merged"; rm -rf "$merged"; }
  fi
  log "########## $b complete ##########"
}

# eval_one <bench> <expdir> <name> <split> <bank-or-empty>
eval_one() {
  local b=$1 exp=$2 name=$3 split=$4 bank=$5
  local dir="$exp/gem_$name"
  if [[ -f "$dir/summary.json" ]]; then
    if $PY -c "import json,sys;sys.exit(0 if json.load(open('$dir/summary.json')).get('pass_rate') is not None else 1)" 2>/dev/null; then
      log "$b: skip eval $name (done)"; return 0
    fi
    log "$b: redoing eval $name (null pass_rate)"; rm -rf "$dir"
  fi
  local spec roll_py
  spec=$(bench_spec "$b"); IFS='|' read -r _ _ roll_py _ <<<"$spec"
  local args=(--split "$split" --output "$dir" --experiment-name "gem_$name"
              --max-parallel 4 --seed 20260822 --model qwen35-tau
              --base-url "http://127.0.0.1:$SERVE_PORT/v1")
  if [[ -n "$bank" ]]; then
    # A missing bank does not stop a rollout: every task errors and a summary
    # is still written with pass_rate null, which the skip check above would
    # read as done. Fail here instead.
    [[ -f "$bank" ]] || { log "$b: ABORT bank not found for $name: $bank"; return 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "$b: eval $name (split=$split)"
  $roll_py -u "scripts/run_${b}_rollout.py" "${args[@]}" > "$exp/gemini_eval_$name.log" 2>&1 \
    || { log "$b: ABORT eval $name (see $exp/gemini_eval_$name.log)"; return 1; }
}

# --- main --------------------------------------------------------------------
BENCHES="${ONLY:-babyai scienceworld sqlgym textcraft webshop}"
log "=== GEMINI ROUTER SWEEP: $BENCHES ==="
rc=0
for b in $BENCHES; do
  run_bench "$b" || { log "!!! $b FAILED -- continuing with the next benchmark"; rc=1; }
done
[[ -n "$SERVE_SESS" ]] && tmux kill-session -t "$SERVE_SESS" 2>/dev/null
log "=== GEMINI ROUTER SWEEP COMPLETE (rc=$rc) ==="
exit $rc
