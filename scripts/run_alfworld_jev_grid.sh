#!/usr/bin/env bash
# Re-run the ALFWorld router arm with TypeSafe's Jev deciding the route.
#
# WHAT THIS DOES AND DOES NOT RE-RUN
#
# `alfworld_summary.md` records six configurations. Only the router arm
# depends on which router is used:
#
#   re-run here          baseline       reused as-is
#   -----------          --------       ------------
#   jev router bank      base_train_v2              (no router involved)
#   jev SFT LoRA+merge   baseline_valid_unseen_v1   (no router involved)
#   6 evals              router_force_memory_v1     (router_mode bypasses
#                                                     decide_route entirely)
#                        router_force_sft_v1        (same)
#
# The two forced arms take the `force_memory` / `force_sft` branches in
# `router_bank_builder`, which never call the router at all, so swapping the
# router cannot change them. The summary also records that ALFWorld rollouts
# are per-task deterministic, so re-running them would reproduce the same
# numbers at the cost of many GPU-hours. They are reused, and the comparison
# against them stays valid because every eval here uses the same serving
# replica, the same seed and the same task ids.
#
# The six evals are {memory-only, sft-only, both} x {train200, unseen57},
# matching the v4 layout:
#   freeze_replay_v4_200 / eval_valid_unseen57 / routerv4_{sftonly,both}_{train200,unseen57}
#
# Task ids come from `train200_ids.txt` / `unseen57_ids.txt`, extracted from
# the existing trajectory directories rather than re-derived with --limit, so
# the jev arm is scored on exactly the tasks the v4 arm was.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src

BASE=/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
PY=.venv/bin/python
ROLL_PY=/nas04/yixuh/alfworld_venv310/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=alfworld_experiment
ARM=router_jev_v1
BANK="$OUT/$ARM/banks/memory_alfworld.json"
MERGED=/nas04/yixuh/jev_v1_merged
TRAIN_IDS=$OUT/train200_ids.txt
UNSEEN_IDS=$OUT/unseen57_ids.txt
export ALFWORLD_DATA=/nas04/yixuh/alfworld_data
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
  if [[ -n "${holder:-}" ]]; then log "serve: port $port held by pid $holder; stopping it"; kill "$holder" 2>/dev/null; fi
  for _ in $(seq 90); do curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null || break; sleep 2; done
  if curl -s -m 3 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null; then
    log "ABORT: $port still serving after 180s"; exit 1
  fi
  sleep 5
  log "serve: $port <- $(basename "$model") on GPU $gpus"
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpus TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 PORT=$port \
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-jev-$port \
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

# --- phase A: build the jev bank + sft pool --------------------------------
phase_a() {
  log "=== PHASE A: jev router bank over the 200-task base pool ==="
  if [[ -f "$OUT/$ARM/summary.json" ]]; then log "skip phase A (done)"; return 0; fi
  [[ -f "$OUT/base_train_v2/summary.json" ]] || { log "ABORT: base_train_v2 missing"; exit 1; }
  [[ -s /nas04/yixuh/.config/continual-memory/jev_api_key ]] || { log "ABORT: no jev api key"; exit 1; }
  serve jev_srv_a 0,1 8030 "$BASE"
  $PY -u scripts/run_alfworld_router_llm_probe.py \
      --output "$OUT/$ARM" --train-rollout "$OUT/base_train_v2" \
      --router-mode jev --sft-writer teacher \
      --base-url http://127.0.0.1:8030/v1 \
    || { log "ABORT: jev bank build failed"; exit 1; }
  [[ -f "$OUT/$ARM/summary.json" ]] || { log "ABORT: bank build wrote no summary"; exit 1; }
  log "=== PHASE A done ==="
}

# --- phase B/C: LoRA + merge -----------------------------------------------
phase_bc() {
  log "=== PHASE B/C: jev SFT LoRA + merge ==="
  local pool="$OUT/$ARM/sft_pool.jsonl"
  local adapter="$OUT/${ARM}_lora/adapter"
  if [[ ! -s "$pool" ]]; then
    log "jev sft pool is empty -- no SFT arm for this router; phase D will run memory-only"
    return 0
  fi
  local n; n=$(wc -l < "$pool")
  # Free GPU 6,7 for training: stop every session holding them, not just the
  # name this script uses. A leftover replica silently CPU-offloaded six tau2
  # LoRAs once ("loaded in 3.2s", lm_head on meta).
  for sess in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    case "$sess" in *srv_c|*server_c|*_c) tmux kill-session -t "$sess" 2>/dev/null;; esac
  done
  sleep 10
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | paste -sd+ | bc)
  if [[ "${used:-99999}" -gt 2000 ]]; then
    log "ABORT: GPU 6,7 still hold ${used}MiB; LoRA would silently CPU-offload"; exit 1
  fi
  if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    log "train LoRA ($n examples)"
    CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/nas04/yixuh/hf_cache \
      $ROUTER_PY -u scripts/train_agent_sft_lora_peft.py --pool "$pool" --output "$adapter" \
        2>&1 | tee -a "$OUT/${ARM}_lora.log"
  else log "skip LoRA (adapter exists)"; fi
  [[ -f "$adapter/adapter_model.safetensors" ]] || { log "ABORT: no adapter"; exit 1; }
  [[ -f "${adapter}_fullmodel/adapter_model.safetensors" ]] || \
    $ROUTER_PY scripts/rekey_lora_to_full_model.py "$adapter" "${adapter}_fullmodel" 2>&1 | tee -a "$OUT/${ARM}_lora.log"
  if [[ ! -f "$MERGED/config.json" ]]; then
    log "merge -> $MERGED"
    HF_HOME=/nas04/yixuh/hf_cache $ROUTER_PY -u scripts/merge_peft_lora.py \
      "$BASE" "${adapter}_fullmodel" "$MERGED" 2>&1 | tee -a "$OUT/${ARM}_merge.log"
    for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
      cp -L "$BASE/$f" "$MERGED/$f" 2>/dev/null
    done
    $ROUTER_PY - "$BASE" "$MERGED" <<'PYV' 2>&1 | tee -a "$OUT/${ARM}_merge.log"
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
  else log "skip merge (exists)"; fi
  log "=== PHASE B/C done ==="
}

# --- phase D: six evaluations ----------------------------------------------
# eval_one <name> <split> <ids-file> <url> [memory-bank]
eval_one() {
  local name=$1 split=$2 ids=$3 url=$4 bank=${5:-}
  local dir="$OUT/jev_$name"
  if [[ -f "$dir/summary.json" ]]; then
    if $PY -c "import json,sys;sys.exit(0 if json.load(open('$dir/summary.json')).get('pass_rate') is not None else 1)"; then
      log "skip eval $name (done)"; return 0
    fi
    log "redoing eval $name (null pass_rate)"; rm -rf "$dir"
  fi
  local args=(--split "$split" --output "$dir" --experiment-name "jev_$name"
              --task-ids $(cat "$ids")
              --max-parallel 4 --max-steps 40 --seed 20260822
              --model qwen35-tau --base-url "$url")
  if [[ -n "$bank" ]]; then
    # A missing bank path does not stop the rollout: every task errors and a
    # summary is still written with pass_rate null, which the skip check
    # above would read as done. Fail here instead.
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $name: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "eval $name ($split)"
  $ROLL_PY -u scripts/run_alfworld_rollout.py "${args[@]}"
}

phase_d() {
  log "=== PHASE D: six evaluations ==="
  serve jev_srv_a 0,1 8030 "$BASE"

  # base model + jev bank  (memory-only)
  ( eval_one memonly_train200  train        "$TRAIN_IDS"  http://127.0.0.1:8030/v1 "$BANK"
    eval_one memonly_unseen57  valid_unseen "$UNSEEN_IDS" http://127.0.0.1:8030/v1 "$BANK"
  ) > "$OUT/jev_phaseD_base.log" 2>&1 &
  local pid_base=$!

  # merged model: sft-only (no bank) and both (bank + weights)
  ( if [[ -f "$MERGED/config.json" ]]; then
      serve jev_srv_b 4,5 8031 "$MERGED"
      eval_one sftonly_train200 train        "$TRAIN_IDS"  http://127.0.0.1:8031/v1
      eval_one sftonly_unseen57 valid_unseen "$UNSEEN_IDS" http://127.0.0.1:8031/v1
      eval_one both_train200    train        "$TRAIN_IDS"  http://127.0.0.1:8031/v1 "$BANK"
      eval_one both_unseen57    valid_unseen "$UNSEEN_IDS" http://127.0.0.1:8031/v1 "$BANK"
      [[ "$KEEP_MERGED" == "1" ]] || { log "rm $MERGED"; rm -rf "$MERGED"; }
    else log "skip sft/both evals (no merged model -- empty sft pool)"; fi
  ) > "$OUT/jev_phaseD_sft.log" 2>&1 &
  local pid_sft=$!

  local rc=0
  for p in $pid_base $pid_sft; do wait "$p" || rc=1; done
  log "=== PHASE D done (rc=$rc) ==="
}

report() {
  log "=== RESULTS: jev router vs the v4 llm router ==="
  $PY - <<'PYR'
import json, os
OUT = "alfworld_experiment"
def read(d):
    p = os.path.join(OUT, d, "summary.json")
    if not os.path.exists(p): return None
    x = json.load(open(p))
    return x["success"], x["tasks"], x["pass_rate"], x["errors"]
rows = [
    ("baseline",            "base_train_v2",             "baseline_valid_unseen_v1"),
    ("v4 llm  memory-only", "freeze_replay_v4_200",      "router_llm_probe_v4/eval_valid_unseen57"),
    ("v4 llm  sft-only",    "routerv4_sftonly_train200", "routerv4_sftonly_unseen57"),
    ("v4 llm  both",        "routerv4_both_train200",    "routerv4_both_unseen57"),
    ("JEV     memory-only", "jev_memonly_train200",      "jev_memonly_unseen57"),
    ("JEV     sft-only",    "jev_sftonly_train200",      "jev_sftonly_unseen57"),
    ("JEV     both",        "jev_both_train200",         "jev_both_unseen57"),
]
print(f"\n{'config':22s} {'train200':>20s} {'unseen57':>20s}")
for label, dtrain, dunseen in rows:
    cells = []
    for d in (dtrain, dunseen):
        r = read(d)
        cells.append("(missing)".rjust(20) if r is None
                     else f"{r[0]:3d}/{r[1]:<3d} {r[2]:.4f} e{r[3]}".rjust(20))
    print(f"{label:22s} {cells[0]} {cells[1]}")
PYR
}

phase_a
phase_bc
phase_d
report
log "=== ALFWORLD JEV GRID COMPLETE ==="
