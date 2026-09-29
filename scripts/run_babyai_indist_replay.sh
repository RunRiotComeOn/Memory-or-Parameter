#!/usr/bin/env bash
# In-distribution replay for the BabyAI grid: the same five frozen
# configurations re-scored on the 200 TRAIN tasks their pools were built from.
#
# This is the section `alfworld_summary.md`, `scienceworld_summary.md` and
# `webshop_summary.md` all carry and that `run_babyai_full_grid.sh` does not
# produce: held-out numbers alone cannot say how much of a config's gain is
# recall of its own training examples. It matters more here than anywhere
# else so far, because the force_sft LoRA drove its 92-example pool to
# loss 0.0000 -- exactly the condition under which a train-pool score is
# inflated -- while the router LoRA saw only 7 examples.
#
# The 200 ids come from `train200_ids.txt` (written from base_train_v1), not
# from `split_task_ids("train")`, which returns all 800 seeds-0-19 tasks. The
# comparison point is base_train_v1 itself: pass 0.3700, score 0.3425.
#
# The merged models were deleted by the grid script once its evals were done,
# so phase M rebuilds them from the surviving re-keyed adapters and phase R
# deletes them again. GPU 2,3 belong to another user and are never touched.
set -uo pipefail
cd /nas04/yixuh/memory
export PYTHONPATH=src

BASE=/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
PY=.venv/bin/python
ROUTER_PY=/nas04/yixuh/router_venv/bin/python
OUT=babyai_experiment
ENV_URL=http://127.0.0.1:36001
IDS=$OUT/train200_ids.txt
EVAL_STEPS=15                    # identical to base_train_v1 and to phase D
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
     MODEL_PATH=$model TRITON_CACHE_DIR=/tmp/det-bai-$port \
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

# --- phase M: rebuild the two merged models --------------------------------
remerge() {
  local tag=$1
  local adapter="$OUT/${tag}_lora/adapter_fullmodel"
  local merged="/nas04/yixuh/ba_${tag}_merged"
  [[ -f "$merged/config.json" ]] && { log "skip merge $tag (exists)"; return 0; }
  [[ -f "$adapter/adapter_model.safetensors" ]] || { log "ABORT: no re-keyed adapter for $tag"; exit 1; }
  log "merge $tag -> $merged"
  HF_HOME=/nas04/yixuh/hf_cache $ROUTER_PY -u scripts/merge_peft_lora.py \
    "$BASE" "$adapter" "$merged" 2>&1 | tee -a "$OUT/${tag}_remerge.log"
  for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
    cp -L "$BASE/$f" "$merged/$f" 2>/dev/null
  done
  # Same two-tensor check the grid script runs: the target layer must have
  # moved and the vision tower must not have. A merge that silently no-ops
  # would make the sft arms read as "no leakage" for the wrong reason.
  $ROUTER_PY - "$BASE" "$merged" <<'PYV' 2>&1 | tee -a "$OUT/${tag}_remerge.log"
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
  [[ -f "$merged/config.json" ]] || { log "ABORT: merge $tag produced no config.json"; exit 1; }
}

phase_m() {
  log "=== PHASE M: rebuild both merged models ==="
  remerge babyai_router_probe
  remerge babyai_force_sft
  log "=== PHASE M done ==="
}

# --- phase R: replay the 200 train tasks -----------------------------------
replay_one() {
  local name=$1 url=$2 bank=${3:-}
  local dir="$OUT/indist_${name}"
  if [[ -f "$dir/summary.json" ]]; then
    if $PY -c "import json,sys;sys.exit(0 if json.load(open('$dir/summary.json')).get('pass_rate') is not None else 1)"; then
      log "skip indist $name (done)"; return 0
    fi
    log "redoing indist $name (previous run has a null pass_rate)"; rm -rf "$dir"
  fi
  local args=(--split train --task-ids-file "$IDS" --output "$dir"
              --experiment-name "babyai_indist_${name}" --env-url "$ENV_URL"
              --max-parallel 4 --max-steps "$EVAL_STEPS" --seed 20260822
              --model qwen35-tau --base-url "$url")
  if [[ -n "$bank" ]]; then
    [[ -f "$bank" ]] || { log "ABORT: bank not found for $name: $bank"; exit 1; }
    args+=(--memory-bank "$bank" --memory-top-k 3)
  fi
  log "indist $name"
  $PY -u scripts/run_babyai_rollout.py "${args[@]}"
}

phase_r() {
  log "=== PHASE R: five in-distribution replays (200 train tasks each) ==="
  serve bai_srv_a 0,1 8030 "$BASE"

  ( replay_one routermem http://127.0.0.1:8030/v1 "$OUT/babyai_router_probe/banks/memory_babyai.json"
    replay_one forcemem  http://127.0.0.1:8030/v1 "$OUT/babyai_force_memory/banks/memory_babyai.json"
  ) > "$OUT/indist_base.log" 2>&1 &
  local pid_base=$!

  ( serve bai_srv_b 4,5 8031 /nas04/yixuh/ba_babyai_router_probe_merged
    replay_one routersftonly http://127.0.0.1:8031/v1
    replay_one routerboth    http://127.0.0.1:8031/v1 "$OUT/babyai_router_probe/banks/memory_babyai.json"
    [[ "$KEEP_MERGED" == "1" ]] || { log "rm router merged"; rm -rf /nas04/yixuh/ba_babyai_router_probe_merged; }
  ) > "$OUT/indist_router.log" 2>&1 &
  local pid_router=$!

  ( serve bai_srv_c 6,7 8032 /nas04/yixuh/ba_babyai_force_sft_merged
    replay_one forcesft http://127.0.0.1:8032/v1
    [[ "$KEEP_MERGED" == "1" ]] || { log "rm force_sft merged"; rm -rf /nas04/yixuh/ba_babyai_force_sft_merged; }
  ) > "$OUT/indist_forcesft.log" 2>&1 &
  local pid_fsft=$!

  local rc=0
  for p in $pid_base $pid_router $pid_fsft; do wait "$p" || rc=1; done
  log "=== PHASE R done (rc=$rc) ==="
}

report() {
  log "=== IN-DISTRIBUTION RESULTS ==="
  $PY - <<'PYR'
import json, os
OUT = "babyai_experiment"
rows = [("baseline (pool source)", "base_train_v1"),
        ("router mem",  "indist_routermem"), ("force_mem",   "indist_forcemem"),
        ("router SFT",  "indist_routersftonly"), ("router both", "indist_routerboth"),
        ("force_sft",   "indist_forcesft")]
print("\n--- babyai (train pool, 200 tasks -- the tasks the pools were built from) ---")
for label, d in rows:
    path = os.path.join(OUT, d, "summary.json")
    if not os.path.exists(path):
        print(f"  {label:24s} (missing)"); continue
    x = json.load(open(path))
    ms = x.get("mean_score")
    tail = f"  score={ms:.4f}" if isinstance(ms, (int, float)) else ""
    print(f"  {label:24s} {x['success']:3d}/{x['tasks']:3d}  pass={x['pass_rate']:.4f}{tail}  errors={x['errors']}")
PYR
}

phase_m
phase_r
report
log "=== BABYAI IN-DISTRIBUTION REPLAY COMPLETE ==="
