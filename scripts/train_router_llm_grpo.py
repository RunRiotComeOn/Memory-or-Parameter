#!/usr/bin/env python3
"""GRPO training for the LLM router (router_llm_trainable.TrainableLLMRouter).

Same loop as `train_router_selfreward.py` -- which stays untouched, this is a
sibling experiment, not a replacement -- with the same reward, the same
batching, the same advantage, and the same survivorship-bias guard:

  K candidates sampled per batch from the SAME canonical bank
    -> each self-scored by a live 10-task AppWorld replay
    -> advantage_k = pass_k - mean_j(pass_j)
    -> ONE gradient step per batch
    -> next batch's canonical bank = a RANDOMLY chosen candidate, not the best

What changes is only what produces the decision and what receives the
gradient: a LoRA adapter on a small LLM that reads the actually-drafted memory
and repair-plan text, rather than `router_policy.RouterPolicy`'s 144-parameter
linear layer over hashed n-grams. The reward pipeline is reused verbatim.

No RL framework and, unlike the usual RLHF setup, NO vLLM for the router
either. The router's action is one categorical draw over four routes, not a
free-form generation -- forcing the assistant turn to open with `{"route": "`
makes the next token identify the route outright -- so a single local forward
pass is the entire policy, differentiable, with an exact entropy. At ~0.2s per
decision against a ~70-minute-per-candidate AppWorld self-eval, there is
nothing for a serving engine to speed up, and sampling and the gradient then
share one set of weights: no adapter hot-swap, no resync window, and the
recomputed logprob is the sampled one exactly rather than approximately. See
router_llm_trainable.py's module docstring for the full argument.

Two-phase-per-batch structure, forced by memory: sampling runs under
`no_grad` and records (prompt_ids, route_index); the update replays each
decision with grad one at a time and accumulates. Retaining 80 live forward
graphs of a 35B model to build one summed loss would not fit. This is exactly
equivalent -- the weights do not move in between -- and `--verify-recompute`
asserts it on the first decision of every batch.

NOT run from this repo's `.venv`: that one holds vLLM 0.17.1, which pins
transformers 4.57.3, and 4.57.3 does not know `qwen3_5_moe` -- `AutoConfig`
refuses the checkpoint outright. The router therefore runs from a separate
`/nas04/yixuh/router_venv` (transformers 5.17.0 + peft + torch 2.10.0), which
leaves the two serving replicas' environment untouched.

Run (see running_log.md for the launched configuration):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/nas04/yixuh/hf_cache \
  APPWORLD_ROOT=/nas04/yixuh/appworld_root_llmgrpo \
  /nas04/yixuh/router_venv/bin/python -u \
      scripts/train_router_llm_grpo.py --yes \
      --rollouts-per-batch 8 --batch-size 10 --lr 1e-5 \
      --base-url http://127.0.0.1:8030/v1 --base-url-probe http://127.0.0.1:8031/v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_bank_builder import RouterBuilderConfig, run_router_chain  # noqa: E402
from trajectory_memory_lab.router_llm_trainable import (  # noqa: E402
    RouterLLMConfig,
    TrainableLLMRouter,
    accumulate_policy_gradient,
    verify_recompute_matches_sample,
)

GROUP = "appworld"
TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
PROBE_BASELINE_ROLLOUT = ROOT / "appworld_experiment/noise_serial_v1/run_a"
PROBE_SET_SIZE = 15  # validation split reporting only, never trained against
OUTPUT_ROOT = ROOT / os.environ.get("ROUTER_OUTPUT_DIR", "router_reward_v1/router_llm_grpo_v1")
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
TRAIN_LOG = OUTPUT_ROOT / "train_log.jsonl"

APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root"
# The router policy is the SAME MoE the task agent is served from
# (Qwen3.5-35B-A3B), loaded locally with transformers+peft rather than through
# vLLM -- see router_llm_trainable.py for why the router needs no serving
# engine. `AutoModelForCausalLM` maps this checkpoint to
# `Qwen3_5MoeForCausalLM`, i.e. the language tower only, which is the
# transformers equivalent of the servers' `--language-model-only`: the 27-block
# vision tower and the MTP head are never built, so ~68GB of the 71.9GB
# checkpoint is loaded.
DEFAULT_ROUTER_MODEL = (
    "/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "59d61f3ce65a6d9863b86d2e96597125219dc754"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_train_trajectories() -> dict[str, dict[str, Any]]:
    protocol = read_json(TRAIN_ROLLOUT / "protocol.json")
    if protocol.get("split") != "train":
        raise ValueError(f"expected split=train, got {protocol.get('split')!r}")
    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted((TRAIN_ROLLOUT / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        if record.get("memory_bank"):
            raise ValueError(f"{path} was rolled out with a memory bank attached")
        trajectory = record["trajectory"]
        trajectory["domain"] = GROUP
        trajectories[record["task_id"]] = trajectory
    return trajectories


def load_probe_set() -> list[str]:
    trajectories: dict[str, bool] = {}
    for path in sorted((PROBE_BASELINE_ROLLOUT / "trajectories").glob("*.json")):
        record = read_json(path)
        trajectories[record["task_id"]] = bool(record["trajectory"]["success"])
    return sorted(trajectories)[:PROBE_SET_SIZE]


def latest_checkpoint_iteration() -> int:
    if not CHECKPOINT_DIR.exists():
        return 0
    iters = []
    for path in CHECKPOINT_DIR.glob("router_iter*"):
        try:
            iters.append(int(path.name.removeprefix("router_iter")))
        except ValueError:
            continue
    return max(iters, default=0)


def split_batches(task_ids: list[str], batch_size: int) -> list[list[str]]:
    return [task_ids[i : i + batch_size] for i in range(0, len(task_ids), batch_size)]


def launch_subset_eval(
    bank_path: Path, eval_dir: Path, experiment_name: str, task_ids: list[str] | None,
    split: str, model: str, base_url: str,
) -> subprocess.Popen:
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_rollout.py"),
        "--split", split,
        "--output", str(eval_dir),
        "--experiment-name", experiment_name,
        "--memory-bank", str(bank_path),
        "--memory-top-k", "3",
        "--max-parallel", "1",
        "--seed", "20260822",
        "--model", model,
        "--base-url", base_url,
    ]
    if task_ids is not None:
        cmd += ["--task-ids", *task_ids]
    print(f"  launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def wait_subset_eval(proc: subprocess.Popen, eval_dir: Path, experiment_name: str) -> float:
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"eval failed ({experiment_name}): exit {returncode}")
    summary = read_json(eval_dir / "summary.json")
    pass_rate = summary["pass_rate"]
    if pass_rate is None:
        raise RuntimeError(f"no pass_rate ({experiment_name}): {summary}")
    return pass_rate


def builder_config_for(output: Path, protocol: str, args: argparse.Namespace, seed: int) -> RouterBuilderConfig:
    return RouterBuilderConfig(
        output=output, record_protocol=protocol,
        model=args.model, base_url=args.base_url, seed=seed,
        sft_writer=args.sft_writer, teacher_model=args.teacher_model,
        teacher_api_key_file=args.teacher_api_key_file,
        router_mode="trained_llm",
    )


def sample_k_candidates(
    router: TrainableLLMRouter, iteration: int, batch_idx: int, batch_task_ids: list[str],
    trajectories: dict[str, dict[str, Any]], canonical_bank: list[dict[str, Any]], position: int,
    total_task_count: int, args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """K stochastic realizations of this one batch, all starting from the SAME
    canonical_bank -- not K independently-diverging full chains."""
    candidates = []
    for k in range(args.rollouts_per_batch):
        torch.manual_seed(20260822 + iteration * 100_000 + batch_idx * 1000 + k)
        cand_dir = OUTPUT_ROOT / f"iter{iteration}" / f"b{batch_idx}" / f"k{k}"
        result = run_router_chain(
            router, GROUP, batch_task_ids, trajectories,
            builder_config_for(cand_dir, "router_llm_grpo_v1_decision", args, 20260822 + k),
            initial_bank=canonical_bank, start_position=position, total_task_count=total_task_count,
        )
        candidates.append({
            "k": k, "dir": cand_dir, "bank": result.bank, "decisions": result.decisions,
            "route_counts": Counter(d["route"] for d in result.decisions),
            "active_entries": len([e for e in result.bank if e["status"] == "active"]),
            "records": result.summary["records"],
        })
    return candidates


def score_candidates_self_only(
    candidates: list[dict[str, Any]], batch_task_ids: list[str], iteration: int,
    batch_idx: int, args: argparse.Namespace,
) -> None:
    """Self-replay only, two candidates at a time across the two replicas."""
    replicas = [args.base_url, args.base_url_probe]
    for start in range(0, len(candidates), len(replicas)):
        pair = candidates[start : start + len(replicas)]
        procs = []
        for cand, base_url in zip(pair, replicas):
            bank_path = cand["dir"] / "banks" / f"memory_{GROUP}.json"
            self_dir = cand["dir"] / "eval_self"
            tag = f"routerllm_iter{iteration}_b{batch_idx}_k{cand['k']}_self"
            procs.append((cand, self_dir, tag, launch_subset_eval(
                bank_path, self_dir, tag, batch_task_ids, "train", args.model, base_url)))
        for cand, self_dir, tag, proc in procs:
            cand["self_pass_rate"] = wait_subset_eval(proc, self_dir, tag)


def run_validation_pass(
    router: TrainableLLMRouter, iteration: int, task_ids: list[str],
    trajectories: dict[str, dict[str, Any]], probe_task_ids: list[str], args: argparse.Namespace,
) -> dict[str, Any]:
    """Deterministic (greedy) full-domain bank, scored on the real 57-task dev
    split. Not used for gradients."""
    val_dir = OUTPUT_ROOT / f"iter{iteration}" / "validation"
    result = run_router_chain(
        router, GROUP, task_ids, trajectories,
        builder_config_for(val_dir, "router_llm_grpo_v1_validation", args, 20260822),
        greedy=True,
    )
    route_counts = Counter(d["route"] for d in result.decisions)
    mean_entropy = (
        float(sum(float(d["entropy"]) for d in result.decisions) / len(result.decisions))
        if result.decisions else 0.0
    )
    bank_path = val_dir / "banks" / f"memory_{GROUP}.json"
    eval_dir = val_dir / "eval_full_dev"
    proc = launch_subset_eval(
        bank_path, eval_dir, f"routerllm_iter{iteration}_validation", None, "dev", args.model, args.base_url)
    pass_rate = wait_subset_eval(proc, eval_dir, "validation")

    probe_set = set(probe_task_ids)
    probe_successes, probe_total, held_out_successes, held_out_total = 0, 0, 0, 0
    for path in sorted((eval_dir / "trajectories").glob("*.json")):
        record = read_json(path)
        success = bool((record.get("trajectory") or {}).get("success"))
        if record["task_id"] in probe_set:
            probe_total += 1
            probe_successes += int(success)
        else:
            held_out_total += 1
            held_out_successes += int(success)
    probe_subset_pass_rate = probe_successes / probe_total if probe_total else None
    held_out_pass_rate = held_out_successes / held_out_total if held_out_total else None

    print(
        f"[iter{iteration} validation] routes={dict(route_counts)} mean_entropy={mean_entropy:.4f} "
        f"active_entries={result.summary['active_entries']} "
        f"full_dev_pass_rate={pass_rate:.4f} probe_subset(15)={probe_subset_pass_rate} "
        f"held_out(42)={held_out_pass_rate}",
        flush=True,
    )
    return {
        "route_counts": dict(route_counts), "mean_entropy": mean_entropy,
        "active_entries": result.summary["active_entries"],
        "full_dev_pass_rate": pass_rate, "probe_subset_pass_rate": probe_subset_pass_rate,
        "held_out_pass_rate": held_out_pass_rate,
    }


def run_one_batch_update(
    router: TrainableLLMRouter, optimizer: torch.optim.Optimizer, iteration: int, batch_idx: int,
    batch_task_ids: list[str], trajectories: dict[str, dict[str, Any]],
    canonical_bank: list[dict[str, Any]], position: int, total_task_count: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Returns the new canonical_bank."""
    candidates = sample_k_candidates(
        router, iteration, batch_idx, batch_task_ids, trajectories, canonical_bank,
        position, total_task_count, args)
    score_candidates_self_only(candidates, batch_task_ids, iteration, batch_idx, args)

    # Same advantage as train_router_selfreward.py, for the same reasons
    # (DESIGN.md section 14.1): subtracting the no-memory self_baseline as a
    # per-candidate constant and then re-centering on the group mean cancels
    # it exactly, so it is computed and logged as a diagnostic only and never
    # reaches the gradient. The K+1-way-mean fix was tried and reverted there;
    # not reintroducing it here keeps the two runs comparable.
    self_baseline = statistics.mean(1.0 if trajectories[t]["success"] else 0.0 for t in batch_task_ids)
    pass_rates = [c["self_pass_rate"] for c in candidates]
    rewards = [p - self_baseline for p in pass_rates]  # diagnostic only
    mean_pass = statistics.mean(pass_rates)
    advantages = [p - mean_pass for p in pass_rates]

    # Flatten to one advantage per DECISION, so the policy-gradient term is a
    # plain sum over all K x batch_size decisions -- identical in shape to the
    # linear-router loss, which is what keeps entropy_coef=0.01 calibrated
    # (both terms grow with K x batch_size, DESIGN.md section 12).
    flat_decisions, flat_advantages = [], []
    for advantage, cand in zip(advantages, candidates):
        for decision in cand["decisions"]:
            sampled = decision.get("sampled_decision")
            if sampled is None:
                raise RuntimeError(
                    "a decision carries no SampledDecision -- run_router_chain was not in "
                    "router_mode='trained_llm', so there is nothing to backprop through"
                )
            flat_decisions.append(sampled)
            flat_advantages.append(advantage)

    # Same weights sampled these decisions and are about to score them, so any
    # disagreement is a real bug (wrong prompt replayed, wrong index stored),
    # not kernel noise. Cheap enough to assert every batch.
    if args.verify_recompute and flat_decisions:
        sampled_lp, recomputed_lp, difference = verify_recompute_matches_sample(
            router, flat_decisions[0].prompt_ids, flat_decisions[0].index)
        print(f"  [verify] logprob sampled={sampled_lp:.6f} recomputed={recomputed_lp:.6f} "
              f"|diff|={difference:.2e}", flush=True)

    entropy_before = statistics.mean(float(d.entropy) for d in flat_decisions) if flat_decisions else 0.0
    optimizer.zero_grad()
    stats = accumulate_policy_gradient(
        router, flat_decisions, flat_advantages,
        entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm)
    optimizer.step()

    # How far the step actually moved the policy, measured on the decisions it
    # was computed from. The single most important number to watch early: the
    # lr sweep in smoke_router_llm_policy_gradient.py showed this mechanism
    # goes from "moves nothing" to "moves everything" over about one order of
    # magnitude of lr, and there are only ~9 steps per iteration to notice.
    with torch.no_grad():
        shifts = []
        for sampled in flat_decisions[: args.drift_probe_size]:
            after = torch.softmax(router.route_logits(sampled.prompt_ids), dim=-1).cpu()
            shifts.append(float((after - sampled.probs).abs().max()))
    max_prob_shift = max(shifts) if shifts else 0.0
    mean_prob_shift = statistics.mean(shifts) if shifts else 0.0

    chosen = random.choice(candidates)
    print(
        f"[iter{iteration} b{batch_idx}] self_pass_rates={[round(c['self_pass_rate'],3) for c in candidates]} "
        f"self_baseline={self_baseline:.3f} mean_pass={mean_pass:.3f} "
        f"rewards={[round(r,3) for r in rewards]} advantages={[round(a,3) for a in advantages]} "
        f"loss={stats['loss']:.4f} pg_term={stats['pg_term']:.4f} "
        f"grad_norm={stats['grad_norm']:.4f} mean_entropy={stats['mean_entropy']:.4f} "
        f"prob_shift(mean/max)={mean_prob_shift:.2e}/{max_prob_shift:.2e} "
        f"chosen_k={chosen['k']} chosen_self_pass_rate={chosen['self_pass_rate']:.4f}",
        flush=True,
    )

    log_record = {
        "iteration": iteration, "batch_idx": batch_idx,
        "route_counts": [dict(c["route_counts"]) for c in candidates],
        "active_entries": [c["active_entries"] for c in candidates],
        "self_pass_rates": [c["self_pass_rate"] for c in candidates],
        "self_baseline": self_baseline,
        "mean_pass": mean_pass,
        "rewards": rewards,
        "advantages": advantages,
        "loss": stats["loss"],
        "pg_term": stats["pg_term"],
        "grad_norm": stats["grad_norm"],
        "mean_entropy": stats["mean_entropy"],
        "entropy_before_step": entropy_before,
        "mean_prob_shift": mean_prob_shift,
        "max_prob_shift": max_prob_shift,
        "entropy_coef": args.entropy_coef,
        "lr": args.lr,
        "decision_count": stats["decision_count"],
        "chosen_k": chosen["k"],
    }

    # DESIGN.md section 15: the sft route only means anything if a plan that
    # routes there is actually replayed and, on success, trained in. Only the
    # CHOSEN candidate's sft/both picks are replayed -- the other K-1 banks
    # are discarded anyway.
    from trajectory_memory_lab.router_sft_pipeline import (
        append_to_pool, collect_batch_sft_examples, maybe_trigger_training,
        sft_candidates_from_records,
    )
    pool_path = OUTPUT_ROOT / "sft_pool.jsonl"
    pool_size_before = sum(1 for _ in pool_path.open()) if pool_path.exists() else 0
    replay_dir = OUTPUT_ROOT / f"iter{iteration}" / f"b{batch_idx}" / "sft_replays"
    sft_examples = collect_batch_sft_examples(
        chosen["records"], replay_dir, args.model, args.base_url, seed=20260822 + batch_idx,
    )
    pool_size_after = append_to_pool(pool_path, sft_examples)
    log_record["sft_candidates_replayed"] = len(sft_candidates_from_records(chosen["records"]))
    log_record["sft_examples_verified"] = len(sft_examples)
    log_record["sft_pool_size"] = pool_size_after
    print(
        f"[iter{iteration} b{batch_idx}] sft: {log_record['sft_candidates_replayed']} replayed, "
        f"{len(sft_examples)} verified (reward=1), pool now {pool_size_after}",
        flush=True,
    )
    if args.enable_sft_lora and maybe_trigger_training(
        pool_path, OUTPUT_ROOT / "lora_work", pool_size_before, pool_size_after
    ):
        log_record["sft_lora_retrained_at_pool_size"] = pool_size_after

    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return chosen["bank"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rollouts-per-batch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="LoRA learning rate. NOT train_router_selfreward.py's 0.01 (tuned for a "
             "144-parameter linear model) and NOT this repo's SFT LoRA 1e-4 either: the "
             "sweep in smoke_router_llm_policy_gradient.py measured 1e-4 moving a route's "
             "probability from 0.00000 to 0.949 in ONE step and diverging the entropy bonus "
             "into a collapsed distribution. 1e-5 reaches ~0.87 only after 9 coherent steps.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="gradient clipping; the measured failure mode is an oversized step, and with "
             "~9 updates per iteration there is no room to recover from one",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="same v4 entropy bonus, same scale (DESIGN.md section 12)")
    parser.add_argument("--router-model-path", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-device", default="cuda:0",
                        help="single-card placement; ignored when --router-device-map is set")
    parser.add_argument("--router-device-map", default="auto",
                        help="shard the router base model across the visible GPUs. 68GB of bf16 "
                             "weights does not fit on one 49GB card, so the 35B MoE needs this; "
                             "pass '' to force the old single-card path for a model that fits")
    parser.add_argument("--policy-temperature", type=float, default=1.0,
                        help="divides the four route logits; part of the policy, not decoding")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="required for the 35B MoE router: without it a 4k-token backward OOMs "
                             "on two 49GB cards, and the median router prompt is ~11.5k")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--model", default="qwen35-tau", help="the 35B task agent + draft writer")
    parser.add_argument("--base-url", default="http://127.0.0.1:8030/v1")
    parser.add_argument("--base-url-probe", default="http://127.0.0.1:8031/v1",
                        help="second task-agent replica, parallelizes self-evals across candidates")
    parser.add_argument("--sft-writer", choices=("teacher", "self"), default="teacher")
    parser.add_argument("--teacher-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--teacher-api-key-file", type=Path,
                        default=Path("/nas04/yixuh/.config/continual-memory/gemini_api_key"))
    parser.add_argument("--enable-sft-lora", action="store_true",
                        help="allow a verified sft pool to trigger router_sft_lora_update.sh, which "
                             "restarts BOTH task-agent replicas; off by default so a first run cannot "
                             "take the servers down mid-experiment")
    parser.add_argument("--verify-recompute", action="store_true", default=True)
    parser.add_argument("--no-verify-recompute", dest="verify_recompute", action="store_false")
    parser.add_argument("--drift-probe-size", type=int, default=8,
                        help="decisions re-scored after each step to report how far the policy moved")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = all; >0 stops after this many batches (use 1 for a single real batch)")
    parser.add_argument("--yes", action="store_true", help="skip the pre-launch cost confirmation")
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    task_ids = sorted(trajectories)
    probe_task_ids = load_probe_set()
    num_batches = len(split_batches(task_ids, args.batch_size))
    if args.max_batches:
        num_batches = min(num_batches, args.max_batches)
    est_minutes_per_eval = 70  # observed single-eval pace, DESIGN.md section 12/14
    rounds_per_batch = -(-args.rollouts_per_batch // 2)  # ceil(K/2), 2-way concurrency
    self_eval_hours = rounds_per_batch * num_batches * est_minutes_per_eval / 60
    # Drafting is real serial wall-clock and train_router_selfreward.py's
    # estimate omits it, which is why its ~49.6h/iteration figure is low for
    # the same shape of run. Every candidate drafts every task before any
    # route is chosen (the content-before-route change, DESIGN.md section
    # 14.3), so it is K x batch_size calls per batch, not batch_size: one 35B
    # memory draft plus one teacher plan each, measured at ~40s per task on
    # the 8030/8031 replicas (router_llm_grpo_v1_probe1: records landing 32-45s
    # apart). It runs on ONE replica, serially, and does not overlap the evals.
    #
    # The router's own forward/backward passes ARE deliberately omitted, but
    # they are no longer free now that the router is the 35B MoE rather than
    # an 8B dense model. Measured on GPU 6,7 at a ~1k-token prompt: 2.2s to
    # sample, 1.5s to recompute with grad, 2.0s to backward -- so ~80 no-grad
    # forwards + ~80 forward/backwards is ~8 min/batch, ~1.2h per iteration.
    # Still ~2% of the ~4.7h of self-eval per batch, which is what keeps the
    # "no serving engine for the router" argument intact, but it is no longer
    # the ~1-2 min/batch that an 8B made it.
    est_seconds_per_draft = 40
    draft_hours = num_batches * args.rollouts_per_batch * args.batch_size * est_seconds_per_draft / 3600
    est_replays_per_batch = 2
    replay_hours = num_batches * est_replays_per_batch * (est_minutes_per_eval / 10) / 60
    lora_hours = 0.0
    if args.enable_sft_lora:
        lora_hours = (num_batches * est_replays_per_batch // 8) * 1.0
    # Validation redraws all 90 tasks greedily (~40s each) and then runs the
    # real 57-task dev eval.
    validation_hours = (3.5 + 90 * est_seconds_per_draft / 3600) if not args.max_batches else 0.0
    est_hours = self_eval_hours + draft_hours + replay_hours + lora_hours + validation_hours
    print(
        f"plan: {num_batches} batches x {args.rollouts_per_batch} rollouts/batch "
        f"({rounds_per_batch} rounds/batch of 2-way concurrent self-evals)\n"
        f"  ~{draft_hours:.1f}h drafting ({num_batches * args.rollouts_per_batch * args.batch_size} "
        f"tasks x ~{est_seconds_per_draft}s: 35B memory draft + teacher plan, serial on one replica)\n"
        f"  ~{self_eval_hours:.1f}h self-eval\n"
        f"  ~{replay_hours:.1f}h sft guided-replay (assumed {est_replays_per_batch}/batch, UNVERIFIED)\n"
        f"  ~{lora_hours:.1f}h LoRA retrain\n"
        f"  ~{validation_hours:.1f}h validation\n"
        f"  = ~{est_hours:.1f}h/iteration x {args.iterations} iteration(s)",
        flush=True,
    )
    if not args.yes:
        raise SystemExit("pass --yes to confirm this budget and actually launch")

    print(f"loaded {len(task_ids)} train trajectories; validation-only probe set: {probe_task_ids}", flush=True)

    router = TrainableLLMRouter(RouterLLMConfig(
        model_path=args.router_model_path, device=args.router_device,
        device_map=args.router_device_map or None,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        policy_temperature=args.policy_temperature,
    ))
    print(
        f"router: {args.router_model_path}\n"
        f"        input_device={router.input_device} logit_device={router.device}\n"
        f"        {router.trainable_parameter_count():,} trainable LoRA params "
        f"(r={args.lora_r}, alpha={args.lora_alpha}), temperature={args.policy_temperature}, "
        f"route tokens={dict(zip(('memory','sft','both','neither'), router.route_token_ids))}",
        flush=True,
    )

    start_iteration = latest_checkpoint_iteration() + 1
    if start_iteration > 1:
        ckpt = CHECKPOINT_DIR / f"router_iter{start_iteration - 1}"
        router.load_adapter(ckpt)
        print(f"resumed from {ckpt}", flush=True)
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=args.lr)

    for offset in range(args.iterations):
        iteration = start_iteration + offset
        batches = split_batches(task_ids, args.batch_size)
        if args.max_batches:
            batches = batches[: args.max_batches]
        canonical_bank: list[dict[str, Any]] = []
        position = 0

        for batch_idx, batch_task_ids in enumerate(batches):
            canonical_bank = run_one_batch_update(
                router, optimizer, iteration, batch_idx, batch_task_ids, trajectories,
                canonical_bank, position, len(task_ids), args,
            )
            position += len(batch_task_ids)

        checkpoint_path = router.save_adapter(CHECKPOINT_DIR / f"router_iter{iteration}")
        if args.max_batches:
            print(f"=== iteration {iteration} stopped after {len(batches)} batch(es) "
                  f"(--max-batches); skipping validation === checkpoint={checkpoint_path}", flush=True)
            continue
        validation = run_validation_pass(router, iteration, task_ids, trajectories, probe_task_ids, args)
        print(f"=== iteration {iteration} done === checkpoint={checkpoint_path} validation={validation}",
              flush=True)


if __name__ == "__main__":
    main()
