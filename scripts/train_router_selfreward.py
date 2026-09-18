#!/usr/bin/env python3
"""Per-batch GRPO training using SELF-replay reward instead of a fixed probe
set (DESIGN.md section 10). Sibling of train_router_cheap_reward.py, which
uses a fixed 15-task dev subset as reward for every single batch of every
iteration -- that script's code is left untouched; this is a separate
experiment, not a replacement.

Why: the fixed-probe design (train_router_cheap_reward.py) scores every
batch's candidates against the SAME 15 dev tasks, every batch, every
iteration. The router has every opportunity and every incentive to overfit
specifically to those 15 tasks rather than learning genuinely general
routing behavior -- and the router_reward_v1/cheap_train_v2 run's own final
validation step exists specifically to catch that (probe_subset(15) vs
held_out(42)).

Here the reward for batch b is instead "replay batch b's OWN 10 tasks with
the bank-so-far" -- literally what the sibling script already computed as
`self_reward` for diagnostics only. Every batch tests a DIFFERENT 10 tasks
(all 90 train tasks get touched across one pass), so there is no fixed
small set to overfit to. Trade-off: self-replay reuses the same tasks a
memory was JUST written from, so a memory's content can trivially help its
own source task without generalizing -- this is the leakage risk the
original pilot (router_reward_v1/pilot_inbatch_v1) measured directly
against a probe set and found correlated at 0.905 (n=4, small-sample) with
true generalization. Not proof it's safe, but not baseless either.

Bonus: dropping the probe eval frees the SECOND server replica entirely.
Instead of pairing self+probe of one candidate across the two replicas
(sibling script), this pairs TWO CANDIDATES' self-evals across the two
replicas at once -- K=4 candidates in 2 rounds of 2-way concurrency instead
of 4 rounds. Cost per batch roughly halves versus the sibling script.

The end-of-iteration validation pass (full greedy 90-task bank, real 57-task
dev eval) is unchanged and still reports the probe(15)/held-out(42) split --
now purely as a curiosity/diagnostic, not an overfitting check, since
training never touches the probe set at all in this version.
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
from trajectory_memory_lab.router_policy import RouterPolicy, load_checkpoint, save_checkpoint  # noqa: E402

GROUP = "appworld"
TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
PROBE_BASELINE_ROLLOUT = ROOT / "appworld_experiment/noise_serial_v1/run_a"
PROBE_SET_SIZE = 15  # only used for the end-of-iteration validation split now, not training
OUTPUT_ROOT = ROOT / os.environ.get("ROUTER_OUTPUT_DIR", "router_reward_v1/cheap_train_v5")
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
TRAIN_LOG = OUTPUT_ROOT / "train_log.jsonl"

APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root"


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
    """Only used for the end-of-iteration validation split now (see module
    docstring) -- training never scores anything against this set."""
    trajectories: dict[str, bool] = {}
    for path in sorted((PROBE_BASELINE_ROLLOUT / "trajectories").glob("*.json")):
        record = read_json(path)
        trajectories[record["task_id"]] = bool(record["trajectory"]["success"])
    return sorted(trajectories)[:PROBE_SET_SIZE]


def latest_checkpoint_iteration() -> int:
    if not CHECKPOINT_DIR.exists():
        return 0
    iters = []
    for path in CHECKPOINT_DIR.glob("router_iter*.pt"):
        try:
            iters.append(int(path.stem.removeprefix("router_iter")))
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


def sample_k_candidates(
    router_model: RouterPolicy, iteration: int, batch_idx: int, batch_task_ids: list[str],
    trajectories: dict[str, dict[str, Any]], canonical_bank: list[dict[str, Any]], position: int,
    total_task_count: int, args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """K stochastic realizations of this one batch, all starting from the
    SAME canonical_bank -- not K independently-diverging full chains."""
    candidates = []
    for k in range(args.rollouts_per_batch):
        torch.manual_seed(20260822 + iteration * 100_000 + batch_idx * 1000 + k)
        cand_dir = OUTPUT_ROOT / f"iter{iteration}" / f"b{batch_idx}" / f"k{k}"
        builder_config = RouterBuilderConfig(
            output=cand_dir, record_protocol="cheap_reward_v4_decision",
            model=args.model, base_url=args.base_url, seed=20260822 + k,
        )
        result = run_router_chain(
            router_model, GROUP, batch_task_ids, trajectories, builder_config,
            initial_bank=canonical_bank, start_position=position, total_task_count=total_task_count,
        )
        candidates.append({
            "k": k, "dir": cand_dir, "bank": result.bank, "decisions": result.decisions,
            "route_counts": Counter(d["route"] for d in result.decisions),
            "active_entries": len([e for e in result.bank if e["status"] == "active"]),
            "records": result.summary["records"],
        })
    return candidates


def score_candidates_self_only(candidates: list[dict[str, Any]], batch_task_ids: list[str], iteration: int, batch_idx: int, args: argparse.Namespace) -> None:
    """Self-replay only. No second server needed per-candidate, so pair TWO
    candidates across the two replicas at once instead of pairing one
    candidate's self+probe -- this is where the 2x speedup vs the
    fixed-probe sibling script comes from."""
    replicas = [args.base_url, args.base_url_probe]
    for start in range(0, len(candidates), len(replicas)):
        pair = candidates[start : start + len(replicas)]
        procs = []
        for cand, base_url in zip(pair, replicas):
            bank_path = cand["dir"] / "banks" / f"memory_{GROUP}.json"
            self_dir = cand["dir"] / "eval_self"
            tag = f"cheap_v4_iter{iteration}_b{batch_idx}_k{cand['k']}_self"
            procs.append((cand, self_dir, tag, launch_subset_eval(bank_path, self_dir, tag, batch_task_ids, "train", args.model, base_url)))
        for cand, self_dir, tag, proc in procs:
            cand["self_pass_rate"] = wait_subset_eval(proc, self_dir, tag)


def run_validation_pass(router_model: RouterPolicy, iteration: int, task_ids: list[str], trajectories: dict[str, dict[str, Any]], probe_task_ids: list[str], args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic (greedy) full-domain bank, scored on the REAL full
    57-task dev split. Not used for gradients. probe(15)/held-out(42) split
    is now just a curiosity, not an overfitting check -- see module
    docstring: this run never scores anything against the probe set."""
    val_dir = OUTPUT_ROOT / f"iter{iteration}" / "validation"
    builder_config = RouterBuilderConfig(
        output=val_dir, record_protocol="cheap_reward_v4_validation",
        model=args.model, base_url=args.base_url, seed=20260822,
    )
    result = run_router_chain(router_model, GROUP, task_ids, trajectories, builder_config, greedy=True)
    route_counts = Counter(d["route"] for d in result.decisions)
    # route_counts here is over argmax routes, which hides a collapsing policy
    # until it has already fully collapsed; the mean entropy of the underlying
    # distribution is the early warning.
    mean_entropy = (
        float(sum(float(d["entropy"]) for d in result.decisions) / len(result.decisions))
        if result.decisions else 0.0
    )
    bank_path = val_dir / "banks" / f"memory_{GROUP}.json"
    eval_dir = val_dir / "eval_full_dev"
    proc = launch_subset_eval(bank_path, eval_dir, f"cheap_v4_iter{iteration}_validation", None, "dev", args.model, args.base_url)
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
        f"full_dev_pass_rate={pass_rate:.4f} probe_subset(15)={probe_subset_pass_rate} held_out(42)={held_out_pass_rate}",
        flush=True,
    )
    return {
        "route_counts": dict(route_counts), "mean_entropy": mean_entropy,
        "active_entries": result.summary["active_entries"],
        "full_dev_pass_rate": pass_rate, "probe_subset_pass_rate": probe_subset_pass_rate,
        "held_out_pass_rate": held_out_pass_rate,
    }


def run_one_batch_update(
    router_model: RouterPolicy, optimizer: torch.optim.Optimizer, iteration: int, batch_idx: int,
    batch_task_ids: list[str], trajectories: dict[str, dict[str, Any]], canonical_bank: list[dict[str, Any]],
    position: int, total_task_count: int, args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Returns the new canonical_bank. No cross-batch reward chaining is
    needed anymore -- each batch's self_baseline is computed fresh from
    that batch's own (already-known, no-memory) recorded success rate, not
    a running previous-checkpoint value."""
    candidates = sample_k_candidates(router_model, iteration, batch_idx, batch_task_ids, trajectories, canonical_bank, position, total_task_count, args)
    score_candidates_self_only(candidates, batch_task_ids, iteration, batch_idx, args)

    # DESIGN.md section 14: subtracting self_baseline as a per-candidate
    # constant and THEN re-centering on the group mean cancels it out exactly
    # -- advantage_k = (pass_k - baseline) - mean_j(pass_j - baseline) =
    # pass_k - mean_j(pass_j). The no-memory baseline never reaches the
    # gradient; the router only ever learns "better than my K-1 siblings",
    # which is a real answer even when every sibling in the batch is worse
    # than doing nothing at all. A K+1-way-mean fix was tried and reverted:
    # folding self_baseline into the group mean as a free extra reference
    # point only gives it 1/(K+1) weight, diluting fast as K grows (1/9 at
    # K=8) -- correct direction, likely too weak to matter, and no better
    # weighting has been worked out yet. Reverted to the plain group-relative
    # advantage; self_baseline/rewards are still computed and logged purely
    # as a diagnostic, not used in the actual gradient.
    self_baseline = statistics.mean(1.0 if trajectories[t]["success"] else 0.0 for t in batch_task_ids)
    pass_rates = [c["self_pass_rate"] for c in candidates]
    rewards = [p - self_baseline for p in pass_rates]  # diagnostic only, see above
    mean_pass = statistics.mean(pass_rates)
    advantages = [p - mean_pass for p in pass_rates]

    optimizer.zero_grad()
    # Policy-gradient term summed over every decision in every candidate, plus an
    # entropy bonus over the same decisions so the two terms share a scale (both
    # grow with K x batch_size). Without it the router collapses to a
    # near-deterministic route well before the reward signal has said anything
    # useful -- v2 ended up greedy-`both` on all 90 decisions and v3_g4's sampled
    # entropy fell from 1.14 to 0.65 nats over 9 batches. Subtracting
    # entropy_coef * H rewards keeping mass on the other routes, so exploration
    # survives long enough for the (small, noisy) advantages to matter.
    pg_term = torch.zeros(())
    entropy_sum = torch.zeros(())
    decision_count = 0
    for advantage, cand in zip(advantages, candidates):
        for decision in cand["decisions"]:
            pg_term = pg_term - advantage * decision["logprob"]
            entropy_sum = entropy_sum + decision["entropy"]
            decision_count += 1
    loss = pg_term - args.entropy_coef * entropy_sum
    loss.backward()
    optimizer.step()

    mean_entropy = float(entropy_sum.item() / decision_count) if decision_count else 0.0

    chosen = random.choice(candidates)
    print(
        f"[iter{iteration} b{batch_idx}] self_pass_rates={[round(c['self_pass_rate'],3) for c in candidates]} "
        f"self_baseline={self_baseline:.3f} mean_pass={mean_pass:.3f} rewards={[round(r,3) for r in rewards]} "
        f"advantages={[round(a,3) for a in advantages]} "
        f"loss={loss.item():.4f} pg_term={pg_term.item():.4f} mean_entropy={mean_entropy:.4f} "
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
        "loss": loss.item(),
        "pg_term": pg_term.item(),
        "mean_entropy": mean_entropy,
        "entropy_coef": args.entropy_coef,
        "chosen_k": chosen["k"],
    }
    # DESIGN.md section 15: actually train the sft route in, instead of just
    # recording repair plans that nothing downstream ever consumed. Only the
    # CHOSEN candidate's sft/both picks are replayed -- replaying all K would
    # multiply AppWorld-eval cost by K for no reward benefit, since only the
    # chosen candidate's bank continues into the next batch anyway.
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
    if maybe_trigger_training(pool_path, OUTPUT_ROOT / "lora_work", pool_size_before, pool_size_after):
        log_record["sft_lora_retrained_at_pool_size"] = pool_size_after

    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return chosen["bank"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rollouts-per-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="weight on the entropy bonus subtracted from the loss; 0 reproduces the "
             "pre-v4 objective. Counteracts the route-distribution collapse seen in "
             "cheap_train_v2/v3_g4 (see the v4 note in DESIGN.md).",
    )
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--base-url-probe", default="http://127.0.0.1:8001/v1", help="second replica, used to parallelize self-evals across candidates now, not for a probe set")
    parser.add_argument("--yes", action="store_true", help="skip the pre-launch cost confirmation")
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    task_ids = sorted(trajectories)
    probe_task_ids = load_probe_set()
    num_batches = len(split_batches(task_ids, args.batch_size))
    est_minutes_per_eval = 70  # observed single-eval pace, DESIGN.md section 12/14
    rounds_per_batch = -(-args.rollouts_per_batch // 2)  # ceil(K/2), 2-way concurrency
    self_eval_hours = rounds_per_batch * num_batches * est_minutes_per_eval / 60
    # DESIGN.md section 14/15 additions, all rough and unverified until observed
    # once: a guided replay is a single-task rollout (~1/10th of a 10-task
    # self-eval); assume ~2 committed sft/both decisions per batch since the
    # actual rate is unknown before this router has ever run. A LoRA-retrain
    # trigger (train + CPU merge + both replicas cold-starting) has never been
    # timed end-to-end -- 60 min is a placeholder, not a measurement.
    est_replays_per_batch = 2
    replay_hours = num_batches * est_replays_per_batch * (est_minutes_per_eval / 10) / 60
    est_examples_total = num_batches * est_replays_per_batch  # optimistic: assumes every replay verifies
    est_lora_triggers = est_examples_total // 8
    est_lora_hours_per_trigger = 1.0
    lora_hours = est_lora_triggers * est_lora_hours_per_trigger
    est_hours = self_eval_hours + replay_hours + lora_hours + 3.5
    print(
        f"plan: {num_batches} batches/iteration x {args.rollouts_per_batch} rollouts/batch "
        f"({rounds_per_batch} rounds/batch of 2-way concurrent self-evals) "
        f"~{self_eval_hours:.1f}h self-eval + ~{replay_hours:.1f}h sft guided-replay "
        f"(assumed {est_replays_per_batch}/batch, UNVERIFIED) + ~{lora_hours:.1f}h LoRA retrain/reload "
        f"(assumed ~{est_lora_triggers} trigger(s) at {est_lora_hours_per_trigger}h each, UNVERIFIED, "
        f"never timed end-to-end) = ~{est_hours:.1f}h/iteration x {args.iterations} iteration(s)",
        flush=True,
    )
    if not args.yes:
        raise SystemExit("pass --yes to confirm this budget and actually launch")

    print(f"loaded {len(task_ids)} train trajectories; validation-only probe set: {probe_task_ids}", flush=True)

    router_model = RouterPolicy()
    start_iteration = latest_checkpoint_iteration() + 1
    if start_iteration > 1:
        ckpt = CHECKPOINT_DIR / f"router_iter{start_iteration - 1}.pt"
        load_checkpoint(router_model, ckpt)
        print(f"resumed from {ckpt}", flush=True)
    optimizer = torch.optim.Adam(router_model.parameters(), lr=args.lr)

    for offset in range(args.iterations):
        iteration = start_iteration + offset
        batches = split_batches(task_ids, args.batch_size)
        canonical_bank: list[dict[str, Any]] = []
        position = 0

        for batch_idx, batch_task_ids in enumerate(batches):
            canonical_bank = run_one_batch_update(
                router_model, optimizer, iteration, batch_idx, batch_task_ids, trajectories,
                canonical_bank, position, len(task_ids), args,
            )
            position += len(batch_task_ids)

        checkpoint_path = save_checkpoint(router_model, iteration, CHECKPOINT_DIR)
        validation = run_validation_pass(router_model, iteration, task_ids, trajectories, probe_task_ids, args)
        print(f"=== iteration {iteration} done === checkpoint={checkpoint_path} validation={validation}", flush=True)


if __name__ == "__main__":
    main()
