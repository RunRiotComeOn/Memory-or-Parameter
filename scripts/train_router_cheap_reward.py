#!/usr/bin/env python3
"""Per-batch GRPO training on the cheap two-tier reward (DESIGN.md section 13).

Supersedes the first cut of this script, which collected all G rollouts over
the WHOLE 90-task domain (9 batches each) and did a single gradient update
at the very end. That wastes 8 of every 9 batches' worth of reward signal
sitting idle before ever touching theta once. Here theta updates after
EVERY batch:

  for each batch (10 tasks, in domain order):
    1. sample K=--rollouts-per-batch stochastic realizations of THIS batch's
       routing decisions, all starting from the SAME canonical bank (the one
       actually carried over from the previous batch's update) -- not K
       independently-diverging full chains.
    2. score each of the K candidate banks: self reward (diagnostic only,
       leakage risk) + probe reward (the real GRPO reward, a fixed 15-task
       disjoint dev subset), both concurrent across two TP=2 replicas.
    3. GRPO advantage = probe_reward_k - mean_k(probe_reward) (group mean
       across just these K samples of just this batch, never a running max
       -- DESIGN.md section 2.1). One gradient step using this batch's
       decisions' logprobs.
    4. Carry ONE of the K sampled banks forward as the canonical state for
       the next batch (uniformly random pick, not the best-scoring one --
       always keeping the lucky sample would bias the chain's content
       toward small-sample noise rather than the policy's typical
       behavior). Free: reuses an eval already paid for, no extra pass.

One iteration = one full pass over all 90 train tasks (9 per-batch updates
if --batch-size 10). After the pass, run one expensive, deterministic
(argmax) full 57-task dev validation -- not used for gradients, purely to
check the cheap per-batch signal against ground truth.

Cost: --rollouts-per-batch K eval-pairs per batch, sequential (only two
server replicas exist, so only one candidate's self+probe pair runs at a
time) at ~78min/pair (observed pace, DESIGN.md section 11/12) -> roughly
K*78min per batch, *9 batches per iteration, + one ~3.5h validation pass.
K=8 -> ~9.4h/batch -> ~84h/iteration. This is a real budget question, not
a default to launch blind -- see the pre-launch cost check in main().
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
PROBE_SET_SIZE = 15
OUTPUT_ROOT = ROOT / "router_reward_v1/cheap_train_v2"
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


def load_probe_set() -> tuple[list[str], float]:
    trajectories: dict[str, bool] = {}
    for path in sorted((PROBE_BASELINE_ROLLOUT / "trajectories").glob("*.json")):
        record = read_json(path)
        trajectories[record["task_id"]] = bool(record["trajectory"]["success"])
    probe_task_ids = sorted(trajectories)[:PROBE_SET_SIZE]
    baseline = statistics.mean(1.0 if trajectories[t] else 0.0 for t in probe_task_ids)
    return probe_task_ids, baseline


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
            output=cand_dir, record_protocol="cheap_reward_v2_decision",
            model=args.model, base_url=args.base_url, seed=20260822 + k,
        )
        result = run_router_chain(
            router_model, GROUP, batch_task_ids, trajectories, builder_config,
            initial_bank=canonical_bank, start_position=position, total_task_count=total_task_count,
        )
        candidates.append({"k": k, "dir": cand_dir, "bank": result.bank, "decisions": result.decisions,
                            "route_counts": Counter(d["route"] for d in result.decisions),
                            "active_entries": len(result.summary and [e for e in result.bank if e["status"] == "active"] or [])})
    return candidates


def score_candidates(candidates: list[dict[str, Any]], batch_task_ids: list[str], probe_task_ids: list[str], iteration: int, batch_idx: int, args: argparse.Namespace) -> None:
    """Mutates each candidate in place with self_pass_rate/probe_pass_rate.
    Only two server replicas exist, so candidates are scored one at a time;
    within a candidate, self and probe run concurrently."""
    for cand in candidates:
        bank_path = cand["dir"] / "banks" / f"memory_{GROUP}.json"
        self_dir, probe_dir = cand["dir"] / "eval_self", cand["dir"] / "eval_probe"
        tag = f"cheap_v2_iter{iteration}_b{batch_idx}_k{cand['k']}"
        self_proc = launch_subset_eval(bank_path, self_dir, f"{tag}_self", batch_task_ids, "train", args.model, args.base_url)
        probe_proc = launch_subset_eval(bank_path, probe_dir, f"{tag}_probe", probe_task_ids, "dev", args.model, args.base_url_probe)
        cand["self_pass_rate"] = wait_subset_eval(self_proc, self_dir, f"{tag}_self")
        cand["probe_pass_rate"] = wait_subset_eval(probe_proc, probe_dir, f"{tag}_probe")


def run_validation_pass(router_model: RouterPolicy, iteration: int, task_ids: list[str], trajectories: dict[str, dict[str, Any]], probe_task_ids: list[str], args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic (greedy) full-domain bank, scored on the REAL full 57-task
    dev split. Expensive, trustworthy, not used for gradients.

    Reported split in two pieces, not just one aggregate number: every single
    batch update this whole run used the SAME fixed `probe_task_ids` (15 of
    the 57 dev tasks) as its reward -- the router had every opportunity to
    overfit specifically to those 15, and the aggregate 57-task pass_rate
    would hide that (15/57 of it IS exactly the tasks training was scored
    against). `probe_subset_pass_rate` (the 15) vs `held_out_pass_rate` (the
    other 42, never touched by any reward computation this run) is the
    actual overfitting check."""
    val_dir = OUTPUT_ROOT / f"iter{iteration}" / "validation"
    builder_config = RouterBuilderConfig(
        output=val_dir, record_protocol="cheap_reward_v2_validation",
        model=args.model, base_url=args.base_url, seed=20260822,
    )
    result = run_router_chain(router_model, GROUP, task_ids, trajectories, builder_config, greedy=True)
    route_counts = Counter(d["route"] for d in result.decisions)
    bank_path = val_dir / "banks" / f"memory_{GROUP}.json"
    eval_dir = val_dir / "eval_full_dev"
    proc = launch_subset_eval(bank_path, eval_dir, f"cheap_v2_iter{iteration}_validation", None, "dev", args.model, args.base_url)
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
        f"[iter{iteration} validation] routes={dict(route_counts)} active_entries={result.summary['active_entries']} "
        f"full_dev_pass_rate={pass_rate:.4f} probe_subset(15)={probe_subset_pass_rate} held_out(42)={held_out_pass_rate}",
        flush=True,
    )
    return {
        "route_counts": dict(route_counts), "active_entries": result.summary["active_entries"],
        "full_dev_pass_rate": pass_rate, "probe_subset_pass_rate": probe_subset_pass_rate,
        "held_out_pass_rate": held_out_pass_rate,
    }


def run_one_batch_update(
    router_model: RouterPolicy, optimizer: torch.optim.Optimizer, iteration: int, batch_idx: int,
    batch_task_ids: list[str], trajectories: dict[str, dict[str, Any]], canonical_bank: list[dict[str, Any]],
    position: int, total_task_count: int, probe_task_ids: list[str], previous_probe_pass_rate: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    """Returns (new_canonical_bank, new_previous_probe_pass_rate)."""
    candidates = sample_k_candidates(router_model, iteration, batch_idx, batch_task_ids, trajectories, canonical_bank, position, total_task_count, args)
    score_candidates(candidates, batch_task_ids, probe_task_ids, iteration, batch_idx, args)

    self_baseline = statistics.mean(1.0 if trajectories[t]["success"] else 0.0 for t in batch_task_ids)
    probe_rewards = [c["probe_pass_rate"] - previous_probe_pass_rate for c in candidates]
    mean_probe_reward = statistics.mean(probe_rewards)
    advantages = [r - mean_probe_reward for r in probe_rewards]

    optimizer.zero_grad()
    loss = torch.zeros(())
    for advantage, cand in zip(advantages, candidates):
        for decision in cand["decisions"]:
            loss = loss - advantage * decision["logprob"]
    loss.backward()
    optimizer.step()

    chosen = random.choice(candidates)
    print(
        f"[iter{iteration} b{batch_idx}] self_rewards={[round(c['self_pass_rate']-self_baseline,3) for c in candidates]} "
        f"probe_rewards={[round(r,3) for r in probe_rewards]} advantages={[round(a,3) for a in advantages]} "
        f"loss={loss.item():.4f} chosen_k={chosen['k']} chosen_probe_pass_rate={chosen['probe_pass_rate']:.4f}",
        flush=True,
    )

    log_record = {
        "iteration": iteration, "batch_idx": batch_idx,
        "route_counts": [dict(c["route_counts"]) for c in candidates],
        "active_entries": [c["active_entries"] for c in candidates],
        "self_pass_rates": [c["self_pass_rate"] for c in candidates],
        "self_baseline": self_baseline,
        "probe_pass_rates": [c["probe_pass_rate"] for c in candidates],
        "previous_probe_pass_rate": previous_probe_pass_rate,
        "probe_rewards": probe_rewards,
        "advantages": advantages,
        "loss": loss.item(),
        "chosen_k": chosen["k"],
    }
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return chosen["bank"], chosen["probe_pass_rate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rollouts-per-batch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--base-url-probe", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--yes", action="store_true", help="skip the pre-launch cost confirmation")
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    task_ids = sorted(trajectories)
    probe_task_ids, probe_baseline = load_probe_set()
    num_batches = len(split_batches(task_ids, args.batch_size))
    est_minutes_per_eval_pair = 78  # observed pace, DESIGN.md section 12
    est_hours = args.rollouts_per_batch * num_batches * est_minutes_per_eval_pair / 60 + 3.5
    print(
        f"plan: {num_batches} batches/iteration x {args.rollouts_per_batch} rollouts/batch = "
        f"{num_batches * args.rollouts_per_batch} eval-pairs, ~{est_hours:.1f}h/iteration "
        f"(at observed ~{est_minutes_per_eval_pair}min/pair) x {args.iterations} iteration(s)",
        flush=True,
    )
    if not args.yes:
        raise SystemExit("pass --yes to confirm this budget and actually launch")

    print(f"loaded {len(task_ids)} train trajectories; probe set: {probe_task_ids} baseline={probe_baseline:.4f}", flush=True)

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
        previous_probe_pass_rate = probe_baseline

        for batch_idx, batch_task_ids in enumerate(batches):
            canonical_bank, previous_probe_pass_rate = run_one_batch_update(
                router_model, optimizer, iteration, batch_idx, batch_task_ids, trajectories,
                canonical_bank, position, len(task_ids), probe_task_ids, previous_probe_pass_rate, args,
            )
            position += len(batch_task_ids)

        checkpoint_path = save_checkpoint(router_model, iteration, CHECKPOINT_DIR)
        validation = run_validation_pass(router_model, iteration, task_ids, trajectories, probe_task_ids, args)
        print(f"=== iteration {iteration} done === checkpoint={checkpoint_path} validation={validation}", flush=True)


if __name__ == "__main__":
    main()
