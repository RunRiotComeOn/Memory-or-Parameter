#!/usr/bin/env python3
"""Per-batch GRPO training using SELF-replay reward instead of a fixed probe
set (DESIGN.md section 14). Sibling of train_router_cheap_reward.py, which
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
OUTPUT_ROOT = ROOT / "router_reward_v1/cheap_train_v3"
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
            output=cand_dir, record_protocol="cheap_reward_v3_decision",
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
            tag = f"cheap_v3_iter{iteration}_b{batch_idx}_k{cand['k']}_self"
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
        output=val_dir, record_protocol="cheap_reward_v3_validation",
        model=args.model, base_url=args.base_url, seed=20260822,
    )
    result = run_router_chain(router_model, GROUP, task_ids, trajectories, builder_config, greedy=True)
    route_counts = Counter(d["route"] for d in result.decisions)
    bank_path = val_dir / "banks" / f"memory_{GROUP}.json"
    eval_dir = val_dir / "eval_full_dev"
    proc = launch_subset_eval(bank_path, eval_dir, f"cheap_v3_iter{iteration}_validation", None, "dev", args.model, args.base_url)
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
    position: int, total_task_count: int, args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Returns the new canonical_bank. No cross-batch reward chaining is
    needed anymore -- each batch's self_baseline is computed fresh from
    that batch's own (already-known, no-memory) recorded success rate, not
    a running previous-checkpoint value."""
    candidates = sample_k_candidates(router_model, iteration, batch_idx, batch_task_ids, trajectories, canonical_bank, position, total_task_count, args)
    score_candidates_self_only(candidates, batch_task_ids, iteration, batch_idx, args)

    self_baseline = statistics.mean(1.0 if trajectories[t]["success"] else 0.0 for t in batch_task_ids)
    rewards = [c["self_pass_rate"] - self_baseline for c in candidates]
    mean_reward = statistics.mean(rewards)
    advantages = [r - mean_reward for r in rewards]

    optimizer.zero_grad()
    loss = torch.zeros(())
    for advantage, cand in zip(advantages, candidates):
        for decision in cand["decisions"]:
            loss = loss - advantage * decision["logprob"]
    loss.backward()
    optimizer.step()

    chosen = random.choice(candidates)
    print(
        f"[iter{iteration} b{batch_idx}] self_pass_rates={[round(c['self_pass_rate'],3) for c in candidates]} "
        f"self_baseline={self_baseline:.3f} rewards={[round(r,3) for r in rewards]} advantages={[round(a,3) for a in advantages]} "
        f"loss={loss.item():.4f} chosen_k={chosen['k']} chosen_self_pass_rate={chosen['self_pass_rate']:.4f}",
        flush=True,
    )

    log_record = {
        "iteration": iteration, "batch_idx": batch_idx,
        "route_counts": [dict(c["route_counts"]) for c in candidates],
        "active_entries": [c["active_entries"] for c in candidates],
        "self_pass_rates": [c["self_pass_rate"] for c in candidates],
        "self_baseline": self_baseline,
        "rewards": rewards,
        "advantages": advantages,
        "loss": loss.item(),
        "chosen_k": chosen["k"],
    }
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return chosen["bank"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rollouts-per-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
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
    est_hours = rounds_per_batch * num_batches * est_minutes_per_eval / 60 + 3.5
    print(
        f"plan: {num_batches} batches/iteration x {args.rollouts_per_batch} rollouts/batch "
        f"({rounds_per_batch} rounds/batch of 2-way concurrent self-evals) ~{est_hours:.1f}h/iteration "
        f"(at observed ~{est_minutes_per_eval}min/eval) x {args.iterations} iteration(s)",
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
