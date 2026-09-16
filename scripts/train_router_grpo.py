#!/usr/bin/env python3
"""GRPO training loop for the learned route router (router_reward_v1/DESIGN.md sections 7, 9).

One rollout = one full pass of the router (current parameters, stochastic
sampling) over all 90 base_train_v2 trajectories, split into `--batches`
sequential chunks along the SAME chain (bank state carries over between
chunks -- only the reward checkpoints are new). After each chunk, the
accumulated bank so far is evaluated on the full 57-task held-out dev split;
the reward attributed to that chunk's decisions is the marginal pass_rate
gain over the previous checkpoint (checkpoint 0 = the known, noise-free
empty-bank baseline from noise_serial_v1, 25/57, reused rather than
re-measured since an empty bank is router-independent by construction).

This replaces v1's single-reward-per-90-decisions design: crediting 90
independent routing decisions with one end-of-chain scalar was too coarse
(see DESIGN.md section 7's own risk note) and gave only `--group-size`
reward observations per iteration. Splitting into batches multiplies both
the number of reward observations and the held-out eval cost by `--batches`
per iteration -- there is no way to get more frequent reward without more
evals, since the proxy reward (DESIGN.md section 2.2) was already
invalidated (section 6) and using a smaller held-out subset per checkpoint
would reintroduce exactly the small-N noise problem this project has
otherwise been careful to avoid.

Advantage is computed per batch index across the group: for batch b,
advantage_{g,b} = marginal_reward_{g,b} - mean_g(marginal_reward at batch b).
This keeps the GRPO group-mean baseline (never a running max, see DESIGN.md
section 2.1) but compares chains at the same point in their chain rather
than only at the very end. One gradient step per iteration, accumulated
over all batches and rollouts -- theta is fixed throughout data collection,
so this stays a clean on-policy update.

SFT routing decisions are recorded and contribute to the policy gradient the
same as memory decisions, but v1 does not build or apply any SFT artifact --
the held-out eval only ever sees the memory bank (DESIGN.md section 8).

Expensive: each held-out eval is a full deterministic-serial AppWorld pass,
~3.5h, and one rollout now runs `--batches` of them. Default `--iterations 1`
on purpose -- this call is meant to produce one checkpoint and stop so the
result can be sanity-checked before spending more GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_bank_builder import (  # noqa: E402
    RouterBuilderConfig,
    run_router_chain,
)
from trajectory_memory_lab.router_policy import RouterPolicy, load_checkpoint, save_checkpoint  # noqa: E402

GROUP = "appworld"
TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
OUTPUT_ROOT = ROOT / "router_reward_v1"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
TRAIN_LOG = OUTPUT_ROOT / "train_log.jsonl"

# noise_serial_v1: two independent full deterministic dev passes with an
# empty bank both scored 25/57 with zero flipped tasks -- an empty bank is
# router-independent by construction, so this is safe to reuse as every
# chain's batch-0 checkpoint instead of re-measuring it every rollout.
BASELINE_PASS_RATE = 25 / 57

APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_train_trajectories() -> dict[str, dict[str, Any]]:
    """Mirrors generate_appworld_alloc_memory_banks_v2.load_rollout: split=train,
    complete + memory-free only."""
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
    if not trajectories:
        raise ValueError(f"no complete trajectories under {TRAIN_ROLLOUT}")
    return trajectories


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


def split_batches(task_ids: list[str], batches: int) -> list[list[str]]:
    n = len(task_ids)
    base, extra = divmod(n, batches)
    chunks = []
    start = 0
    for i in range(batches):
        size = base + (1 if i < extra else 0)
        chunks.append(task_ids[start : start + size])
        start += size
    return chunks


def run_held_out_eval(bank_path: Path, eval_dir: Path, experiment_name: str, args: argparse.Namespace) -> float:
    # run_appworld_rollout.py needs the *appworld* package (a separate venv,
    # scripts/run_appworld_serial_noise.sh is the known-good invocation this
    # mirrors) and refuses to run unless APPWORLD_ROOT resolves to a path
    # without the substring "memory" in it -- this project's own root
    # (/nas04/yixuh/memory) fails that check, so APPWORLD_ROOT must point
    # elsewhere (the appworld data root) rather than default to ".".
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON,
        "-u",
        str(ROOT / "scripts/run_appworld_rollout.py"),
        "--split", "dev",
        "--output", str(eval_dir),
        "--experiment-name", experiment_name,
        "--memory-bank", str(bank_path),
        "--memory-top-k", "3",
        "--max-parallel", "1",
        "--seed", "20260822",
        "--model", args.model,
        "--base-url", args.base_url,
    ]
    print(f"  launching held-out eval: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"held-out eval failed ({experiment_name}): exit {proc.returncode}")
    eval_summary = read_json(eval_dir / "summary.json")
    pass_rate = eval_summary["pass_rate"]
    if pass_rate is None:
        raise RuntimeError(f"held-out eval produced no pass_rate ({experiment_name}): {eval_summary}")
    return pass_rate


def run_one_rollout(
    router_model: RouterPolicy,
    iteration: int,
    g: int,
    task_ids: list[str],
    trajectories: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """One full chain over all train tasks, split into `args.batches` reward
    checkpoints. Returns one entry per batch: pass_rate, marginal_reward
    (vs. the previous checkpoint), and the live decisions made in that batch.
    """
    rollout_dir = OUTPUT_ROOT / "rollouts" / f"iter{iteration}" / f"g{g}"
    torch.manual_seed(20260822 + iteration * 1000 + g)

    batches = split_batches(task_ids, args.batches)
    bank: list[dict[str, Any]] = []
    position = 0
    previous_pass_rate = BASELINE_PASS_RATE
    batch_results: list[dict[str, Any]] = []

    for batch_idx, batch_task_ids in enumerate(batches):
        builder_config = RouterBuilderConfig(
            output=rollout_dir / f"b{batch_idx}",
            record_protocol="router_grpo_v1_decision",
            model=args.model,
            base_url=args.base_url,
            seed=20260822 + g,
        )
        print(
            f"[iter{iteration} g{g} b{batch_idx}] routing {len(batch_task_ids)} tasks "
            f"(domain position {position}/{len(task_ids)})",
            flush=True,
        )
        result = run_router_chain(
            router_model,
            GROUP,
            batch_task_ids,
            trajectories,
            builder_config,
            initial_bank=bank,
            start_position=position,
            total_task_count=len(task_ids),
        )
        bank = result.bank
        position += len(batch_task_ids)
        route_counts = Counter(d["route"] for d in result.decisions)
        print(
            f"[iter{iteration} g{g} b{batch_idx}] routes={dict(route_counts)} "
            f"active_entries={result.summary['active_entries']}",
            flush=True,
        )

        bank_path = builder_config.output / "banks" / f"memory_{GROUP}.json"
        eval_dir = builder_config.output / "eval"
        pass_rate = run_held_out_eval(
            bank_path, eval_dir, f"router_grpo_iter{iteration}_g{g}_b{batch_idx}", args
        )
        marginal_reward = pass_rate - previous_pass_rate
        print(
            f"[iter{iteration} g{g} b{batch_idx}] checkpoint pass_rate={pass_rate:.4f} "
            f"marginal_reward={marginal_reward:+.4f}",
            flush=True,
        )
        batch_results.append(
            {
                "batch_idx": batch_idx,
                "pass_rate": pass_rate,
                "marginal_reward": marginal_reward,
                "decisions": result.decisions,
            }
        )
        previous_pass_rate = pass_rate

    return batch_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    task_ids = sorted(trajectories)
    print(f"loaded {len(task_ids)} train trajectories from {TRAIN_ROLLOUT}", flush=True)

    router_model = RouterPolicy()
    start_iteration = latest_checkpoint_iteration() + 1
    if start_iteration > 1:
        ckpt = CHECKPOINT_DIR / f"router_iter{start_iteration - 1}.pt"
        load_checkpoint(router_model, ckpt)
        print(f"resumed from {ckpt}", flush=True)
    optimizer = torch.optim.Adam(router_model.parameters(), lr=args.lr)

    for offset in range(args.iterations):
        iteration = start_iteration + offset
        rollouts: list[list[dict[str, Any]]] = []
        for g in range(args.group_size):
            rollouts.append(run_one_rollout(router_model, iteration, g, task_ids, trajectories, args))

        # advantage per batch index, compared across the group at that same
        # point in the chain -- not one advantage per whole rollout.
        optimizer.zero_grad()
        loss = torch.zeros(())
        advantages: list[list[float]] = [[] for _ in range(args.group_size)]
        for batch_idx in range(args.batches):
            marginals_at_b = [rollouts[g][batch_idx]["marginal_reward"] for g in range(args.group_size)]
            mean_b = statistics.mean(marginals_at_b)
            for g in range(args.group_size):
                advantage = marginals_at_b[g] - mean_b
                advantages[g].append(advantage)
                for decision in rollouts[g][batch_idx]["decisions"]:
                    loss = loss - advantage * decision["logprob"]
        loss.backward()
        optimizer.step()

        checkpoint_path = save_checkpoint(router_model, iteration, CHECKPOINT_DIR)
        route_dist = Counter(
            d["route"] for g_batches in rollouts for batch in g_batches for d in batch["decisions"]
        )
        log_record = {
            "iteration": iteration,
            "final_pass_rates": [g_batches[-1]["pass_rate"] for g_batches in rollouts],
            "checkpoint_pass_rates": [[b["pass_rate"] for b in g_batches] for g_batches in rollouts],
            "marginal_rewards": [[b["marginal_reward"] for b in g_batches] for g_batches in rollouts],
            "advantages": advantages,
            "loss": loss.item(),
            "route_distribution_counts": dict(route_dist),
            "checkpoint": str(checkpoint_path),
        }
        TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TRAIN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        print(f"=== iteration {iteration} done === {json.dumps(log_record, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
