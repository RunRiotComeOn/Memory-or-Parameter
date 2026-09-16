#!/usr/bin/env python3
"""Pilot: is a cheap in-batch reward usable, or is it just self-leakage?

router_reward_v1/DESIGN.md section 9's design evaluates every checkpoint on
the full 57-task held-out dev split (~3.5h each) -- correct but expensive.
The proposal under test here: skip the dev split entirely and score a
checkpoint's bank by rerunning it on the SAME small batch of train tasks
that produced it (batch_size=10), which is ~5.7x cheaper per eval (10/57 of
the tasks).

Risk: a memory entry is written from one specific task's own trajectory, so
its content is often close enough to that task's instruction that BM25
retrieval hands it straight back when the SAME task is rerun -- inflating
"accuracy" via self-recognition rather than measuring whether the entry
generalizes to anything else. This script checks for that directly: for
each of K stochastic router realizations over batch_task_ids, it measures
both

  in_batch_pass_rate  = rerun batch_task_ids with the resulting bank
  probe_pass_rate     = rerun a DISJOINT set of train tasks (never a memory
                         source in this bank) with the same bank

If in_batch reward and probe reward don't move together across the K
realizations, the cheap in-batch signal is not a usable proxy for the thing
we actually care about (does this routing generalize), no matter how much
variance it has on its own.

Baselines (no-memory pass rate on each task set) are read for free from
base_train_v2's own recorded `success` field -- no rerun needed, since an
empty bank's behavior doesn't depend on the router.
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

from trajectory_memory_lab.router_bank_builder import RouterBuilderConfig, run_router_chain  # noqa: E402
from trajectory_memory_lab.router_policy import RouterPolicy, load_checkpoint  # noqa: E402

GROUP = "appworld"
TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
OUTPUT_ROOT = ROOT / "router_reward_v1/pilot_inbatch_v1"
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


def launch_subset_eval(bank_path: Path, eval_dir: Path, experiment_name: str, task_ids: list[str], model: str, base_url: str) -> subprocess.Popen:
    """Starts the eval subprocess and returns immediately (does not wait) --
    lets the caller run several of these concurrently against independent
    server replicas. Each replica still serves with --max-num-seqs 1, so
    determinism is per-replica, not violated by running replicas in parallel
    (see router_reward_v1/DESIGN.md section 11)."""
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_rollout.py"),
        "--split", "train",
        "--output", str(eval_dir),
        "--experiment-name", experiment_name,
        "--memory-bank", str(bank_path),
        "--memory-top-k", "3",
        "--task-ids", *task_ids,
        "--max-parallel", "1",
        "--seed", "20260822",
        "--model", model,
        "--base-url", base_url,
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--checkpoint", default=str(ROOT / "router_reward_v1/checkpoints/router_iter3.pt"))
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="used for bank-building content generation, and as the self-eval server")
    parser.add_argument("--base-url-probe", default="http://127.0.0.1:8001/v1", help="independent server replica for the probe-eval, run concurrently with self-eval")
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    all_task_ids = sorted(trajectories)
    batch_task_ids = all_task_ids[: args.batch_size]
    probe_task_ids = all_task_ids[args.batch_size : 2 * args.batch_size]
    print(f"batch (memory source) tasks: {batch_task_ids}", flush=True)
    print(f"probe (disjoint, no memory from these) tasks: {probe_task_ids}", flush=True)

    baseline_batch = statistics.mean(1.0 if trajectories[t]["success"] else 0.0 for t in batch_task_ids)
    baseline_probe = statistics.mean(1.0 if trajectories[t]["success"] else 0.0 for t in probe_task_ids)
    print(f"baseline (no-memory) pass_rate: batch={baseline_batch:.4f} probe={baseline_probe:.4f}", flush=True)

    router_model = RouterPolicy()
    load_checkpoint(router_model, Path(args.checkpoint))
    print(f"loaded router checkpoint: {args.checkpoint}", flush=True)

    records = []
    for k in range(args.k):
        torch.manual_seed(90000 + k)
        run_dir = OUTPUT_ROOT / f"k{k}"
        builder_config = RouterBuilderConfig(
            output=run_dir, record_protocol="pilot_inbatch_v1_decision",
            model=args.model, base_url=args.base_url, seed=20260822 + k,
        )
        result = run_router_chain(
            router_model, GROUP, batch_task_ids, trajectories, builder_config,
            total_task_count=len(batch_task_ids),
        )
        route_counts = Counter(d["route"] for d in result.decisions)
        print(f"[k={k}] routes={dict(route_counts)} active_entries={result.summary['active_entries']}", flush=True)

        bank_path = run_dir / "banks" / f"memory_{GROUP}.json"
        self_dir, probe_dir = run_dir / "eval_in_batch", run_dir / "eval_probe"
        self_proc = launch_subset_eval(bank_path, self_dir, f"pilot_inbatch_k{k}_self", batch_task_ids, args.model, args.base_url)
        probe_proc = launch_subset_eval(bank_path, probe_dir, f"pilot_inbatch_k{k}_probe", probe_task_ids, args.model, args.base_url_probe)
        in_batch_pass_rate = wait_subset_eval(self_proc, self_dir, f"pilot_inbatch_k{k}_self")
        probe_pass_rate = wait_subset_eval(probe_proc, probe_dir, f"pilot_inbatch_k{k}_probe")

        record = {
            "k": k,
            "route_counts": dict(route_counts),
            "active_entries": result.summary["active_entries"],
            "in_batch_pass_rate": in_batch_pass_rate,
            "in_batch_reward": in_batch_pass_rate - baseline_batch,
            "probe_pass_rate": probe_pass_rate,
            "probe_reward": probe_pass_rate - baseline_probe,
        }
        records.append(record)
        print(f"[k={k}] in_batch_reward={record['in_batch_reward']:+.4f} probe_reward={record['probe_reward']:+.4f}", flush=True)

    write_json(OUTPUT_ROOT / "results.json", {
        "batch_task_ids": batch_task_ids, "probe_task_ids": probe_task_ids,
        "baseline_batch": baseline_batch, "baseline_probe": baseline_probe,
        "records": records,
    })

    in_batch = [r["in_batch_reward"] for r in records]
    probe = [r["probe_reward"] for r in records]
    print("\n=== summary ===")
    print(f"in_batch_reward: {in_batch}")
    print(f"probe_reward:    {probe}")
    if len(records) >= 2:
        try:
            correlation = statistics.correlation(in_batch, probe)
        except statistics.StatisticsError:
            correlation = float("nan")
        print(f"correlation(in_batch_reward, probe_reward) across k={args.k} realizations: {correlation:.3f}")
    print(f"wrote {OUTPUT_ROOT / 'results.json'}")


if __name__ == "__main__":
    main()
