#!/usr/bin/env python3
"""Build AppWorld memory banks under the v2 allocation rubrics.

Same engine as the tau-bench run (`trajectory_memory_lab.alloc_bank_builder`);
only trajectory assembly differs.  AppWorld tasks all share one environment of
11 apps, so there is a single bank rather than one per domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trajectory_memory_lab.alloc_bank_builder import BuilderConfig, build_banks


GROUP = "appworld"


def load_rollout(rollout: Path, expected_split: str) -> dict[str, dict[str, Any]]:
    """Load completed rollout trajectories, refusing the wrong split."""
    protocol_path = rollout / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("split") != expected_split:
            raise ValueError(
                f"rollout split is {protocol.get('split')!r}, expected {expected_split!r}: "
                "memory must never be written from dev or test trajectories"
            )
    trajectories: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for path in sorted((rollout / "trajectories").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            skipped.append(record.get("task_id", path.stem))
            continue
        if record.get("memory_bank"):
            raise ValueError(
                f"{path} was rolled out with a memory bank attached; source "
                "trajectories for bank building must be memory-free"
            )
        trajectory = record["trajectory"]
        trajectory["domain"] = GROUP
        trajectories[record["task_id"]] = trajectory
    if skipped:
        print(f"skipped {len(skipped)} incomplete rollouts: {skipped[:5]}", flush=True)
    if not trajectories:
        raise ValueError(f"no complete trajectories under {rollout}")
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel-chains", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--budget-fraction", type=float, default=0.4)
    args = parser.parse_args()

    trajectories = load_rollout(args.rollout, args.split)
    task_ids = sorted(trajectories)
    successes = sum(1 for value in trajectories.values() if value["success"])

    config = BuilderConfig(
        output=args.output,
        record_protocol="appworld_alloc_memory_decision_v2",
        summary_protocol="appworld_alloc_memory_banks_v2",
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        seed=args.seed,
        budget_fraction=args.budget_fraction,
        max_parallel_chains=args.max_parallel_chains,
    )
    summary = build_banks(
        {GROUP: task_ids},
        {GROUP: trajectories},
        config,
        extra_summary={
            "benchmark": "appworld",
            "source_split": args.split,
            "source_tasks": len(task_ids),
            "source_base_successes": successes,
            "source_rollout": str(args.rollout),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
