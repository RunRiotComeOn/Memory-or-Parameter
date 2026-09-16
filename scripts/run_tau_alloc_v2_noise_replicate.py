#!/usr/bin/env python3
"""Re-run the no-memory control a second time to measure the replay noise floor.

Byte-identical inputs at temperature 0 are not bitwise reproducible on a shared
vLLM server: continuous batching changes reduction order.  This quantifies how
many dev tasks flip between two runs of the same configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_tau_alloc_dev_matrix_v2 import DOMAINS, SIMULATIONS, read_json, run_one, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="rep2")
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--memory-top-k", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)

    flips = {}
    for domain in DOMAINS:
        task_ids = list(map(str, manifest["dev"][domain]["task_ids"]))
        run = {
            "memory_rubric": f"none_{args.tag}",
            "domain": domain,
            "task_ids": task_ids,
            "memory_path": None,
            "save_name": f"qwen35_alloc_v2_m-none_{domain}_dev_v1_{args.tag}",
        }
        print(run_one(run, args), flush=True)
        first = {
            str(simulation["task_id"]): float(simulation["reward_info"]["reward"])
            for simulation in read_json(
                SIMULATIONS / f"qwen35_alloc_v2_m-none_{domain}_dev_v1/results.json"
            )["simulations"]
        }
        second = {
            str(simulation["task_id"]): float(simulation["reward_info"]["reward"])
            for simulation in read_json(SIMULATIONS / run["save_name"] / "results.json")[
                "simulations"
            ]
        }
        flips[domain] = {
            "tasks": len(first),
            "run1_reward": sum(first.values()),
            "run2_reward": sum(second.values()),
            "flipped": sorted(key for key in first if first[key] != second[key]),
        }
    total_tasks = sum(value["tasks"] for value in flips.values())
    total_flips = sum(len(value["flipped"]) for value in flips.values())
    summary = {
        "protocol": "tau_alloc_v2_noise_replicate",
        "design": "identical no-memory configuration replayed twice",
        "per_domain": flips,
        "tasks": total_tasks,
        "flipped_tasks": total_flips,
        "flip_rate": total_flips / total_tasks if total_tasks else None,
        "run1_reward": sum(value["run1_reward"] for value in flips.values()),
        "run2_reward": sum(value["run2_reward"] for value in flips.values()),
    }
    write_json(args.output / "noise_replicate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
