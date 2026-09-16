#!/usr/bin/env python3
"""Create leakage-safe splits for the tau SFT-data-writer experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_DATA = ROOT / "third_party/tau2-bench/data"
DOMAINS = ("airline", "retail", "telecom")
META_COUNTS = {"airline": 15, "retail": 30, "telecom": 30}
META_FAILURE_COUNTS = {"airline": 5, "retail": 12, "telecom": 15}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()

    manifest: dict[str, Any] = {
        "protocol": "tau_sft_data_writer_split_v1",
        "seed": args.seed,
        "teacher_writer": {},
        "writer_generation": {},
        "test": {},
    }
    global_meta: set[tuple[str, str]] = set()
    global_generation: set[tuple[str, str]] = set()
    global_test: set[tuple[str, str]] = set()

    for domain_index, domain in enumerate(DOMAINS):
        domain_root = TAU_DATA / f"tau2/domains/{domain}"
        split = read_json(domain_root / "split_tasks.json")
        tasks = {str(item["id"]): item for item in read_json(domain_root / "tasks.json")}
        result_path = TAU_DATA / f"simulations/qwen35_base_{domain}_full_v1/results.json"
        simulations = {
            str(item["task_id"]): item
            for item in read_json(result_path)["simulations"]
        }
        train_ids = list(map(str, split["train"]))
        test_ids = list(map(str, split["test"]))
        if set(train_ids) & set(test_ids):
            raise ValueError(f"{domain}: upstream train/test overlap")
        if set(train_ids) - simulations.keys():
            raise ValueError(f"{domain}: source results are missing train tasks")

        failed_with_actions = []
        successful = []
        for task_id in train_ids:
            reward = simulations[task_id].get("reward_info", {}).get("reward")
            actions = (
                (tasks[task_id].get("evaluation_criteria") or {}).get("actions") or []
            )
            if reward == 1:
                successful.append(task_id)
            elif actions:
                failed_with_actions.append(task_id)

        rng = random.Random(args.seed + domain_index * 1009)
        rng.shuffle(failed_with_actions)
        rng.shuffle(successful)
        failure_count = META_FAILURE_COUNTS[domain]
        meta_count = META_COUNTS[domain]
        meta_ids = failed_with_actions[:failure_count] + successful[
            : meta_count - failure_count
        ]
        rng.shuffle(meta_ids)
        if len(meta_ids) != meta_count:
            raise ValueError(f"{domain}: unable to construct requested teacher subset")
        generation_ids = [task_id for task_id in train_ids if task_id not in meta_ids]

        meta_set = {(domain, task_id) for task_id in meta_ids}
        generation_set = {(domain, task_id) for task_id in generation_ids}
        test_set = {(domain, task_id) for task_id in test_ids}
        if meta_set & generation_set or meta_set & test_set or generation_set & test_set:
            raise ValueError(f"{domain}: leakage detected while constructing split")
        global_meta |= meta_set
        global_generation |= generation_set
        global_test |= test_set

        manifest["teacher_writer"][domain] = {
            "task_ids": meta_ids,
            "count": len(meta_ids),
            "source_successes": sum(
                simulations[task_id].get("reward_info", {}).get("reward") == 1
                for task_id in meta_ids
            ),
            "source_failures": sum(
                simulations[task_id].get("reward_info", {}).get("reward") != 1
                for task_id in meta_ids
            ),
        }
        manifest["writer_generation"][domain] = {
            "task_ids": generation_ids,
            "count": len(generation_ids),
        }
        manifest["test"][domain] = {"task_ids": test_ids, "count": len(test_ids)}

    if global_meta & global_generation or global_meta & global_test or global_generation & global_test:
        raise ValueError("global leakage detected")
    manifest["counts"] = {
        "teacher_writer": len(global_meta),
        "writer_generation": len(global_generation),
        "test": len(global_test),
    }
    manifest["leakage_audit"] = {
        "teacher_writer_vs_writer_generation": len(global_meta & global_generation),
        "teacher_writer_vs_test": len(global_meta & global_test),
        "writer_generation_vs_test": len(global_generation & global_test),
        "passed": True,
    }
    write_json(args.output, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["leakage_audit"], indent=2))


if __name__ == "__main__":
    main()
