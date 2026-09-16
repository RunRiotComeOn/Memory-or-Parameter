#!/usr/bin/env python3
"""Create a frozen train/dev/test split for the rubric end-to-end experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_DATA = ROOT / "third_party/tau2-bench/data"
DOMAINS = ("airline", "retail", "telecom")
DEV_COUNTS = {"airline": 3, "retail": 10, "telecom": 10}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    source = read_json(args.source_manifest)
    manifest: dict[str, Any] = {
        "protocol": "tau_rubric_e2e_split_v1",
        "seed": args.seed,
        "rubric_train": {},
        # Compatibility alias for the existing guided-replay/collector scripts.
        "writer_generation": {},
        "dev": {},
        "test": {},
    }
    source_manifest = {
        "protocol": "tau_rubric_e2e_train_sources_v1",
        "seed": args.seed,
        "source_split": "upstream_train/rubric_train",
        "final_test_used": False,
        "sources": [],
    }
    audit = []
    global_train: set[tuple[str, str]] = set()
    global_dev: set[tuple[str, str]] = set()
    global_test: set[tuple[str, str]] = set()

    for domain_index, domain in enumerate(DOMAINS):
        pool = list(map(str, source["writer_generation"][domain]["task_ids"]))
        final_test = list(map(str, source["test"][domain]["task_ids"]))
        upstream = read_json(TAU_DATA / f"tau2/domains/{domain}/split_tasks.json")
        upstream_train = set(map(str, upstream["train"]))
        upstream_test = set(map(str, upstream["test"]))
        if set(pool) - upstream_train or set(pool) & upstream_test:
            raise ValueError(f"{domain}: source pool is not wholly upstream-train")
        if set(final_test) != upstream_test:
            raise ValueError(f"{domain}: final test differs from the upstream frozen test")

        results = read_json(
            TAU_DATA / f"simulations/qwen35_base_{domain}_full_v1/results.json"
        )
        rewards = {
            str(item["task_id"]): (item.get("reward_info") or {}).get("reward")
            for item in results["simulations"]
        }
        successes = [task_id for task_id in pool if rewards.get(task_id) == 1]
        failures = [task_id for task_id in pool if rewards.get(task_id) != 1]
        rng = random.Random(args.seed + domain_index * 1009)
        rng.shuffle(successes)
        rng.shuffle(failures)
        dev_count = DEV_COUNTS[domain]
        dev_success_count = round(dev_count * len(successes) / len(pool))
        dev_ids = successes[:dev_success_count] + failures[: dev_count - dev_success_count]
        if len(dev_ids) < dev_count:
            remainder = [task_id for task_id in pool if task_id not in dev_ids]
            rng.shuffle(remainder)
            dev_ids.extend(remainder[: dev_count - len(dev_ids)])
        rng.shuffle(dev_ids)
        train_ids = [task_id for task_id in pool if task_id not in set(dev_ids)]

        train_set = {(domain, task_id) for task_id in train_ids}
        dev_set = {(domain, task_id) for task_id in dev_ids}
        test_set = {(domain, task_id) for task_id in final_test}
        if train_set & dev_set or train_set & test_set or dev_set & test_set:
            raise ValueError(f"{domain}: leakage while creating rubric split")
        global_train |= train_set
        global_dev |= dev_set
        global_test |= test_set

        def section(task_ids: list[str]) -> dict[str, Any]:
            return {
                "task_ids": task_ids,
                "count": len(task_ids),
                "source_successes": sum(rewards.get(task_id) == 1 for task_id in task_ids),
                "source_failures": sum(rewards.get(task_id) != 1 for task_id in task_ids),
            }

        manifest["rubric_train"][domain] = section(train_ids)
        manifest["writer_generation"][domain] = section(train_ids)
        manifest["dev"][domain] = section(dev_ids)
        manifest["test"][domain] = {"task_ids": final_test, "count": len(final_test)}
        source_manifest["sources"].extend(
            {
                "source_task_id": f"tau2.{domain}.{task_id}",
                "domain": domain,
                "task_id": task_id,
                "success": rewards.get(task_id) == 1,
                "reward": rewards.get(task_id),
            }
            for task_id in train_ids
        )
        audit.extend(
            {"domain": domain, "task_id": task_id, "partition": "rubric_train", "source_reward": rewards.get(task_id)}
            for task_id in train_ids
        )
        audit.extend(
            {"domain": domain, "task_id": task_id, "partition": "dev", "source_reward": rewards.get(task_id)}
            for task_id in dev_ids
        )

    if global_train & global_dev or global_train & global_test or global_dev & global_test:
        raise ValueError("global split leakage")
    if len(global_train) != 80 or len(global_dev) != 23 or len(global_test) != 100:
        raise ValueError(
            f"unexpected split sizes: train={len(global_train)} dev={len(global_dev)} test={len(global_test)}"
        )
    manifest["counts"] = {
        "rubric_train": len(global_train),
        "writer_generation": len(global_train),
        "dev": len(global_dev),
        "test": len(global_test),
    }
    manifest["leakage_audit"] = {
        "train_vs_dev": len(global_train & global_dev),
        "train_vs_test": len(global_train & global_test),
        "dev_vs_test": len(global_dev & global_test),
        "passed": True,
    }
    write_json(args.output, manifest)
    write_json(args.output.parent / "train_source_manifest.json", source_manifest)
    write_json(args.output.parent / "split_audit.json", audit)
    print(json.dumps({"counts": manifest["counts"], "leakage_audit": manifest["leakage_audit"]}, indent=2))


if __name__ == "__main__":
    main()
