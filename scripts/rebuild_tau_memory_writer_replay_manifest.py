#!/usr/bin/env python3
"""Rebuild a writer replay manifest with source-shared validation tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_memory_lab.memory_writer_harness import (
    select_validation_task_ids,
    validate_writer_candidate,
)
from trajectory_memory_lab.storage import write_json


DOMAINS = ("airline", "retail", "telecom")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("third_party/tau2-bench/data")
    )
    parser.add_argument("--save-prefix", default="qwen35_mw_utility_v2")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    records = json.loads((experiment / "candidates.json").read_text())
    sources = json.loads((experiment / "source_manifest.json").read_text())[
        "sources"
    ]
    source_ids_by_domain = {
        domain: {
            str(source["task_id"])
            for source in sources
            if source["domain"] == domain
        }
        for domain in DOMAINS
    }
    domain_tasks = {}
    domain_train_ids = {}
    domain_rewards = {}
    for domain in DOMAINS:
        tasks = json.loads(
            (args.data_root / f"tau2/domains/{domain}/tasks.json").read_text()
        )
        domain_tasks[domain] = {str(task["id"]): task for task in tasks}
        domain_train_ids[domain] = [
            str(task_id)
            for task_id in json.loads(
                (
                    args.data_root
                    / f"tau2/domains/{domain}/split_tasks.json"
                ).read_text()
            )["train"]
        ]
        baseline = json.loads(
            (
                args.data_root
                / "simulations"
                / f"qwen35_base_{domain}_full_v1/results.json"
            ).read_text()
        )
        domain_rewards[domain] = {
            str(simulation["task_id"]): float(
                simulation["reward_info"]["reward"]
            )
            for simulation in baseline["simulations"]
            if isinstance(simulation.get("reward_info"), dict)
            and simulation["reward_info"].get("reward") is not None
        }

    for source_index, source in enumerate(sources):
        domain = source["domain"]
        trajectory = json.loads(
            (
                experiment
                / "sources"
                / f"{source_index:03d}"
                / "trajectory.json"
            ).read_text()
        )
        for record in records:
            if record["source_index"] == source_index:
                record["hard_validation"] = validate_writer_candidate(
                    record["candidate"], trajectory
                )
        source_query = {
            "memory": {
                "scope": domain,
                "content": json.dumps(trajectory["task"], ensure_ascii=False),
                "conditions": [],
                "exceptions": [],
            }
        }
        selected = select_validation_task_ids(
            source_query,
            [
                domain_tasks[domain][task_id]
                for task_id in domain_train_ids[domain]
            ],
            excluded_ids=source_ids_by_domain[domain],
            baseline_by_task=domain_rewards[domain],
        )
        for record in records:
            if (
                record["source_index"] == source_index
                and record["hard_validation"]["replay_required"]
            ):
                record["validation_tasks"] = selected
                record["save_name"] = (
                    f"{args.save_prefix}_{record['candidate_id']}"
                )

    write_json(experiment / "candidates.json", records)
    replay = [
        {
            "candidate_id": record["candidate_id"],
            "source_task_id": record["source_task_id"],
            "domain": record["domain"],
            "memory_path": record["memory_path"],
            "save_name": record["save_name"],
            "task_ids": record["validation_tasks"]["related"]
            + record["validation_tasks"]["scope_controls"],
            "related_task_ids": record["validation_tasks"]["related"],
            "scope_control_task_ids": record["validation_tasks"][
                "scope_controls"
            ],
        }
        for record in records
        if record["hard_validation"]["replay_required"]
    ]
    write_json(experiment / "replay_manifest.json", replay)
    grouped = {}
    for item in replay:
        grouped.setdefault(item["source_task_id"], set()).add(
            tuple(item["task_ids"])
        )
    if any(len(task_sets) != 1 for task_sets in grouped.values()):
        raise RuntimeError("Candidates from one source received different tasks")
    print(
        json.dumps(
            {
                "replay_candidates": len(replay),
                "sources_with_replay": len(grouped),
                "shared_task_sets_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
