#!/usr/bin/env python3
"""Prepare oracle-free inputs for trained SFT-data-writer generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_codex_tau_sft_data_teacher import DOMAINS, compact_message, tool_schemas
from trajectory_memory_lab.tau_sft_data_writer import SFT_DATA_WRITER_SYSTEM


ROOT = Path(__file__).resolve().parents[1]
TAU_DATA = ROOT / "third_party/tau2-bench/data"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    test = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["test"][domain]["task_ids"]
    }
    records = []
    schemas = {domain: tool_schemas(domain) for domain in DOMAINS}
    for domain in DOMAINS:
        result_data = read_json(
            TAU_DATA / f"simulations/qwen35_base_{domain}_full_v1/results.json"
        )
        simulations = {
            str(item["task_id"]): item for item in result_data["simulations"]
        }
        tasks = {
            str(item["id"]): item
            for item in read_json(TAU_DATA / f"tau2/domains/{domain}/tasks.json")
        }
        for raw_task_id in manifest["writer_generation"][domain]["task_ids"]:
            task_id = str(raw_task_id)
            if (domain, task_id) in test:
                raise ValueError("test task reached writer generation inputs")
            task = {
                key: value
                for key, value in tasks[task_id].items()
                if key not in {"evaluation_criteria", "initial_state"}
            }
            simulation = simulations[task_id]
            student_input = {
                "task": task,
                "source_trajectory": {
                    "messages": [
                        compact_message(message) for message in simulation["messages"]
                    ],
                    "reward": simulation.get("reward_info", {}).get("reward"),
                    "reward_feedback": simulation.get("reward_info"),
                    "termination_reason": simulation.get("termination_reason"),
                    "review": simulation.get("review"),
                },
                "policy": simulation.get("policy"),
                "tool_schemas": schemas[domain],
            }
            records.append(
                {
                    "domain": domain,
                    "source_task_id": task_id,
                    "source_reward": simulation.get("reward_info", {}).get("reward"),
                    "system": SFT_DATA_WRITER_SYSTEM,
                    "student_input": student_input,
                }
            )
    if len(records) != manifest["counts"]["writer_generation"]:
        raise ValueError("writer generation input count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "test_overlap": 0,
                "by_domain": {
                    domain: sum(item["domain"] == domain for item in records)
                    for domain in DOMAINS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
