#!/usr/bin/env python3
"""Prepare identical, train-only SFT-writer inputs for four rubric variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_codex_tau_sft_data_teacher import compact_message, tool_schemas
from trajectory_memory_lab.tau_sft_data_writer import SFT_DATA_WRITER_SYSTEM
from trajectory_memory_lab.writer_rubrics import RUBRIC_IDS, rubric_block


ROOT = Path(__file__).resolve().parents[1]
TAU_DATA = ROOT / "third_party/tau2-bench/data"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(args.source_manifest)
    selected = {domain: [] for domain in ("airline", "retail", "telecom")}
    for source in source_manifest["sources"]:
        selected[source["domain"]].append(str(source["task_id"]))

    final_test = {}
    all_rows = {rubric_id: [] for rubric_id in RUBRIC_IDS}
    source_audit = []
    for domain, task_ids in selected.items():
        split = read_json(TAU_DATA / f"tau2/domains/{domain}/split_tasks.json")
        train = set(map(str, split["train"]))
        test = list(map(str, split["test"]))
        final_test[domain] = test
        if not set(task_ids) <= train or set(task_ids) & set(test):
            raise ValueError(f"{domain}: rubric source escaped the train split")

        result = read_json(TAU_DATA / f"simulations/qwen35_base_{domain}_full_v1/results.json")
        simulations = {str(item["task_id"]): item for item in result["simulations"]}
        tasks = {str(item["id"]): item for item in read_json(TAU_DATA / f"tau2/domains/{domain}/tasks.json")}
        schemas = tool_schemas(domain)
        for task_id in task_ids:
            simulation = simulations[task_id]
            task = {k: v for k, v in tasks[task_id].items() if k not in {"evaluation_criteria", "initial_state"}}
            student_input = {
                "task": task,
                "source_trajectory": {
                    "messages": [compact_message(message) for message in simulation["messages"]],
                    "reward": (simulation.get("reward_info") or {}).get("reward"),
                    "reward_feedback": simulation.get("reward_info"),
                    "termination_reason": simulation.get("termination_reason"),
                    "review": simulation.get("review"),
                },
                "policy": simulation.get("policy"),
                "tool_schemas": schemas,
            }
            source_audit.append({
                "domain": domain,
                "task_id": task_id,
                "source_reward": (simulation.get("reward_info") or {}).get("reward"),
            })
            for rubric_id in RUBRIC_IDS:
                all_rows[rubric_id].append({
                    "domain": domain,
                    "source_task_id": task_id,
                    "source_reward": (simulation.get("reward_info") or {}).get("reward"),
                    "rubric_id": rubric_id,
                    "system": SFT_DATA_WRITER_SYSTEM + rubric_block("sft", rubric_id),
                    "student_input": student_input,
                })

    for rubric_id, rows in all_rows.items():
        path = args.output / f"sft_inputs_{rubric_id}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    smoke_manifest = {
        "protocol": "tau_writer_rubric_smoke_v1",
        "teacher_writer": {domain: {"task_ids": [], "count": 0} for domain in selected},
        "writer_generation": {
            domain: {"task_ids": task_ids, "count": len(task_ids)}
            for domain, task_ids in selected.items()
        },
        "test": {
            domain: {"task_ids": task_ids, "count": len(task_ids)}
            for domain, task_ids in final_test.items()
        },
        "counts": {"writer_generation": sum(map(len, selected.values()))},
        "rubrics": list(RUBRIC_IDS),
        "final_test_used": False,
        "source_audit": source_audit,
    }
    write_json(args.output / "smoke_manifest.json", smoke_manifest)
    print(json.dumps({
        "sources": len(source_audit),
        "by_domain": {domain: len(task_ids) for domain, task_ids in selected.items()},
        "source_successes": sum(item["source_reward"] == 1 for item in source_audit),
        "source_failures": sum(item["source_reward"] == 0 for item in source_audit),
        "rubrics": list(RUBRIC_IDS),
        "final_test_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
