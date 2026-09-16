#!/usr/bin/env python3
"""Prepare a leakage-safe 5x5 memory/SFT-rubric joint replay manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trajectory_memory_lab.writer_rubrics import RUBRIC_IDS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ready: dict[str, set[tuple[str, str]]] = {}
    for rubric_id in RUBRIC_IDS:
        records = [
            read_json(path)
            for path in sorted((experiment / f"sft_candidates/{rubric_id}/tasks").glob("*/candidate.json"))
        ]
        ready[rubric_id] = {
            (record["domain"], str(record["source_task_id"]))
            for record in records
            if record.get("status") == "prediction_ready"
        }
    common = set.intersection(*ready.values())
    source_audit = read_json(experiment / "prepared/smoke_manifest.json")["source_audit"]
    ordered_tasks = [
        (item["domain"], str(item["task_id"]))
        for item in source_audit
        if (item["domain"], str(item["task_id"])) in common
    ]
    if len(ordered_tasks) != 9:
        raise ValueError(f"expected 9 common SFT tasks, found {len(ordered_tasks)}")

    memory_records = read_json(experiment / "memory_candidates/candidates.json")
    memory_by_rubric_domain: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in memory_records:
        rubric_id = record.get("rubric_id")
        if rubric_id not in RUBRIC_IDS:
            continue
        if not record.get("hard_validation", {}).get("replay_required"):
            continue
        memory = record.get("candidate", {}).get("memory") or {}
        if not memory.get("content") or not memory.get("scope"):
            continue
        source_task = str(record["source_task_id"]).split(".", 2)[2]
        entry = {
            "id": record["candidate_id"],
            "scope": memory["scope"],
            "content": memory["content"],
            "conditions": memory.get("conditions") or [],
            "exceptions": memory.get("exceptions") or [],
            "status": "active",
            "source_task_id": source_task,
            "rubric_id": rubric_id,
        }
        memory_by_rubric_domain.setdefault((rubric_id, record["domain"]), []).append(entry)

    memory_paths: dict[tuple[str, str, str], Path] = {}
    memory_audit = []
    for rubric_id in RUBRIC_IDS:
        for domain, task_id in ordered_tasks:
            entries = [
                entry
                for entry in memory_by_rubric_domain.get((rubric_id, domain), [])
                if entry["source_task_id"] != task_id
            ]
            if any(entry["source_task_id"] == task_id for entry in entries):
                raise ValueError("same-task memory leakage")
            path = output / "memory_banks" / rubric_id / domain / f"task_{len(memory_audit):03d}.json"
            write_json(path, entries)
            memory_paths[(rubric_id, domain, task_id)] = path
            memory_audit.append(
                {
                    "memory_rubric": rubric_id,
                    "domain": domain,
                    "task_id": task_id,
                    "entries": len(entries),
                    "source_task_ids": [entry["source_task_id"] for entry in entries],
                    "same_task_leakage": False,
                }
            )

    levels = ("none",) + RUBRIC_IDS
    runs = []
    for memory_rubric in levels:
        for sft_rubric in levels:
            for task_index, (domain, task_id) in enumerate(ordered_tasks):
                guidance_path = None
                if sft_rubric != "none":
                    guidance_path = experiment / f"sft_replays/{sft_rubric}/guidance_{domain}.json"
                    guidance = read_json(guidance_path)
                    if task_id not in guidance["tasks"]:
                        raise ValueError(f"missing guidance {sft_rubric} {domain}.{task_id}")
                memory_path = (
                    memory_paths[(memory_rubric, domain, task_id)]
                    if memory_rubric != "none"
                    else None
                )
                runs.append(
                    {
                        "memory_rubric": memory_rubric,
                        "sft_rubric": sft_rubric,
                        "domain": domain,
                        "task_id": task_id,
                        "task_index": task_index,
                        "guidance_path": str(guidance_path) if guidance_path else None,
                        "memory_path": str(memory_path) if memory_path else None,
                        "save_name": (
                            f"qwen35_joint_rubric_m-{memory_rubric}_s-{sft_rubric}_"
                            f"{domain}_{task_index:02d}_v1"
                        ),
                    }
                )
    write_json(output / "run_manifest.json", runs)
    write_json(output / "memory_audit.json", memory_audit)
    write_json(
        output / "protocol.json",
        {
            "protocol": "tau_joint_writer_rubric_replay_v1",
            "final_test_used": False,
            "design": "5 memory levels x 5 SFT levels x 9 common tasks",
            "memory_levels": list(levels),
            "sft_levels": list(levels),
            "tasks": [{"domain": domain, "task_id": task_id} for domain, task_id in ordered_tasks],
            "runs": len(runs),
            "same_task_memory_excluded": True,
            "memory_retrieval": "bm25_top3",
            "seed": 20260821,
        },
    )
    print(json.dumps({"runs": len(runs), "tasks": len(ordered_tasks), "memory_audits": len(memory_audit), "final_test_used": False}, indent=2))


if __name__ == "__main__":
    main()
