#!/usr/bin/env python3
"""Reduce rubric-screening memory replay to one uplift and one retention task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    candidates = {
        item["candidate_id"]: item
        for item in read_json(args.experiment / "candidates.json")
    }
    original = read_json(args.experiment / "replay_manifest.json")
    reduced = []
    for item in original:
        record = candidates[item["candidate_id"]]
        validation = record["validation_tasks"]
        outcomes = validation.get("baseline_outcomes") or {}
        related = list(validation.get("related") or [])
        selected = []
        for wanted in (0.0, 1.0):
            match = next(
                (task_id for task_id in related if outcomes.get(task_id) == wanted),
                None,
            )
            if match is not None and match not in selected:
                selected.append(match)
        for task_id in related:
            if len(selected) >= 2:
                break
            if task_id not in selected:
                selected.append(task_id)
        reduced.append(
            {
                **item,
                "task_ids": selected,
                "related_task_ids": selected,
                "scope_control_task_ids": [],
                "timeout": args.timeout,
                "max_concurrency": 2,
                "rubric_id": record.get("rubric_id"),
                "baseline_outcomes": {task_id: outcomes.get(task_id) for task_id in selected},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reduced, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidates": len(reduced),
                "episodes": sum(len(item["task_ids"]) for item in reduced),
                "all_train_only": True,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
