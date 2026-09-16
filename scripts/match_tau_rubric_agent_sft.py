#!/usr/bin/env python3
"""Match four rubric SFT datasets on the same successful live-replay tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from trajectory_memory_lab.writer_rubrics import RUBRIC_IDS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-common", type=int, default=12)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    allowed = {
        (domain, str(task_id))
        for domain, section in manifest["rubric_train"].items()
        for task_id in section["task_ids"]
    }
    dev_test = {
        (domain, str(task_id))
        for split in ("dev", "test")
        for domain, section in manifest[split].items()
        for task_id in section["task_ids"]
    }

    rows_by_rubric = {}
    keys_by_rubric = {}
    for rubric_id in RUBRIC_IDS:
        rows = read_jsonl(args.input_root / rubric_id / "all.jsonl")
        keyed = {(row["domain"], str(row["source_task_id"])): row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"{rubric_id}: duplicate source tasks")
        if set(keyed) - allowed or set(keyed) & dev_test:
            raise ValueError(f"{rubric_id}: split leakage in collected SFT data")
        rows_by_rubric[rubric_id] = keyed
        keys_by_rubric[rubric_id] = set(keyed)
    common = set.intersection(*(keys_by_rubric[rubric_id] for rubric_id in RUBRIC_IDS))
    if len(common) < args.minimum_common:
        raise ValueError(f"only {len(common)} common successful tasks; minimum is {args.minimum_common}")
    ordered = sorted(common, key=lambda key: (key[0], key[1]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    for rubric_id in RUBRIC_IDS:
        output = args.output_root / rubric_id / "matched.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(rows_by_rubric[rubric_id][key], ensure_ascii=False) + "\n" for key in ordered),
            encoding="utf-8",
        )
    audit = {
        "protocol": "tau_rubric_matched_agent_sft_v1",
        "rubrics": list(RUBRIC_IDS),
        "successes_before_matching": {rubric_id: len(keys_by_rubric[rubric_id]) for rubric_id in RUBRIC_IDS},
        "common_tasks": len(common),
        "common_by_domain": dict(Counter(domain for domain, _ in common)),
        "task_keys": [{"domain": domain, "task_id": task_id} for domain, task_id in ordered],
        "train_only": True,
        "dev_overlap": 0,
        "test_overlap": 0,
    }
    write_json(args.output_root / "matching_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
