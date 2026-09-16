#!/usr/bin/env python3
"""Compare two AppWorld runs task-by-task.

Reports the paired flip rate (the noise measure used in the alloc_v1 report) and,
because the runs are meant to be bit-reproducible here, also whether the agent
took a literally identical trajectory.  Success can coincide while the path
differs, which would still mean the run is not reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load(run_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted((run_dir / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        trajectory = record["trajectory"]
        assistant = [s["content"] for s in trajectory["steps"] if s["role"] == "assistant"]
        records[record["task_id"]] = {
            "success": bool(trajectory["success"]),
            "steps": len(trajectory["steps"]),
            "termination": trajectory.get("termination_reason"),
            "digest": hashlib.sha256("\x00".join(assistant).encode()).hexdigest()[:16],
        }
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    keys = sorted(a.keys() & b.keys())
    flipped = [k for k in keys if a[k]["success"] != b[k]["success"]]
    diverged = [k for k in keys if a[k]["digest"] != b[k]["digest"]]
    report = {
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "paired_tasks": len(keys),
        "only_in_a": sorted(a.keys() - b.keys()),
        "only_in_b": sorted(b.keys() - a.keys()),
        "reward_a": sum(a[k]["success"] for k in keys),
        "reward_b": sum(b[k]["success"] for k in keys),
        "flipped_tasks": len(flipped),
        "flip_rate": len(flipped) / len(keys) if keys else None,
        "flipped_task_ids": flipped,
        "trajectory_diverged_tasks": len(diverged),
        "trajectory_divergence_rate": len(diverged) / len(keys) if keys else None,
        "trajectory_diverged_task_ids": diverged,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
