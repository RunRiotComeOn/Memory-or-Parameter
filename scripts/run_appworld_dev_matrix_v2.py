#!/usr/bin/env python3
"""Score v2 allocation rubrics on AppWorld dev: one agent, several memory levels.

Mirrors the tau-bench dev matrix, with one addition learned there: a second
no-memory run is included by default so the benchmark's own run-to-run noise is
measured in the same session as the effects it would otherwise swamp.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from trajectory_memory_lab.writer_rubrics import ALLOC_RUBRIC_IDS


ROOT = Path(__file__).resolve().parents[1]
NONE = "none"
REPLICATE = "none_replicate"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def level_rewards(level_dir: Path) -> dict[str, float]:
    rewards = {}
    for path in sorted((level_dir / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        rewards[record["task_id"]] = 1.0 if record["trajectory"]["success"] else 0.0
    return rewards


def comparison(control: dict[str, float], treatment: dict[str, float]) -> dict[str, Any]:
    keys = sorted(control.keys() & treatment.keys())
    helped = [key for key in keys if treatment[key] > control[key]]
    hurt = [key for key in keys if treatment[key] < control[key]]
    return {
        "paired_tasks": len(keys),
        "control_reward": sum(control[key] for key in keys),
        "treatment_reward": sum(treatment[key] for key in keys),
        "helped": helped,
        "hurt": hurt,
        "flipped": len(helped) + len(hurt),
        "net_utility": (len(helped) - 2 * len(hurt)) / len(keys) if keys else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--no-replicate", action="store_true")
    args = parser.parse_args()

    bank_summary = read_json(args.banks / "summary.json")
    if bank_summary.get("source_split") != "train":
        raise ValueError("memory banks must be built from the train split")

    levels = [NONE, *ALLOC_RUBRIC_IDS]
    if not args.no_replicate:
        levels.append(REPLICATE)

    for level in levels:
        bank = None
        if level not in {NONE, REPLICATE}:
            bank = args.banks / "banks" / level / "memory_appworld.json"
            if not bank.exists():
                raise FileNotFoundError(bank)
        level_dir = args.output / level
        command = [
            str(ROOT.parent / "appworld_venv/bin/python"),
            "-u",
            str(ROOT / "scripts/run_appworld_rollout.py"),
            "--split", args.split,
            "--output", str(level_dir),
            "--experiment-name", f"alloc_dev_{level}_v1",
            "--model", args.model,
            "--max-parallel", str(args.max_parallel),
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
        ]
        if bank is not None:
            command += ["--memory-bank", str(bank), "--memory-top-k", str(args.memory_top_k)]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        log_path = args.output / "logs" / f"{level}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== level {level} ===", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode != 0:
            raise RuntimeError(f"level {level} failed; see {log_path}")

    rewards = {level: level_rewards(args.output / level) for level in levels}
    control = rewards[NONE]
    arms = bank_summary["arms"]
    summary_levels = {}
    for level in levels:
        values = rewards[level]
        arm = arms.get(level, {})
        bank_total = arm.get("bank_total", 0)
        delta = sum(values.values()) - sum(control[key] for key in values if key in control)
        summary_levels[level] = {
            "reward_one": sum(values.values()),
            "episodes": len(values),
            "pass_rate": sum(values.values()) / len(values) if values else None,
            "bank_entries": bank_total,
            "sft_selected": arm.get("sft_selected"),
            "routes": arm.get("routes"),
            "delta_vs_none": delta,
            "delta_per_bank_entry": (delta / bank_total) if bank_total else None,
            "vs_none": comparison(control, values),
        }
    noise = None
    if REPLICATE in rewards:
        paired = comparison(control, rewards[REPLICATE])
        noise = {
            "design": "identical no-memory configuration replayed twice",
            "paired_tasks": paired["paired_tasks"],
            "run1_reward": paired["control_reward"],
            "run2_reward": paired["treatment_reward"],
            "flipped_tasks": paired["flipped"],
            "flip_rate": paired["flipped"] / paired["paired_tasks"]
            if paired["paired_tasks"]
            else None,
            "flipped_task_ids": sorted(paired["helped"] + paired["hurt"]),
        }
    summary = {
        "protocol": "appworld_alloc_dev_matrix_v2",
        "benchmark": "appworld",
        "split": args.split,
        "agent": f"code-acting loop, {args.model}, no SFT",
        "memory_retrieval": f"bm25_top{args.memory_top_k}",
        "levels": summary_levels,
        "noise_replicate": noise,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
