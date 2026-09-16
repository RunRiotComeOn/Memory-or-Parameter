#!/usr/bin/env python3
"""Summarize R0-R3 writer candidates and their train-only live replays."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from trajectory_memory_lab.writer_rubrics import RUBRIC_IDS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()

    result: dict[str, Any] = {
        "protocol": "tau_writer_rubric_smoke_summary_v1",
        "final_test_used": False,
        "memory": {},
        "sft": {},
    }

    memory_candidates_path = experiment / "memory_candidates/candidates.json"
    memory_utilities_path = experiment / "memory_candidates/candidate_utilities.json"
    if memory_candidates_path.exists():
        records = (
            read_json(memory_utilities_path)
            if memory_utilities_path.exists()
            else read_json(memory_candidates_path)
        )
        for rubric_id in RUBRIC_IDS:
            selected = [item for item in records if item.get("rubric_id") == rubric_id]
            contents = [
                item.get("candidate", {}).get("memory", {}).get("content", "")
                for item in selected
            ]
            summary: dict[str, Any] = {
                "candidates": len(selected),
                "hard_accepted": sum(
                    bool(item.get("hard_validation", {}).get("accepted"))
                    for item in selected
                ),
                "mean_content_chars": (
                    sum(map(len, contents)) / len(contents) if contents else 0.0
                ),
            }
            if memory_utilities_path.exists():
                utilities = [item["utility"] for item in selected]
                summary.update(
                    {
                        "paired_episodes": sum(item["paired_tasks"] for item in utilities),
                        "helped": sum(len(item["helped"]) for item in utilities),
                        "hurt": sum(len(item["hurt"]) for item in utilities),
                        "net_utility": sum(float(item["net_utility"]) for item in utilities),
                        "harm_weighted_score": sum(
                            len(item["helped"]) - 2 * len(item["hurt"])
                            for item in utilities
                        ),
                    }
                )
            result["memory"][rubric_id] = summary

    source_rewards: dict[tuple[str, str], float] = {}
    manifest_path = experiment / "prepared/smoke_manifest.json"
    if manifest_path.exists():
        for item in read_json(manifest_path).get("source_audit", []):
            source_rewards[(item["domain"], str(item["task_id"]))] = float(item["source_reward"])

    for rubric_id in RUBRIC_IDS:
        candidate_dir = experiment / "sft_candidates" / rubric_id
        candidate_records = [
            read_json(path)
            for path in sorted(candidate_dir.glob("tasks/*/candidate.json"))
        ]
        ready = [item for item in candidate_records if item.get("status") == "prediction_ready"]
        turns = [
            turn
            for item in ready
            for turn in item.get("candidate", {}).get("assistant_turns", [])
        ]
        content_chars = sum(
            len(turn.get("content") or "") for turn in turns
        )
        tool_calls = sum(len(turn.get("tool_calls") or []) for turn in turns)
        summary = {
            "candidate_records": len(candidate_records),
            "prediction_ready": len(ready),
            "errors": len(candidate_records) - len(ready),
            "assistant_turns": len(turns),
            "tool_calls": tool_calls,
            "content_chars": content_chars,
            "mean_turns_per_ready_candidate": len(turns) / len(ready) if ready else 0.0,
        }

        outcomes = Counter()
        by_domain: dict[str, Any] = {}
        total_episodes = 0
        for domain in ("airline", "retail", "telecom"):
            replay_summary_path = experiment / f"sft_replays/{rubric_id}/summary_{domain}.json"
            if not replay_summary_path.exists():
                continue
            replay_summary = read_json(replay_summary_path)
            simulations = read_json(Path(replay_summary["results_path"]))["simulations"]
            domain_counts = Counter()
            terminations = Counter()
            for simulation in simulations:
                task_id = str(simulation["task_id"])
                source = source_rewards[(domain, task_id)]
                reward_info = simulation.get("reward_info") or {}
                replay = float(reward_info.get("reward") or 0.0)
                if source == 0 and replay == 1:
                    key = "repaired_failure"
                elif source == 1 and replay == 0:
                    key = "hurt_success"
                elif source == replay == 1:
                    key = "retained_success"
                else:
                    key = "unchanged_failure"
                outcomes[key] += 1
                domain_counts[key] += 1
                terminations[simulation.get("termination_reason") or "unknown"] += 1
                total_episodes += 1
            by_domain[domain] = {
                **domain_counts,
                "episodes": len(simulations),
                "reward_one": sum(
                    float((item.get("reward_info") or {}).get("reward") or 0.0)
                    for item in simulations
                ),
                "terminations": dict(terminations),
            }
        if total_episodes:
            summary.update(
                {
                    "live_replay_episodes": total_episodes,
                    **outcomes,
                    "harm_weighted_score": outcomes["repaired_failure"]
                    - 2 * outcomes["hurt_success"],
                    "by_domain": by_domain,
                }
            )
        result["sft"][rubric_id] = summary

    output = args.output or (experiment / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
