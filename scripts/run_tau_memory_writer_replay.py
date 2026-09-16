#!/usr/bin/env python3
"""Run candidate-memory treatments and compute paired utility."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trajectory_memory_lab.memory_writer_harness import paired_utility
from trajectory_memory_lab.storage import write_json


ROOT = Path("/nas04/yixuh/memory")
SIMULATIONS = ROOT / "third_party/tau2-bench/data/simulations"


def _rewards(
    path: Path, task_ids: list[str] | None = None
) -> tuple[dict[str, float], int]:
    result = json.loads(path.read_text(encoding="utf-8"))
    rewards = {}
    infrastructure_errors = 0
    requested = set(map(str, task_ids)) if task_ids is not None else None
    for simulation in result["simulations"]:
        task_id = str(simulation["task_id"])
        if requested is not None and task_id not in requested:
            continue
        if simulation.get("termination_reason") == "infrastructure_error":
            infrastructure_errors += 1
        reward_info = simulation.get("reward_info")
        if isinstance(reward_info, dict) and reward_info.get("reward") is not None:
            rewards[task_id] = float(reward_info["reward"])
    return rewards, infrastructure_errors


def _run_one(item: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    result_path = SIMULATIONS / item["save_name"] / "results.json"
    if result_path.exists():
        rewards, errors = _rewards(result_path, item["task_ids"])
        if len(rewards) == len(item["task_ids"]) and errors == 0:
            return {"candidate_id": item["candidate_id"], "status": "cached"}
    env = os.environ.copy()
    env.update(
        {
            "TAU2_AGENT_LLM": "openai/qwen35-tau",
            "TAU2_AGENT_MEMORY_PATH": item["memory_path"],
            "TAU2_AGENT_MEMORY_RETRIEVAL": "bm25",
            "TAU2_AGENT_MEMORY_TOP_K": "1",
        }
    )
    command = [
        str(ROOT / "scripts/run_tau_bench.sh"),
        item["domain"],
        item["save_name"],
        "--task-ids",
        *item["task_ids"],
        "--num-trials",
        "1",
        "--max-concurrency",
        str(item.get("max_concurrency", 2)),
        "--max-steps",
        "200",
        "--timeout",
        str(item.get("timeout", 1200)),
        "--max-retries",
        "2",
        "--retry-delay",
        "2",
        "--seed",
        str(item.get("seed", 300)),
        "--verbose-logs",
        "--llm-log-mode",
        "latest",
        "--auto-resume",
        "--log-level",
        "INFO",
    ]
    log_path = log_dir / f"{item['candidate_id']}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{item['candidate_id']} exited {completed.returncode}; see {log_path}"
        )
    rewards, errors = _rewards(result_path, item["task_ids"])
    if errors or len(rewards) != len(item["task_ids"]):
        raise RuntimeError(
            f"{item['candidate_id']} scored={len(rewards)}/{len(item['task_ids'])} "
            f"infrastructure_errors={errors}"
        )
    return {"candidate_id": item["candidate_id"], "status": "complete"}


def _summarize(experiment: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = json.loads((experiment / "candidates.json").read_text())
    by_candidate = {item["candidate_id"]: item for item in manifest}
    baseline_cache = {}
    utility_records = []
    for record in candidates:
        candidate_id = record["candidate_id"]
        if record["candidate"]["operation"] == "noop":
            utility_records.append(
                {
                    **record,
                    "utility": {
                        "paired_tasks": 0,
                        "helped": [],
                        "hurt": [],
                        "pass_rate_delta": 0.0,
                        "net_utility": 0.0,
                        "basis": "explicit_noop_control",
                    },
                }
            )
            continue
        if not record["hard_validation"]["replay_required"]:
            utility_records.append(
                {
                    **record,
                    "utility": {
                        "paired_tasks": 0,
                        "helped": [],
                        "hurt": [],
                        "pass_rate_delta": None,
                        "net_utility": -1.0,
                        "basis": "hard_validation_rejection",
                    },
                }
            )
            continue
        item = by_candidate[candidate_id]
        domain = item["domain"]
        if domain not in baseline_cache:
            baseline_cache[domain], baseline_errors = _rewards(
                SIMULATIONS / f"qwen35_base_{domain}_full_v1/results.json"
            )
            if baseline_errors:
                raise RuntimeError(f"{domain} baseline has infrastructure errors")
        treatment, treatment_errors = _rewards(
            SIMULATIONS / item["save_name"] / "results.json",
            item["task_ids"],
        )
        if treatment_errors:
            raise RuntimeError(f"{candidate_id} treatment has infrastructure errors")
        selected_baseline = {
            task_id: baseline_cache[domain][task_id] for task_id in item["task_ids"]
        }
        utility = paired_utility(selected_baseline, treatment)
        utility["basis"] = "paired_base_without_vs_candidate_memory_with"
        utility["related_task_ids"] = item["related_task_ids"]
        utility["scope_control_task_ids"] = item["scope_control_task_ids"]
        utility_records.append({**record, "utility": utility})

    preferences = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in utility_records:
        by_source.setdefault(record["source_task_id"], []).append(record)
    for source_task_id, records in by_source.items():
        source_index = records[0]["source_index"]
        trajectory = json.loads(
            (
                experiment
                / "sources"
                / f"{source_index:03d}"
                / "trajectory.json"
            ).read_text()
        )
        ranked = sorted(
            records,
            key=lambda item: (
                -float(item["utility"]["net_utility"]),
                item["candidate_id"],
            ),
        )
        best = ranked[0]
        for rejected in ranked[1:]:
            if best["utility"]["net_utility"] <= rejected["utility"]["net_utility"]:
                continue
            preferences.append(
                {
                    "source_task_id": source_task_id,
                    "writer_input": {
                        "current_memory": [],
                        "trajectory": trajectory,
                    },
                    "chosen": best["candidate"],
                    "chosen_candidate_id": best["candidate_id"],
                    "chosen_utility": best["utility"],
                    "rejected": rejected["candidate"],
                    "rejected_candidate_id": rejected["candidate_id"],
                    "rejected_utility": rejected["utility"],
                }
            )
    write_json(experiment / "candidate_utilities.json", utility_records)
    with (experiment / "preference_pairs.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for preference in preferences:
            handle.write(json.dumps(preference, ensure_ascii=False) + "\n")
    accepted = [
        record
        for record in utility_records
        if record["hard_validation"]["replay_required"]
    ]
    summary = {
        "candidates": len(utility_records),
        "replayed_candidates": len(accepted),
        "positive_utility": sum(
            record["utility"]["net_utility"] > 0 for record in accepted
        ),
        "harmful": sum(bool(record["utility"]["hurt"]) for record in accepted),
        "neutral": sum(
            record["utility"]["net_utility"] == 0 for record in accepted
        ),
        "preference_pairs": len(preferences),
    }
    write_json(experiment / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=6)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    manifest = json.loads((experiment / "replay_manifest.json").read_text())
    log_dir = ROOT / "runtime_logs" / f"tau_memory_writer_{experiment.name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel_runs) as executor:
        futures = {
            executor.submit(_run_one, item, log_dir): item for item in manifest
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                print(
                    f"[{result['status']}] {result['candidate_id']}", flush=True
                )
            except Exception as exc:
                failures.append({"candidate_id": item["candidate_id"], "error": repr(exc)})
                print(f"[failed] {item['candidate_id']}: {exc}", flush=True)
    if failures:
        write_json(experiment / "replay_failures.json", failures)
        raise SystemExit(1)
    summary = _summarize(experiment, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
