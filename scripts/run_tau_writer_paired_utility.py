#!/usr/bin/env python3
"""Run and score the writer-only paired τ-bench utility experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trajectory_memory_lab.memory_writer_harness import paired_utility
from trajectory_memory_lab.storage import write_json


ROOT = Path("/nas04/yixuh/memory")
SIMULATIONS = ROOT / "third_party/tau2-bench/data/simulations"
ARMS = ("baseline", "base_writer", "sft_writer")


def _rewards(path: Path, task_ids: list[str]) -> tuple[dict[str, float], int]:
    result = json.loads(path.read_text(encoding="utf-8"))
    wanted = set(map(str, task_ids))
    rewards: dict[str, float] = {}
    errors = 0
    for simulation in result["simulations"]:
        task_id = str(simulation["task_id"])
        if task_id not in wanted:
            continue
        if simulation.get("termination_reason") == "infrastructure_error":
            errors += 1
        reward_info = simulation.get("reward_info")
        if isinstance(reward_info, dict) and reward_info.get("reward") is not None:
            rewards[task_id] = float(reward_info["reward"])
    return rewards, errors


def _run_one(run: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    result_path = SIMULATIONS / run["save_name"] / "results.json"
    if result_path.exists():
        rewards, errors = _rewards(result_path, run["task_ids"])
        if len(rewards) == len(run["task_ids"]) and errors == 0:
            return {"run_id": run["run_id"], "status": "cached"}
    env = os.environ.copy()
    env.update(
        {
            "TAU2_AGENT_LLM": "openai/qwen35-tau",
            "TAU2_AGENT_MEMORY_PATH": run["memory_path"],
            "TAU2_AGENT_MEMORY_RETRIEVAL": "bm25",
            "TAU2_AGENT_MEMORY_TOP_K": "3",
        }
    )
    command = [
        str(ROOT / "scripts/run_tau_bench.sh"),
        run["domain"],
        run["save_name"],
        "--task-ids",
        *run["task_ids"],
        "--num-trials",
        "1",
        "--max-concurrency",
        "2",
        "--max-steps",
        "200",
        "--timeout",
        "1200",
        "--max-retries",
        "2",
        "--retry-delay",
        "2",
        "--seed",
        "300",
        "--verbose-logs",
        "--llm-log-mode",
        "latest",
        "--auto-resume",
        "--log-level",
        "INFO",
    ]
    log_path = log_dir / f"{run['run_id']}.log"
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
        raise RuntimeError(f"exit={completed.returncode}; see {log_path}")
    rewards, errors = _rewards(result_path, run["task_ids"])
    if errors or len(rewards) != len(run["task_ids"]):
        raise RuntimeError(
            f"scored={len(rewards)}/{len(run['task_ids'])} infrastructure_errors={errors}"
        )
    return {"run_id": run["run_id"], "status": "complete"}


def _retrieval_stats(log_path: Path, update_ids: list[str]) -> dict[str, Any]:
    if not update_ids or not log_path.exists():
        return {"events": 0, "events_selecting_updated_memory": 0, "exposed": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    selections = re.findall(r"External memory retrieval selected ids=(\[[^\n]*?\])", text)
    hits = sum(any(memory_id in selection for memory_id in update_ids) for selection in selections)
    return {
        "events": len(selections),
        "events_selecting_updated_memory": hits,
        "exposed": hits > 0,
    }


def _pooled(records: list[dict[str, Any]], baseline_arm: str, treatment_arm: str) -> dict[str, Any]:
    baseline: dict[str, float] = {}
    treatment: dict[str, float] = {}
    by_source: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for record in records:
        by_source[record["source_index"]][record["arm"]] = record["rewards"]
    source_metrics = []
    for source_index, arm_rewards in sorted(by_source.items()):
        if baseline_arm not in arm_rewards or treatment_arm not in arm_rewards:
            continue
        metric = paired_utility(arm_rewards[baseline_arm], arm_rewards[treatment_arm])
        source_metrics.append({"source_index": source_index, **metric})
        for task_id, reward in arm_rewards[baseline_arm].items():
            key = f"s{source_index:03d}:{task_id}"
            baseline[key] = reward
            treatment[key] = arm_rewards[treatment_arm][task_id]
    result = paired_utility(baseline, treatment)
    result["comparison"] = f"{treatment_arm}_vs_{baseline_arm}"
    result["sources"] = source_metrics
    return result


def _summarize(experiment: Path, log_dir: Path) -> dict[str, Any]:
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    run_by_id = {run["run_id"]: run for run in manifest["runs"]}
    source_by_index = {
        source["source_index"]: source for source in manifest["sources"]
    }
    records = []
    for arm in manifest["arms"]:
        run = run_by_id[arm["run_id"]]
        rewards, errors = _rewards(
            SIMULATIONS / run["save_name"] / "results.json", run["task_ids"]
        )
        if errors or len(rewards) != len(run["task_ids"]):
            raise RuntimeError(f"incomplete result for {run['run_id']}")
        records.append(
            {
                **arm,
                "source_success": source_by_index[arm["source_index"]]["source_success"],
                "task_ids": run["task_ids"],
                "related_task_ids": run["related_task_ids"],
                "scope_control_task_ids": run["scope_control_task_ids"],
                "rewards": rewards,
                "pass_rate": sum(rewards.values()) / len(rewards),
                "retrieval": _retrieval_stats(
                    log_dir / f"{run['run_id']}.log", arm["update_memory_ids"]
                ),
            }
        )
    comparisons = {
        "base_writer_vs_baseline": _pooled(records, "baseline", "base_writer"),
        "sft_writer_vs_baseline": _pooled(records, "baseline", "sft_writer"),
        "sft_writer_vs_base_writer": _pooled(records, "base_writer", "sft_writer"),
    }
    arm_pass_rates = {
        arm: sum(
            sum(record["rewards"].values())
            for record in records
            if record["arm"] == arm
        )
        / sum(
            len(record["rewards"])
            for record in records
            if record["arm"] == arm
        )
        for arm in ARMS
    }
    retrieval = {
        arm: {
            "applied_updates": sum(
                bool(record["application"]["applied"])
                for record in records
                if record["arm"] == arm
            ),
            "sources_with_updated_memory_retrieved": sum(
                record["retrieval"]["exposed"]
                for record in records
                if record["arm"] == arm
            ),
            "retrieval_events_selecting_update": sum(
                record["retrieval"]["events_selecting_updated_memory"]
                for record in records
                if record["arm"] == arm
            ),
        }
        for arm in ("base_writer", "sft_writer")
    }
    summary = {
        "protocol": manifest["protocol"],
        "sources": len(manifest["sources"]),
        "logical_task_simulations_per_arm": sum(
            len(source["validation_tasks"]["related"])
            + len(source["validation_tasks"]["scope_controls"])
            for source in manifest["sources"]
        ),
        "arm_pass_rates": arm_pass_rates,
        "comparisons": comparisons,
        "retrieval": retrieval,
    }
    write_json(experiment / "arm_results.json", records)
    write_json(experiment / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=4)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    runs = json.loads((experiment / "replay_manifest.json").read_text(encoding="utf-8"))
    if args.max_sources is not None:
        source_indexes = sorted({run["source_index"] for run in runs})[: args.max_sources]
        runs = [run for run in runs if run["source_index"] in source_indexes]
    log_dir = ROOT / "runtime_logs" / f"tau_writer_pair_{experiment.name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    if not args.summarize_only:
        with ThreadPoolExecutor(max_workers=args.max_parallel_runs) as executor:
            futures = {executor.submit(_run_one, run, log_dir): run for run in runs}
            for future in as_completed(futures):
                run = futures[future]
                try:
                    result = future.result()
                    print(f"[{result['status']}] {result['run_id']}", flush=True)
                except Exception as exc:
                    failures.append({"run_id": run["run_id"], "error": repr(exc)})
                    print(f"[failed] {run['run_id']}: {exc}", flush=True)
    if failures:
        write_json(experiment / "replay_failures.json", failures)
        raise SystemExit(1)
    if args.max_sources is None:
        summary = _summarize(experiment, log_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"smoke_sources": args.max_sources, "failures": 0}, indent=2))


if __name__ == "__main__":
    main()
