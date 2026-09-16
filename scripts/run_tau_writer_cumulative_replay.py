#!/usr/bin/env python3
"""Replay τ-bench test tasks with Base-Writer and SFT-Writer memory banks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trajectory_memory_lab.memory_writer_harness import paired_utility
from trajectory_memory_lab.storage import write_json


ROOT = Path("/nas04/yixuh/memory")
DATA = ROOT / "third_party/tau2-bench/data"
SIMULATIONS = DATA / "simulations"
DOMAINS = ("airline", "retail", "telecom")


def _task_ids(domain: str) -> list[str]:
    split = json.loads(
        (DATA / f"tau2/domains/{domain}/split_tasks.json").read_text(encoding="utf-8")
    )
    return [str(task_id) for task_id in split["test"]]


def _rewards(path: Path, task_ids: list[str]) -> tuple[dict[str, float], int]:
    result = json.loads(path.read_text(encoding="utf-8"))
    wanted = set(task_ids)
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


def _run_one(run: dict[str, Any], log_dir: Path) -> dict[str, str]:
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


def _pooled(
    rewards: dict[str, dict[str, dict[str, float]]],
    baseline_arm: str,
    treatment_arm: str,
) -> dict[str, Any]:
    baseline: dict[str, float] = {}
    treatment: dict[str, float] = {}
    domains = {}
    for domain in DOMAINS:
        domain_metric = paired_utility(
            rewards[domain][baseline_arm], rewards[domain][treatment_arm]
        )
        domains[domain] = domain_metric
        for task_id, reward in rewards[domain][baseline_arm].items():
            key = f"{domain}:{task_id}"
            baseline[key] = reward
            treatment[key] = rewards[domain][treatment_arm][task_id]
    result = paired_utility(baseline, treatment)
    result["comparison"] = f"{treatment_arm}_vs_{baseline_arm}"
    result["domains"] = domains
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-writer-retention", type=Path, required=True)
    parser.add_argument("--sft-writer-retention", type=Path, required=True)
    parser.add_argument("--base-arm-name", default="base_writer")
    parser.add_argument("--sft-arm-name", default="sft_writer")
    parser.add_argument(
        "--run-tag",
        default="cumwriter_v1",
        help="Unique suffix for tau result directories; avoids stale cached runs.",
    )
    parser.add_argument("--max-parallel-runs", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "runtime_logs" / f"tau_cumulative_replay_{output.name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    retention_by_arm = {
        args.base_arm_name: args.base_writer_retention.resolve(),
        args.sft_arm_name: args.sft_writer_retention.resolve(),
    }
    runs = []
    for arm, retention in retention_by_arm.items():
        for domain in DOMAINS:
            memory_path = retention / f"memory_{domain}.json"
            if not memory_path.exists():
                raise FileNotFoundError(memory_path)
            runs.append(
                {
                    "run_id": f"{arm}_{domain}",
                    "arm": arm,
                    "domain": domain,
                    "memory_path": str(memory_path),
                    "task_ids": _task_ids(domain),
                    "save_name": (
                        f"qwen35_baseagent_{arm}_memory_{domain}_test_{args.run_tag}"
                    ),
                }
            )
    write_json(output / "replay_manifest.json", runs)
    failures = []
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
        write_json(output / "replay_failures.json", failures)
        raise SystemExit(1)

    rewards: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for domain in DOMAINS:
        task_ids = _task_ids(domain)
        baseline, errors = _rewards(
            SIMULATIONS / f"qwen35_base_{domain}_full_v1/results.json", task_ids
        )
        if errors or len(baseline) != len(task_ids):
            raise RuntimeError(f"invalid no-memory baseline for {domain}")
        rewards[domain]["no_memory"] = baseline
    for run in runs:
        arm_rewards, errors = _rewards(
            SIMULATIONS / run["save_name"] / "results.json", run["task_ids"]
        )
        if errors or len(arm_rewards) != len(run["task_ids"]):
            raise RuntimeError(f"invalid result for {run['run_id']}")
        rewards[run["domain"]][run["arm"]] = arm_rewards

    arm_pass_rates = {}
    for arm in ("no_memory", args.base_arm_name, args.sft_arm_name):
        values = [
            reward
            for domain in DOMAINS
            for reward in rewards[domain][arm].values()
        ]
        arm_pass_rates[arm] = sum(values) / len(values)
    summary = {
        "protocol": "tau_cumulative_memory_writer_replay_v1",
        "agent_model": "untrained Qwen/Qwen3.5-35B-A3B",
        "user_simulator_model": "untrained Qwen/Qwen3.5-35B-A3B",
        "retrieval": {"method": "bm25", "top_k": 3},
        "test_tasks": sum(len(_task_ids(domain)) for domain in DOMAINS),
        "arm_pass_rates": arm_pass_rates,
        "comparisons": {
            f"{args.base_arm_name}_vs_no_memory": _pooled(
                rewards, "no_memory", args.base_arm_name
            ),
            f"{args.sft_arm_name}_vs_no_memory": _pooled(
                rewards, "no_memory", args.sft_arm_name
            ),
            f"{args.sft_arm_name}_vs_{args.base_arm_name}": _pooled(
                rewards, args.base_arm_name, args.sft_arm_name
            ),
        },
    }
    write_json(output / "rewards.json", rewards)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
