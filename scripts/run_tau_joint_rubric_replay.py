#!/usr/bin/env python3
"""Run and summarize the 5x5 joint memory/SFT rubric replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
SIMULATIONS = TAU_ROOT / "data/simulations"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def llm_args() -> str:
    return json.dumps(
        {
            "temperature": 0.0,
            "max_tokens": 1024,
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        separators=(",", ":"),
    )


def valid_result(path: Path, task_id: str) -> bool:
    if not path.exists():
        return False
    matches = [
        simulation
        for simulation in read_json(path)["simulations"]
        if str(simulation["task_id"]) == task_id
    ]
    return len(matches) == 1 and matches[0].get("termination_reason") != "infrastructure_error" and (matches[0].get("reward_info") or {}).get("reward") is not None


def run_one(run: dict[str, Any], output: Path, timeout: int) -> str:
    result_path = SIMULATIONS / run["save_name"] / "results.json"
    if valid_result(result_path, run["task_id"]):
        return f"cached {run['memory_rubric']} x {run['sft_rubric']} {run['domain']}.{run['task_index']}"
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain", run["domain"],
        "--agent", "sft_data_guided_agent" if run["sft_rubric"] != "none" else "llm_agent",
        "--agent-llm", "openai/qwen35-tau",
        "--agent-llm-args", llm_args(),
        "--user-llm", "openai/qwen35-tau",
        "--user-llm-args", llm_args(),
        "--task-ids", run["task_id"],
        "--num-trials", "1",
        "--max-concurrency", "1",
        "--max-steps", "200",
        "--timeout", str(timeout),
        "--max-retries", "2",
        "--retry-delay", "2",
        "--seed", "20260821",
        "--save-to", run["save_name"],
        "--verbose-logs",
        "--llm-log-mode", "latest",
        "--auto-resume",
        "--log-level", "INFO",
    ]
    environment = dict(os.environ)
    environment["TAU2_LLM_NL_ASSERTIONS"] = "openai/qwen35-tau"
    environment["TAU2_LLM_NL_ASSERTIONS_ARGS"] = llm_args()
    if run["guidance_path"]:
        environment["TAU_SFT_DATA_GUIDANCE_PATH"] = run["guidance_path"]
    else:
        environment.pop("TAU_SFT_DATA_GUIDANCE_PATH", None)
    if run["memory_path"]:
        environment["TAU2_AGENT_MEMORY_PATH"] = run["memory_path"]
        environment["TAU2_AGENT_MEMORY_RETRIEVAL"] = "bm25"
        environment["TAU2_AGENT_MEMORY_TOP_K"] = "3"
    else:
        environment.pop("TAU2_AGENT_MEMORY_PATH", None)
        environment.pop("TAU2_AGENT_MEMORY_RETRIEVAL", None)
        environment.pop("TAU2_AGENT_MEMORY_TOP_K", None)
    log = output / "logs" / f"{run['save_name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=TAU_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not valid_result(result_path, run["task_id"]):
        raise RuntimeError(f"invalid replay {run['save_name']}; see {log}")
    return f"complete {run['memory_rubric']} x {run['sft_rubric']} {run['domain']}.{run['task_index']}"


def summarize(runs: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rewards: dict[tuple[str, str, str, str], float] = {}
    terminations: dict[tuple[str, str], Counter] = {}
    for run in runs:
        result = read_json(SIMULATIONS / run["save_name"] / "results.json")
        simulation = next(item for item in result["simulations"] if str(item["task_id"]) == run["task_id"])
        key = (run["memory_rubric"], run["sft_rubric"], run["domain"], run["task_id"])
        rewards[key] = float(simulation["reward_info"]["reward"])
        terminations.setdefault((run["memory_rubric"], run["sft_rubric"]), Counter())[simulation["termination_reason"]] += 1
    levels = sorted({run["memory_rubric"] for run in runs}, key=lambda x: (x != "none", x))
    tasks = sorted({(run["domain"], run["task_id"]) for run in runs})
    cells = {}
    for memory_rubric in levels:
        for sft_rubric in levels:
            values = [rewards[(memory_rubric, sft_rubric, domain, task_id)] for domain, task_id in tasks]
            base = [rewards[("none", "none", domain, task_id)] for domain, task_id in tasks]
            sft_only = [rewards[("none", sft_rubric, domain, task_id)] for domain, task_id in tasks]
            memory_only = [rewards[(memory_rubric, "none", domain, task_id)] for domain, task_id in tasks]
            cells[f"{memory_rubric}__{sft_rubric}"] = {
                "reward_one": sum(values),
                "episodes": len(values),
                "helped_vs_sft_only": sum(value > control for value, control in zip(values, sft_only)),
                "hurt_vs_sft_only": sum(value < control for value, control in zip(values, sft_only)),
                "helped_vs_memory_only": sum(value > control for value, control in zip(values, memory_only)),
                "hurt_vs_memory_only": sum(value < control for value, control in zip(values, memory_only)),
                "synergy_sum": sum(value - sft - memory + baseline for value, sft, memory, baseline in zip(values, sft_only, memory_only, base)),
                "terminations": dict(terminations[(memory_rubric, sft_rubric)]),
            }
    summary = {
        "protocol": "tau_joint_writer_rubric_replay_v1",
        "final_test_used": False,
        "tasks": len(tasks),
        "runs": len(runs),
        "cells": cells,
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    runs = read_json(experiment / "run_manifest.json")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_runs) as pool:
        futures = {pool.submit(run_one, run, experiment, args.timeout): run for run in runs}
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append({"run": run, "error": repr(exc)})
                print(f"failed {run['save_name']}: {exc}", flush=True)
    if failures:
        write_json(experiment / "failures.json", failures)
        raise SystemExit(1)
    print(json.dumps(summarize(runs, experiment), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
