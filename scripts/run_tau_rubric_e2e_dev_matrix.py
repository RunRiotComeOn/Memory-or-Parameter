#!/usr/bin/env python3
"""Run a true task-agent-LoRA x external-memory rubric matrix on frozen dev tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from trajectory_memory_lab.writer_rubrics import RUBRIC_IDS


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
SIMULATIONS = TAU_ROOT / "data/simulations"
DOMAINS = ("airline", "retail", "telecom")
BASE = "base"
LEVELS = (BASE,) + RUBRIC_IDS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def llm_args(max_tokens: int) -> str:
    return json.dumps(
        {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        separators=(",", ":"),
    )


def result_rewards(path: Path, task_ids: list[str]) -> dict[str, float]:
    if not path.exists():
        return {}
    wanted = set(task_ids)
    rewards = {}
    for simulation in read_json(path).get("simulations", []):
        task_id = str(simulation.get("task_id"))
        reward_info = simulation.get("reward_info")
        reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
        if task_id in wanted and isinstance(reward, (int, float)) and simulation.get("termination_reason") != "infrastructure_error":
            rewards[task_id] = float(reward)
    return rewards


def run_one(run: dict[str, Any], args: argparse.Namespace) -> str:
    result_path = SIMULATIONS / run["save_name"] / "results.json"
    if len(result_rewards(result_path, run["task_ids"])) == len(run["task_ids"]):
        return f"resume-skip {run['agent_rubric']} x {run['memory_rubric']} {run['domain']}"
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain", run["domain"],
        "--agent", "llm_agent",
        "--agent-llm", f"openai/{run['model']}",
        "--agent-llm-args", llm_args(args.max_tokens),
        "--user-llm", "openai/qwen35-tau",
        "--user-llm-args", llm_args(args.max_tokens),
        "--task-ids", *run["task_ids"],
        "--num-trials", "1",
        "--max-concurrency", str(args.max_concurrency),
        "--max-steps", "200",
        "--timeout", str(args.timeout),
        "--max-retries", "2",
        "--retry-delay", "2",
        "--seed", str(args.seed),
        "--save-to", run["save_name"],
        "--verbose-logs",
        "--llm-log-mode", "latest",
        "--auto-resume",
        "--log-level", "INFO",
    ]
    environment = dict(os.environ)
    environment["TAU2_LLM_NL_ASSERTIONS"] = "openai/qwen35-tau"
    environment["TAU2_LLM_NL_ASSERTIONS_ARGS"] = llm_args(args.max_tokens)
    environment.pop("TAU_SFT_DATA_GUIDANCE_PATH", None)
    if run["memory_path"]:
        environment["TAU2_AGENT_MEMORY_PATH"] = run["memory_path"]
        environment["TAU2_AGENT_MEMORY_RETRIEVAL"] = "bm25"
        environment["TAU2_AGENT_MEMORY_TOP_K"] = str(args.memory_top_k)
    else:
        environment.pop("TAU2_AGENT_MEMORY_PATH", None)
        environment.pop("TAU2_AGENT_MEMORY_RETRIEVAL", None)
        environment.pop("TAU2_AGENT_MEMORY_TOP_K", None)
    log_path = args.output / "logs" / f"{run['save_name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=TAU_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    final = result_rewards(result_path, run["task_ids"])
    if completed.returncode != 0 or len(final) != len(run["task_ids"]):
        raise RuntimeError(
            f"invalid dev replay {run['save_name']}: status={completed.returncode} rewards={len(final)}/{len(run['task_ids'])}"
        )
    return f"complete {run['agent_rubric']} x {run['memory_rubric']} {run['domain']}"


def comparison(control: dict[str, float], treatment: dict[str, float]) -> dict[str, Any]:
    keys = sorted(control.keys() & treatment.keys())
    return {
        "paired_tasks": len(keys),
        "control_reward": sum(control[key] for key in keys),
        "treatment_reward": sum(treatment[key] for key in keys),
        "helped": [key for key in keys if treatment[key] > control[key]],
        "hurt": [key for key in keys if treatment[key] < control[key]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--memory-top-k", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    if manifest["counts"]["dev"] != 23 or manifest["counts"]["test"] != 100:
        raise ValueError("expected frozen 23-task dev and 100-task test")

    model_by_agent = {BASE: "qwen35-tau"}
    model_by_agent.update({rubric_id: f"qwen35-agent-{rubric_id}" for rubric_id in RUBRIC_IDS})
    runs = []
    for agent_rubric in LEVELS:
        for memory_rubric in LEVELS:
            for domain in DOMAINS:
                task_ids = list(map(str, manifest["dev"][domain]["task_ids"]))
                memory_path = None
                if memory_rubric != BASE:
                    memory_path = args.memory_root / memory_rubric / f"memory_{domain}.json"
                    if not memory_path.exists():
                        raise FileNotFoundError(memory_path)
                runs.append(
                    {
                        "agent_rubric": agent_rubric,
                        "memory_rubric": memory_rubric,
                        "model": model_by_agent[agent_rubric],
                        "domain": domain,
                        "task_ids": task_ids,
                        "memory_path": str(memory_path.resolve()) if memory_path else None,
                        "save_name": f"qwen35_rubric_e2e_a-{agent_rubric}_m-{memory_rubric}_{domain}_dev_v1",
                    }
                )
    write_json(args.output / "run_manifest.json", runs)
    write_json(
        args.output / "protocol.json",
        {
            "protocol": "tau_rubric_e2e_dev_matrix_v1",
            "design": "5 task-agent levels x 5 memory levels x 23 frozen dev tasks",
            "agent_levels": list(LEVELS),
            "memory_levels": list(LEVELS),
            "dev_tasks": 23,
            "runs": 575,
            "same_task_guidance": False,
            "memory_source_split": "rubric_train",
            "memory_retrieval": f"bm25_top{args.memory_top_k}",
            "user_simulator": "frozen raw Qwen3.5-35B-A3B",
            "final_test_used": False,
            "seed": args.seed,
        },
    )
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_runs) as pool:
        futures = {pool.submit(run_one, run, args): run for run in runs}
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append({"run": run, "error": repr(exc)})
                print(f"failed {run['save_name']}: {exc}", flush=True)
    if failures:
        write_json(args.output / "failures.json", failures)
        raise SystemExit(1)

    cell_rewards: dict[str, dict[str, float]] = {}
    terminations: dict[str, Counter[str]] = {}
    for run in runs:
        cell = f"{run['agent_rubric']}__{run['memory_rubric']}"
        cell_rewards.setdefault(cell, {})
        terminations.setdefault(cell, Counter())
        result = read_json(SIMULATIONS / run["save_name"] / "results.json")
        for simulation in result["simulations"]:
            task_id = str(simulation["task_id"])
            if task_id not in set(run["task_ids"]):
                continue
            key = f"{run['domain']}:{task_id}"
            cell_rewards[cell][key] = float(simulation["reward_info"]["reward"])
            terminations[cell][simulation["termination_reason"]] += 1
    base = cell_rewards[f"{BASE}__{BASE}"]
    cells = {}
    for agent_rubric in LEVELS:
        for memory_rubric in LEVELS:
            cell = f"{agent_rubric}__{memory_rubric}"
            values = cell_rewards[cell]
            agent_only = cell_rewards[f"{agent_rubric}__{BASE}"]
            memory_only = cell_rewards[f"{BASE}__{memory_rubric}"]
            keys = sorted(values)
            cells[cell] = {
                "reward_one": sum(values.values()),
                "episodes": len(values),
                "pass_rate": sum(values.values()) / len(values),
                "vs_base": comparison(base, values),
                "vs_agent_only": comparison(agent_only, values),
                "vs_memory_only": comparison(memory_only, values),
                "synergy_sum": sum(values[key] - agent_only[key] - memory_only[key] + base[key] for key in keys),
                "terminations": dict(terminations[cell]),
            }
    summary = {
        "protocol": "tau_rubric_e2e_dev_matrix_v1",
        "final_test_used": False,
        "dev_tasks": 23,
        "runs": 575,
        "cells": cells,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
