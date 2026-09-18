#!/usr/bin/env python3
"""Score v2 allocation rubrics: frozen base agent x five memory levels on dev.

The agent is held at raw base for every cell, so the only thing that differs
between cells is which trajectories the allocation rubric chose to turn into
memory.  Bank sizes differ by design, so utility is reported per bank entry as
well as in absolute pass rate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from trajectory_memory_lab.writer_rubrics import ALLOC_RUBRIC_IDS


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
SIMULATIONS = TAU_ROOT / "data/simulations"
DOMAINS = ("airline", "retail", "telecom")
NONE = "none"
LEVELS = (NONE,) + ALLOC_RUBRIC_IDS


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
        if (
            task_id in wanted
            and isinstance(reward, (int, float))
            and simulation.get("termination_reason") != "infrastructure_error"
        ):
            rewards[task_id] = float(reward)
    return rewards


def run_one(run: dict[str, Any], args: argparse.Namespace) -> str:
    result_path = SIMULATIONS / run["save_name"] / "results.json"
    if len(result_rewards(result_path, run["task_ids"])) == len(run["task_ids"]):
        return f"resume-skip {run['memory_rubric']} {run['domain']}"
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain", run["domain"],
        "--agent", "llm_agent",
        "--agent-llm", "openai/qwen35-tau",
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
        for key in (
            "TAU2_AGENT_MEMORY_PATH",
            "TAU2_AGENT_MEMORY_RETRIEVAL",
            "TAU2_AGENT_MEMORY_TOP_K",
        ):
            environment.pop(key, None)
    log_path = args.output / "logs" / f"{run['save_name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=TAU_ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    final = result_rewards(result_path, run["task_ids"])
    if completed.returncode != 0 or len(final) != len(run["task_ids"]):
        raise RuntimeError(
            f"invalid dev replay {run['save_name']}: status={completed.returncode} "
            f"rewards={len(final)}/{len(run['task_ids'])}"
        )
    return f"complete {run['memory_rubric']} {run['domain']}"


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
        "net_utility": (len(helped) - 2 * len(hurt)) / len(keys) if keys else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--bank-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=5)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--memory-top-k", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    if manifest["counts"]["dev"] != 23 or manifest["counts"]["test"] != 100:
        raise ValueError("expected frozen 23-task dev and 100-task test")
    bank_summary = read_json(args.bank_summary)

    runs = []
    for memory_rubric in LEVELS:
        for domain in DOMAINS:
            task_ids = list(map(str, manifest["dev"][domain]["task_ids"]))
            memory_path = None
            if memory_rubric != NONE:
                memory_path = args.memory_root / memory_rubric / f"memory_{domain}.json"
                if not memory_path.exists():
                    raise FileNotFoundError(memory_path)
            runs.append(
                {
                    "memory_rubric": memory_rubric,
                    "domain": domain,
                    "task_ids": task_ids,
                    "memory_path": str(memory_path.resolve()) if memory_path else None,
                    "save_name": f"qwen35_alloc_v2_m-{memory_rubric}_{domain}_dev_v1",
                }
            )
    write_json(args.output / "run_manifest.json", runs)
    write_json(
        args.output / "protocol.json",
        {
            "protocol": "tau_alloc_dev_matrix_v2",
            "design": "frozen raw base agent x 5 memory levels x 23 frozen dev tasks",
            "variable_under_test": "per-trajectory allocation rubric",
            "writing_style": "frozen (causal-minimal memory, faithful sft)",
            "task_agent": "raw Qwen3.5-35B-A3B, no LoRA",
            "memory_levels": list(LEVELS),
            "dev_tasks": 23,
            "runs": len(LEVELS) * 23,
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

    level_rewards: dict[str, dict[str, float]] = {}
    terminations: dict[str, Counter[str]] = {}
    for run in runs:
        level = run["memory_rubric"]
        level_rewards.setdefault(level, {})
        terminations.setdefault(level, Counter())
        result = read_json(SIMULATIONS / run["save_name"] / "results.json")
        for simulation in result["simulations"]:
            task_id = str(simulation["task_id"])
            if task_id not in set(run["task_ids"]):
                continue
            level_rewards[level][f"{run['domain']}:{task_id}"] = float(
                simulation["reward_info"]["reward"]
            )
            terminations[level][simulation["termination_reason"]] += 1

    control = level_rewards[NONE]
    levels = {}
    for level in LEVELS:
        values = level_rewards[level]
        arm = bank_summary["arms"].get(level, {})
        bank_total = arm.get("bank_total", 0)
        delta = sum(values.values()) - sum(control[key] for key in values)
        levels[level] = {
            "reward_one": sum(values.values()),
            "episodes": len(values),
            "pass_rate": sum(values.values()) / len(values),
            "bank_entries": bank_total,
            "memory_written_fraction": (
                bank_total / arm["decisions"] if arm.get("decisions") else None
            ),
            "sft_selected": arm.get("sft_selected"),
            "routes": arm.get("routes"),
            "delta_vs_none": delta,
            "delta_per_bank_entry": (delta / bank_total) if bank_total else None,
            "vs_none": comparison(control, values),
            "terminations": dict(terminations[level]),
        }
    summary = {
        "protocol": "tau_alloc_dev_matrix_v2",
        "final_test_used": False,
        "dev_tasks": 23,
        "runs": len(LEVELS) * 23,
        "levels": levels,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
