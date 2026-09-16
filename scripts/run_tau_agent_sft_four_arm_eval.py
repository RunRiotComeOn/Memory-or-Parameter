#!/usr/bin/env python3
"""Evaluate base/SFT task agents with and without trained-writer memory."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
TAU_DATA = TAU_ROOT / "data"
SIMULATIONS = TAU_DATA / "simulations"
DOMAINS = ("airline", "retail", "telecom")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_ids(domain: str) -> list[str]:
    split = read_json(TAU_DATA / f"tau2/domains/{domain}/split_tasks.json")
    train = set(map(str, split["train"]))
    test = list(map(str, split["test"]))
    if train & set(test):
        raise ValueError(f"{domain}: upstream train/test overlap")
    return test


def rewards(path: Path, wanted: list[str]) -> dict[str, float]:
    wanted_set = set(wanted)
    result = {}
    for simulation in read_json(path)["simulations"]:
        task_id = str(simulation["task_id"])
        reward_info = simulation.get("reward_info")
        reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
        if task_id in wanted_set and isinstance(reward, (int, float)):
            result[task_id] = float(reward)
    return result


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


def run_one(run: dict[str, Any], args: argparse.Namespace) -> str:
    result_path = SIMULATIONS / run["save_name"] / "results.json"
    if result_path.exists():
        current = rewards(result_path, run["task_ids"])
        if len(current) == len(run["task_ids"]):
            return f"resume-skip {run['arm']}_{run['domain']}"
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain",
        run["domain"],
        "--agent-llm",
        "openai/qwen35-tau-agent-sft",
        "--agent-llm-args",
        llm_args(args.max_tokens),
        "--user-llm",
        "openai/qwen35-tau",
        "--user-llm-args",
        llm_args(args.max_tokens),
        "--task-ids",
        *run["task_ids"],
        "--num-trials",
        "1",
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-steps",
        "200",
        "--timeout",
        str(args.timeout),
        "--max-retries",
        "2",
        "--retry-delay",
        "2",
        "--seed",
        str(args.seed),
        "--save-to",
        run["save_name"],
        "--verbose-logs",
        "--llm-log-mode",
        "latest",
        "--auto-resume",
        "--log-level",
        "INFO",
    ]
    environment = dict(os.environ)
    # The tau retail evaluator otherwise defaults to an external GPT model.
    # This experiment intentionally keeps both the user simulator and the NL
    # assertion judge on the frozen, untrained base Qwen served alongside the
    # task-agent LoRA, so judge behavior is stable across all arms.
    environment["TAU2_LLM_NL_ASSERTIONS"] = "openai/qwen35-tau"
    environment["TAU2_LLM_NL_ASSERTIONS_ARGS"] = llm_args(args.max_tokens)
    if run.get("memory_path"):
        environment["TAU_EXTERNAL_MEMORY_PATH"] = run["memory_path"]
        environment["TAU2_AGENT_MEMORY_RETRIEVAL"] = "1"
        environment["TAU2_AGENT_MEMORY_TOP_K"] = str(args.memory_top_k)
    else:
        environment.pop("TAU_EXTERNAL_MEMORY_PATH", None)
        environment.pop("TAU2_AGENT_MEMORY_RETRIEVAL", None)
        environment.pop("TAU2_AGENT_MEMORY_TOP_K", None)
    log_path = args.output / "logs" / f"{run['arm']}_{run['domain']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=TAU_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{run['arm']}_{run['domain']} exited {completed.returncode}")
    final = rewards(result_path, run["task_ids"])
    if len(final) != len(run["task_ids"]):
        raise RuntimeError(
            f"{run['arm']}_{run['domain']} scored {len(final)}/{len(run['task_ids'])}"
        )
    return f"complete {run['arm']}_{run['domain']}"


def comparison(base: dict[str, float], treatment: dict[str, float]) -> dict[str, Any]:
    keys = sorted(base.keys() & treatment.keys())
    helped = [key for key in keys if treatment[key] > base[key]]
    hurt = [key for key in keys if treatment[key] < base[key]]
    return {
        "paired_tasks": len(keys),
        "helped": helped,
        "hurt": hurt,
        "unchanged_success": [key for key in keys if base[key] == treatment[key] == 1],
        "unchanged_failure": [key for key in keys if base[key] == treatment[key] == 0],
        "pass_rate_delta": (
            sum(treatment.values()) / len(treatment)
            - sum(base.values()) / len(base)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--max-parallel-runs", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--run-tag", default="v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = []
    for arm in ("sft_no_memory", "sft_with_memory"):
        for domain in DOMAINS:
            memory_path = (
                args.memory_dir / f"memory_{domain}.json"
                if arm == "sft_with_memory"
                else None
            )
            if memory_path is not None and not memory_path.exists():
                raise FileNotFoundError(memory_path)
            runs.append(
                {
                    "arm": arm,
                    "domain": domain,
                    "task_ids": test_ids(domain),
                    "memory_path": str(memory_path.resolve()) if memory_path else None,
                    "save_name": f"qwen35_agent_sftdata_{arm}_{domain}_test_{args.run_tag}",
                }
            )
    write_json(args.output / "run_manifest.json", runs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_runs) as pool:
        futures = [pool.submit(run_one, run, args) for run in runs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)

    arm_domain_rewards: dict[str, dict[str, dict[str, float]]] = {
        arm: {} for arm in ("base_no_memory", "base_with_memory", "sft_no_memory", "sft_with_memory")
    }
    for domain in DOMAINS:
        ids = test_ids(domain)
        arm_domain_rewards["base_no_memory"][domain] = rewards(
            SIMULATIONS / f"qwen35_base_{domain}_full_v1/results.json", ids
        )
        arm_domain_rewards["base_with_memory"][domain] = rewards(
            SIMULATIONS
            / f"qwen35_baseagent_codex_sft_writer_memory_{domain}_test_codex56_v1/results.json",
            ids,
        )
        for arm in ("sft_no_memory", "sft_with_memory"):
            arm_domain_rewards[arm][domain] = rewards(
                SIMULATIONS
                / f"qwen35_agent_sftdata_{arm}_{domain}_test_{args.run_tag}/results.json",
                ids,
            )
    flat = {
        arm: {
            f"{domain}:{task_id}": reward
            for domain, domain_rewards in domains.items()
            for task_id, reward in domain_rewards.items()
        }
        for arm, domains in arm_domain_rewards.items()
    }
    rates = {
        arm: sum(values.values()) / len(values) for arm, values in flat.items()
    }
    summary = {
        "protocol": "tau_agent_sft_four_arm_eval_v1",
        "test_tasks": 100,
        "task_agent_training_test_overlap": 0,
        "arm_pass_rates": rates,
        "domain_pass_rates": {
            arm: {
                domain: sum(values.values()) / len(values)
                for domain, values in domains.items()
            }
            for arm, domains in arm_domain_rewards.items()
        },
        "comparisons": {
            "sft_vs_base_no_memory": comparison(flat["base_no_memory"], flat["sft_no_memory"]),
            "sft_memory_vs_base_memory": comparison(flat["base_with_memory"], flat["sft_with_memory"]),
            "memory_effect_on_base": comparison(flat["base_no_memory"], flat["base_with_memory"]),
            "memory_effect_on_sft": comparison(flat["sft_no_memory"], flat["sft_with_memory"]),
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
