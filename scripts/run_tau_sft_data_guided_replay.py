#!/usr/bin/env python3
"""Replay SFT-data-writer trajectory candidates in the live tau environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=("airline", "retail", "telecom"), required=True)
    parser.add_argument("--split-section", choices=("teacher_writer", "writer_generation"), required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    domain = args.domain
    allowed = set(map(str, manifest[args.split_section][domain]["task_ids"]))
    test = set(map(str, manifest["test"][domain]["task_ids"]))
    if allowed & test:
        raise ValueError("candidate replay split overlaps the held-out test set")

    tasks: dict[str, Any] = {}
    records = sorted((args.candidates / "tasks").glob(f"*_{domain}/candidate.json"))
    for path in records:
        record = read_json(path)
        task_id = str(record["source_task_id"])
        if task_id not in allowed:
            continue
        if record.get("status") not in {"candidate_ready", "prediction_ready"}:
            continue
        candidate = record.get("candidate") or record.get("prediction")
        if not candidate or not candidate.get("assistant_turns"):
            continue
        tasks[task_id] = {
            "assistant_turns": candidate["assistant_turns"],
            "writer_rationale": candidate.get("rationale", ""),
            "risk_checks": candidate.get("risk_checks", []),
        }
    if not tasks:
        raise ValueError(f"{domain}: no ready candidates to replay")
    if set(tasks) & test:
        raise ValueError("refusing to replay test tasks during data generation")

    args.output.mkdir(parents=True, exist_ok=True)
    guidance_path = args.output / f"guidance_{domain}.json"
    write_json(
        guidance_path,
        {
            "protocol": "tau_sft_data_guidance_v1",
            "domain": domain,
            "source_split": args.split_section,
            "test_overlap": 0,
            "tasks": tasks,
        },
    )
    llm_args = json.dumps(
        {
            "temperature": 0.0,
            "max_tokens": 4096,
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        separators=(",", ":"),
    )
    save_name = f"qwen35_sftdata_guided_{args.run_tag}_{domain}"
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain",
        domain,
        "--agent",
        "sft_data_guided_agent",
        "--agent-llm",
        "openai/qwen35-tau",
        "--agent-llm-args",
        llm_args,
        "--user-llm",
        "openai/qwen35-tau",
        "--user-llm-args",
        llm_args,
        "--task-ids",
        *tasks.keys(),
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
        save_name,
        "--verbose-logs",
        "--llm-log-mode",
        "latest",
        "--auto-resume",
        "--log-level",
        "INFO",
    ]
    environment = dict(os.environ)
    environment["TAU_SFT_DATA_GUIDANCE_PATH"] = str(guidance_path.resolve())
    # Some retail tasks require an LLM judge for natural-language assertions.
    # Keep that judge on the same local raw Qwen server instead of silently
    # falling back to the unavailable upstream GPT-4.1 default.
    environment["TAU2_LLM_NL_ASSERTIONS"] = "openai/qwen35-tau"
    environment["TAU2_LLM_NL_ASSERTIONS_ARGS"] = json.dumps(
        {
            "temperature": 0.0,
            "max_tokens": 4096,
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        separators=(",", ":"),
    )
    environment.pop("TAU_EXTERNAL_MEMORY_PATH", None)
    environment.pop("TAU2_AGENT_MEMORY_RETRIEVAL", None)
    print(f"{domain}: replaying {len(tasks)} writer candidates", flush=True)
    completed = subprocess.run(command, cwd=TAU_ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    results_path = TAU_ROOT / f"data/simulations/{save_name}/results.json"
    simulations = read_json(results_path)["simulations"]
    summary = {
        "protocol": "tau_sft_data_guided_replay_v1",
        "domain": domain,
        "source_split": args.split_section,
        "candidates": len(tasks),
        "completed": len(simulations),
        "reward_one": sum(
            isinstance(item.get("reward_info"), dict)
            and item["reward_info"].get("reward") == 1
            for item in simulations
        ),
        "results_path": str(results_path.resolve()),
        "test_overlap": 0,
    }
    write_json(args.output / f"summary_{domain}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
