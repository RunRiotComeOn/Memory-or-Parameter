#!/usr/bin/env python3
"""Run environment-grounded GT candidates for failed writer-teacher tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domain", choices=("airline", "retail", "telecom"), required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    domain = args.domain
    test_ids = set(map(str, manifest["test"][domain]["task_ids"]))
    selected = list(map(str, manifest["teacher_writer"][domain]["task_ids"]))
    source_path = (
        TAU_ROOT / f"data/simulations/qwen35_base_{domain}_full_v1/results.json"
    )
    source = {
        str(item["task_id"]): item for item in read_json(source_path)["simulations"]
    }
    task_ids = [
        task_id
        for task_id in selected
        if source[task_id].get("reward_info", {}).get("reward") != 1
    ]
    if set(task_ids) & test_ids:
        raise ValueError("refusing to run a test task as a teacher candidate")
    if not task_ids:
        print(f"{domain}: no failed teacher tasks need GT candidates", flush=True)
        return

    llm_args = json.dumps(
        {
            "temperature": 0.0,
            "max_tokens": 1024,
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        separators=(",", ":"),
    )
    save_name = f"qwen35_gt_sftdata_teacher_{domain}_v1"
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain",
        domain,
        "--agent",
        "llm_agent_gt",
        "--agent-llm",
        "openai/qwen35-tau",
        "--agent-llm-args",
        llm_args,
        "--user-llm",
        "openai/qwen35-tau",
        "--user-llm-args",
        llm_args,
        "--task-ids",
        *task_ids,
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
    print(f"{domain}: running {len(task_ids)} failed teacher tasks", flush=True)
    completed = subprocess.run(command, cwd=TAU_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
