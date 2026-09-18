#!/usr/bin/env python3
"""Repair tau guided replays that used an unavailable NL-assertion judge."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
SIMULATIONS = TAU_ROOT / "data/simulations"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--source-save", required=True)
    parser.add_argument("--repair-save", required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    source_path = SIMULATIONS / args.source_save / "results.json"
    source = read_json(source_path)
    bad_ids = [
        str(item["task_id"])
        for item in source["simulations"]
        if item.get("termination_reason") == "infrastructure_error"
        or not isinstance(item.get("reward_info"), dict)
    ]
    if not bad_ids:
        print("no infrastructure-error tasks need repair", flush=True)
        return
    guidance = read_json(args.guidance)
    missing = set(bad_ids) - set(guidance["tasks"])
    if missing:
        raise ValueError(f"guidance is missing repair tasks: {sorted(missing)}")

    common_args = {
        "temperature": 0.0,
        "max_tokens": 4096,
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    encoded_args = json.dumps(common_args, separators=(",", ":"))
    command = [
        str(TAU_ROOT / ".venv/bin/tau2"),
        "run",
        "--domain",
        args.domain,
        "--agent",
        "sft_data_guided_agent",
        "--agent-llm",
        "openai/qwen35-tau",
        "--agent-llm-args",
        encoded_args,
        "--user-llm",
        "openai/qwen35-tau",
        "--user-llm-args",
        encoded_args,
        "--task-ids",
        *bad_ids,
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
        "300",
        "--save-to",
        args.repair_save,
        "--verbose-logs",
        "--llm-log-mode",
        "latest",
        "--auto-resume",
        "--log-level",
        "INFO",
    ]
    environment = dict(os.environ)
    environment["TAU_SFT_DATA_GUIDANCE_PATH"] = str(args.guidance.resolve())
    environment["TAU2_LLM_NL_ASSERTIONS"] = "openai/qwen35-tau"
    environment["TAU2_LLM_NL_ASSERTIONS_ARGS"] = encoded_args
    environment.pop("TAU_EXTERNAL_MEMORY_PATH", None)
    environment.pop("TAU2_AGENT_MEMORY_RETRIEVAL", None)
    print(f"repairing {len(bad_ids)} tasks: {bad_ids}", flush=True)
    completed = subprocess.run(command, cwd=TAU_ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    repair_path = SIMULATIONS / args.repair_save / "results.json"
    repair = read_json(repair_path)
    repaired_by_id = {str(item["task_id"]): item for item in repair["simulations"]}
    invalid = [
        task_id
        for task_id in bad_ids
        if task_id not in repaired_by_id
        or repaired_by_id[task_id].get("termination_reason") == "infrastructure_error"
        or not isinstance(repaired_by_id[task_id].get("reward_info"), dict)
    ]
    if invalid:
        raise RuntimeError(f"repair still has infrastructure errors: {invalid}")

    backup = source_path.with_name("results.pre_nl_judge_repair.json")
    if not backup.exists():
        shutil.copy2(source_path, backup)
    source["simulations"] = [
        repaired_by_id.get(str(item["task_id"]), item)
        for item in source["simulations"]
    ]
    atomic_write(source_path, source)
    print(
        json.dumps(
            {
                "repaired": len(bad_ids),
                "task_ids": bad_ids,
                "backup": str(backup),
                "source": str(source_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
