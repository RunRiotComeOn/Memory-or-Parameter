#!/usr/bin/env python3
"""Guided replay for one ALFWorld task -- ALFWorld counterpart of
run_appworld_guided_replay.py (DESIGN.md section 15).

Runs a FRESH attempt at `--task-id` (within `--split`), injecting `--guidance`
into the initial user message exactly the way `--memory-bank` retrieval
already does (same `memory_block` parameter, see `alfworld_agent.run_task`).
Not a scripted/deterministic replay of a candidate's exact command sequence --
a live agent reads its own admissible-commands list each turn and picks its
own actions, using the guidance as a hint. Only a `success=True` result is
meant to be kept as SFT data by the caller; this script just produces the
trajectory and lets the caller decide.

Must run under the alfworld_venv310 interpreter (Python 3.10) -- same
constraint as run_alfworld_rollout.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALFWORLD_DATA = Path(os.environ.get("ALFWORLD_DATA", "/nas04/yixuh/alfworld_data"))
DEFAULT_CONFIG = DEFAULT_ALFWORLD_DATA / "base_config.yaml"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split", choices=("train", "valid_seen", "valid_unseen"), default="train")
    parser.add_argument("--guidance", required=True, help="plan text, injected like memory_block")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--previous-success", action="store_true",
        help="the attempt this plan came from SUCCEEDED; the plan consolidates it rather than repairing it",
    )
    args = parser.parse_args()

    import yaml

    from trajectory_memory_lab.alfworld_agent import (
        SPLIT_TO_TRAIN_EVAL,
        list_available_tasks,
        load_task_env,
        run_task,
    )
    from trajectory_memory_lab.alfworld_sft_writer import guidance_block
    from trajectory_memory_lab.model_client import ModelClient

    os.environ.setdefault("ALFWORLD_DATA", str(args.config.resolve().parent))
    config = yaml.safe_load(args.config.read_text())
    train_eval = SPLIT_TO_TRAIN_EVAL[args.split]
    split_dir = Path(os.path.expandvars(
        {"train": config["dataset"]["data_path"],
         "eval_in_distribution": config["dataset"]["eval_id_data_path"],
         "eval_out_of_distribution": config["dataset"]["eval_ood_data_path"]}[train_eval]
    ))
    game_file = list_available_tasks(split_dir)[args.task_id]

    client = ModelClient(
        base_url=args.base_url,
        api_key="EMPTY",
        model=args.model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=False,
        timeout=args.timeout,
    )
    guidance_text = guidance_block({"plan": args.guidance}, previous_success=args.previous_success)
    try:
        env = load_task_env(config, train_eval, game_file, args.seed)
        trajectory = run_task(
            env, args.task_id, client, memory_block=guidance_text, max_steps=args.max_steps,
        )
        record = {
            "protocol": "alfworld_guided_replay_v1",
            "status": "complete",
            "task_id": args.task_id,
            "split": args.split,
            "guidance": args.guidance,
            "trajectory": trajectory,
        }
    except Exception as exc:
        import traceback

        record = {
            "protocol": "alfworld_guided_replay_v1",
            "status": "error",
            "task_id": args.task_id,
            "split": args.split,
            "guidance": args.guidance,
            "error": repr(exc),
            "traceback": traceback.format_exc()[-4_000:],
        }
    write_json(args.output, record)
    trajectory = record.get("trajectory") or {}
    print(json.dumps({
        "task_id": args.task_id, "status": record["status"],
        "success": trajectory.get("success"), "steps": len(trajectory.get("steps") or []),
    }))


if __name__ == "__main__":
    main()
