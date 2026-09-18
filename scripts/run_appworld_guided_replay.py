#!/usr/bin/env python3
"""Guided replay for one AppWorld task (DESIGN.md section 15).

Runs a FRESH attempt at `--task-id`, injecting `--guidance` into the initial
user message exactly the way `--memory-bank` retrieval already does (same
`memory_block` parameter, see `appworld_agent.build_initial_user_message`).
Not a scripted/deterministic replay of a candidate's exact code -- a live
agent writes its own code and reads real API responses, using the guidance
as a hint. Only a `success=True` result is meant to be kept as SFT data by
the caller; this script just produces the trajectory and lets the caller
decide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--guidance", required=True, help="plan text, injected like memory_block")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    # v6: the plan is a repair only when the previous attempt failed. Optional
    # so a caller that predates the flag still gets the original wording.
    parser.add_argument(
        "--previous-success", action="store_true",
        help="the attempt this plan came from SUCCEEDED; the plan consolidates it rather than repairing it",
    )
    args = parser.parse_args()

    from appworld import AppWorld

    from trajectory_memory_lab.appworld_agent import run_task
    from trajectory_memory_lab.appworld_sft_writer import guidance_block
    from trajectory_memory_lab.model_client import ModelClient

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
        with AppWorld(
            task_id=args.task_id, experiment_name=args.experiment_name, random_seed=args.seed,
        ) as world:
            trajectory = run_task(
                world, world.task, client, memory_block=guidance_text, max_steps=args.max_steps,
            )
        record = {
            "protocol": "appworld_guided_replay_v1",
            "status": "complete",
            "task_id": args.task_id,
            "guidance": args.guidance,
            "trajectory": trajectory,
        }
    except Exception as exc:
        import traceback

        record = {
            "protocol": "appworld_guided_replay_v1",
            "status": "error",
            "task_id": args.task_id,
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
