#!/usr/bin/env python3
"""Replay ONE ScienceWorld task with a teacher plan injected, and record it.

ScienceWorld counterpart of `run_alfworld_guided_replay.py` /
`run_appworld_guided_replay.py`, and invoked the same way: the sft pipeline
runs this as a subprocess per committed `sft`/`both` decision, and keeps the
resulting trajectory as a training example only if the replay actually
succeeded.

Simpler than the ALFWorld one in one respect: a ScienceWorld task_id
("<task_name>::<variation_id>") identifies a task outright, because the
train/dev/test variation lists are disjoint id ranges of the same task names.
So no `--split` is needed to disambiguate it, unlike ALFWorld where task ids
only unique within a split directory -- `router_sft_pipeline._REPLAY_CONFIG`
correspondingly passes no split argument for this domain.

Must run under `/nas04/yixuh/scienceworld_venv/bin/python`, not the repo's
default `.venv`: the `scienceworld` package and its JVM-backed environment
live only in that venv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="<task_name>::<variation_id>")
    parser.add_argument("--guidance", required=True, help="plan text, injected like memory_block")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--previous-success", action="store_true",
        help="the attempt this plan came from SUCCEEDED; the plan consolidates it rather than repairing it",
    )
    args = parser.parse_args()

    from scienceworld import ScienceWorldEnv

    from trajectory_memory_lab.model_client import ModelClient
    from trajectory_memory_lab.scienceworld_agent import parse_task_id, run_task
    from trajectory_memory_lab.scienceworld_sft_writer import guidance_block

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
        task_name, variation_id = parse_task_id(args.task_id)
        env = ScienceWorldEnv()
        env.load(task_name, variation_id, simplificationStr="")
        trajectory = run_task(
            env, args.task_id, client, memory_block=guidance_text, max_steps=args.max_steps,
        )
        record = {
            "protocol": "scienceworld_guided_replay_v1",
            "status": "complete",
            "task_id": args.task_id,
            "guidance": args.guidance,
            "trajectory": trajectory,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        record = {
            "protocol": "scienceworld_guided_replay_v1",
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
