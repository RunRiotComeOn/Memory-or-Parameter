#!/usr/bin/env python3
"""Replay ONE WebShop task with a teacher plan injected, and record it.

WebShop counterpart of `run_scienceworld_guided_replay.py` /
`run_alfworld_guided_replay.py`, invoked the same way: the sft pipeline runs
this as a subprocess per committed `sft`/`both` decision and keeps the
resulting trajectory as a training example only if the replay scores 1.

No `--split` is needed: a task id (`goal_<index>`) names one goal of the
env server's goal list outright, and splits are just index ranges of it.
The replay must hit the SAME env server (same world seed) the plan's source
trajectory came from -- `router_sft_pipeline` passes the address through
`WEBSHOP_ENV_URL`.

Runs under the repo's own .venv; only the env server needs webshop_venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="goal_<index>")
    parser.add_argument("--guidance", required=True, help="plan text, injected like memory_block")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-url", default=None, help="default $WEBSHOP_ENV_URL or http://127.0.0.1:3100")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--previous-success", action="store_true",
        help="the attempt this plan came from SUCCEEDED; the plan consolidates it rather than repairing it",
    )
    args = parser.parse_args()

    from trajectory_memory_lab.model_client import ModelClient
    from trajectory_memory_lab.webshop_agent import WebShopEnvClient, run_task
    from trajectory_memory_lab.webshop_sft_writer import guidance_block

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
        trajectory = run_task(
            WebShopEnvClient(args.env_url), args.task_id, client,
            memory_block=guidance_text, max_steps=args.max_steps,
        )
        trajectory.pop("retrieved_memory", None)
        record = {
            "protocol": "webshop_guided_replay_v1",
            "status": "complete",
            "task_id": args.task_id,
            "guidance": args.guidance,
            "trajectory": trajectory,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        record = {
            "protocol": "webshop_guided_replay_v1",
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
        "success": trajectory.get("success"), "score": trajectory.get("reward"),
        "steps": len(trajectory.get("steps") or []),
    }))


if __name__ == "__main__":
    main()
