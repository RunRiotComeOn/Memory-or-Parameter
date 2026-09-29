#!/usr/bin/env python3
"""Replay ONE TextCraft task with a teacher plan injected, and record it.

Counterpart of `run_babyai_guided_replay.py`: the sft pipeline runs this as
a subprocess per committed `sft`/`both` decision and keeps the resulting
trajectory as a training example only if the fresh attempt actually
succeeded.

Runs under the repo's own `.venv`, not `textcraft_venv`: the environment
lives in a long-lived HTTP server (AgentGym's `agentenv-textcraft`) and this
is only a client of it. It must be the same server the plan's source
trajectory ran against -- a TextCraft goal is a pure function of `data_idx`
given the recipe set, so any server started from the same checkout
reproduces it, but a differently-built crafting tree would silently compare
different tasks.
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
    parser.add_argument("--task-id", required=True, help="textcraft::<data_idx>")
    parser.add_argument("--guidance", required=True, help="plan text, injected like memory_block")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-url", default=None,
                        help="AgentGym textcraft env server; default $TEXTCRAFT_ENV_URL or http://127.0.0.1:36002")
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

    import sys

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))

    from trajectory_memory_lab.textcraft_agent import TextCraftEnvClient, run_task
    from trajectory_memory_lab.textcraft_sft_writer import guidance_block
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
        trajectory = run_task(
            TextCraftEnvClient(args.env_url), args.task_id, client,
            memory_block=guidance_text, max_steps=args.max_steps,
        )
        trajectory.pop("retrieved_memory", None)
        record = {
            "protocol": "textcraft_guided_replay_v1",
            "status": "complete",
            "task_id": args.task_id,
            "guidance": args.guidance,
            "trajectory": trajectory,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        record = {
            "protocol": "textcraft_guided_replay_v1",
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
