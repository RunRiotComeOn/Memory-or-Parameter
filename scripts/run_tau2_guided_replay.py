#!/usr/bin/env python3
"""Replay ONE tau2 task with a teacher plan injected, and record it.

tau2 counterpart of `run_webshop_guided_replay.py` /
`run_alfworld_guided_replay.py`, invoked the same way by
`router_sft_pipeline.replay_and_verify`: one subprocess per committed
`sft`/`both` decision; the trajectory becomes a training example only if
tau2 scores the replay 1.

The plan goes into the agent's system prompt (`tau2_agent.MemoryAgent`'s
fixed block), where retrieved memory also goes. The training example is the
trajectory's `sft_example`, whose system prompt is the BASE one -- so the
plan is already absent from it, without the text-stripping the text
benchmarks need.

No `--split`: a task id (`<domain>::<tau2 id>`) names one task outright.
Must run under tau2's venv.
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="<domain>::<tau2 task id>")
    parser.add_argument("--guidance", required=True, help="plan text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--user-llm", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument(
        "--previous-success", action="store_true",
        help="the attempt this plan came from SUCCEEDED; the plan consolidates it rather than repairing it",
    )
    args = parser.parse_args()

    from trajectory_memory_lab import tau2_agent
    from trajectory_memory_lab.tau2_sft_writer import guidance_block

    tau2_agent.quiet_logs()
    tau2_agent.configure_gemini()
    # Same memo as the rollouts (default path), so a replay's customer answers
    # a given conversation prefix exactly as it did in the rollout.
    tau2_agent.install_llm_cache()
    guidance_text = guidance_block({"plan": args.guidance}, previous_success=args.previous_success)
    try:
        trajectory = tau2_agent.run_task(
            args.task_id,
            agent_base_url=args.base_url,
            agent_llm=f"openai/{args.model}",
            agent_max_tokens=args.max_tokens,
            user_llm=args.user_llm or tau2_agent.DEFAULT_USER_LLM,
            memory_block=guidance_text,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        trajectory.pop("retrieved_memory", None)
        record = {
            "protocol": "tau2_guided_replay_v1",
            "status": "complete",
            "task_id": args.task_id,
            "guidance": args.guidance,
            "trajectory": trajectory,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        record = {
            "protocol": "tau2_guided_replay_v1",
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
        "success": trajectory.get("success"), "reward": trajectory.get("reward"),
        "steps": len(trajectory.get("steps") or []),
    }))


if __name__ == "__main__":
    main()
