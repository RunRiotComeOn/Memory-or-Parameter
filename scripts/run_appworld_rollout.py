#!/usr/bin/env python3
"""Roll out AppWorld tasks with the local Qwen agent, optionally with memory.

This is the AppWorld counterpart of the tau2 replay runner: it produces both the
scored results and the canonical trajectories that the v2 allocation writer
reads.  Tasks run in separate processes because AppWorld keeps process-global DB
engine caches.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Runs in a worker process: imports AppWorld fresh, drives one task."""
    from appworld import AppWorld

    from trajectory_memory_lab.appworld_agent import run_task
    from trajectory_memory_lab.memory_retrieval import retrieved_block
    from trajectory_memory_lab.model_client import ModelClient

    task_id = job["task_id"]
    output_path = Path(job["output_dir"]) / f"{task_id}.json"
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("status") in {"complete", "error"}:
            return {"task_id": task_id, "status": f"resume-{existing['status']}"}

    client = ModelClient(
        base_url=job["base_url"],
        api_key="EMPTY",
        model=job["model"],
        temperature=0.0,
        top_p=1.0,
        max_tokens=job["max_tokens"],
        seed=job["seed"],
        enable_thinking=False,
        timeout=job["timeout"],
    )
    try:
        with AppWorld(
            task_id=task_id,
            experiment_name=job["experiment_name"],
            random_seed=job["seed"],
        ) as world:
            memory_block, selection = retrieved_block(
                job["memory_bank"], world.task.instruction, job["memory_top_k"]
            )
            trajectory = run_task(
                world,
                world.task,
                client,
                memory_block=memory_block,
                max_steps=job["max_steps"],
            )
        record = {
            "protocol": "appworld_rollout_v1",
            "status": "complete",
            "task_id": task_id,
            "split": job["split"],
            "memory_bank": job["memory_bank"],
            "retrieved_memory": selection,
            "trajectory": trajectory,
        }
    except Exception as exc:
        record = {
            "protocol": "appworld_rollout_v1",
            "status": "error",
            "task_id": task_id,
            "split": job["split"],
            "memory_bank": job["memory_bank"],
            "error": repr(exc),
            "traceback": traceback.format_exc()[-4_000:],
        }
    write_json(output_path, record)
    trajectory = record.get("trajectory") or {}
    return {
        "task_id": task_id,
        "status": record["status"],
        "success": trajectory.get("success"),
        "termination": trajectory.get("termination_reason"),
        "steps": len(trajectory.get("steps") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--memory-bank", type=Path, default=None)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    if "memory" in str(Path(os.environ.get("APPWORLD_ROOT", ".")).resolve()):
        raise SystemExit(
            "APPWORLD_ROOT must not contain the substring 'memory': AppWorld uses "
            "a naive `\"memory\" in path` test to detect in-memory SQLite and will "
            "mis-handle the on-disk task databases."
        )

    from appworld import load_task_ids

    task_ids = args.task_ids or list(load_task_ids(args.split))
    if args.limit:
        task_ids = task_ids[: args.limit]
    trajectories_dir = args.output / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "task_id": task_id,
            "split": args.split,
            "output_dir": str(trajectories_dir),
            "experiment_name": args.experiment_name,
            "memory_bank": str(args.memory_bank.resolve()) if args.memory_bank else None,
            "memory_top_k": args.memory_top_k,
            "model": args.model,
            "base_url": args.base_url,
            "max_steps": args.max_steps,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "seed": args.seed,
        }
        for task_id in task_ids
    ]
    write_json(
        args.output / "protocol.json",
        {
            "protocol": "appworld_rollout_v1",
            "split": args.split,
            "tasks": len(jobs),
            "agent": f"code-acting loop, {args.model}, temperature 0, thinking disabled",
            "max_steps": args.max_steps,
            "memory_bank": str(args.memory_bank) if args.memory_bank else None,
            "memory_retrieval": f"bm25_top{args.memory_top_k}" if args.memory_bank else None,
            "seed": args.seed,
        },
    )

    results = []
    if args.max_parallel <= 1:
        for job in jobs:
            result = run_one(job)
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{len(results)}/{len(jobs)}] {result}", flush=True)

    records = [read_json(path) for path in sorted(trajectories_dir.glob("*.json"))]
    complete = [r for r in records if r.get("status") == "complete"]
    successes = [r for r in complete if (r.get("trajectory") or {}).get("success")]
    terminations: dict[str, int] = {}
    for record in complete:
        reason = (record.get("trajectory") or {}).get("termination_reason", "unknown")
        terminations[reason] = terminations.get(reason, 0) + 1
    summary = {
        "protocol": "appworld_rollout_v1",
        "split": args.split,
        "tasks": len(jobs),
        "complete": len(complete),
        "errors": len(records) - len(complete),
        "success": len(successes),
        "pass_rate": len(successes) / len(complete) if complete else None,
        "terminations": terminations,
        "memory_bank": str(args.memory_bank) if args.memory_bank else None,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
