#!/usr/bin/env python3
"""Roll out ScienceWorld tasks with the local Qwen agent, optionally with memory.

Third-benchmark counterpart of run_appworld_rollout.py / run_alfworld_rollout.py:
same job shape, same canonical trajectory output, same per-task subprocess
isolation. ScienceWorld's JVM-backed env is cheap to construct (~0.8s,
verified) so per-task subprocesses cost little extra here, same as ALFWorld.

Train/test separation: `--split train/dev/test` maps onto ScienceWorld's own
`env.get_variations_{train,dev,test}()` per task_name -- these are disjoint
variation-ID ranges within EACH of the 30 task types (e.g. "boil" has 14
train / 7 dev / 9 test variations; some task types have hundreds). `dev` is
the strictest held-out generalization split ScienceWorld ships (its own
convention names it "dev", not "valid_unseen", but it plays the same role:
never touched when building a training-pool bank).

Must run under scienceworld_venv (Python 3.10, matching the pattern already
used for appworld_venv/alfworld_venv310 -- one dedicated venv per benchmark).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Runs in a worker process: constructs a fresh ScienceWorldEnv, drives one task."""
    from scienceworld import ScienceWorldEnv

    from trajectory_memory_lab.scienceworld_agent import parse_task_id, run_task
    from trajectory_memory_lab.memory_retrieval import retrieved_block
    from trajectory_memory_lab.model_client import ModelClient

    task_id = job["task_id"]
    output_path = Path(job["output_dir"]) / f"{task_id.replace('::', '__')}.json"
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("status") in {"complete", "error"}:
            return {"task_id": task_id, "status": f"resume-{existing['status']}"}

    task_name, variation_id = parse_task_id(task_id)
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
        env = ScienceWorldEnv()
        env.load(task_name, variation_id, simplificationStr="")
        memory_block, selection = retrieved_block(
            job["memory_bank"], task_id, job["memory_top_k"]
        )
        trajectory = run_task(
            env, task_id, client, memory_block=memory_block, max_steps=job["max_steps"],
        )
        record = {
            "protocol": "scienceworld_rollout_v1",
            "status": "complete",
            "task_id": task_id,
            "split": job["split"],
            "memory_bank": job["memory_bank"],
            "retrieved_memory": selection,
            "trajectory": trajectory,
        }
    except Exception as exc:
        import traceback

        record = {
            "protocol": "scienceworld_rollout_v1",
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
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--memory-bank", type=Path, default=None)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    from scienceworld import ScienceWorldEnv

    from trajectory_memory_lab.scienceworld_agent import list_available_tasks

    available = list_available_tasks(ScienceWorldEnv(), args.split)
    task_ids = args.task_ids or sorted(available)
    if args.limit:
        task_ids = task_ids[: args.limit]
    unknown = set(task_ids) - set(available)
    if unknown:
        raise SystemExit(f"{len(unknown)} task_id(s) not found under split={args.split}: {sorted(unknown)[:5]}...")

    trajectories_dir = args.output / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "task_id": task_id,
            "split": args.split,
            "output_dir": str(trajectories_dir),
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
            "protocol": "scienceworld_rollout_v1",
            "split": args.split,
            "tasks": len(jobs),
            "agent": f"valid-action text agent, {args.model}, temperature 0, thinking disabled",
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
        "protocol": "scienceworld_rollout_v1",
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
