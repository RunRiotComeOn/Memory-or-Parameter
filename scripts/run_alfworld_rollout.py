#!/usr/bin/env python3
"""Roll out ALFWorld tasks with the local Qwen agent, optionally with memory.

Second-benchmark counterpart of `run_appworld_rollout.py`: same job shape,
same canonical trajectory output, same per-task subprocess isolation (ALFWorld
registers a gym env per process; running many tasks in one process would mean
tearing down and re-registering textworld gym envs repeatedly, which is the
same class of process-global-state problem AppWorld has with its DB engine
cache, just for a different reason).

MUST run under the alfworld_venv310 interpreter (Python 3.10) -- the
project's default .venv is 3.11+/3.13 and `textworld==1.7.0`'s
`EvalSymbol.derive()` breaks under newer CPython's local-variable
optimizations. This script does not re-exec itself into that interpreter;
invoke it directly with:

    /nas04/yixuh/alfworld_venv310/bin/python scripts/run_alfworld_rollout.py ...

Train/test separation (the one thing explicitly asked for): `--split` is one
of `train` / `valid_seen` / `valid_unseen`, mapped to ALFWorld's own
`json_2.1.1/{train,valid_seen,valid_unseen}` directories via
`alfworld_agent.SPLIT_TO_TRAIN_EVAL`. `train` is where base rollouts get
collected and memory/SFT content gets drafted from (this project's
`base_train_v2` equivalent); `valid_unseen` is the held-out generalization
metric (room LAYOUTS never seen in train, ALFWorld's strict OOD split) and is
the one that should be quoted as "the eval number"; `valid_seen` (same room
types, unseen instances/tasks within them) is a softer, in-distribution
check, kept available but not the default for `--split` in an eval context.
No task_id ever crosses between these directories: `alfworld_agent.list_available_tasks`
only ever looks inside the one split directory it is given.
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
DEFAULT_ALFWORLD_DATA = Path(os.environ.get("ALFWORLD_DATA", "/nas04/yixuh/alfworld_data"))
DEFAULT_CONFIG = DEFAULT_ALFWORLD_DATA / "base_config.yaml"

SPLIT_DIRNAME = {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Runs in a worker process: registers a fresh textworld gym env, drives one task."""
    import yaml

    from trajectory_memory_lab.alfworld_agent import (
        SPLIT_TO_TRAIN_EVAL,
        list_available_tasks,
        load_task_env,
        run_task,
    )
    from trajectory_memory_lab.memory_retrieval import retrieved_block
    from trajectory_memory_lab.model_client import ModelClient

    task_id = job["task_id"]
    output_path = Path(job["output_dir"]) / f"{task_id.replace('/', '__')}.json"
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("status") in {"complete", "error"}:
            return {"task_id": task_id, "status": f"resume-{existing['status']}"}

    os.environ.setdefault("ALFWORLD_DATA", str(Path(job["config"]).resolve().parent))
    config = yaml.safe_load(Path(job["config"]).read_text())
    train_eval = SPLIT_TO_TRAIN_EVAL[job["split"]]
    split_dir = Path(os.path.expandvars(
        {"train": config["dataset"]["data_path"],
         "eval_in_distribution": config["dataset"]["eval_id_data_path"],
         "eval_out_of_distribution": config["dataset"]["eval_ood_data_path"]}[train_eval]
    ))
    game_file = list_available_tasks(split_dir)[task_id]

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
        env = load_task_env(config, train_eval, game_file, job["seed"])
        memory_block, selection = retrieved_block(
            job["memory_bank"], task_id, job["memory_top_k"]
        )
        # NOTE: retrieval query is the task_id, not the instruction, because the
        # instruction is only known AFTER env.reset() inside run_task. Good
        # enough while task_ids are descriptive folder names; revisit if a
        # trained router/memory pass needs the real instruction text for
        # retrieval quality.
        trajectory = run_task(
            env,
            task_id,
            client,
            memory_block=memory_block,
            max_steps=job["max_steps"],
        )
        record = {
            "protocol": "alfworld_rollout_v1",
            "status": "complete",
            "task_id": task_id,
            "split": job["split"],
            "memory_bank": job["memory_bank"],
            "retrieved_memory": selection,
            "trajectory": trajectory,
        }
    except Exception as exc:
        record = {
            "protocol": "alfworld_rollout_v1",
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
    parser.add_argument("--split", choices=sorted(SPLIT_DIRNAME), default="valid_unseen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--memory-bank", type=Path, default=None)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    import yaml

    from trajectory_memory_lab.alfworld_agent import SPLIT_TO_TRAIN_EVAL, list_available_tasks

    os.environ.setdefault("ALFWORLD_DATA", str(args.config.resolve().parent))
    config = yaml.safe_load(args.config.read_text())
    train_eval = SPLIT_TO_TRAIN_EVAL[args.split]
    split_dir = Path(os.path.expandvars(
        {"train": config["dataset"]["data_path"],
         "eval_in_distribution": config["dataset"]["eval_id_data_path"],
         "eval_out_of_distribution": config["dataset"]["eval_ood_data_path"]}[train_eval]
    ))
    available = list_available_tasks(split_dir)
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
            "config": str(args.config.resolve()),
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
            "protocol": "alfworld_rollout_v1",
            "split": args.split,
            "tasks": len(jobs),
            "agent": f"admissible-command text agent, {args.model}, temperature 0, thinking disabled",
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
        "protocol": "alfworld_rollout_v1",
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
