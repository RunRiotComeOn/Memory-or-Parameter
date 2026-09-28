#!/usr/bin/env python3
"""Roll out WebShop tasks with the local Qwen agent, optionally with memory.

Fourth-benchmark counterpart of run_alfworld_rollout.py /
run_scienceworld_rollout.py: same job shape, same canonical trajectory
output. One difference in structure: the environment is NOT built per task.
It lives in a separately started `scripts/webshop_env_server.py` (the full
catalog takes minutes and tens of GB to load), and this script's workers only
hold an HTTP client to it -- so this runs under the repo's own .venv, not
webshop_venv.

Splits are WebShop's own, by goal index: `test` 0-499, `dev` 500-1499,
`train` 1500 onward. `--limit` takes the first N of a split.

The env server's world seed (`/info`) is recorded in protocol.json; every
comparison should use trajectories from the same seed (see the server's
docstring for why stock WebShop is not reproducible without one).

  # once, in another terminal/tmux:
  PYTHONPATH=third_party/WebShop /nas04/yixuh/webshop_venv/bin/python -u scripts/webshop_env_server.py
  # then:
  PYTHONPATH=src .venv/bin/python -u scripts/run_webshop_rollout.py --split test \\
      --output webshop_experiment/baseline_test_v1 --experiment-name baseline_test_v1 \\
      --base-url http://127.0.0.1:8030/v1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    from trajectory_memory_lab.memory_retrieval import retrieved_block
    from trajectory_memory_lab.model_client import ModelClient
    from trajectory_memory_lab.webshop_agent import WebShopEnvClient, run_task

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
        # Retrieval query is the instruction, which WebShop only reveals on
        # reset -- hence a callback rather than a precomputed block.
        trajectory = run_task(
            WebShopEnvClient(job["env_url"]), task_id, client,
            memory_lookup=lambda instruction: retrieved_block(
                job["memory_bank"], instruction, job["memory_top_k"]),
            max_steps=job["max_steps"],
        )
        record = {
            "protocol": "webshop_rollout_v1",
            "status": "complete",
            "task_id": task_id,
            "split": job["split"],
            "memory_bank": job["memory_bank"],
            "retrieved_memory": trajectory.pop("retrieved_memory"),
            "trajectory": trajectory,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        record = {
            "protocol": "webshop_rollout_v1",
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
        "score": trajectory.get("reward"),
        "termination": trajectory.get("termination_reason"),
        "steps": len(trajectory.get("steps") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--env-url", default=None, help="env server; default $WEBSHOP_ENV_URL or http://127.0.0.1:3100")
    parser.add_argument("--memory-bank", type=Path, default=None)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--task-ids-file", type=Path, default=None, help="whitespace-separated task ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    from trajectory_memory_lab.webshop_agent import WebShopEnvClient, split_task_ids

    env_client = WebShopEnvClient(args.env_url)
    env_info = env_client.info()
    available = split_task_ids(args.split, env_info["num_goals"])
    if args.task_ids or args.task_ids_file:
        task_ids = list(args.task_ids) if args.task_ids else args.task_ids_file.read_text().split()
        unknown = set(task_ids) - set(available)
        if unknown:
            raise SystemExit(f"{len(unknown)} task_id(s) not in split={args.split}: {sorted(unknown)[:5]}...")
    else:
        task_ids = available
    if args.limit:
        task_ids = task_ids[: args.limit]

    trajectories_dir = args.output / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "task_id": task_id,
            "split": args.split,
            "env_url": env_client.base_url,
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
            "protocol": "webshop_rollout_v1",
            "split": args.split,
            "tasks": len(jobs),
            "env_world": {"seed": env_info["seed"], "num_products": env_info["num_products"] or "all",
                          "num_goals": env_info["num_goals"]},
            "agent": f"available-action text agent, {args.model}, temperature 0, thinking disabled",
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

    wanted = set(task_ids)
    records = [read_json(p) for p in sorted(trajectories_dir.glob("*.json")) if p.stem in wanted]
    complete = [r for r in records if r.get("status") == "complete"]
    successes = [r for r in complete if r["trajectory"]["success"]]
    terminations = Counter(r["trajectory"].get("termination_reason", "unknown") for r in complete)
    scores = [float(r["trajectory"]["reward"]) for r in complete]
    summary = {
        "protocol": "webshop_rollout_v1",
        "split": args.split,
        "tasks": len(jobs),
        "complete": len(complete),
        "errors": len(records) - len(complete),
        "success": len(successes),
        "pass_rate": len(successes) / len(complete) if complete else None,
        # WebShop's second headline metric: mean score, where a partial match
        # still earns partial credit and never buying scores 0.
        "mean_score": sum(scores) / len(scores) if scores else None,
        "terminations": dict(terminations),
        "memory_bank": str(args.memory_bank) if args.memory_bank else None,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
