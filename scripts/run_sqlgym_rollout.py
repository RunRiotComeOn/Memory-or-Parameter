#!/usr/bin/env python3
"""Roll out SQLGym / BIRD tasks with the local Qwen agent, optionally with memory.

Eighth benchmark. Unlike the AgentGym environments this one has NO HTTP hop:
`sqlgym` is a library, so this runs under /nas04/yixuh/sqlgym_venv (which
also carries `openai` for the model client) and opens the SQLite databases
in-process, read-only.

Three splits, from `sqlgym_agent.split_task_ids`:

- `train` (200) and `test` (80) are both drawn from BIRD train, over the
  same 69 databases and disjoint from each other -- the same-distribution
  pair.
- `xdb` (149) comes from BIRD dev, whose 11 databases do not appear in
  train at all. That is the cross-schema line, and the one that tests
  whether SFT gains come from transferable clause patterns rather than
  memorized schemas.

BIRD ships difficulty labels on dev only, so `by_difficulty` is populated
for `xdb` and empty elsewhere. Reward is binary (result sets compared as
sets), so `mean_score` equals `pass_rate`; both are reported to keep the
cross-benchmark tables uniform.

  PYTHONPATH=src /nas04/yixuh/sqlgym_venv/bin/python -u scripts/run_sqlgym_rollout.py \
      --split xdb --output sqlgym_experiment/baseline_xdb --experiment-name baseline_xdb \
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


def difficulty_breakdown(complete: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pass rate per BIRD difficulty tier.

    Measured on the evaluation side because BIRD labels dev only. The probe
    found the tiers do NOT separate evenly under this multi-turn protocol
    (simple 0.80, moderate 1.00, challenging 0.20): letting the agent inspect
    the data dissolves the "what is actually in this column" difficulty that
    the labels partly encode, leaving the genuinely complex queries. So the
    aggregate hides where the resolution is, and this breakdown is the number
    to read.
    """
    buckets: dict[str, list[bool]] = {}
    for record in complete:
        tier = (record["trajectory"].get("evaluation") or {}).get("difficulty") or "unlabeled"
        buckets.setdefault(tier, []).append(bool(record["trajectory"]["success"]))
    order = {"simple": 0, "moderate": 1, "challenging": 2, "unlabeled": 3}
    return {
        tier: {"tasks": len(v), "success": sum(v), "pass_rate": sum(v) / len(v)}
        for tier, v in sorted(buckets.items(), key=lambda kv: order.get(kv[0], 9))
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    from trajectory_memory_lab.memory_retrieval import retrieved_block
    from trajectory_memory_lab.model_client import ModelClient
    from trajectory_memory_lab.sqlgym_agent import SqlGymTaskEnv, run_task, parse_task_id

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
        # Retrieval query is the question, which BIRD only reveals on
        # reset -- hence a callback rather than a precomputed block.
        trajectory = run_task(
            SqlGymTaskEnv(job["bird_path"], parse_task_id(task_id)[0]), task_id, client,
            memory_lookup=lambda instruction: retrieved_block(
                job["memory_bank"], instruction, job["memory_top_k"]),
            max_steps=job["max_steps"],
        )
        record = {
            "protocol": "sqlgym_rollout_v1",
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
            "protocol": "sqlgym_rollout_v1",
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
    parser.add_argument("--split", choices=("train", "test", "xdb"), default="xdb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--bird-path", default="/nas04/yixuh/bird", help="BIRD dataset root (contains train/ and dev/)")
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

    from trajectory_memory_lab.sqlgym_agent import split_task_ids

    # Splits are defined in code (by BIRD mode and database), not read off
    # any server -- there is no server here.
    available = split_task_ids(args.split)
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
            "bird_path": args.bird_path,
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
            "protocol": "sqlgym_rollout_v1",
            "split": args.split,
            "tasks": len(jobs),
            "env_world": {"bird_path": args.bird_path, "databases": "69 train / 11 dev, disjoint",
                          "split_rule": "train+test from BIRD train (same schemas); xdb from BIRD dev (unseen schemas)"},
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
        "protocol": "sqlgym_rollout_v1",
        "split": args.split,
        "tasks": len(jobs),
        "complete": len(complete),
        "errors": len(records) - len(complete),
        "success": len(successes),
        "pass_rate": len(successes) / len(complete) if complete else None,
        # Binary reward here (result sets compared as sets), so this equals
        # pass_rate by construction. Kept so every benchmark's summary carries
        # the same two fields; a run where they diverge means something changed.
        "mean_score": sum(scores) / len(scores) if scores else None,
        "terminations": dict(terminations),
        "by_difficulty": difficulty_breakdown(complete),
        "memory_bank": str(args.memory_bank) if args.memory_bank else None,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
