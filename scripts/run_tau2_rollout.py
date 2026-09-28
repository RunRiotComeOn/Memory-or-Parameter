#!/usr/bin/env python3
"""Roll out tau2-bench tasks (airline / retail / telecom) with the local Qwen
agent and a Gemini user simulator, optionally with memory.

Same job shape and canonical output as run_alfworld_rollout.py /
run_webshop_rollout.py; the tau2-specific parts live in
`trajectory_memory_lab.tau2_agent` (see its docstring for the choice of user
simulator and judge). Splits are tau2's own `split_tasks.json`.

Workers are threads, not processes: a simulation spends nearly all its time
waiting on the two LLM endpoints, and tau2 itself runs batches the same way.

The Gemini side (user simulator, NL judge) is memoized on disk by default
(`tau2_agent.install_llm_cache`): Gemini at temperature 0 is not
deterministic over whole conversations, and without the memo two arms of an
experiment would differ by simulator draws as well as by the agent.

`--repeat N` runs every task N times (seeds seed, seed+1, ...) and reports
how often the outcome agrees across repeats. With the cache on this checks
the pipeline is deterministic end to end; with `--no-llm-cache` it measures
the simulator's own spread.

Must run under tau2's venv:
  PYTHONPATH=src third_party/tau2-bench/.venv/bin/python -u scripts/run_tau2_rollout.py \\
      --domain airline --split test --output tau2_experiment/airline_baseline_test_v1 \\
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    from trajectory_memory_lab import tau2_agent
    from trajectory_memory_lab.memory_retrieval import retrieved_block

    task_id = job["task_id"]
    output_path = Path(job["output_path"])
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("status") in {"complete", "error"}:
            return {"task_id": task_id, "repeat": job["repeat"], "status": f"resume-{existing['status']}"}
    try:
        trajectory = tau2_agent.run_task(
            task_id,
            agent_base_url=job["base_url"],
            agent_llm=f"openai/{job['model']}",
            agent_max_tokens=job["max_tokens"],
            user_llm=job["user_llm"],
            memory_lookup=(
                (lambda query: retrieved_block(job["memory_bank"], query, job["memory_top_k"]))
                if job["memory_bank"] else None
            ),
            seed=job["seed"],
            max_steps=job["max_steps"],
        )
        record = {
            "protocol": "tau2_rollout_v1",
            "status": "complete",
            "task_id": task_id,
            "split": job["split"],
            "repeat": job["repeat"],
            "memory_bank": job["memory_bank"],
            "retrieved_memory": trajectory.pop("retrieved_memory"),
            "trajectory": trajectory,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        record = {
            "protocol": "tau2_rollout_v1",
            "status": "error",
            "task_id": task_id,
            "split": job["split"],
            "repeat": job["repeat"],
            "memory_bank": job["memory_bank"],
            "error": repr(exc),
            "traceback": traceback.format_exc()[-4_000:],
        }
    write_json(output_path, record)
    trajectory = record.get("trajectory") or {}
    return {
        "task_id": task_id,
        "repeat": job["repeat"],
        "status": record["status"],
        "reward": trajectory.get("reward"),
        "termination": trajectory.get("termination_reason"),
        "steps": len(trajectory.get("steps") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("airline", "retail", "telecom"), required=True)
    parser.add_argument("--split", choices=("train", "test", "base"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-bank", type=Path, default=None)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--task-ids", nargs="*", default=None, help="full ids, <domain>::<tau2 id>")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1, help="run each task N times with seeds seed..seed+N-1")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--user-llm", default=None, help="default: tau2_agent.DEFAULT_USER_LLM")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument(
        "--llm-cache", type=Path, default=None,
        help="sqlite memo of user-simulator / NL-judge calls (default tau2_agent.DEFAULT_LLM_CACHE)",
    )
    parser.add_argument(
        "--no-llm-cache", action="store_true",
        help="call Gemini fresh every time -- only for measuring the simulator's own spread",
    )
    args = parser.parse_args()

    from trajectory_memory_lab import tau2_agent

    tau2_agent.quiet_logs()
    tau2_agent.configure_gemini()
    llm_cache = None if args.no_llm_cache else (args.llm_cache or tau2_agent.DEFAULT_LLM_CACHE)
    tau2_agent.install_llm_cache(llm_cache)
    user_llm = args.user_llm or tau2_agent.DEFAULT_USER_LLM

    available = tau2_agent.split_task_ids(args.domain, args.split)
    task_ids = list(args.task_ids) if args.task_ids else available
    unknown = set(task_ids) - set(available)
    if unknown:
        raise SystemExit(f"{len(unknown)} task id(s) not in {args.domain}/{args.split}: {sorted(unknown)[:3]}")
    if args.limit:
        task_ids = task_ids[: args.limit]

    trajectories_dir = args.output / "trajectories"
    jobs = []
    for repeat in range(args.repeat):
        suffix = f"__r{repeat}" if args.repeat > 1 else ""
        for task_id in task_ids:
            jobs.append({
                "task_id": task_id,
                "split": args.split,
                "repeat": repeat,
                "output_path": str(trajectories_dir / f"{tau2_agent.task_file_stem(task_id)}{suffix}.json"),
                "memory_bank": str(args.memory_bank.resolve()) if args.memory_bank else None,
                "memory_top_k": args.memory_top_k,
                "model": args.model,
                "base_url": args.base_url,
                "user_llm": user_llm,
                "max_steps": args.max_steps,
                "max_tokens": args.max_tokens,
                "seed": args.seed + repeat,
            })
    write_json(args.output / "protocol.json", {
        "protocol": "tau2_rollout_v1",
        "domain": args.domain,
        "split": args.split,
        "tasks": len(task_ids),
        "repeat": args.repeat,
        "agent": f"tau2 LLMAgent + memory block, openai/{args.model}, temperature 0, thinking disabled",
        "user_simulator": {"llm": user_llm, "llm_args": tau2_agent.DEFAULT_USER_LLM_ARGS},
        "llm_cache": str(llm_cache) if llm_cache else None,
        "nl_assertion_judge": {"llm": tau2_agent.DEFAULT_JUDGE_LLM, "llm_args": tau2_agent.DEFAULT_JUDGE_LLM_ARGS},
        "max_steps": args.max_steps,
        "memory_bank": str(args.memory_bank) if args.memory_bank else None,
        "memory_retrieval": f"bm25_top{args.memory_top_k}_on_first_customer_message" if args.memory_bank else None,
        "seed": args.seed,
    })

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run_one, job) for job in jobs]):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result}", flush=True)

    records = [read_json(Path(job["output_path"])) for job in jobs if Path(job["output_path"]).exists()]
    complete = [r for r in records if r.get("status") == "complete"]
    first = [r for r in complete if r["repeat"] == 0]
    summary: dict[str, Any] = {
        "protocol": "tau2_rollout_v1",
        "domain": args.domain,
        "split": args.split,
        "tasks": len(task_ids),
        "complete": len(complete),
        "errors": len(records) - len(complete),
        "success": sum(bool(r["trajectory"]["success"]) for r in first),
        "pass_rate": (sum(bool(r["trajectory"]["success"]) for r in first) / len(first)) if first else None,
        "terminations": dict(Counter(r["trajectory"]["termination_reason"] for r in first)),
        "memory_bank": str(args.memory_bank) if args.memory_bank else None,
    }
    if args.repeat > 1:
        by_task: dict[str, list[bool]] = {}
        for record in complete:
            by_task.setdefault(record["task_id"], []).append(bool(record["trajectory"]["success"]))
        full = {t: v for t, v in by_task.items() if len(v) == args.repeat}
        agree = sum(len(set(v)) == 1 for v in full.values())
        summary["repeat_check"] = {
            "tasks_with_all_repeats": len(full),
            "outcome_agrees_across_repeats": agree,
            "agreement_rate": agree / len(full) if full else None,
            "pass_rate_per_repeat": [
                sum(v[i] for v in full.values()) / len(full) if full else None for i in range(args.repeat)
            ],
            "flipping_tasks": sorted(t for t, v in full.items() if len(set(v)) > 1),
        }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
