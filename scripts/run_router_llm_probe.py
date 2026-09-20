#!/usr/bin/env python3
"""First look at the LLM router (router_llm_policy.py), NOT trained.

Builds one bank over the 90 base_train_v2 tasks using `RouterBuilderConfig
(router_mode="llm")` -- the SAME model that plays the task agent (qwen35-tau,
the Qwen3.5-35B-A3B MoE, served by det_server_a/b) decides each route by
prompted judgment, reading the FULL task instruction + trajectory + the
drafted memory/sft candidates -- not just a hashed numeric summary. No GRPO,
no gradient, no logprob -- this is a single deterministic (temperature=0)
pass to see what an untrained LLM router's own judgment looks like before
deciding whether it's worth training at all. No separate server: this reuses
whatever --base-url/--model the task agent itself is served on.

Reports the route distribution and, optionally, the real 57-task dev
pass_rate for the resulting bank -- the same number every trained-router
run reports, so this is directly comparable to running_log.md's other
entries (v2/v3/v5/...).

Run:
  PYTHONPATH=src python3 -u scripts/run_router_llm_probe.py \
      --output router_reward_v1/router_llm_probe_v1 \
      --base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_bank_builder import RouterBuilderConfig, run_router_chain  # noqa: E402

GROUP = "appworld"
TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_train_trajectories() -> dict[str, dict[str, Any]]:
    protocol = read_json(TRAIN_ROLLOUT / "protocol.json")
    if protocol.get("split") != "train":
        raise ValueError(f"expected split=train, got {protocol.get('split')!r}")
    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted((TRAIN_ROLLOUT / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        trajectory = record["trajectory"]
        trajectory["domain"] = GROUP
        trajectories[record["task_id"]] = trajectory
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "router_reward_v1/router_llm_probe_v1")
    parser.add_argument("--model", default="qwen35-tau", help="task agent + memory/sft draft writer + router")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--sft-writer", choices=("teacher", "self"), default="teacher")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 90 train tasks; >0 truncates for a quick look")
    parser.add_argument("--run-dev-eval", action="store_true", help="also score the resulting bank on the real 57-task dev split")
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    task_ids = sorted(trajectories)
    if args.limit > 0:
        task_ids = task_ids[: args.limit]
    print(f"router_mode=llm ({args.model}), {len(task_ids)} train tasks, sft_writer={args.sft_writer}", flush=True)

    config = RouterBuilderConfig(
        output=args.output, record_protocol="router_llm_probe_v1",
        model=args.model, base_url=args.base_url,
        sft_writer=args.sft_writer,
        router_mode="llm",
    )
    result = run_router_chain(
        None, GROUP, task_ids, trajectories, config, total_task_count=len(task_ids),
    )
    route_counts = Counter(d["route"] for d in result.decisions)
    by_success: dict[bool, Counter] = {True: Counter(), False: Counter()}
    for record in result.summary["records"]:
        by_success[bool(record.get("base_agent_success"))][record.get("router_route")] += 1
    print(f"route_counts={dict(route_counts)} active_entries={result.summary['active_entries']}", flush=True)
    print(f"  base_agent success -> routes: {dict(by_success[True])}", flush=True)
    print(f"  base_agent failure -> routes: {dict(by_success[False])}", flush=True)

    summary = {
        "protocol": "router_llm_probe_v1",
        "router_llm_model": args.model,
        "task_count": len(task_ids),
        "route_counts": dict(route_counts),
        "active_entries": result.summary["active_entries"],
        "routes_when_base_agent_succeeded": dict(by_success[True]),
        "routes_when_base_agent_failed": dict(by_success[False]),
    }

    if args.run_dev_eval:
        bank_path = args.output / "banks" / f"memory_{GROUP}.json"
        eval_dir = args.output / "eval_full_dev"
        env = dict(os.environ)
        env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
        env["PYTHONPATH"] = str(ROOT / "src")
        cmd = [
            APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_rollout.py"),
            "--split", "dev", "--output", str(eval_dir), "--experiment-name", "router_llm_probe_v1_dev",
            "--memory-bank", str(bank_path), "--memory-top-k", "3",
            "--max-parallel", "1", "--seed", "20260822", "--model", args.model, "--base-url", args.base_url,
        ]
        print(f"launching real dev eval: {' '.join(cmd)}", flush=True)
        result_proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if result_proc.returncode == 0:
            dev_summary = read_json(eval_dir / "summary.json")
            summary["full_dev_pass_rate"] = dev_summary["pass_rate"]
            print(f"full_dev_pass_rate={dev_summary['pass_rate']:.4f}", flush=True)

    write_json(args.output / "summary.json", summary)
    print(f"wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
