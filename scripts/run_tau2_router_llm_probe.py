#!/usr/bin/env python3
"""tau2-bench counterpart of run_webshop_router_llm_probe.py.

Builds one bank per tau2 domain (`--domain airline|retail|telecom`, bank
`memory_tau2_<domain>.json`) over a tau2 TRAIN-split rollout
(`scripts/run_tau2_rollout.py`), with `RouterBuilderConfig(domain="tau2")` --
same model, router prompt and payload builder as the other probes; per
domain only the writer's framing sentence and the `tau2_sft_writer` prompts
change.

`--router-mode` selects the ablation exactly as for the other benchmarks:
"llm", "force_memory", "force_sft".

Each trajectory's `sft_example` (the agent's full prompt: policy + tool
schemas, several thousand tokens) is dropped before routing: the router and
memory writer see the trajectory itself, and would otherwise be handed the
same policy text again for every task. Guided replays produce their own.

The optional held-out eval (`--run-eval`) scores the bank on the domain's
tau2 `test` split with the same user simulator.

Run from the repo's default .venv (sft replays and the eval subprocess
launch under tau2's venv themselves):
  PYTHONPATH=src .venv/bin/python -u scripts/run_tau2_router_llm_probe.py --domain airline \
      --train-rollout tau2_experiment/airline_base_train_v1 \
      --output tau2_experiment/airline_router_llm_probe_v1 \
      --base-url http://127.0.0.1:8030/v1 --run-eval
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
from trajectory_memory_lab.router_sft_pipeline import (  # noqa: E402
    append_to_pool,
    collect_batch_sft_examples,
    sft_candidates_from_records,
)

TAU2_PYTHON = str(ROOT / "third_party/tau2-bench/.venv/bin/python")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_train_trajectories(train_rollout: Path, group: str) -> dict[str, dict[str, Any]]:
    protocol = read_json(train_rollout / "protocol.json")
    if protocol.get("split") != "train":
        raise ValueError(f"expected split=train, got {protocol.get('split')!r}")
    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted((train_rollout / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        trajectory = record["trajectory"]
        trajectory.pop("sft_example", None)
        trajectory["domain"] = group
        trajectories[record["task_id"]] = trajectory
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rollout", type=Path, required=True, help="a run_tau2_rollout.py --split train output")
    parser.add_argument("--model", default="qwen35-tau", help="task agent + memory draft writer + router")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0, help="0 = every train task in the rollout; >0 truncates for a quick look")
    parser.add_argument("--sft-writer", choices=("teacher", "self", "none"), default="teacher")
    parser.add_argument(
        "--router-mode", choices=("llm", "force_memory", "force_sft"), default="llm",
        help="'llm' = normal prompted routing (default); 'force_memory' = skip routing, every task commits its drafted memory unconditionally (ablation); "
             "'force_sft' = the mirror ablation, every task commits its drafted sft plan and nothing ever reaches the memory bank",
    )
    parser.add_argument(
        "--skip-sft-replay", action="store_true",
        help="don't replay+verify committed sft/both decisions (route counts and the memory bank still work without this)",
    )
    parser.add_argument("--domain", choices=("airline", "retail", "telecom"), required=True)
    parser.add_argument(
        "--run-eval", action="store_true",
        help="also score the resulting bank on the domain's tau2 test split",
    )
    args = parser.parse_args()

    group = f"tau2_{args.domain}"
    protocol = read_json(args.train_rollout / "protocol.json")
    if protocol.get("domain") != args.domain:
        raise SystemExit(f"--domain {args.domain} but the train rollout is {protocol.get('domain')!r}")
    trajectories = load_train_trajectories(args.train_rollout, group)
    task_ids = sorted(trajectories)
    if args.limit > 0:
        task_ids = task_ids[: args.limit]
    print(
        f"router_mode={args.router_mode} domain=tau2/{args.domain} ({args.model}), {len(task_ids)} train tasks, sft_writer={args.sft_writer}",
        flush=True,
    )

    config = RouterBuilderConfig(
        output=args.output, record_protocol="tau2_router_llm_probe_v1",
        model=args.model, base_url=args.base_url,
        sft_writer=args.sft_writer,
        router_mode=args.router_mode, domain="tau2",
    )
    result = run_router_chain(
        None, group, task_ids, trajectories, config, total_task_count=len(task_ids),
    )
    route_counts = Counter(d["route"] for d in result.decisions)
    by_success: dict[bool, Counter] = {True: Counter(), False: Counter()}
    for record in result.summary["records"]:
        by_success[bool(record.get("base_agent_success"))][record.get("router_route")] += 1
    print(f"route_counts={dict(route_counts)} active_entries={result.summary['active_entries']}", flush=True)
    print(f"  base_agent success -> routes: {dict(by_success[True])}", flush=True)
    print(f"  base_agent failure -> routes: {dict(by_success[False])}", flush=True)

    summary = {
        "protocol": "tau2_router_llm_probe_v1",
        "tau2_domain": args.domain,
        "router_llm_model": args.model,
        "task_count": len(task_ids),
        "route_counts": dict(route_counts),
        "active_entries": result.summary["active_entries"],
        "routes_when_base_agent_succeeded": dict(by_success[True]),
        "routes_when_base_agent_failed": dict(by_success[False]),
    }

    if not args.skip_sft_replay:
        candidates = sft_candidates_from_records(result.summary["records"])
        print(f"sft/both committed: {len(candidates)} candidate(s) to replay+verify", flush=True)
        replay_dir = args.output / "sft_replays"
        sft_examples = collect_batch_sft_examples(
            result.summary["records"], replay_dir, args.model, args.base_url,
            seed=20260922, domain="tau2",
        )
        summary["sft_candidates_replayed"] = len(candidates)
        summary["sft_examples_verified"] = len(sft_examples)
        summary["sft_rescue_yield"] = (
            len(sft_examples) / len(candidates) if candidates else None
        )
        pool_path = args.output / "sft_pool.jsonl"
        pool_size = append_to_pool(pool_path, sft_examples)
        summary["sft_pool_path"] = str(pool_path)
        summary["sft_pool_size"] = pool_size
        print(
            f"sft replay: {len(candidates)} replayed, {len(sft_examples)} verified success "
            f"(yield={summary['sft_rescue_yield']})",
            flush=True,
        )

    if args.run_eval:
        bank_path = args.output / "banks" / f"memory_{group}.json"
        eval_dir = args.output / "eval_test"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        cmd = [
            TAU2_PYTHON, "-u", str(ROOT / "scripts/run_tau2_rollout.py"),
            "--domain", args.domain, "--split", "test", "--output", str(eval_dir),
            "--memory-bank", str(bank_path), "--memory-top-k", "3",
            "--max-parallel", "4",
            "--model", args.model, "--base-url", args.base_url,
        ]
        print(f"launching {args.domain} test-split eval", flush=True)
        result_proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if result_proc.returncode == 0:
            eval_summary = read_json(eval_dir / "summary.json")
            summary["eval_test_pass_rate"] = eval_summary["pass_rate"]
            print(f"eval_test: pass_rate={eval_summary['pass_rate']:.4f}", flush=True)

    write_json(args.output / "summary.json", summary)
    print(f"wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
