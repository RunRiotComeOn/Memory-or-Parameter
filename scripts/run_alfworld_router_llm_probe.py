#!/usr/bin/env python3
"""ALFWorld counterpart of run_router_llm_probe.py -- first look at the LLM
router (router_llm_policy.py) on the second benchmark, NOT trained.

Builds one bank over `alfworld_experiment/base_train_v1`'s 40 train
trajectories using `RouterBuilderConfig(router_mode="llm", domain="alfworld")`
-- same model, same prompt, same payload builder as the AppWorld probe; only
the writer's framing sentence changes (see `router_bank_builder.
RouterBuilderConfig.domain`'s docstring). SFT drafting is skipped entirely
for this domain (no ALFWorld teacher prompt or guided-replay script exists
yet), so decisions can only ever land on `memory` or `neither` -- an `sft`/
`both` pick would simply fail validation (`missing_plan`) and be recorded as
such, not silently miscounted as a real SFT commit.

Reports the route distribution and, optionally, the real 57-task
valid_unseen pass_rate for the resulting bank, directly comparable to
`alfworld_experiment/baseline_valid_unseen_v1/summary.json` (no-memory
baseline, same 57 task_ids, same replica/config) -- running_log.md section
15 already flags that any "did memory help" claim must reuse that exact
baseline rather than a fresh one.

Run (from the repo's default .venv, NOT alfworld_venv310 -- this script only
calls into trajectory_memory_lab and subprocess-launches the ALFWorld eval
under the right interpreter itself):
  PYTHONPATH=src python3 -u scripts/run_alfworld_router_llm_probe.py \
      --output alfworld_experiment/router_llm_probe_v1 \
      --base-url http://127.0.0.1:8000/v1 --run-dev-eval
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
    collect_batch_sft_examples,
    sft_candidates_from_records,
)

GROUP = "alfworld"
ALFWORLD_PYTHON = "/nas04/yixuh/alfworld_venv310/bin/python"
ALFWORLD_DATA_DEFAULT = "/nas04/yixuh/alfworld_data"
EVAL_TASK_IDS_FILE = Path("/tmp/alfworld_unseen57_lines.txt")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_train_trajectories(train_rollout: Path) -> dict[str, dict[str, Any]]:
    protocol = read_json(train_rollout / "protocol.json")
    if protocol.get("split") != "train":
        raise ValueError(f"expected split=train, got {protocol.get('split')!r}")
    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted((train_rollout / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        trajectory = record["trajectory"]
        trajectory["domain"] = GROUP
        trajectories[record["task_id"]] = trajectory
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "alfworld_experiment/router_llm_probe_v1")
    parser.add_argument("--train-rollout", type=Path, default=ROOT / "alfworld_experiment/base_train_v2")
    parser.add_argument("--model", default="qwen35-tau", help="task agent + memory draft writer + router")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 40 train tasks; >0 truncates for a quick look")
    parser.add_argument("--sft-writer", choices=("teacher", "self"), default="teacher")
    parser.add_argument(
        "--skip-sft-replay", action="store_true",
        help="don't replay+verify committed sft/both decisions (route counts and the memory bank still work without this)",
    )
    parser.add_argument(
        "--run-dev-eval", action="store_true",
        help="also score the resulting bank on the same 57-task valid_unseen sample as baseline_valid_unseen_v1",
    )
    args = parser.parse_args()

    trajectories = load_train_trajectories(args.train_rollout)
    task_ids = sorted(trajectories)
    if args.limit > 0:
        task_ids = task_ids[: args.limit]
    print(
        f"router_mode=llm domain=alfworld ({args.model}), {len(task_ids)} train tasks, sft_writer={args.sft_writer}",
        flush=True,
    )

    config = RouterBuilderConfig(
        output=args.output, record_protocol="alfworld_router_llm_probe_v1",
        model=args.model, base_url=args.base_url,
        sft_writer=args.sft_writer,
        router_mode="llm", domain="alfworld",
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
        "protocol": "alfworld_router_llm_probe_v1",
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
            seed=20260919, domain="alfworld",
        )
        summary["sft_candidates_replayed"] = len(candidates)
        summary["sft_examples_verified"] = len(sft_examples)
        summary["sft_rescue_yield"] = (
            len(sft_examples) / len(candidates) if candidates else None
        )
        write_json(args.output / "sft_pool_v1.jsonl.preview", sft_examples)
        print(
            f"sft replay: {len(candidates)} replayed, {len(sft_examples)} verified success "
            f"(yield={summary['sft_rescue_yield']})",
            flush=True,
        )

    if args.run_dev_eval:
        bank_path = args.output / "banks" / f"memory_{GROUP}.json"
        eval_dir = args.output / "eval_valid_unseen57"
        eval_task_ids = EVAL_TASK_IDS_FILE.read_text().split()
        env = dict(os.environ)
        env["ALFWORLD_DATA"] = env.get("ALFWORLD_DATA", ALFWORLD_DATA_DEFAULT)
        cmd = [
            ALFWORLD_PYTHON, "-u", str(ROOT / "scripts/run_alfworld_rollout.py"),
            "--split", "valid_unseen", "--output", str(eval_dir),
            "--experiment-name", "alfworld_router_llm_probe_v1_eval",
            "--task-ids", *eval_task_ids,
            "--memory-bank", str(bank_path), "--memory-top-k", "3",
            "--max-parallel", "4", "--max-steps", "40",
            "--model", args.model, "--base-url", args.base_url,
        ]
        print(f"launching real valid_unseen eval ({len(eval_task_ids)} tasks, same sample as baseline_valid_unseen_v1)", flush=True)
        result_proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if result_proc.returncode == 0:
            dev_summary = read_json(eval_dir / "summary.json")
            summary["valid_unseen_pass_rate"] = dev_summary["pass_rate"]
            print(f"valid_unseen_pass_rate={dev_summary['pass_rate']:.4f}", flush=True)

    write_json(args.output / "summary.json", summary)
    print(f"wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
