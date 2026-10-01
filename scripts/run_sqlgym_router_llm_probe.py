#!/usr/bin/env python3
"""SQLGym / BIRD counterpart of `run_textcraft_router_llm_probe.py`.

Builds one bank over `sqlgym_experiment/base_train_v1`'s train trajectories
with `RouterBuilderConfig(domain="sqlgym")` -- same model, same router
prompt, same payload builder as the other seven benchmarks.

Runs under the repo .venv (the router chain needs torch); only the sft
replay subprocesses run under sqlgym_venv, since they are what actually
opens the BIRD databases.

This benchmark was chosen to discriminate between two accounts the earlier
results left fitted rather than tested -- see `sqlgym_agent` for both, and
for the predictions recorded before the run.
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

GROUP = "sqlgym"
SQLGYM_PYTHON = "/nas04/yixuh/sqlgym_venv/bin/python"


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
    parser.add_argument("--output", type=Path, default=ROOT / "sqlgym_experiment/router_llm_probe_v1")
    parser.add_argument("--train-rollout", type=Path, default=ROOT / "sqlgym_experiment/base_train_v1")
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
    parser.add_argument("--bird-path", default="/nas04/yixuh/bird", help="BIRD dataset root")
    parser.add_argument(
        "--run-eval", action="store_true",
        help="also score the resulting bank on --eval-split (first --eval-limit goals)",
    )
    parser.add_argument("--eval-split", choices=("test", "xdb"), default="xdb")
    parser.add_argument("--eval-limit", type=int, default=500)
    args = parser.parse_args()

    os.environ["SQLGYM_BIRD_PATH"] = args.bird_path  # read by the sft replay subprocesses
    
    trajectories = load_train_trajectories(args.train_rollout)

    # BIRD is a fixed dataset on disk rather than a live server, so the
    # equivalent of the AgentGym world check is that a recorded task still
    # resolves to the same question -- a different BIRD release renumbers the
    # items and would silently compare different tasks.
    #
    # Read the question straight out of BIRD's json rather than through
    # SqlGymTaskEnv: this probe runs under the repo .venv (the router chain
    # needs torch) and `sqlgym` is installed only in sqlgym_venv, which is
    # what the replay subprocesses use. The first version of this guard
    # imported sqlgym here and killed all three build arms on the first run.
    probe_task = sorted(trajectories)[0]
    recorded = (trajectories[probe_task]["task"]["instruction"] or "").strip()
    mode, idx = probe_task.split("::")[1], int(probe_task.split("::")[2])
    bird_rows = json.loads(
        (Path(args.bird_path) / mode / f"{mode}.json").read_text(encoding="utf-8")
    )
    if idx >= len(bird_rows):
        raise SystemExit(f"BIRD {mode} at {args.bird_path} has {len(bird_rows)} rows, "
                         f"but {probe_task} needs index {idx}")
    question = (bird_rows[idx].get("question") or "").strip()
    # The recorded instruction is "<schema description>\n\n<question>", so the
    # question is its tail; comparing the tail is enough to catch renumbering
    # without depending on how the schema text is rendered.
    if not question or not recorded.endswith(question):
        raise SystemExit(
            f"BIRD at {args.bird_path} does not reproduce {probe_task}:\n"
            f"  expected the recorded instruction to end with: {question[:200]!r}\n"
            f"  recorded instruction tail:                     {recorded[-200:]!r}"
        )

    task_ids = sorted(trajectories)
    if args.limit > 0:
        task_ids = task_ids[: args.limit]
    print(
        f"router_mode={args.router_mode} domain=sqlgym ({args.model}), {len(task_ids)} train tasks, sft_writer={args.sft_writer}",
        flush=True,
    )

    config = RouterBuilderConfig(
        output=args.output, record_protocol="sqlgym_router_llm_probe_v1",
        model=args.model, base_url=args.base_url,
        sft_writer=args.sft_writer,
        router_mode=args.router_mode, domain="sqlgym",
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
        "protocol": "sqlgym_router_llm_probe_v1",
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
            seed=20260922, domain="sqlgym",
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
        bank_path = args.output / "banks" / f"memory_{GROUP}.json"
        eval_dir = args.output / f"eval_{args.eval_split}{args.eval_limit}"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        cmd = [
            SQLGYM_PYTHON, "-u", str(ROOT / "scripts/run_sqlgym_rollout.py"),
            "--split", args.eval_split, "--limit", str(args.eval_limit), "--output", str(eval_dir),
            "--experiment-name", "sqlgym_router_llm_probe_v1_eval", "--bird-path", args.bird_path,
            "--memory-bank", str(bank_path), "--memory-top-k", "3",
            "--max-parallel", "4",
            "--model", args.model, "--base-url", args.base_url,
        ]
        print(f"launching {args.eval_split}-split eval (first {args.eval_limit} goals)", flush=True)
        result_proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if result_proc.returncode == 0:
            eval_summary = read_json(eval_dir / "summary.json")
            summary[f"eval_{args.eval_split}_pass_rate"] = eval_summary["pass_rate"]
            summary[f"eval_{args.eval_split}_mean_score"] = eval_summary["mean_score"]
            print(f"eval_{args.eval_split}: pass_rate={eval_summary['pass_rate']:.4f} "
                  f"mean_score={eval_summary['mean_score']:.4f}", flush=True)

    write_json(args.output / "summary.json", summary)
    print(f"wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
