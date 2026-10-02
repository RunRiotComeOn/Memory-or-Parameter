#!/usr/bin/env python3
"""TextCraft counterpart of `run_babyai_router_llm_probe.py`.

Builds one bank over `textcraft_experiment/base_train_v1`'s train
trajectories with `RouterBuilderConfig(domain="textcraft")` -- same model,
same router prompt, same payload builder as the other six benchmarks; what
changes per domain is the writer's framing sentence and which
`*_sft_writer` module supplies the teacher prompt, both looked up by
`router_bank_builder`'s domain tables.

`--router-mode` selects the ablation: "llm" is the router deciding, and
"force_memory"/"force_sft" are the two forced arms defined in
`alfworld_summary.md`.

Note what the pool is here. The train split is recipe-tree depth 1-2, where
the base agent succeeds about 95% of the time, so the pool is dominated by
CONSOLIDATION rather than rescue -- far more so than in any earlier domain.
That is deliberate: the held-out line that matters is `deep` (depth 3-4),
which no pool ever sees, making this the suite's first compositional
generalization test. It also puts `babyai_summary.md` conclusion 2
("consolidation samples buy memorization, not capability") to a direct test.

Run from the repo's default .venv; the AgentGym env server must already be
up on :36002.
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

GROUP = "textcraft"
TEXTCRAFT_PYTHON = str(ROOT / ".venv/bin/python")


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
    parser.add_argument("--output", type=Path, default=ROOT / "textcraft_experiment/router_llm_probe_v1")
    parser.add_argument("--train-rollout", type=Path, default=ROOT / "textcraft_experiment/base_train_v1")
    parser.add_argument("--model", default="qwen35-tau", help="task agent + memory draft writer + router")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0, help="0 = every train task in the rollout; >0 truncates for a quick look")
    parser.add_argument("--sft-writer", choices=("teacher", "self", "none"), default="teacher")
    parser.add_argument(
        "--router-mode", choices=("llm", "gemini", "force_memory", "force_sft"), default="llm",
        help="'llm' = normal prompted routing (default); 'gemini' = the same prompt and payload read by a cheap hosted Gemini Flash model instead of the local task model (see router_gemini_policy); 'force_memory' = skip routing, every task commits its drafted memory unconditionally (ablation); "
             "'force_sft' = the mirror ablation, every task commits its drafted sft plan and nothing ever reaches the memory bank",
    )
    parser.add_argument(
        "--skip-sft-replay", action="store_true",
        help="don't replay+verify committed sft/both decisions (route counts and the memory bank still work without this)",
    )
    parser.add_argument("--env-url", default="http://127.0.0.1:36002", help="AgentGym TextCraft env server")
    parser.add_argument(
        "--run-eval", action="store_true",
        help="also score the resulting bank on --eval-split (first --eval-limit goals)",
    )
    parser.add_argument("--eval-split", choices=("test", "deep"), default="deep")
    parser.add_argument("--eval-limit", type=int, default=500)
    args = parser.parse_args()

    os.environ["TEXTCRAFT_ENV_URL"] = args.env_url  # read by the sft replay subprocesses
    from trajectory_memory_lab.textcraft_agent import TextCraftEnvClient

    trajectories = load_train_trajectories(args.train_rollout)

    # Verify this server rebuilds the SAME tasks the train rollout saw: every
    # sft replay compares a fresh attempt against a stored one. AgentGym has
    # no /info and TextCraft has no server seed -- a goal is a pure function
    # of data_idx GIVEN the crafting tree -- so the check is an observation
    # comparison, which also catches a differently-built recipe set.
    probe_task = sorted(trajectories)[0]
    recorded = (trajectories[probe_task]["steps"][0]["content"] or "").strip()
    env_probe = TextCraftEnvClient(args.env_url)
    try:
        env_probe.create()
        env_probe.assert_reproduces(int(probe_task.rsplit("::", 1)[1]), recorded)
    finally:
        env_probe.close()

    task_ids = sorted(trajectories)
    if args.limit > 0:
        task_ids = task_ids[: args.limit]
    print(
        f"router_mode={args.router_mode} domain=textcraft ({args.model}), {len(task_ids)} train tasks, sft_writer={args.sft_writer}",
        flush=True,
    )

    config = RouterBuilderConfig(
        output=args.output, record_protocol="textcraft_router_llm_probe_v1",
        model=args.model, base_url=args.base_url,
        sft_writer=args.sft_writer,
        router_mode=args.router_mode, domain="textcraft",
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
        "protocol": "textcraft_router_llm_probe_v1",
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
            seed=20260922, domain="textcraft",
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
            TEXTCRAFT_PYTHON, "-u", str(ROOT / "scripts/run_textcraft_rollout.py"),
            "--split", args.eval_split, "--limit", str(args.eval_limit), "--output", str(eval_dir),
            "--experiment-name", "textcraft_router_llm_probe_v1_eval", "--env-url", args.env_url,
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
