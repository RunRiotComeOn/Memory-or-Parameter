#!/usr/bin/env python3
"""Validate the memory-side logprob proxy reward from router_reward_v1/DESIGN.md.

For each candidate memory entry M already written by the alloc_v1 writer runs,
and each already-verified-successful probe trajectory p in the same AppWorld
task group (same task_id prefix, excluding M's own source task), compute

    reward(M, p) = logP(p's real assistant turns | prompt_p + M)
                 - logP(p's real assistant turns | prompt_p)

via teacher forcing (no sampling). As a control, also score a "placebo" memory
M' -- a real memory entry pulled from an unrelated task group -- against the
same probe, using the same cached baseline. If the proxy reward is doing
anything sensible, the real M should score higher than the mismatched M' on
average, since M' is topically irrelevant to p.

This does not touch SFT-side reward (design doc section 2.2, deferred) and
does not train anything. It only asks: is this signal worth building a router
on top of.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.appworld_agent import AGENT_SYSTEM  # noqa: E402
from trajectory_memory_lab.logprob_scoring import score_assistant_turns  # noqa: E402
from trajectory_memory_lab.memory_retrieval import render_memory_block  # noqa: E402

DEFAULT_MODEL_PATH = (
    "/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/"
    "snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
)
DEFAULT_TRAJECTORY_GLOB = "appworld_experiment/base_train_v2/trajectories/*.json"
DEFAULT_ARMS = ("a1_outcome", "a2_counterfactual", "a3_budgeted")
DEFAULT_BANK_DIR = ROOT / "appworld_experiment/alloc_banks_v1/banks"


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def group_of(task_id: str) -> str:
    return task_id.split("_")[0]


def load_trajectories(pattern: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(ROOT / pattern))):
        record = read_json(path)
        trajectory = record.get("trajectory")
        if not trajectory:
            continue
        records[trajectory["task"]["id"]] = trajectory
    return records


def load_candidate_memories(arms: tuple[str, ...]) -> list[dict[str, Any]]:
    candidates = []
    for arm in arms:
        bank_path = DEFAULT_BANK_DIR / arm / "memory_appworld.json"
        if not bank_path.exists():
            continue
        for entry in read_json(bank_path):
            if entry.get("status") != "active":
                continue
            entry = dict(entry)
            entry["_arm"] = arm
            candidates.append(entry)
    return candidates


def build_messages(trajectory: dict[str, Any], memory_block: str) -> list[dict[str, str]]:
    """Reconstruct the exact message history run_task would have built.

    steps[0]["content"] already equals build_initial_user_message(task, "")
    for every base_train_v2 trajectory (that run had no memory bank), so the
    memory-injected variant is produced the same way build_initial_user_message
    itself appends it: `"\n".join([..., memory_block])`.
    """
    steps = trajectory["steps"]
    user_0 = steps[0]["content"]
    if memory_block:
        user_0 = user_0 + "\n" + memory_block
    messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_0}]
    for step in steps[1:]:
        if step["role"] == "assistant":
            messages.append({"role": "assistant", "content": step["content"]})
        else:  # "tool" (and defensively "user") both map to the user role
            messages.append({"role": "user", "content": step["content"]})
    return messages


def memory_block_for(entry: dict[str, Any]) -> str:
    rendered = {"id": entry["id"], "scope": entry["scope"], "content": entry["content"]}
    return render_memory_block([rendered])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    parser.add_argument("--probes-per-entry", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "router_reward_v1/proxy_reward_v1_results.json",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"loading tokenizer from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1200)

    trajectories = load_trajectories(DEFAULT_TRAJECTORY_GLOB)
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, trajectory in trajectories.items():
        groups[group_of(task_id)].append(task_id)

    def successful_probes(source_task_id: str) -> list[str]:
        group = group_of(source_task_id)
        return [
            task_id
            for task_id in groups.get(group, [])
            if task_id != source_task_id and trajectories[task_id]["success"]
        ]

    candidates = load_candidate_memories(tuple(args.arms))
    usable = [c for c in candidates if successful_probes(c["source_task_id"])]
    print(f"candidate memories: {len(candidates)}, usable (>=1 in-group probe): {len(usable)}", flush=True)

    # placebo pool: any usable candidate from a different group than the one
    # being tested, so the swap is guaranteed topically unrelated.
    def placebo_for(entry: dict[str, Any]) -> dict[str, Any]:
        own_group = group_of(entry["source_task_id"])
        pool = [c for c in usable if group_of(c["source_task_id"]) != own_group]
        return rng.choice(pool)

    baseline_cache: dict[str, float] = {}

    def baseline_score(probe_task_id: str) -> float:
        if probe_task_id not in baseline_cache:
            trajectory = trajectories[probe_task_id]
            messages = build_messages(trajectory, "")
            result = score_assistant_turns(client, tokenizer, args.model, messages)
            baseline_cache[probe_task_id] = result.total_logprob
            print(
                f"  baseline[{probe_task_id}] logprob={result.total_logprob:.2f} "
                f"tokens={result.scored_token_count}",
                flush=True,
            )
        return baseline_cache[probe_task_id]

    def with_memory_score(entry: dict[str, Any], probe_task_id: str) -> float:
        trajectory = trajectories[probe_task_id]
        block = memory_block_for(entry)
        messages = build_messages(trajectory, block)
        result = score_assistant_turns(client, tokenizer, args.model, messages)
        return result.total_logprob

    records = []
    for i, entry in enumerate(usable):
        probes = successful_probes(entry["source_task_id"])
        rng.shuffle(probes)
        probes = probes[: args.probes_per_entry]
        placebo = placebo_for(entry)
        print(
            f"[{i + 1}/{len(usable)}] entry={entry['id']} arm={entry['_arm']} "
            f"group={group_of(entry['source_task_id'])} probes={probes}",
            flush=True,
        )
        for probe_task_id in probes:
            base = baseline_score(probe_task_id)
            real_with = with_memory_score(entry, probe_task_id)
            placebo_with = with_memory_score(placebo, probe_task_id)
            record = {
                "entry_id": entry["id"],
                "arm": entry["_arm"],
                "entry_group": group_of(entry["source_task_id"]),
                "probe_task_id": probe_task_id,
                "probe_group": group_of(probe_task_id),
                "placebo_entry_id": placebo["id"],
                "placebo_group": group_of(placebo["source_task_id"]),
                "baseline_logprob": base,
                "real_logprob": real_with,
                "placebo_logprob": placebo_with,
                "reward_real": real_with - base,
                "reward_placebo": placebo_with - base,
            }
            records.append(record)
            print(
                f"    probe={probe_task_id} reward_real={record['reward_real']:+.3f} "
                f"reward_placebo={record['reward_placebo']:+.3f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    reward_real = [r["reward_real"] for r in records]
    reward_placebo = [r["reward_placebo"] for r in records]
    wins = sum(1 for r in records if r["reward_real"] > r["reward_placebo"])
    print("\n=== summary ===")
    print(f"pairs: {len(records)}")
    print(f"reward_real   mean={statistics.mean(reward_real):+.3f} median={statistics.median(reward_real):+.3f} "
          f"positive_frac={sum(1 for x in reward_real if x > 0) / len(reward_real):.2f}")
    print(f"reward_placebo mean={statistics.mean(reward_placebo):+.3f} median={statistics.median(reward_placebo):+.3f} "
          f"positive_frac={sum(1 for x in reward_placebo if x > 0) / len(reward_placebo):.2f}")
    print(f"real beats placebo on {wins}/{len(records)} pairs ({wins / len(records):.2%})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
