#!/usr/bin/env python3
"""Build SFT-data-writer supervision from reward-one guided replays."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from trajectory_memory_lab.tau_sft_data_writer import (
    SFT_DATA_WRITER_SYSTEM,
    assistant_turns,
)


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
DOMAINS = ("airline", "retail", "telecom")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def encode_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in record.items() if key not in {"messages", "tools"}},
        "messages": json.dumps(record["messages"], ensure_ascii=False, separators=(",", ":")),
        "tools": "[]",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--replay-tag", default="teacher_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=45056)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.manifest)
    test = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["test"][domain]["task_ids"]
    }
    candidate_by_key = {}
    for path in sorted((args.candidates / "tasks").glob("*/candidate.json")):
        item = read_json(path)
        candidate_by_key[(item.get("domain"), str(item.get("source_task_id")))] = item

    records = []
    rejected = []
    for domain in DOMAINS:
        results_path = (
            TAU_ROOT
            / f"data/simulations/qwen35_sftdata_guided_{args.replay_tag}_{domain}/results.json"
        )
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        simulations = read_json(results_path)["simulations"]
        for simulation in simulations:
            task_id = str(simulation["task_id"])
            key = (domain, task_id)
            if key in test:
                raise ValueError("test task reached writer SFT preparation")
            candidate = candidate_by_key.get(key)
            reward_info = simulation.get("reward_info")
            reward = (
                reward_info.get("reward") if isinstance(reward_info, dict) else None
            )
            if candidate is None or reward != 1:
                rejected.append(
                    {
                        "domain": domain,
                        "source_task_id": task_id,
                        "reason": "missing_candidate" if candidate is None else "guided_replay_reward_not_one",
                        "reward": reward,
                    }
                )
                continue
            target = {
                "assistant_turns": assistant_turns(simulation["messages"]),
                "rationale": "Complete trajectory validated by a live tau replay with reward 1.",
                "risk_checks": [
                    "live tool results used",
                    "policy evaluated by tau environment",
                    "terminal reward equals 1",
                ],
            }
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": SFT_DATA_WRITER_SYSTEM},
                        {
                            "role": "user",
                            "content": json.dumps(
                                candidate["student_input"],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                target, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    ],
                    "tools": [],
                    "enable_thinking": False,
                    "source_task_id": task_id,
                    "domain": domain,
                    "source_reward": candidate.get("source_reward"),
                    "guided_replay_reward": reward,
                    "validation_status": "accepted",
                    "validation_basis": "live_guided_replay_reward_one",
                }
            )

    if not records:
        raise ValueError("no reward-one guided replays available")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    length_accepted = []
    length_rejected = []
    accepted = []
    for record in records:
        ids = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        length = len(ids["input_ids"] if isinstance(ids, dict) else ids)
        if length <= args.max_length:
            record["tokens"] = length
            accepted.append(record)
            length_accepted.append(length)
        else:
            length_rejected.append(length)
            rejected.append(
                {
                    "domain": record["domain"],
                    "source_task_id": record["source_task_id"],
                    "reason": "exceeds_max_length",
                    "tokens": length,
                }
            )
    if not accepted:
        raise ValueError("all writer examples exceed max length")

    rng = random.Random(args.seed)
    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    for record in accepted:
        by_domain[record["domain"]].append(record)
    train = []
    validation = []
    for domain in DOMAINS:
        rows = by_domain[domain]
        rng.shuffle(rows)
        validation_count = max(1, round(len(rows) * args.validation_fraction))
        validation.extend(rows[:validation_count])
        train.extend(rows[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)

    train_keys = {(item["domain"], item["source_task_id"]) for item in train}
    validation_keys = {(item["domain"], item["source_task_id"]) for item in validation}
    if train_keys & validation_keys or (train_keys | validation_keys) & test:
        raise ValueError("writer SFT split leakage detected")

    write_jsonl(args.output / "all.jsonl", accepted)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "validation.jsonl", validation)
    Dataset.from_list([encode_row(item) for item in train]).to_parquet(
        str(args.output / "train.parquet")
    )
    Dataset.from_list([encode_row(item) for item in validation]).to_parquet(
        str(args.output / "validation.parquet")
    )
    summary = {
        "protocol": "tau_sft_data_writer_supervision_v1",
        "live_reward_one_examples": len(records),
        "training_examples": len(train),
        "validation_examples": len(validation),
        "rejected": len(rejected),
        "rejection_details": rejected,
        "domains": dict(Counter(item["domain"] for item in accepted)),
        "source_rewards": dict(Counter(str(item["source_reward"]) for item in accepted)),
        "max_length": args.max_length,
        "min_tokens": min(length_accepted),
        "max_tokens": max(length_accepted),
        "mean_tokens": sum(length_accepted) / len(length_accepted),
        "writer_train_vs_validation_overlap": 0,
        "writer_examples_vs_test_overlap": 0,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
