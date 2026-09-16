#!/usr/bin/env python3
"""Build audited Memory Writer warm-start SFT data from tau retention runs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from datasets import Dataset

from trajectory_memory_lab.memory_writer_harness import (
    MEMORY_WRITER_POLICY_SYSTEM,
    rank_task_documents,
    validate_writer_candidate,
)
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    memory_for_agent,
    normalize_memory_operations,
)
from trajectory_memory_lab.storage import write_json


DOMAINS = ("airline", "retail", "telecom")


def _excerpt(value: Any, limit: int = 4_000) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    head = limit // 2
    return value[:head] + "\n...[middle omitted]...\n" + value[-head:]


def _compact_json_value(value: Any, string_limit: int = 400) -> Any:
    if isinstance(value, str):
        return _excerpt(value, string_limit)
    if isinstance(value, list):
        return [_compact_json_value(item, string_limit) for item in value]
    if isinstance(value, dict):
        return {
            key: _compact_json_value(item, string_limit)
            for key, item in value.items()
        }
    return value


def _trajectory(domain: str, task: dict, simulation: dict, policy: str) -> dict:
    reward_info = simulation.get("reward_info")
    return {
        "source_task_id": f"tau2.{domain}.{simulation['task_id']}",
        "domain": domain,
        "task": task,
        "policy": policy,
        "success": bool(
            isinstance(reward_info, dict) and reward_info.get("reward") == 1.0
        ),
        "reward": None
        if not isinstance(reward_info, dict)
        else reward_info.get("reward"),
        "termination_reason": simulation.get("termination_reason"),
        "evaluation": reward_info,
        "steps": [
            {
                "index": index,
                "role": message.get("role"),
                "content": _excerpt(message.get("content")),
                "tool_calls": message.get("tool_calls"),
                "tool_error": message.get("error"),
            }
            for index, message in enumerate(simulation.get("messages") or [])
        ],
    }


def _training_trajectory(
    trajectory: dict,
    evidence_steps: set[int],
    *,
    message_char_budget: int,
) -> dict:
    """Compact non-evidence messages while retaining cited source evidence."""
    result = deepcopy(trajectory)
    task = result.get("task") or {}
    result["task"] = {
        "id": task.get("id"),
        "description": _compact_json_value(task.get("description"), 600),
        "user_scenario": _compact_json_value(task.get("user_scenario"), 800),
    }
    evaluation = result.get("evaluation") or {}
    result["evaluation"] = {
        "reward": evaluation.get("reward"),
        "db_check": _compact_json_value(evaluation.get("db_check"), 300),
        "reward_breakdown": evaluation.get("reward_breakdown"),
        "action_checks": [
            {
                "action": {
                    "name": (item.get("action") or {}).get("name"),
                    "arguments": _compact_json_value(
                        (item.get("action") or {}).get("arguments"), 200
                    ),
                },
                "action_match": item.get("action_match"),
                "action_reward": item.get("action_reward"),
                "tool_type": item.get("tool_type"),
            }
            for item in (evaluation.get("action_checks") or [])[:24]
            if isinstance(item, dict)
        ],
        "nl_assertions": _compact_json_value(
            evaluation.get("nl_assertions"), 500
        ),
        "communicate_checks": _compact_json_value(
            evaluation.get("communicate_checks"), 500
        ),
    }
    steps = result["steps"]
    evidence = [step for step in steps if step["index"] in evidence_steps]
    other = [step for step in steps if step["index"] not in evidence_steps]
    evidence_budget = int(message_char_budget * 0.7) if evidence else 0
    other_budget = message_char_budget - evidence_budget
    evidence_cap = max(120, evidence_budget // max(1, len(evidence)))
    # Long telecom trajectories can contain nearly 200 messages. Keep the
    # complete indexed role skeleton, but reserve text budget for the cited
    # authoritative evidence instead of spending >=40 chars on every turn.
    other_cap = max(8, other_budget // max(1, len(other)))
    for step in steps:
        is_evidence = step["index"] in evidence_steps
        cap = evidence_cap if is_evidence else other_cap
        step["content"] = _excerpt(step.get("content"), cap)
        if is_evidence:
            step["tool_calls"] = _compact_json_value(
                step.get("tool_calls"), 200
            )
        else:
            step["tool_calls"] = [
                {"name": call.get("name")}
                for call in (step.get("tool_calls") or [])
                if isinstance(call, dict) and call.get("name")
            ] or None
    return result


def _relevant_policy(policy: str, query: str, max_chars: int) -> str:
    if len(policy) <= max_chars:
        return policy
    paragraphs = [item.strip() for item in policy.split("\n\n") if item.strip()]
    if len(paragraphs) < 2:
        paragraphs = [policy[index : index + 2_000] for index in range(0, len(policy), 2_000)]
    selected = []
    used = 0
    for index, _ in rank_task_documents(query, paragraphs):
        paragraph = paragraphs[index]
        if selected and used + len(paragraph) > max_chars:
            continue
        selected.append((index, paragraph))
        used += len(paragraph)
        if used >= max_chars:
            break
    selected.sort()
    return "[Selected relevant policy excerpts]\n\n" + "\n\n".join(
        paragraph for _, paragraph in selected
    )


def _source_data(results_root: Path, retention_root: Path) -> list[dict]:
    metadata = json.loads((retention_root / "manifest.json").read_text())["domains"]
    records = []
    ordinal = 0
    for domain in DOMAINS:
        result_path = results_root / f"qwen35_base_{domain}_full_v1/results.json"
        result = json.loads(result_path.read_text())
        tasks = {str(item["id"]): item for item in result["tasks"]}
        simulations = {
            str(item["task_id"]): item for item in result["simulations"]
        }
        policy = result["info"]["environment_info"]["policy"]
        for raw_task_id in metadata[domain]["train_task_ids"]:
            task_id = str(raw_task_id)
            decision_path = (
                retention_root
                / "tasks"
                / f"{ordinal:03d}_{domain}"
                / "retention_decision.json"
            )
            if decision_path.exists():
                records.append(
                    {
                        "ordinal": ordinal,
                        "domain": domain,
                        "task_id": task_id,
                        "task": tasks[task_id],
                        "simulation": simulations[task_id],
                        "policy": policy,
                        "decision": json.loads(decision_path.read_text()),
                    }
                )
            ordinal += 1
    return records


def _strictly_supported(operation: dict, trajectory: dict) -> tuple[bool, list[str]]:
    candidate = {
        "operation": "add",
        "memory": {
            **operation["memory"],
            "conditions": [],
            "exceptions": [],
        },
    }
    validation = validate_writer_candidate(candidate, trajectory)
    return validation["accepted"], validation["reasons"]


def _accepted_operations(
    bank: list[dict], operations: list[dict], trajectory: dict
) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    for operation in operations:
        supported, reasons = _strictly_supported(operation, trajectory)
        if not supported:
            rejected.append({"operation": operation, "reasons": reasons})
            continue
        result = apply_memory_operations(bank, [operation], trajectory=trajectory)
        if result["applied"]:
            accepted.append(operation)
        else:
            rejected.append(
                {
                    "operation": operation,
                    "reasons": [item["reason"] for item in result["rejected"]],
                }
            )
    return accepted, rejected


def _split(records: list[dict], seed: int, val_fraction: float) -> tuple[list, list]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["domain"], record["label_class"])].append(record)
    rng = random.Random(seed)
    train, validation = [], []
    for items in groups.values():
        rng.shuffle(items)
        validation_count = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 0
        validation.extend(items[:validation_count])
        train.extend(items[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retention-root",
        type=Path,
        default=Path("tau_experiment/v1/retention"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("training/tau_memory_writer_sft_v2")
    )
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--memory-max-chars", type=int, default=4_000)
    parser.add_argument("--trajectory-message-char-budget", type=int, default=8_000)
    parser.add_argument("--policy-max-chars", type=int, default=5_000)
    parser.add_argument(
        "--max-train-refines-per-target",
        type=int,
        default=0,
        help="Cap repeated refine labels for one (domain, target_memory_id); 0 disables.",
    )
    parser.add_argument("--train-add-repeat", type=int, default=1)
    parser.add_argument("--train-replace-repeat", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    banks = {domain: [] for domain in DOMAINS}
    examples = []
    routing_outcomes = []
    rejected_operations = []
    writer_calls = 0
    for source in _source_data(args.results_root, args.retention_root):
        decision = source["decision"]
        tool = (decision.get("tools") or {}).get("edit_memory")
        if not isinstance(tool, dict):
            continue
        writer_calls += 1
        trajectory = _trajectory(
            source["domain"],
            source["task"],
            source["simulation"],
            source["policy"],
        )
        if decision.get("source_task_id") != trajectory["source_task_id"]:
            raise ValueError(
                f"ordinal {source['ordinal']}: source task mismatch "
                f"{decision.get('source_task_id')} != {trajectory['source_task_id']}"
            )
        bank_before = deepcopy(banks[source["domain"]])
        proposed = normalize_memory_operations(
            {"operations": tool.get("final_operations") or []}
        )
        accepted, rejected = _accepted_operations(
            banks[source["domain"]], proposed, trajectory
        )
        current_memory = memory_for_agent(
            bank_before, max_chars=args.memory_max_chars
        )
        required_targets = {
            operation.get("target_memory_id")
            for operation in accepted
            if operation["op"] in {"refine", "replace"}
        }
        visible_ids = {entry["id"] for entry in current_memory}
        if required_targets - visible_ids:
            full_memory = {
                entry["id"]: entry
                for entry in memory_for_agent(bank_before, max_chars=1_000_000)
            }
            current_memory.extend(
                full_memory[target]
                for target in sorted(required_targets - visible_ids)
                if target in full_memory
            )
        rejected_operations.extend(
            {
                "source_task_id": trajectory["source_task_id"],
                **item,
            }
            for item in rejected
        )
        routing_outcomes.append(
            {
                "source_task_id": trajectory["source_task_id"],
                "domain": source["domain"],
                "controller_called_edit_memory": True,
                "writer_call_yielded_accepted_operation": bool(accepted),
                "accepted_operation_count": len(accepted),
                "rejected_operation_count": len(rejected),
                "label_status": "controller_candidate_requires_separate_evaluation",
            }
        )
        # The routing controller owns the decision to invoke the writer. Calls
        # that yield no audited operation are controller negatives, not writer
        # noop targets. The writer dataset is conditioned on a correct call.
        if not accepted:
            continue
        label_class = accepted[0]["op"]
        if len({operation["op"] for operation in accepted}) > 1:
            label_class = "mixed"
        target = {"operations": accepted}
        evidence_steps = {
            step
            for operation in accepted
            for step in operation["memory"]["evidence_steps"]
        }
        training_trajectory = _training_trajectory(
            trajectory,
            evidence_steps,
            message_char_budget=args.trajectory_message_char_budget,
        )
        policy_query = json.dumps(trajectory["task"], ensure_ascii=False) + " " + " ".join(
            operation["memory"]["content"] + " " + operation["memory"]["scope"]
            for operation in accepted
        )
        training_trajectory["policy"] = _relevant_policy(
            trajectory["policy"], policy_query, args.policy_max_chars
        )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": MEMORY_WRITER_POLICY_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "controller_suggested_evidence_steps": next(
                                    (
                                        call.get("arguments", {}).get(
                                            "evidence_steps", []
                                        )
                                        for call in (
                                            decision.get("controller", {})
                                            .get("decision", {})
                                            .get("tool_calls", [])
                                        )
                                        if call.get("name") == "edit_memory"
                                    ),
                                    [],
                                ),
                                "controller_suggested_steps_are_not_evidence": True,
                                "current_memory": current_memory,
                                "trajectory": training_trajectory,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(target, ensure_ascii=False),
                    },
                ],
                "enable_thinking": False,
                "domain": source["domain"],
                "source_task_id": trajectory["source_task_id"],
                "source_success": trajectory["success"],
                "label_class": label_class,
                "operation_count": len(accepted),
                "validation_status": "accepted",
                "validation_basis": "audited_then_deterministically_applied",
            }
        )

    train, validation = _split(examples, args.seed, args.val_fraction)
    original_train = list(train)
    if args.max_train_refines_per_target > 0:
        refine_counts = Counter()
        capped_train = []
        for item in train:
            if item["label_class"] != "refine":
                capped_train.append(item)
                continue
            target = json.loads(item["messages"][2]["content"])["operations"][0].get(
                "target_memory_id"
            )
            key = (item["domain"], target)
            if refine_counts[key] >= args.max_train_refines_per_target:
                continue
            refine_counts[key] += 1
            capped_train.append(item)
        train = capped_train
    repeats = {
        "add": max(1, args.train_add_repeat),
        "replace": max(1, args.train_replace_repeat),
    }
    expanded_train = []
    for item in train:
        expanded_train.extend(deepcopy(item) for _ in range(repeats.get(item["label_class"], 1)))
    random.Random(args.seed + 1).shuffle(expanded_train)
    train = expanded_train
    _write_jsonl(args.output / "all.jsonl", examples)
    _write_jsonl(args.output / "train.jsonl", train)
    _write_jsonl(args.output / "validation.jsonl", validation)
    _write_jsonl(args.output / "controller_routing_outcomes.jsonl", routing_outcomes)
    Dataset.from_list(train).to_parquet(str(args.output / "train.parquet"))
    Dataset.from_list(validation).to_parquet(
        str(args.output / "validation.parquet")
    )
    write_json(args.output / "rejected_operations.json", rejected_operations)
    summary = {
        "protocol": "tau_memory_writer_sft_v2_router_conditioned",
        "writer_calls_observed": writer_calls,
        "examples": len(examples),
        "train_examples": len(train),
        "original_train_examples": len(original_train),
        "validation_examples": len(validation),
        "label_classes": dict(Counter(item["label_class"] for item in examples)),
        "train_label_classes": dict(Counter(item["label_class"] for item in train)),
        "validation_label_classes": dict(
            Counter(item["label_class"] for item in validation)
        ),
        "domains": dict(Counter(item["domain"] for item in examples)),
        "source_success": dict(
            Counter(str(item["source_success"]).lower() for item in examples)
        ),
        "rejected_operations": len(rejected_operations),
        "replace_positive_examples": sum(
            item["label_class"] == "replace" for item in examples
        ),
        "writer_calls_with_accepted_operation": sum(
            item["writer_call_yielded_accepted_operation"]
            for item in routing_outcomes
        ),
        "writer_calls_without_accepted_operation": sum(
            not item["writer_call_yielded_accepted_operation"]
            for item in routing_outcomes
        ),
        "memory_max_chars": args.memory_max_chars,
        "trajectory_message_char_budget": args.trajectory_message_char_budget,
        "policy_max_chars": args.policy_max_chars,
        "max_train_refines_per_target": args.max_train_refines_per_target,
        "train_add_repeat": args.train_add_repeat,
        "train_replace_repeat": args.train_replace_repeat,
        "note": "Writer SFT is conditioned on a correct controller call and contains only audited, deterministically applicable add/refine/replace targets. Empty outcomes are saved separately as controller-training candidates, not trusted negative labels; they require independent controller evaluation because writer or auditor failure may also explain the empty outcome.",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
