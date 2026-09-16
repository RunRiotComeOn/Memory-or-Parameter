#!/usr/bin/env python3
"""Build a source-paired utility replay for base and SFT memory writers."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from trajectory_memory_lab.memory_writer_harness import select_validation_task_ids
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    memory_for_agent,
    normalize_memory_bank,
    normalize_memory_operations,
)
from trajectory_memory_lab.storage import write_json


ROOT = Path("/nas04/yixuh/memory")
DOMAINS = ("airline", "retail", "telecom")
ARMS = {
    "qwen35-tau": "base_writer",
    "qwen35-tau-writer-sft-v2": "sft_writer",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_id(source_task_id: str, domain: str) -> str:
    prefix = f"tau2.{domain}."
    if not source_task_id.startswith(prefix):
        raise ValueError(f"unexpected source_task_id: {source_task_id}")
    return source_task_id[len(prefix) :]


def _agent_bank(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return memory_for_agent(normalize_memory_bank(entries), max_chars=1_000_000)


def _bank_hash(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        _agent_bank(entries), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _domain_data(data_root: Path, domain: str) -> dict[str, Any]:
    tasks = json.loads(
        (data_root / f"tau2/domains/{domain}/tasks.json").read_text(encoding="utf-8")
    )
    task_by_id = {str(task["id"]): task for task in tasks}
    train_ids = [
        str(task_id)
        for task_id in json.loads(
            (data_root / f"tau2/domains/{domain}/split_tasks.json").read_text(
                encoding="utf-8"
            )
        )["train"]
    ]
    baseline = json.loads(
        (
            data_root
            / "simulations"
            / f"qwen35_base_{domain}_full_v1"
            / "results.json"
        ).read_text(encoding="utf-8")
    )
    rewards = {
        str(simulation["task_id"]): float(simulation["reward_info"]["reward"])
        for simulation in baseline["simulations"]
        if simulation.get("termination_reason") != "infrastructure_error"
        and isinstance(simulation.get("reward_info"), dict)
        and simulation["reward_info"].get("reward") is not None
    }
    return {
        "tasks": task_by_id,
        "train_ids": [task_id for task_id in train_ids if task_id in rewards],
        "baseline_rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("training/tau_memory_writer_sft_v2/validation.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "tau_experiment/memory_writer_sft_eval_v2_positive_20260815/"
            "predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tau_experiment/memory_writer_paired_utility_20260816"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("third_party/tau2-bench/data")
    )
    parser.add_argument("--save-prefix", default="qwen35_writer_pair_20260816")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "memory_banks").mkdir(exist_ok=True)
    validation = _read_jsonl(args.validation)
    predictions = _read_jsonl(args.predictions)
    prediction_by_key = {
        (record["model"], record["source_task_id"]): record
        for record in predictions
    }
    domain_data = {
        domain: _domain_data(args.data_root, domain) for domain in DOMAINS
    }
    excluded_by_domain = {domain: set() for domain in DOMAINS}
    for row in validation:
        excluded_by_domain[row["domain"]].add(
            _task_id(row["source_task_id"], row["domain"])
        )

    sources: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    unique_runs: dict[tuple[int, str], dict[str, Any]] = {}
    for source_index, row in enumerate(validation):
        domain = row["domain"]
        source_task_id = row["source_task_id"]
        writer_input = json.loads(row["messages"][1]["content"])
        trajectory = writer_input["trajectory"]
        baseline_bank = normalize_memory_bank(deepcopy(writer_input["current_memory"]))
        source_query = {
            "memory": {
                "scope": domain,
                "content": json.dumps(trajectory["task"], ensure_ascii=False),
                "conditions": [],
                "exceptions": [],
            }
        }
        data = domain_data[domain]
        selected = select_validation_task_ids(
            source_query,
            [data["tasks"][task_id] for task_id in data["train_ids"]],
            excluded_ids=excluded_by_domain[domain],
            baseline_by_task=data["baseline_rewards"],
        )
        task_ids = selected["related"] + selected["scope_controls"]
        source_record = {
            "source_index": source_index,
            "source_task_id": source_task_id,
            "source_task_id_within_domain": _task_id(source_task_id, domain),
            "domain": domain,
            "source_success": bool(row.get("source_success")),
            "baseline_memory_count": len(_agent_bank(baseline_bank)),
            "validation_tasks": selected,
        }
        sources.append(source_record)

        arm_specs: list[tuple[str, dict[str, Any] | None]] = [("baseline", None)]
        arm_specs.extend(
            (
                arm,
                prediction_by_key.get((model, source_task_id)),
            )
            for model, arm in ARMS.items()
        )
        for arm, prediction in arm_specs:
            bank = normalize_memory_bank(deepcopy(baseline_bank))
            operations: list[dict[str, Any]] = []
            application = {"applied": [], "rejected": []}
            if prediction is not None:
                operations = normalize_memory_operations(prediction.get("parsed") or {})
                if operations:
                    application = apply_memory_operations(
                        bank, operations, trajectory=trajectory
                    )
                else:
                    application["rejected"].append(
                        {"operation": prediction.get("parsed"), "reason": "no_normalized_operation"}
                    )
            visible_bank = _agent_bank(bank)
            bank_hash = _bank_hash(bank)
            bank_path = output / "memory_banks" / f"source_{source_index:03d}_{arm}.json"
            write_json(bank_path, visible_bank)
            update_memory_ids = [
                item["memory_id"] for item in application["applied"]
            ]
            arm_record = {
                "source_index": source_index,
                "source_task_id": source_task_id,
                "domain": domain,
                "arm": arm,
                "writer_model": prediction.get("model") if prediction else None,
                "memory_path": str(bank_path.resolve()),
                "memory_hash": bank_hash,
                "memory_count": len(visible_bank),
                "operations": operations,
                "application": application,
                "update_memory_ids": update_memory_ids,
                "writer_schema_valid": prediction.get("schema_valid") if prediction else None,
                "writer_executable": prediction.get("executable") if prediction else None,
            }
            run_key = (source_index, bank_hash)
            if run_key not in unique_runs:
                save_name = (
                    f"{args.save_prefix}_s{source_index:03d}_{arm}_{bank_hash[:8]}"
                )
                unique_runs[run_key] = {
                    "run_id": f"s{source_index:03d}_{bank_hash[:12]}",
                    "source_index": source_index,
                    "source_task_id": source_task_id,
                    "domain": domain,
                    "memory_path": str(bank_path.resolve()),
                    "memory_hash": bank_hash,
                    "save_name": save_name,
                    "task_ids": task_ids,
                    "related_task_ids": selected["related"],
                    "scope_control_task_ids": selected["scope_controls"],
                }
            arm_record["run_id"] = unique_runs[run_key]["run_id"]
            arm_record["save_name"] = unique_runs[run_key]["save_name"]
            arms.append(arm_record)

    manifest = {
        "protocol": "writer_only_paired_utility_v1",
        "writer_decision": "controller_assumed_to_have_called_writer",
        "agent_model": "untrained Qwen/Qwen3.5-35B-A3B served as qwen35-tau",
        "user_simulator_model": "untrained Qwen/Qwen3.5-35B-A3B served as qwen35-tau",
        "grader_model": "untrained Qwen/Qwen3.5-35B-A3B served as qwen35-tau",
        "retrieval": {"method": "bm25", "top_k": 3},
        "source_split": "writer_sft_v2_validation",
        "utility_split": "tau_train_excluding_all_writer_validation_sources",
        "final_test_used": False,
        "selection": {
            "related_baseline_failures": 2,
            "related_baseline_successes": 1,
            "scope_control_baseline_successes": 2,
            "query": "source task only; writer target and writer output are not used",
        },
        "sources": sources,
        "arms": arms,
        "runs": list(unique_runs.values()),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "replay_manifest.json", list(unique_runs.values()))
    print(
        json.dumps(
            {
                "sources": len(sources),
                "logical_arms": len(arms),
                "unique_replay_runs": len(unique_runs),
                "task_simulations": sum(
                    len(run["task_ids"]) for run in unique_runs.values()
                ),
                "applied_updates": {
                    arm: sum(
                        bool(record["application"]["applied"])
                        for record in arms
                        if record["arm"] == arm
                    )
                    for arm in ("base_writer", "sft_writer")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
