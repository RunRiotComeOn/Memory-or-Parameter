"""Benchmark-agnostic engine for building memory banks under allocation rubrics.

The v2 experiment's only benchmark-specific parts were assembling trajectories
and naming the protocol; the decision loop itself never depended on tau-bench.
This module holds that loop so any benchmark can drive it by supplying canonical
trajectories:

    {"source_task_id", "domain", "task", "success", "reward",
     "termination_reason", "evaluation", "steps": [{"index", "role", "content"}]}

Chains are sequential per (rubric, group) because a routing decision that cannot
see the current bank is not a routing decision.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .alloc_writer_harness import (
    REFINE_TOPIC_OVERLAP_MIN,
    active_entries,
    apply_memory_operation,
    normalize_alloc_decision,
    render_bank,
    topic_overlap,
    validate_alloc_decision,
    writes_memory,
)
from .model_client import ModelClient
from .writer_rubrics import ALLOC_RUBRIC_IDS, ALLOC_WRITER_SYSTEM, alloc_rubric_block


RESUMABLE_STATUSES = {"committed", "memory_rejected", "no_write", "error"}


@dataclass
class BuilderConfig:
    """Everything the chain loop needs that is not a trajectory."""

    output: Path
    record_protocol: str
    summary_protocol: str
    model: str = "qwen35-tau"
    base_url: str = "http://127.0.0.1:8000/v1"
    max_tokens: int = 4096
    timeout: float = 1200
    seed: int = 20260822
    budget_fraction: float = 0.4
    max_parallel_chains: int = 6
    rubric_ids: tuple[str, ...] = ALLOC_RUBRIC_IDS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def budget_capacity(rubric_id: str, task_count: int, fraction: float) -> int | None:
    if rubric_id != "a3_budgeted":
        return None
    return max(1, math.ceil(task_count * fraction))


def run_chain(
    rubric_id: str,
    group: str,
    task_ids: list[str],
    trajectories: dict[str, dict[str, Any]],
    config: BuilderConfig,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Process one group's trajectories in order against a single running bank."""
    records_dir = config.output / "records" / rubric_id / group
    capacity = budget_capacity(rubric_id, len(task_ids), config.budget_fraction)
    bank: list[dict[str, Any]] = []
    slots_used = 0
    records: list[dict[str, Any]] = []

    for position, task_id in enumerate(task_ids):
        record_path = records_dir / f"{position:03d}_{task_id}.json"
        trajectory = trajectories[task_id]
        record: dict[str, Any] | None = None
        if record_path.exists():
            existing = read_json(record_path)
            if existing.get("status") in RESUMABLE_STATUSES:
                record = existing
        if record is None:
            budget_state = None
            if capacity is not None:
                budget_state = {
                    "domain_capacity": capacity,
                    "slots_used": slots_used,
                    "slots_remaining": capacity - slots_used,
                    "trajectories_remaining": len(task_ids) - position,
                }
            payload = {
                "allocation_state": {
                    "active_memory_count": len(active_entries(bank)),
                    "trajectories_seen": position,
                    "memory_budget": budget_state,
                },
                "active_memory": render_bank(bank),
                "base_agent_outcome": {
                    "success": trajectory["success"],
                    "reward": trajectory["reward"],
                    "termination_reason": trajectory["termination_reason"],
                },
                "trajectory": trajectory,
            }
            client = ModelClient(
                base_url=config.base_url,
                api_key="EMPTY",
                model=config.model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=config.max_tokens,
                seed=config.seed + position,
                enable_thinking=False,
                timeout=config.timeout,
            )
            reply = None
            try:
                reply = client.json_chat(
                    system=ALLOC_WRITER_SYSTEM + alloc_rubric_block(rubric_id),
                    user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                )
                decision = normalize_alloc_decision(reply.parsed)
                record = {
                    "protocol": config.record_protocol,
                    "rubric_id": rubric_id,
                    "domain": group,
                    "position": position,
                    "source_task_id": task_id,
                    "base_agent_success": trajectory["success"],
                    "decision": decision,
                    "enforcement": [],
                    "usage": reply.usage,
                }
            except Exception as exc:
                record = {
                    "protocol": config.record_protocol,
                    "rubric_id": rubric_id,
                    "domain": group,
                    "position": position,
                    "source_task_id": task_id,
                    "base_agent_success": trajectory["success"],
                    "status": "error",
                    "error": repr(exc),
                    "raw_prediction": reply.parsed if reply is not None else None,
                }
                write_json(record_path, record)
                records.append(record)
                continue

            decision = record["decision"]
            enforcement = record["enforcement"]
            # a0 is the control arm: its rubric prescribes `both` for every
            # trajectory, so a different route is a rubric violation, not a
            # measurement.  Coerce when the artifact is present, count always.
            if rubric_id == "a0_always_both" and decision["route"] != "both":
                enforcement.append(f"route_violation:{decision['route']}")
                if decision.get("memory"):
                    decision["route"] = "both"
                    enforcement.append("route_coerced_to_both")
            # a3's budget is a hard constraint, not a suggestion.
            if (
                capacity is not None
                and writes_memory(decision["route"])
                and decision.get("memory_operation") == "add"
                and slots_used >= capacity
            ):
                enforcement.append("budget_exhausted")
                decision["route"] = "sft" if decision.get("sft_plan") else "neither"
                decision["memory"] = None
                decision["memory_operation"] = None
                decision["target_memory_id"] = None

            # An edit aimed at an unrelated entry is a mis-chosen operation, not a
            # bad memory: keep the content, make it an add, and record the coercion.
            if decision.get("memory") and decision.get("memory_operation") in {
                "refine",
                "replace",
            }:
                target = next(
                    (
                        entry
                        for entry in active_entries(bank)
                        if entry["id"] == decision.get("target_memory_id")
                    ),
                    None,
                )
                if target is not None:
                    overlap = topic_overlap(decision["memory"], target)
                    record["refine_topic_overlap"] = round(overlap, 4)
                    if overlap < REFINE_TOPIC_OVERLAP_MIN:
                        enforcement.append(
                            f"refine_topic_mismatch_coerced_to_add:{overlap:.3f}"
                        )
                        decision["memory_operation"] = "add"
                        decision["target_memory_id"] = None

            validation = validate_alloc_decision(decision, trajectory, active_entries(bank))
            record["hard_validation"] = validation
            if not validation["memory"]["required"]:
                record["status"] = "no_write"
            elif validation["memory"]["accepted"]:
                record["status"] = "committed"
            else:
                record["status"] = "memory_rejected"
            if not validation["sft"]["required"]:
                record["sft_status"] = "not_selected"
            elif validation["sft"]["accepted"]:
                record["sft_status"] = "selected"
            else:
                record["sft_status"] = "sft_rejected"

        # Replay the committed effect on the bank (fresh or resumed alike).
        decision = record.get("decision") or {}
        if record.get("status") == "committed" and writes_memory(decision.get("route", "")):
            entry = apply_memory_operation(
                bank,
                decision,
                entry_id=f"{rubric_id}_{group}_{len(bank):03d}",
                source_task_id=task_id,
                rubric_id=rubric_id,
            )
            record["committed_entry_id"] = entry["id"]
            if decision.get("memory_operation") == "add":
                slots_used += 1
        write_json(record_path, record)
        records.append(record)
        log(
            f"  {rubric_id}/{group} [{position + 1}/{len(task_ids)}] task={task_id} "
            f"route={decision.get('route')} status={record.get('status')} "
            f"active={len(active_entries(bank))}"
        )

    write_json(config.output / "banks" / rubric_id / f"memory_{group}.json", active_entries(bank))
    write_json(config.output / "banks" / rubric_id / f"full_{group}.json", bank)
    return {
        "rubric_id": rubric_id,
        "domain": group,
        "capacity": capacity,
        "records": records,
        "active_entries": len(active_entries(bank)),
        "total_entries": len(bank),
    }


def summarize(
    results: list[dict[str, Any]], config: BuilderConfig, extra: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate per-arm allocation behaviour across all chains."""
    summary: dict[str, Any] = {
        "protocol": config.summary_protocol,
        "writer_rubric_version": 2,
        "variable_under_test": "allocation (memory/sft/both/neither), writing style frozen",
        **extra,
        "budget_fraction": config.budget_fraction,
        "arms": {},
    }
    for rubric_id in config.rubric_ids:
        arm_records = [
            record
            for result in results
            if result["rubric_id"] == rubric_id
            for record in result["records"]
        ]
        routes = Counter(
            (record.get("decision") or {}).get("route", record.get("status"))
            for record in arm_records
        )
        gaps = Counter(
            (record.get("decision") or {}).get("gap_type") for record in arm_records
        )
        operations = Counter(
            (record.get("decision") or {}).get("memory_operation")
            for record in arm_records
            if record.get("status") == "committed"
        )
        enforcement = Counter(
            item for record in arm_records for item in (record.get("enforcement") or [])
        )
        rejects = Counter(
            f"{branch}:{reason}"
            for record in arm_records
            for branch in ("memory", "sft")
            for reason in ((record.get("hard_validation") or {}).get(branch) or {}).get(
                "reasons", []
            )
        )
        by_base_outcome = Counter(
            (
                "base_success" if record.get("base_agent_success") else "base_failure",
                (record.get("decision") or {}).get("route"),
            )
            for record in arm_records
        )
        summary["arms"][rubric_id] = {
            "decisions": len(arm_records),
            "routes": dict(sorted(routes.items())),
            "gap_types": {
                str(key): value for key, value in sorted(gaps.items(), key=lambda kv: str(kv[0]))
            },
            "committed_operations": {
                str(key): value for key, value in sorted(operations.items(), key=lambda kv: str(kv[0]))
            },
            "enforcement": dict(sorted(enforcement.items())),
            "reject_reasons": dict(sorted(rejects.items())),
            "route_by_base_outcome": {
                f"{outcome}__{route}": count
                for (outcome, route), count in sorted(
                    by_base_outcome.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
                )
            },
            "statuses": dict(
                sorted(Counter(record.get("status") for record in arm_records).items())
            ),
            "sft_statuses": dict(
                sorted(Counter(record.get("sft_status") for record in arm_records).items())
            ),
            "sft_selected": sum(
                1 for record in arm_records if record.get("sft_status") == "selected"
            ),
            "memory_rejected": sum(
                1 for record in arm_records if record.get("status") == "memory_rejected"
            ),
            "bank_entries": {
                result["domain"]: result["active_entries"]
                for result in sorted(results, key=lambda item: item["domain"])
                if result["rubric_id"] == rubric_id
            },
            "bank_total": sum(
                result["active_entries"]
                for result in results
                if result["rubric_id"] == rubric_id
            ),
        }
    return summary


def build_banks(
    groups: dict[str, list[str]],
    trajectories: dict[str, dict[str, dict[str, Any]]],
    config: BuilderConfig,
    extra_summary: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every (rubric, group) chain and write banks plus a summary."""
    config.output.mkdir(parents=True, exist_ok=True)
    chains: Iterable[tuple[str, str]] = [
        (rubric_id, group) for rubric_id in config.rubric_ids for group in groups
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.max_parallel_chains
    ) as pool:
        futures = {
            pool.submit(
                run_chain, rubric_id, group, groups[group], trajectories[group], config, log
            ): (rubric_id, group)
            for rubric_id, group in chains
        }
        for future in concurrent.futures.as_completed(futures):
            rubric_id, group = futures[future]
            result = future.result()
            results.append(result)
            log(f"chain complete {rubric_id}/{group} active={result['active_entries']}")

    # `as_completed` yields in finish order; sort so the summary serializes the
    # same way on every run.
    results.sort(key=lambda result: (result["rubric_id"], result["domain"]))
    summary = summarize(results, config, extra_summary or {})
    write_json(config.output / "summary.json", summary)
    return summary
