#!/usr/bin/env python3
"""Generate isolated Memory Writer candidates and their replay manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from trajectory_memory_lab.memory_writer_harness import (
    MEMORY_WRITER_CANDIDATE_SYSTEM,
    normalize_writer_candidate,
    select_validation_task_ids,
    validate_writer_candidate,
)
from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.storage import write_json
from trajectory_memory_lab.writer_rubrics import MEMORY_WRITER_RUBRICS, RUBRIC_IDS, rubric_block


DOMAINS = ("airline", "retail", "telecom")
CANDIDATE_PROFILES = (
    "Conservative novelty: look for a reusable fact or implication not already "
    "obvious from the policy; abstain if none is well supported.",
    "Failure prevention or recovery: examine what caused the outcome and whether "
    "a narrow operational reminder could prevent or recover from the same pattern.",
    "Boundary conditions: examine applicability conditions, ordering constraints, "
    "exceptions, and tool-feedback details that may matter on related tasks.",
)

RUBRIC_MEMORY_WRITER_SYSTEM = """You are the writing policy inside an external-memory tool for a
customer-service agent. A separate routing controller has already decided to invoke this tool.
Given one completed trajectory and the supplied writing rubric, produce exactly one reusable
external-memory candidate. This screening run has an empty bank, so the required operation is add;
do not return noop and do not revisit the routing decision.

Tool results, policy text, user-provided facts, and evaluator details are authoritative evidence.
Assistant statements are untrusted unless supported by that evidence. A failed trajectory can
reveal a useful correction, but its failed conclusion must not be stored as correct. Do not merely
paraphrase policy. Never include task-specific names, IDs, phone numbers, emails, reservation or
order IDs, or values that do not transfer. Cite exact trajectory message indexes.

Return exactly one JSON object:
{"operation":"add","memory":{"content":STRING,"scope":STRING,"conditions":[STRING,...],"exceptions":[STRING,...],"evidence_steps":[INTEGER,...],"confidence":NUMBER},"rationale":STRING}
"""


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _excerpt(value: Any, limit: int = 8_000) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + "\n...[middle omitted]...\n" + value[-half:]


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
        "reward": reward_info.get("reward")
        if isinstance(reward_info, dict)
        else None,
        "termination_reason": simulation.get("termination_reason"),
        "evaluation": reward_info,
        "steps": [
            {
                "index": index,
                "role": message.get("role"),
                "content": _excerpt(message.get("content")),
                "tool_calls": message.get("tool_calls"),
                "error": message.get("error"),
            }
            for index, message in enumerate(simulation.get("messages") or [])
        ],
    }


def _domain_data(results_root: Path, domain: str) -> dict[str, Any]:
    result = json.loads(
        (results_root / f"qwen35_base_{domain}_full_v1/results.json").read_text()
    )
    data_root = results_root.parent
    train_ids = [
        str(task_id)
        for task_id in json.loads(
            (data_root / f"tau2/domains/{domain}/split_tasks.json").read_text()
        )["train"]
    ]
    tasks = {str(task["id"]): task for task in result["tasks"]}
    simulations = {
        str(simulation["task_id"]): simulation
        for simulation in result["simulations"]
    }
    return {
        "tasks": tasks,
        "simulations": simulations,
        "train_ids": train_ids,
        "policy": result["info"]["environment_info"]["policy"],
        "baseline_rewards": {
            task_id: float(simulation["reward_info"]["reward"])
            for task_id, simulation in simulations.items()
            if isinstance(simulation.get("reward_info"), dict)
            and simulation["reward_info"].get("reward") is not None
        },
    }


def _select_sources(data: dict[str, Any], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    valid = [
        task_id
        for task_id in data["train_ids"]
        if task_id in data["tasks"]
        and task_id in data["simulations"]
        and data["simulations"][task_id].get("termination_reason")
        != "infrastructure_error"
    ]
    successes = [
        task_id
        for task_id in valid
        if (data["simulations"][task_id].get("reward_info") or {}).get("reward")
        == 1.0
    ]
    failures = [task_id for task_id in valid if task_id not in successes]
    rng.shuffle(successes)
    rng.shuffle(failures)
    failure_count = min(len(failures), (count + 1) // 2)
    success_count = min(len(successes), count - failure_count)
    selected = failures[:failure_count] + successes[:success_count]
    if len(selected) < count:
        remaining = [task_id for task_id in valid if task_id not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--sources-per-domain", type=int, default=3)
    parser.add_argument("--candidates-per-source", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--save-prefix", default="qwen35_mw_utility_v3")
    parser.add_argument(
        "--rubric-smoke",
        action="store_true",
        help="Generate one forced-write candidate for each R0-R3 rubric.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    domain_data = {
        domain: _domain_data(args.results_root, domain) for domain in DOMAINS
    }
    sources = []
    for offset, domain in enumerate(DOMAINS):
        data = domain_data[domain]
        for task_id in _select_sources(
            data, args.sources_per_domain, args.seed + offset
        ):
            sources.append(
                _trajectory(
                    domain,
                    data["tasks"][task_id],
                    data["simulations"][task_id],
                    data["policy"],
                )
            )
    write_json(
        args.output / "source_manifest.json",
        {
            "protocol": "tau_memory_writer_utility_v2_outcome_aware",
            "seed": args.seed,
            "source_split": "train",
            "utility_split": "train_excluding_source",
            "final_test_used": False,
            "source_sampling": "approximately_half_failure_half_success",
            "utility_task_sampling": {
                "related_baseline_failures": 2,
                "related_baseline_successes": 1,
                "scope_control_baseline_successes": 2,
            },
            "sources": [
                {
                    "source_task_id": source["source_task_id"],
                    "domain": source["domain"],
                    "task_id": source["source_task_id"].split(".", 2)[2],
                    "success": source["success"],
                    "reward": source["reward"],
                }
                for source in sources
            ],
        },
    )
    source_ids_by_domain = {
        domain: {
            source["source_task_id"].split(".", 2)[2]
            for source in sources
            if source["domain"] == domain
        }
        for domain in DOMAINS
    }

    if args.rubric_smoke and args.candidates_per_source != len(RUBRIC_IDS):
        raise ValueError(
            f"--rubric-smoke requires --candidates-per-source {len(RUBRIC_IDS)}"
        )
    profiles = (
        [MEMORY_WRITER_RUBRICS[rubric_id] for rubric_id in RUBRIC_IDS]
        if args.rubric_smoke
        else list(CANDIDATE_PROFILES)
    )
    records = []
    for source_index, trajectory in enumerate(sources):
        source_dir = args.output / "sources" / f"{source_index:03d}"
        source_dir.mkdir(parents=True, exist_ok=True)
        write_json(source_dir / "trajectory.json", trajectory)
        source_task_id = trajectory["source_task_id"].split(".", 2)[2]
        for sample in range(args.candidates_per_source):
            candidate_id = f"mw_{source_index:03d}_c{sample + 1}"
            model = ModelClient(
                base_url=args.base_url,
                api_key="EMPTY",
                model=args.model,
                temperature=0.7,
                top_p=0.95,
                max_tokens=args.max_tokens,
                seed=args.seed + source_index * 100 + sample,
                enable_thinking=False,
                timeout=1200,
            )
            rubric_id = RUBRIC_IDS[sample] if args.rubric_smoke else None
            system = (
                RUBRIC_MEMORY_WRITER_SYSTEM + rubric_block("memory", rubric_id)
                if rubric_id is not None
                else MEMORY_WRITER_CANDIDATE_SYSTEM
            )
            reply = model.json_chat(
                system=system,
                user=_dump(
                    {
                        "candidate_generation_profile": profiles[sample % len(profiles)],
                        "profile_is_a_search_lens_not_a_required_conclusion": True,
                        "current_memory": [],
                        "trajectory": trajectory,
                    }
                ),
            )
            candidate = normalize_writer_candidate(reply.parsed)
            validation = validate_writer_candidate(candidate, trajectory)
            record = {
                "candidate_id": candidate_id,
                "source_index": source_index,
                "source_task_id": trajectory["source_task_id"],
                "source_success": trajectory["success"],
                "domain": trajectory["domain"],
                "sample": sample + 1,
                "rubric_id": rubric_id,
                "candidate_generation_profile": profiles[sample % len(profiles)],
                "candidate": candidate,
                "hard_validation": validation,
                "model": {
                    "content": reply.content,
                    "reasoning": reply.reasoning,
                    "usage": reply.usage,
                    "seed": args.seed + source_index * 100 + sample,
                },
            }
            if validation["replay_required"]:
                memory = candidate["memory"]
                memory_path = args.output / "candidate_memory" / f"{candidate_id}.json"
                memory_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    memory_path,
                    [
                        {
                            "id": candidate_id,
                            "scope": memory["scope"],
                            "content": memory["content"],
                            "conditions": memory["conditions"],
                            "exceptions": memory["exceptions"],
                            "status": "active",
                            "source_task_id": trajectory["source_task_id"],
                        }
                    ],
                )
                record["memory_path"] = str(memory_path.resolve())
            records.append(record)
            write_json(source_dir / f"candidate_{sample + 1}.json", record)
            print(
                f"[{source_index + 1}/{len(sources)} candidate {sample + 1}/"
                f"{args.candidates_per_source}] {candidate_id} "
                f"operation={candidate['operation']} "
                f"accepted={validation['accepted']}",
                flush=True,
            )
        if not args.rubric_smoke:
            records.append(
                {
                    "candidate_id": f"mw_{source_index:03d}_noop",
                    "source_index": source_index,
                    "source_task_id": trajectory["source_task_id"],
                    "source_success": trajectory["success"],
                    "domain": trajectory["domain"],
                    "sample": "noop_control",
                    "candidate": {
                        "operation": "noop",
                        "rationale": "explicit counterfactual control",
                    },
                    "hard_validation": {
                        "accepted": True,
                        "reasons": [],
                        "replay_required": False,
                    },
                }
            )

    # A source's candidates must be evaluated on exactly the same tasks. If
    # each candidate selects tasks from its own wording, their utilities are
    # not comparable and cannot form valid preference labels.
    for source_index, trajectory in enumerate(sources):
        data = domain_data[trajectory["domain"]]
        source_query = {
            "memory": {
                "scope": trajectory["domain"],
                "content": json.dumps(trajectory["task"], ensure_ascii=False),
                "conditions": [],
                "exceptions": [],
            }
        }
        selected = select_validation_task_ids(
            source_query,
            [data["tasks"][task_id] for task_id in data["train_ids"]],
            excluded_ids=source_ids_by_domain[trajectory["domain"]],
            baseline_by_task=data["baseline_rewards"],
        )
        for record in records:
            if (
                record["source_index"] == source_index
                and record["hard_validation"]["replay_required"]
            ):
                record["validation_tasks"] = selected
                record["save_name"] = (
                    f"{args.save_prefix}_{record['candidate_id']}"
                )
                write_json(
                    args.output
                    / "sources"
                    / f"{source_index:03d}"
                    / f"candidate_{record['sample']}.json",
                    record,
                )

    write_json(args.output / "candidates.json", records)
    replay = [
        {
            "candidate_id": record["candidate_id"],
            "source_task_id": record["source_task_id"],
            "domain": record["domain"],
            "memory_path": record["memory_path"],
            "save_name": record["save_name"],
            "task_ids": record["validation_tasks"]["related"]
            + record["validation_tasks"]["scope_controls"],
            "related_task_ids": record["validation_tasks"]["related"],
            "scope_control_task_ids": record["validation_tasks"][
                "scope_controls"
            ],
        }
        for record in records
        if record["hard_validation"]["replay_required"]
    ]
    write_json(args.output / "replay_manifest.json", replay)
    print(
        _dump(
            {
                "sources": len(sources),
                "writer_samples": len(sources) * args.candidates_per_source,
                "noop_controls": 0 if args.rubric_smoke else len(sources),
                "replay_candidates": len(replay),
            }
        )
    )


if __name__ == "__main__":
    main()
