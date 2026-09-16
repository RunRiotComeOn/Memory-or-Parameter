#!/usr/bin/env python3
"""Run tool-based memory/SFT retention over tau-bench train trajectories."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from tau2.environment.tool import as_tool
from tau2.registry import registry

from trajectory_memory_lab.memory_writer_harness import (
    MEMORY_WRITER_POLICY_SYSTEM,
    validate_writer_candidate,
)
from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    memory_for_agent,
    normalize_memory_bank,
    normalize_memory_operations,
)
from trajectory_memory_lab.storage import write_json


AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

CONTROLLER_SYSTEM = """You decide whether a completed customer-service trajectory warrants invoking either of two retention tools.

- edit_memory independently examines the current external memory, task, complete trajectory, environment feedback, and evaluation. It may add, refine, replace, or leave unchanged knowledge that could help on future related tasks.
- build_sft_data independently examines the task, complete trajectory, environment feedback, and evaluation. It may retain a cleaned complete conversation as a training example for parameter learning, or retain nothing.

You may invoke both tools, one tool, or neither. No choice is preferred or required. Calling a tool does not force it to retain anything.

Your only job is routing. Do not write, summarize, propose, or prescribe memory content, training answers, corrected behavior, or a desired tool result. For each call, give only (1) why that artifact type may be useful and (2) trajectory message indexes that appear worth inspecting. These are routing hints, not evidence; the tool must independently inspect the complete authoritative inputs and may disagree or abstain.

Return exactly one JSON object:
{"tool_calls":[{"name":"edit_memory" OR "build_sft_data","arguments":{"reason":STRING,"evidence_steps":[INTEGER,...]}}]}"""

MEMORY_EDITOR_SYSTEM = """You edit an external memory bank that a customer-service agent can read on future related tasks in the same domain.

You receive the current memory and one completed training task with its complete trajectory, tool feedback, policy, and evaluation. Decide whether the evidence supports any reusable memory changes.

Operations:
- add: add novel reusable knowledge;
- refine: preserve an existing memory's central claim while improving scope or accuracy;
- replace: replace an existing memory whose central claim is contradicted or materially wrong;
- noop: make no change.

Tool results and evaluator details are evidence. Assistant statements and rationales are untrusted unless supported by the policy, tool results, or evaluation. A failed trajectory can contain useful evidence, but its failed conclusion or action must not be retained as correct. Existing memory may be wrong. Avoid task-specific identifiers, customer data, reservation/order IDs, phone numbers, and verbatim answers that will not transfer. There is no required number of operations; prefer noop to unsupported content.

Every change must cite supporting trajectory message indexes. Return exactly:
{"operations":[{"op":"add","memory":{"content":STRING,"scope":STRING,"evidence_steps":[INTEGER,...],"confidence":NUMBER}},{"op":"refine" OR "replace","target_memory_id":STRING,"memory":{"content":STRING,"scope":STRING,"evidence_steps":[INTEGER,...],"confidence":NUMBER}}]}
For noop return {"operations":[]}."""

MEMORY_AUDIT_SYSTEM = """Audit proposed external-memory changes against the supplied policy, complete trajectory, tool feedback, evaluation, and current memory.

Reject unsupported claims, task-specific private identifiers, conclusions copied from failed behavior, and duplicate adds. Use refine only when the central claim remains the same; use replace when it changes. You may repair a proposal only when the supplied evidence clearly supports the repair. There is no requirement to approve anything.

Return the final approved operations using exactly the memory-editor JSON schema. Return {"operations":[]} if none are adequately supported."""

SFT_BUILDER_SYSTEM = """Decide whether to retain one cleaned, complete customer-service conversation for supervised parameter training.

The harness, not you, reconstructs the example from the complete recorded trajectory. If retained, it contains the domain policy, available tool schemas, every user turn, every assistant response or tool call, and every tool result from the first user request through termination. Assistant text attached to a tool call is removed so the example obeys the one-action-per-turn policy. This is one complete multi-turn example, never isolated intermediate actions.

Retain only if the recorded trajectory is successful according to the environment evaluator, complete, free of infrastructure errors, and useful for learning how to perform future related tasks. Do not retain a failed or merely plausible trajectory. There is no requirement to retain anything.

Return exactly one JSON object:
{"retain":BOOLEAN,"rationale":STRING}"""

DOMAINS = ("airline", "retail", "telecom")


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _excerpt(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    head = limit // 2
    return value[:head] + "\n...[middle omitted]...\n" + value[-head:]


def _compact_trajectory(
    domain: str,
    task: dict,
    simulation: dict,
    policy: str,
    *,
    per_message_chars: int,
) -> dict:
    steps = []
    for index, message in enumerate(simulation.get("messages") or []):
        steps.append(
            {
                "index": index,
                "role": message.get("role"),
                "content": _excerpt(message.get("content"), per_message_chars),
                "tool_calls": message.get("tool_calls"),
                "tool_error": message.get("error"),
            }
        )
    reward_info = simulation.get("reward_info")
    return {
        "source_task_id": f"tau2.{domain}.{simulation['task_id']}",
        "domain": domain,
        "task": task,
        "policy": policy,
        "success": bool(
            isinstance(reward_info, dict) and reward_info.get("reward") == 1.0
        ),
        "reward": None if not isinstance(reward_info, dict) else reward_info.get("reward"),
        "termination_reason": simulation.get("termination_reason"),
        "evaluation": reward_info,
        "steps": steps,
    }


def _controller_decision(value: Any) -> list[dict]:
    calls = []
    seen = set()
    if not isinstance(value, dict) or not isinstance(value.get("tool_calls"), list):
        return calls
    for raw in value["tool_calls"]:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if name not in {"edit_memory", "build_sft_data"} or name in seen:
            continue
        arguments = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
        raw_steps = arguments.get("evidence_steps")
        evidence_steps = []
        if isinstance(raw_steps, list):
            evidence_steps = sorted(
                {
                    step
                    for step in raw_steps
                    if isinstance(step, int)
                    and not isinstance(step, bool)
                    and step >= 0
                }
            )[:16]
        reason = arguments.get("reason")
        calls.append(
            {
                "name": name,
                "arguments": {
                    "reason": str(reason).strip()[:500]
                    if isinstance(reason, str)
                    else "",
                    "evidence_steps": evidence_steps,
                },
            }
        )
        seen.add(name)
    return calls


def _system_prompt(policy: str) -> str:
    return (
        "<instructions>\n"
        f"{AGENT_INSTRUCTION}\n"
        "</instructions>\n"
        "<policy>\n"
        f"{policy}\n"
        "</policy>"
    )


def _training_messages(policy: str, simulation: dict) -> list[dict]:
    messages = [{"role": "system", "content": _system_prompt(policy)}]
    source = simulation.get("messages") or []
    first_user = next(
        (index for index, message in enumerate(source) if message.get("role") == "user"),
        len(source),
    )
    for message in source[first_user:]:
        role = message.get("role")
        if role == "user":
            messages.append({"role": "user", "content": message.get("content") or ""})
        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            item: dict[str, Any] = {
                "role": "assistant",
                "content": None if tool_calls else (message.get("content") or ""),
            }
            if tool_calls:
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "id": call["id"],
                        "function": {
                            "name": call["name"],
                            "arguments": call.get("arguments") or {},
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(item)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": message.get("content") or "",
                    "tool_call_id": message.get("id"),
                }
            )
    return messages


def _tools_for_domain(domain: str) -> list[dict]:
    environment = registry.get_env_constructor(domain)()
    return [as_tool(tool).openai_schema for tool in environment.get_tools()]


def _hard_sft_valid(simulation: dict, messages: list[dict]) -> bool:
    reward_info = simulation.get("reward_info")
    return (
        isinstance(reward_info, dict)
        and reward_info.get("reward") == 1.0
        and simulation.get("termination_reason") in {"user_stop", "agent_stop"}
        and any(message.get("role") == "assistant" for message in messages[1:])
    )


def _task_manifest(results_root: Path, seed: int) -> tuple[list[dict], dict]:
    manifest = []
    metadata = {}
    ordinal = 0
    for offset, domain in enumerate(DOMAINS):
        result_path = results_root / f"qwen35_base_{domain}_full_v1/results.json"
        results = json.loads(result_path.read_text(encoding="utf-8"))
        split_path = (
            results_root.parent / "tau2/domains" / domain / "split_tasks.json"
        )
        train_ids = json.loads(split_path.read_text(encoding="utf-8"))["train"]
        rng = random.Random(seed + offset)
        rng.shuffle(train_ids)
        task_by_id = {task["id"]: task for task in results["tasks"]}
        simulation_by_id = {sim["task_id"]: sim for sim in results["simulations"]}
        if set(train_ids) - simulation_by_id.keys():
            raise RuntimeError(f"{domain}: baseline is missing train tasks")
        metadata[domain] = {
            "result_path": str(result_path),
            "train_task_ids": train_ids,
            "count": len(train_ids),
        }
        for task_id in train_ids:
            manifest.append(
                {
                    "ordinal": ordinal,
                    "domain": domain,
                    "task_id": task_id,
                    "task": task_by_id[task_id],
                    "simulation": simulation_by_id[task_id],
                    "policy": results["info"]["environment_info"]["policy"],
                }
            )
            ordinal += 1
    return manifest, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument(
        "--writer-model",
        help="Optional separately served model for edit_memory calls.",
    )
    parser.add_argument(
        "--auditor-model",
        help="Optional separately served model for memory audit calls.",
    )
    parser.add_argument(
        "--routing-source",
        type=Path,
        help=(
            "Reuse controller tool calls from another retention directory, "
            "matched by ordinal and source_task_id."
        ),
    )
    parser.add_argument(
        "--routing-require-source-memory-change",
        action="store_true",
        help=(
            "When reusing routing, keep edit_memory only when the source "
            "decision actually applied a memory operation. This turns an "
            "offline teacher's accepted edits into a frozen positive router "
            "without teaching noop behavior to the writer."
        ),
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Execute only edit_memory calls from the selected routing decisions.",
    )
    parser.add_argument(
        "--deterministic-memory-validation",
        action="store_true",
        help=(
            "Use evidence/privacy/application gates without a model audit. "
            "This is useful when the writer is served as a merged model and "
            "must remain the only learned component under evaluation."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--limit-per-domain",
        type=int,
        default=0,
        help="Keep the first N shuffled tasks from each domain; 0 disables.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, metadata = _task_manifest(args.results_root, args.seed)
    if args.limit_per_domain > 0:
        kept = Counter()
        limited = []
        for item in manifest:
            if kept[item["domain"]] >= args.limit_per_domain:
                continue
            limited.append(item)
            kept[item["domain"]] += 1
        manifest = limited
    if args.limit > 0:
        manifest = manifest[: args.limit]
    write_json(
        args.output / "manifest.json",
        {
            "protocol": "tau_tool_retention_v2_router_specialist_candidates",
            "seed": args.seed,
            "explicitly_shuffled": True,
            "split": "train",
            "domains": metadata,
            "total": len(manifest),
            "models": {
                "controller": args.model,
                "writer": args.writer_model or args.model,
                "auditor": args.auditor_model or args.model,
            },
            "routing_source": str(args.routing_source.resolve())
            if args.routing_source
            else None,
            "routing_require_source_memory_change": (
                args.routing_require_source_memory_change
            ),
            "memory_only": args.memory_only,
            "memory_validation": "deterministic"
            if args.deterministic_memory_validation
            else "model_audit",
        },
    )

    def client(model_name: str) -> ModelClient:
        return ModelClient(
            base_url=args.base_url,
            api_key="EMPTY",
            model=model_name,
            temperature=args.temperature,
            top_p=0.95 if args.temperature else 1.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            enable_thinking=False,
            timeout=1200,
        )

    controller_model = client(args.model)
    writer_model = client(args.writer_model or args.model)
    auditor_model = client(args.auditor_model or args.model)
    memories = {}
    for domain in DOMAINS:
        path = args.output / f"memory_{domain}.json"
        memories[domain] = (
            normalize_memory_bank(json.loads(path.read_text(encoding="utf-8")))
            if path.exists()
            else []
        )
    tools_by_domain = {domain: _tools_for_domain(domain) for domain in DOMAINS}

    for run_index, item in enumerate(manifest):
        ordinal = item["ordinal"]
        domain = item["domain"]
        task_dir = args.output / "tasks" / f"{ordinal:03d}_{domain}"
        decision_path = task_dir / "retention_decision.json"
        if decision_path.exists():
            print(f"[{run_index + 1}/{len(manifest)}] resume-skip {domain}.{item['task_id']}", flush=True)
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        trajectory = _compact_trajectory(
            domain,
            item["task"],
            item["simulation"],
            item["policy"],
            per_message_chars=4_000,
        )
        controller_view = deepcopy(trajectory)
        for step in controller_view["steps"]:
            step["content"] = _excerpt(step.get("content"), 1_000)
        common = {
            "current_memory": memory_for_agent(memories[domain], max_chars=24_000),
            "trajectory": trajectory,
        }
        record: dict[str, Any] = {
            "protocol": "tau_tool_retention_v2_router_specialist_candidates",
            "source_task_id": trajectory["source_task_id"],
            "controller": None,
            "tools": {},
        }
        if args.routing_source:
            routing_path = (
                args.routing_source
                / "tasks"
                / f"{ordinal:03d}_{domain}"
                / "retention_decision.json"
            )
            routing = json.loads(routing_path.read_text(encoding="utf-8"))
            if routing.get("source_task_id") != trajectory["source_task_id"]:
                raise RuntimeError(
                    f"routing mismatch at ordinal {ordinal}: "
                    f"{routing.get('source_task_id')} != {trajectory['source_task_id']}"
                )
            calls = _controller_decision(
                {"tool_calls": (routing.get("controller") or {}).get("tool_calls", [])}
            )
            if args.routing_require_source_memory_change:
                source_applied = bool(
                    ((routing.get("tools") or {}).get("edit_memory") or {})
                    .get("application", {})
                    .get("applied", [])
                )
                if not source_applied:
                    calls = [call for call in calls if call["name"] != "edit_memory"]
            if args.memory_only:
                calls = [call for call in calls if call["name"] == "edit_memory"]
            record["controller"] = {
                "tool_calls": calls,
                "routing_source": str(routing_path.resolve()),
                "frozen": True,
            }
        else:
            try:
                controller_reply = controller_model.json_chat(
                    system=CONTROLLER_SYSTEM,
                    user=_dump(
                        {
                            "current_memory": common["current_memory"],
                            "trajectory": controller_view,
                        }
                    ),
                )
                calls = _controller_decision(controller_reply.parsed)
                if args.memory_only:
                    calls = [call for call in calls if call["name"] == "edit_memory"]
                record["controller"] = {
                    "tool_calls": calls,
                    "model_content": controller_reply.content,
                    "usage": controller_reply.usage,
                }
            except Exception as exc:
                calls = []
                record["controller"] = {"tool_calls": [], "error": repr(exc)}

        retained_episode = None
        for call in calls:
            name = call["name"]
            payload = {
                "controller_suggested_evidence_steps": call["arguments"][
                    "evidence_steps"
                ],
                "controller_suggested_steps_are_not_evidence": True,
                **common,
            }
            try:
                if name == "edit_memory":
                    draft_reply = writer_model.json_chat(
                        system=MEMORY_WRITER_POLICY_SYSTEM
                        if args.writer_model
                        else MEMORY_EDITOR_SYSTEM,
                        user=_dump(payload),
                    )
                    draft = normalize_memory_operations(draft_reply.parsed)
                    final = draft
                    audit_record = None
                    deterministic_rejections = []
                    if draft and args.deterministic_memory_validation:
                        final = []
                        for operation in draft:
                            validation = validate_writer_candidate(
                                {
                                    "operation": "add",
                                    "memory": {
                                        **operation["memory"],
                                        "conditions": [],
                                        "exceptions": [],
                                    },
                                },
                                trajectory,
                            )
                            if validation["accepted"]:
                                final.append(operation)
                            else:
                                deterministic_rejections.append(
                                    {
                                        "operation": operation,
                                        "reasons": validation["reasons"],
                                    }
                                )
                    elif draft:
                        audit_reply = auditor_model.json_chat(
                            system=MEMORY_AUDIT_SYSTEM,
                            user=_dump({**payload, "proposed_operations": draft}),
                        )
                        final = normalize_memory_operations(audit_reply.parsed)
                        audit_record = {
                            "parsed": audit_reply.parsed,
                            "model_content": audit_reply.content,
                            "usage": audit_reply.usage,
                        }
                    application = apply_memory_operations(
                        memories[domain], final, trajectory=trajectory
                    )
                    record["tools"][name] = {
                        "candidate": draft,
                        "audit": audit_record,
                        "deterministic_rejections": deterministic_rejections,
                        "final_operations": final,
                        "application": application,
                    }
                    write_json(
                        args.output / f"memory_{domain}.json", memories[domain]
                    )
                else:
                    reply = controller_model.json_chat(
                        system=SFT_BUILDER_SYSTEM, user=_dump(payload)
                    )
                    requested = reply.parsed.get("retain") is True
                    messages = _training_messages(item["policy"], item["simulation"])
                    accepted = requested and _hard_sft_valid(
                        item["simulation"], messages
                    )
                    record["tools"][name] = {
                        "candidate": reply.parsed,
                        "model_content": reply.content,
                        "usage": reply.usage,
                        "validation": {
                            "requested": requested,
                            "accepted": accepted,
                            "reason": "complete_successful_recorded_episode"
                            if accepted
                            else "builder_abstained_or_hard_validation_failed",
                        },
                    }
                    if accepted:
                        retained_episode = {
                            "messages": messages,
                            "tools": tools_by_domain[domain],
                            "enable_thinking": False,
                            "domain": domain,
                            "source_task_id": trajectory["source_task_id"],
                            "source_reward": trajectory["reward"],
                            "validation_status": "accepted",
                            "selection_rationale": str(
                                reply.parsed.get("rationale", "")
                            ),
                        }
            except Exception as exc:
                record["tools"][name] = {"error": repr(exc)}

        memory_changed = bool(
            (record["tools"].get("edit_memory") or {})
            .get("application", {})
            .get("applied")
        )
        record["controller_choice"] = (
            "both"
            if {call["name"] for call in calls}
            == {"edit_memory", "build_sft_data"}
            else "context_only"
            if any(call["name"] == "edit_memory" for call in calls)
            else "sft_only"
            if any(call["name"] == "build_sft_data" for call in calls)
            else "neither"
        )
        record["artifact_choice"] = (
            "both"
            if memory_changed and retained_episode
            else "context_only"
            if memory_changed
            else "sft_only"
            if retained_episode
            else "neither"
        )
        if retained_episode:
            write_json(task_dir / "sft_episode.json", retained_episode)
        write_json(decision_path, record)
        print(
            f"[{run_index + 1}/{len(manifest)}] {domain}.{item['task_id']} "
            f"controller={record['controller_choice']} artifact={record['artifact_choice']} "
            f"memories={len(memories[domain])}",
            flush=True,
        )

    episodes = []
    decisions = []
    for path in sorted((args.output / "tasks").glob("*/retention_decision.json")):
        decisions.append(json.loads(path.read_text(encoding="utf-8")))
        episode_path = path.with_name("sft_episode.json")
        if episode_path.exists():
            episodes.append(json.loads(episode_path.read_text(encoding="utf-8")))
    with (args.output / "sft_examples.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    summary = {
        "tasks": len(decisions),
        "controller_choices": dict(Counter(x["controller_choice"] for x in decisions)),
        "artifact_choices": dict(Counter(x["artifact_choice"] for x in decisions)),
        "accepted_sft_episodes": len(episodes),
        "memory_writer": {
            "calls": sum("edit_memory" in x.get("tools", {}) for x in decisions),
            "errors": sum(
                "error" in (x.get("tools", {}).get("edit_memory") or {})
                for x in decisions
            ),
            "candidate_operations": sum(
                len((x.get("tools", {}).get("edit_memory") or {}).get("candidate", []))
                for x in decisions
            ),
            "deterministic_rejections": sum(
                len(
                    (x.get("tools", {}).get("edit_memory") or {}).get(
                        "deterministic_rejections", []
                    )
                )
                for x in decisions
            ),
            "applied_operations": sum(
                len(
                    (x.get("tools", {}).get("edit_memory") or {})
                    .get("application", {})
                    .get("applied", [])
                )
                for x in decisions
            ),
            "application_rejections": sum(
                len(
                    (x.get("tools", {}).get("edit_memory") or {})
                    .get("application", {})
                    .get("rejected", [])
                )
                for x in decisions
            ),
        },
        "memory_entries": {
            domain: len(memories[domain]) for domain in DOMAINS
        },
    }
    write_json(args.output / "summary.json", summary)
    print(_dump(summary), flush=True)


if __name__ == "__main__":
    main()
