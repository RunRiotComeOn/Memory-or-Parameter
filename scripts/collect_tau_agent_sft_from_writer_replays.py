#!/usr/bin/env python3
"""Collect reward-one writer-guided replays as complete task-agent SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tau2.agent.llm_agent import AGENT_INSTRUCTION
from tau2.environment.tool import as_tool
from tau2.registry import registry


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
DOMAINS = ("airline", "retail", "telecom")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def system_prompt(policy: str) -> str:
    return (
        "<instructions>\n"
        f"{AGENT_INSTRUCTION}\n"
        "</instructions>\n"
        "<policy>\n"
        f"{policy}\n"
        "</policy>"
    )


def training_messages(policy: str, simulation: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(policy)}
    ]
    source = simulation.get("messages") or []
    first_user = next(
        (
            index
            for index, message in enumerate(source)
            if message.get("role") == "user"
        ),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay-tag", default="generation_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    test = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["test"][domain]["task_ids"]
    }
    allowed = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["writer_generation"][domain]["task_ids"]
    }
    records = []
    rejected = []
    for domain in DOMAINS:
        environment = registry.get_env_constructor(domain)()
        tools = [as_tool(tool).openai_schema for tool in environment.get_tools()]
        path = (
            TAU_ROOT
            / f"data/simulations/qwen35_sftdata_guided_{args.replay_tag}_{domain}/results.json"
        )
        simulations = read_json(path)["simulations"]
        for simulation in simulations:
            task_id = str(simulation["task_id"])
            key = (domain, task_id)
            if key in test or key not in allowed:
                raise ValueError("unexpected or test task reached agent SFT collection")
            reward_info = simulation.get("reward_info")
            reward = (
                reward_info.get("reward") if isinstance(reward_info, dict) else None
            )
            termination = simulation.get("termination_reason")
            if reward != 1 or termination not in {"user_stop", "agent_stop"}:
                rejected.append(
                    {
                        "domain": domain,
                        "source_task_id": task_id,
                        "reward": reward,
                        "termination_reason": termination,
                    }
                )
                continue
            messages = training_messages(simulation.get("policy") or "", simulation)
            records.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "enable_thinking": False,
                    "source_task_id": task_id,
                    "domain": domain,
                    "validation_status": "accepted",
                    "validation_basis": "trained_writer_guided_live_replay_reward_one",
                }
            )
    if not records:
        raise ValueError("no reward-one writer-guided replays")
    if {(item["domain"], item["source_task_id"]) for item in records} & test:
        raise ValueError("test leakage in final agent SFT records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    summary = {
        "protocol": "tau_agent_sft_from_trained_writer_v1",
        "accepted": len(records),
        "rejected": len(rejected),
        "rejection_details": rejected,
        "test_overlap": 0,
        "by_domain": {
            domain: sum(item["domain"] == domain for item in records)
            for domain in DOMAINS
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
