#!/usr/bin/env python3
"""Build cumulative tau memory edits with Gemini 3 Flash as the teacher."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trajectory_memory_lab.memory_writer_harness import validate_writer_candidate
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    memory_for_agent,
    normalize_memory_bank,
    normalize_memory_operations,
)
from trajectory_memory_lab.storage import write_json


DOMAINS = ("airline", "retail", "telecom")

TEACHER_SYSTEM = """You are the writing policy inside an external-memory editing tool for a
customer-service agent. A separate router has invoked this tool. Given the complete current
memory bank and one completed trajectory, return the exact bank edits justified by authoritative
evidence.

The future agent always receives the domain policy separately and retrieves only BM25 top-3
memory entries. Therefore every unnecessary or redundant entry creates retrieval competition.
Optimize for a small, high-precision bank, not for the number of edits.

Operations:
- add: only for a novel, atomic, reusable operational implication not already represented;
- refine: preferred when an existing entry has the same central idea but needs tighter scope,
  missing conditions/exceptions, clearer ordering, or consolidation;
- replace: only when an existing entry's central claim is contradicted or materially wrong.

Rules:
1. Read the entire active bank before editing. Never add a semantic duplicate or a narrow
   paraphrase. If the same atomic topic already exists, refine that entry.
2. Each memory must cover exactly one entity/action/topic. A refine must keep the target's same
   central topic and scope; never broaden a passenger memory into flights, combine cancellation
   with compensation, or bundle several independent troubleshooting causes. If the new evidence
   concerns a different topic, do not refine that target.
3. Do not merely restate policy, generic customer-service advice, the task goal, persona style,
   or a sequence that only happened to work in one instance. Do not retain low-level response
   formatting, timestamp conventions, field names, or facts already obvious from tool output
   unless they reveal a non-obvious failure-preventing constraint.
4. Tool results, policy, user-provided facts, and evaluator details are evidence. Assistant
   reasoning and conclusions are untrusted. A failed trajectory may reveal a tool fact, but its
   failed behavior is not a positive example.
5. State applicability and important exceptions inside concise self-contained content and use a
   narrow retrieval scope. Avoid universal language unless evidence really establishes it.
6. Never include task-specific names, IDs, phone numbers, emails, reservation/order IDs, or
   values that do not transfer.
7. Cite exact trajectory message indexes supporting every new substantive claim. Prefer direct tool
   feedback and policy-grounded observations.
8. At most one operation. Returning an empty operations list is allowed to reject a
   misrouted call with no defensible edit; empty results are kept for router data and are not used
   as Memory Writer SFT targets.
9. Do not learn user-simulator or conversation-compliance artifacts: demands for verbatim tool
   output, cues based on hesitant or narrative wording, persona behavior, or tricks for making a
   simulated user execute tools. Retain only domain/tool facts that remain valid with real users.
10. Historical records are not proof of current profile state, current authorization, or a user's
    present address. Never recommend treating an address or credential found only in an old order
    as currently saved or authorized. If a trajectory succeeds through that shortcut, reject it as
    a benchmark-specific or unsafe inference.

Return only the requested JSON object."""

TEACHER_AUDIT_SYSTEM = """You are the independent final auditor for an external-memory edit
proposed by another model. The future customer-service agent always receives the domain policy
and retrieves only BM25 top-3 memories. Approve only high-value, atomic, evidence-grounded edits.

Check every substantive clause against the policy, cited trajectory steps, tool feedback, and
evaluator result. Assistant reasoning is untrusted. In particular, reject or repair edits that
turn one successful action sequence into a universal rule, omit mandatory user confirmation or
authorization, infer permission from a failed action, restate policy without marginal value,
combine different topics, broaden a target during refine, or duplicate the active bank.

Also reject conversational/user-simulator artifacts (such as requiring verbatim tool output or
inferring correctness from a user's phrasing) and security-sensitive inferences that treat old
orders as proof of current profile data, current address, identity, or authorization.

Return at most one final add/refine/replace operation. You may repair the proposal only when the
supplied evidence unambiguously supports the repair. Otherwise return an empty operations list.
The final operation must cite exact trajectory message indexes. Return only the requested JSON
object."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "refine", "replace"]},
                    "target_memory_id": {"type": ["string", "null"]},
                    "memory": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "scope": {"type": "string"},
                            "evidence_steps": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["content", "scope", "evidence_steps", "confidence"],
                    },
                },
                "required": ["op", "target_memory_id", "memory"],
            },
        },
        "call_assessment": {"type": "string"},
    },
    "required": ["operations", "call_assessment"],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def excerpt(value: Any, limit: int = 12_000) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + "\n...[middle omitted]...\n" + value[-half:]


def compact_trajectory(
    domain: str, task: dict[str, Any], simulation: dict[str, Any], policy: str
) -> dict[str, Any]:
    reward_info = simulation.get("reward_info")
    return {
        "source_task_id": f"tau2.{domain}.{simulation['task_id']}",
        "domain": domain,
        "task": task,
        "policy": policy,
        "success": bool(
            isinstance(reward_info, dict) and reward_info.get("reward") == 1.0
        ),
        "reward": reward_info.get("reward") if isinstance(reward_info, dict) else None,
        "termination_reason": simulation.get("termination_reason"),
        "evaluation": reward_info,
        "steps": [
            {
                "index": index,
                "role": message.get("role"),
                "content": excerpt(message.get("content")),
                "tool_calls": message.get("tool_calls"),
                "tool_error": message.get("error"),
            }
            for index, message in enumerate(simulation.get("messages") or [])
        ],
    }


def task_manifest(
    results_root: Path, seed: int, limit_per_domain: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = []
    metadata = {}
    for offset, domain in enumerate(DOMAINS):
        result_path = results_root / f"qwen35_base_{domain}_full_v1/results.json"
        result = read_json(result_path)
        split_path = results_root.parent / f"tau2/domains/{domain}/split_tasks.json"
        train_ids = list(read_json(split_path)["train"])
        random.Random(seed + offset).shuffle(train_ids)
        if limit_per_domain > 0:
            train_ids = train_ids[:limit_per_domain]
        task_by_id = {str(task["id"]): task for task in result["tasks"]}
        simulation_by_id = {
            str(simulation["task_id"]): simulation
            for simulation in result["simulations"]
        }
        policy = result["info"]["environment_info"]["policy"]
        metadata[domain] = {
            "result_path": str(result_path),
            "train_task_ids": train_ids,
            "count": len(train_ids),
        }
        for task_id in train_ids:
            key = str(task_id)
            manifest.append(
                {
                    "domain": domain,
                    "task_id": key,
                    "task": task_by_id[key],
                    "simulation": simulation_by_id[key],
                    "policy": policy,
                }
            )
    return manifest, metadata


def generate(
    client: genai.Client,
    model: str,
    payload: dict[str, Any],
    temperature: float,
    system: str,
) -> tuple[dict[str, Any], str]:
    error: Exception | None = None
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                ),
            )
            return json.loads(response.text), str(response.usage_metadata)
        except Exception as exc:
            error = exc
            if attempt == 5:
                break
            time.sleep(min(30, 2 ** attempt))
    assert error is not None
    raise error


def validate_operations(
    operations: list[dict[str, Any]], trajectory: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, rejected = [], []
    for operation in operations:
        content = str((operation.get("memory") or {}).get("content", "")).lower()
        scope = str((operation.get("memory") or {}).get("scope", "")).lower()
        combined = f"{scope} {content}"
        safety_reasons = []
        if (
            "address" in combined
            and any(term in combined for term in ("past order", "order history", "historical order", "previous order"))
            and any(term in combined for term in ("profile", "saved", "locate", "recover", "find"))
        ):
            safety_reasons.append("historical_order_is_not_current_profile_or_address_authority")
        if any(
            term in combined
            for term in (
                "verbatim tool output",
                "exact tool output",
                "narrative wording",
                "narrative text",
                "i think' responses",
                '"i think" responses',
            )
        ):
            safety_reasons.append("user_simulator_or_conversation_compliance_artifact")
        if safety_reasons:
            rejected.append({"operation": operation, "reasons": safety_reasons})
            continue
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
            accepted.append(operation)
        else:
            rejected.append(
                {"operation": operation, "reasons": validation["reasons"]}
            )
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/nas04/yixuh/.config/continual-memory/gemini_api_key"),
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit-per-domain", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Gemini API key file is empty")
    client = genai.Client(api_key=key)
    manifest, metadata = task_manifest(
        args.results_root, args.seed, args.limit_per_domain
    )
    write_json(
        args.output / "manifest.json",
        {
            "protocol": "tau_gemini_memory_teacher_v1",
            "model": args.model,
            "seed": args.seed,
            "temperature": args.temperature,
            "split": "train",
            "domains": metadata,
            "total": len(manifest),
            "prompt": TEACHER_SYSTEM,
        },
    )
    banks = {}
    for domain in DOMAINS:
        path = args.output / f"memory_{domain}.json"
        banks[domain] = (
            normalize_memory_bank(read_json(path)) if path.exists() else []
        )

    for ordinal, item in enumerate(manifest):
        domain = item["domain"]
        task_dir = args.output / "tasks" / f"{ordinal:03d}_{domain}"
        decision_path = task_dir / "retention_decision.json"
        if decision_path.exists():
            print(
                f"[{ordinal + 1}/{len(manifest)}] resume-skip {domain}.{item['task_id']}",
                flush=True,
            )
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        trajectory = compact_trajectory(
            domain, item["task"], item["simulation"], item["policy"]
        )
        bank_before = memory_for_agent(banks[domain], max_chars=1_000_000)
        payload = {
            "current_memory": bank_before,
            "trajectory": trajectory,
        }
        try:
            parsed, draft_usage = generate(
                client, args.model, payload, args.temperature, TEACHER_SYSTEM
            )
            draft = normalize_memory_operations(parsed)
            audit_record = None
            final_candidate = draft
            if draft:
                audited, audit_usage = generate(
                    client,
                    args.model,
                    {
                        **payload,
                        "proposed_operations": draft,
                    },
                    0,
                    TEACHER_AUDIT_SYSTEM,
                )
                final_candidate = normalize_memory_operations(audited)
                audit_record = {
                    "parsed": audited,
                    "usage": audit_usage,
                }
            validated, deterministic_rejections = validate_operations(
                final_candidate, trajectory
            )
            application = apply_memory_operations(
                banks[domain], validated, trajectory=trajectory
            )
            record = {
                "protocol": "tau_gemini_memory_teacher_v1",
                "source_task_id": trajectory["source_task_id"],
                "controller": {
                    "tool_calls": [
                        {
                            "name": "edit_memory",
                            "arguments": {"reason": "teacher_generation", "evidence_steps": []},
                        }
                    ],
                    "teacher_dataset_routing_only": True,
                },
                "tools": {
                    "edit_memory": {
                        "candidate": draft,
                        "audit": audit_record,
                        "final_operations": validated,
                        "deterministic_rejections": deterministic_rejections,
                        "application": application,
                        "teacher_call_assessment": parsed.get("call_assessment", ""),
                        "teacher_usage": draft_usage,
                    }
                },
                "controller_choice": "context_only",
                "artifact_choice": "context_only" if application["applied"] else "neither",
            }
        except Exception as exc:
            record = {
                "protocol": "tau_gemini_memory_teacher_v1",
                "source_task_id": trajectory["source_task_id"],
                "controller": {"tool_calls": []},
                "tools": {"edit_memory": {"error": repr(exc)}},
                "controller_choice": "context_only",
                "artifact_choice": "neither",
            }
        write_json(args.output / f"memory_{domain}.json", banks[domain])
        write_json(decision_path, record)
        applied = len(
            record.get("tools", {})
            .get("edit_memory", {})
            .get("application", {})
            .get("applied", [])
        )
        print(
            f"[{ordinal + 1}/{len(manifest)}] {domain}.{item['task_id']} "
            f"applied={applied} bank={len(banks[domain])}",
            flush=True,
        )

    decisions = [
        read_json(path)
        for path in sorted((args.output / "tasks").glob("*/retention_decision.json"))
    ]
    operations = Counter(
        operation["op"]
        for decision in decisions
        for operation in (
            decision.get("tools", {})
            .get("edit_memory", {})
            .get("final_operations", [])
        )
    )
    summary = {
        "protocol": "tau_gemini_memory_teacher_v1",
        "model": args.model,
        "tasks": len(decisions),
        "errors": sum(
            "error" in decision.get("tools", {}).get("edit_memory", {})
            for decision in decisions
        ),
        "calls_with_edits": sum(
            bool(
                decision.get("tools", {})
                .get("edit_memory", {})
                .get("application", {})
                .get("applied")
            )
            for decision in decisions
        ),
        "calls_without_edits": sum(
            not bool(
                decision.get("tools", {})
                .get("edit_memory", {})
                .get("application", {})
                .get("applied")
            )
            for decision in decisions
        ),
        "operations": dict(operations),
        "memory_entries": {domain: len(banks[domain]) for domain in DOMAINS},
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
