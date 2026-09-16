#!/usr/bin/env python3
"""Generate complete tau assistant-trajectory candidates with Codex/GPT-5.6."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "third_party/tau2-bench"
TAU_DATA = TAU_ROOT / "data"
SCHEMA = ROOT / "schemas/tau_sft_data_candidate.schema.json"
DOMAINS = ("airline", "retail", "telecom")

TEACHER_SYSTEM = """You are the teacher for an SFT-data writing tool used after a customer-service trajectory has completed.

A separate router has already chosen to call this tool. Produce one complete candidate assistant trajectory for the same task, from the first assistant response through the final resolution. The candidate will not be trusted directly: a live agent will receive it as guidance and must replay the task in the real environment. Only a replay with reward 1 can become training data.

Use the policy, tool schemas, task, recorded trajectory, tool feedback, evaluator feedback, and teacher-only evaluation oracle. Fix the recorded trajectory when it failed or violated policy. A successful source may be made shorter and clearer, but preserve all actions necessary for success. Do not output isolated actions, a partial prefix, a summary, general advice, user turns, or tool-result turns.

Each assistant turn must contain either natural-language content or one or more tool calls, never both. Tool names and argument shapes must match the schemas. In each tool call, encode `arguments` as a JSON-object string, for example `{"user_id":"example"}`. Include concrete task-specific arguments when supported. Do not invent tool outputs. Put user-facing confirmations and questions in content turns. End with a content turn that clearly communicates the supported outcome. Do not claim that an operation succeeded until a preceding tool call would establish it. Follow confirmation and authorization requirements in the policy.

Return exactly the schema object. `assistant_turns` is the ordered complete candidate. `rationale` briefly explains the repairs. `risk_checks` lists policy, grounding, and completion checks you performed."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def compact_message(message: dict[str, Any], max_content: int = 7000) -> dict[str, Any]:
    item: dict[str, Any] = {"role": message.get("role")}
    content = message.get("content")
    if content is not None:
        text = str(content)
        item["content"] = text[:max_content]
        if len(text) > max_content:
            item["content_truncated"] = True
    calls = []
    for call in message.get("tool_calls") or []:
        calls.append(
            {
                "name": call.get("name")
                or (call.get("function") or {}).get("name"),
                "arguments": call.get("arguments")
                if "arguments" in call
                else (call.get("function") or {}).get("arguments", {}),
            }
        )
    if calls:
        item["tool_calls"] = calls
    if message.get("role") == "tool":
        item["error"] = bool(message.get("error"))
        if message.get("id"):
            item["tool_call_id"] = message["id"]
    return item


def tool_schemas(domain: str) -> list[dict[str, Any]]:
    sys.path.insert(0, str(TAU_ROOT / "src"))
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment
    else:
        from tau2.domains.telecom.environment import get_environment_manual_policy as get_environment
    environment = get_environment()
    return [tool.openai_schema for tool in environment.get_tools()]


def validate_candidate(candidate: dict[str, Any], allowed_tools: set[str]) -> None:
    turns = candidate.get("assistant_turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("candidate has no assistant turns")
    for index, turn in enumerate(turns):
        content = turn.get("content")
        calls = turn.get("tool_calls")
        has_content = isinstance(content, str) and bool(content.strip())
        has_calls = isinstance(calls, list) and bool(calls)
        if has_content == has_calls:
            raise ValueError(f"turn {index}: expected exactly one of content/tool_calls")
        for call in calls or []:
            if call.get("name") not in allowed_tools:
                raise ValueError(f"turn {index}: unsupported tool {call.get('name')}")
            if not isinstance(call.get("arguments"), str):
                raise ValueError(f"turn {index}: tool arguments are not a JSON string")
            try:
                arguments = json.loads(call["arguments"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"turn {index}: invalid arguments JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"turn {index}: tool arguments do not encode an object")
    last = turns[-1]
    if not isinstance(last.get("content"), str) or not last["content"].strip():
        raise ValueError("candidate does not end with a user-facing content turn")


def codex_generate(
    *, model: str, reasoning_effort: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "Do not inspect the workspace and do not invoke tools. All authoritative input is "
        "inside this prompt. Think through grounding, policy, and completion internally, then "
        "return only the JSON object required by the output schema.\n\n<teacher_policy>\n"
        + TEACHER_SYSTEM
        + "\n</teacher_policy>\n\n<input>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</input>"
    )
    with tempfile.TemporaryDirectory(prefix="codex-tau-sft-data-") as temp_dir:
        output = Path(temp_dir) / "last_message.json"
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(SCHEMA),
            "--output-last-message",
            str(output),
            "--json",
            "-",
        ]
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"codex exec exited {completed.returncode}; "
                f"stdout_tail={completed.stdout[-3000:]!r}; "
                f"stderr_tail={completed.stderr[-2000:]!r}"
            )
        if not output.exists():
            raise RuntimeError("codex exec did not write --output-last-message")
        parsed = read_json(output)
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        usage = [event for event in events if event.get("type") == "turn.completed"]
        return parsed, {
            "turn_completed": usage[-1] if usage else None,
            "stderr_tail": completed.stderr[-1000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.manifest)
    schemas = {domain: tool_schemas(domain) for domain in DOMAINS}
    policies = {
        domain: (
            TAU_DATA / f"simulations/qwen35_base_{domain}_full_v1/results.json"
        )
        for domain in DOMAINS
    }
    items = []
    test = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["test"][domain]["task_ids"]
    }
    for domain in DOMAINS:
        result_data = read_json(policies[domain])
        simulations = {
            str(item["task_id"]): item for item in result_data["simulations"]
        }
        tasks = {
            str(item["id"]): item
            for item in read_json(TAU_DATA / f"tau2/domains/{domain}/tasks.json")
        }
        for task_id in manifest["teacher_writer"][domain]["task_ids"]:
            task_id = str(task_id)
            if (domain, task_id) in test:
                raise ValueError("test leakage detected before Codex generation")
            items.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "task": tasks[task_id],
                    "simulation": simulations[task_id],
                }
            )
    if args.limit > 0:
        items = items[: args.limit]

    write_json(
        args.output / "manifest.json",
        {
            "protocol": "tau_codex_sft_data_teacher_v1",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "source_split_manifest": str(args.manifest.resolve()),
            "teacher_tasks": len(items),
            "test_overlap": 0,
            "prompt": TEACHER_SYSTEM,
        },
    )

    def run_one(ordinal_item: tuple[int, dict[str, Any]]) -> tuple[int, str]:
        ordinal, item = ordinal_item
        domain = item["domain"]
        task_id = item["task_id"]
        task_dir = args.output / "tasks" / f"{ordinal:03d}_{domain}"
        output_path = task_dir / "candidate.json"
        if output_path.exists():
            existing = read_json(output_path)
            if existing.get("status") == "candidate_ready":
                return ordinal, f"resume-skip {domain}.{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        simulation = item["simulation"]
        student_task = {
            key: value
            for key, value in item["task"].items()
            if key not in {"evaluation_criteria", "initial_state"}
        }
        payload = {
            "domain": domain,
            "source_task_id": task_id,
            "student_visible": {
                "task": student_task,
                "source_trajectory": {
                    "messages": [compact_message(message) for message in simulation["messages"]],
                    "reward": simulation.get("reward_info", {}).get("reward"),
                    "reward_feedback": simulation.get("reward_info"),
                    "termination_reason": simulation.get("termination_reason"),
                    "review": simulation.get("review"),
                },
                "policy": simulation.get("policy"),
                "tool_schemas": schemas[domain],
            },
            "teacher_only_oracle": {
                "initial_state": item["task"].get("initial_state"),
                "evaluation_criteria": item["task"].get("evaluation_criteria"),
            },
        }
        try:
            candidate, usage = codex_generate(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                payload=payload,
                timeout=args.timeout,
            )
            allowed = {
                schema["function"]["name"] for schema in schemas[domain]
            }
            validate_candidate(candidate, allowed)
            record = {
                "protocol": "tau_codex_sft_data_teacher_v1",
                "domain": domain,
                "source_task_id": task_id,
                "source_reward": simulation.get("reward_info", {}).get("reward"),
                "student_input": payload["student_visible"],
                "candidate": candidate,
                "teacher_usage": usage,
                "status": "candidate_ready",
            }
        except Exception as exc:
            record = {
                "protocol": "tau_codex_sft_data_teacher_v1",
                "domain": domain,
                "source_task_id": task_id,
                "source_reward": simulation.get("reward_info", {}).get("reward"),
                "error": repr(exc),
                "status": "error",
            }
        write_json(output_path, record)
        return ordinal, f"{domain}.{task_id} status={record['status']}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in enumerate(items)]
        for future in concurrent.futures.as_completed(futures):
            ordinal, message = future.result()
            print(f"[{ordinal + 1}/{len(items)}] {message}", flush=True)

    records = [
        read_json(path)
        for path in sorted((args.output / "tasks").glob("*/candidate.json"))
    ]
    summary = {
        "protocol": "tau_codex_sft_data_teacher_v1",
        "tasks": len(records),
        "candidate_ready": sum(item.get("status") == "candidate_ready" for item in records),
        "errors": sum(item.get("status") == "error" for item in records),
        "by_domain": {
            domain: sum(item.get("domain") == domain for item in records)
            for domain in DOMAINS
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
