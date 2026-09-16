"""Shared prompt and normalization helpers for the tau SFT-data writer."""

from __future__ import annotations

import json
from typing import Any


SFT_DATA_WRITER_SYSTEM = """You are the generation policy inside an SFT-data writing tool for a customer-service agent.

A separate routing controller has already decided to invoke this tool. You receive one completed training task with its full recorded conversation, tool feedback, policy, tool schemas, evaluator outcome, and no private evaluation oracle. Your only job is to produce one complete candidate assistant trajectory for the same task. Do not repeat the routing decision and do not return a summary or isolated middle actions.

The candidate must cover the assistant's behavior from its first response through the final resolution. Repair failed, unsupported, incomplete, or policy-violating behavior. A later harness gives the candidate to a live agent as guidance and replays the task in the real environment; only a replay with reward 1 is retained for task-agent SFT.

Each assistant turn must contain either natural-language content or one or more tool calls, never both. In a tool call, `arguments` must be a JSON-object string. Use only listed tools and supported task-specific values. Do not invent tool results. Put confirmations and questions in content turns. Do not claim an operation succeeded until a preceding live tool call can establish it. End with a content turn communicating the supported outcome.

Some trajectories, especially telecom troubleshooting, contain operations that the user performs
on their own device. Those are not assistant tools. If an operation is not present in the supplied
assistant tool schemas, express it as a concise natural-language instruction or question and wait
for the user's response; never copy or invent it as an assistant tool call. Ensure the returned
object is complete valid JSON even for long trajectories.

Return exactly one JSON object:
{"assistant_turns":[{"content":STRING_OR_NULL,"tool_calls":[{"name":STRING,"arguments":JSON_OBJECT_STRING},...] OR NULL},...],"rationale":STRING,"risk_checks":[STRING,...]}"""


def assistant_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a successful live transcript to the writer's guidance schema."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        calls = []
        for call in message.get("tool_calls") or []:
            name = call.get("name") or (call.get("function") or {}).get("name")
            arguments = (
                call.get("arguments")
                if "arguments" in call
                else (call.get("function") or {}).get("arguments", {})
            )
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append(
                {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        # The tau logs may contain narration and a tool call in one API message.
        # The live protocol requires them to be separate, so preserve both in
        # their natural order as two candidate turns.
        if content:
            turns.append({"content": content, "tool_calls": None})
        if calls:
            turns.append({"content": None, "tool_calls": calls})
    if not turns:
        raise ValueError("successful transcript contains no assistant turns")
    if turns[-1]["tool_calls"]:
        raise ValueError("successful transcript ends with an unacknowledged tool call")
    return turns


def validate_writer_output(
    value: dict[str, Any], allowed_tools: set[str]
) -> dict[str, Any]:
    """Validate and normalize one writer output without evaluating utility."""
    turns = value.get("assistant_turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("writer output has no assistant turns")
    normalized = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {index} is not an object")
        content = turn.get("content")
        calls = turn.get("tool_calls")
        has_content = isinstance(content, str) and bool(content.strip())
        has_calls = isinstance(calls, list) and bool(calls)
        # Generation models sometimes emit a structurally present but empty
        # placeholder turn.  It carries no behavior, so discard it.  If a
        # provider combines narration and a tool call in one assistant turn,
        # split it exactly as ``assistant_turns`` does for recorded tau logs.
        if not has_content and not has_calls:
            continue
        if has_content and has_calls:
            normalized.append({"content": content.strip(), "tool_calls": None})
        normalized_calls = None
        if has_calls:
            normalized_calls = []
            for call in calls:
                if call.get("name") not in allowed_tools:
                    raise ValueError(f"turn {index} uses unsupported tool")
                raw_arguments = call.get("arguments")
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                elif isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments)
                else:
                    raise ValueError(
                        f"turn {index} arguments must be a JSON string or object"
                    )
                if not isinstance(arguments, dict):
                    raise ValueError(f"turn {index} arguments must encode an object")
                normalized_calls.append(
                    {
                        "name": call["name"],
                        "arguments": json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
        normalized.append(
            {
                "content": content.strip() if has_content and not has_calls else None,
                "tool_calls": normalized_calls,
            }
        )
    if not normalized:
        raise ValueError("writer output has no non-empty assistant turns")
    if normalized[-1]["tool_calls"]:
        raise ValueError("writer output must end with content")
    return {
        "assistant_turns": normalized,
        "rationale": str(value.get("rationale") or "").strip(),
        "risk_checks": [str(item) for item in value.get("risk_checks") or []],
    }
