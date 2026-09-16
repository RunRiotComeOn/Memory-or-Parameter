from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

from .model_client import ModelClient, ModelReply
from .prompts import (
    MEMORY_AUDIT_SYSTEM,
    MEMORY_EDITOR_SYSTEM,
    RETENTION_CONTROLLER_SYSTEM,
    SFT_AUDIT_SYSTEM,
    SFT_BUILDER_SYSTEM,
)


PROTOCOL_VERSION = "tool_v5_router_specialist_candidates"
MEMORY_TOOL = "edit_memory"
SFT_TOOL = "build_sft_examples"
_MEMORY_OPS = {"add", "refine", "replace"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reply_record(reply: ModelReply) -> dict[str, Any]:
    return {
        "parsed": reply.parsed,
        "model_content": reply.content,
        "model_reasoning": reply.reasoning,
        "model_usage": reply.usage,
    }


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _task_id(trajectory: dict[str, Any]) -> str:
    if "source_task_id" in trajectory:
        return str(trajectory["source_task_id"])
    if "task_id" in trajectory:
        return str(trajectory["task_id"])
    task = trajectory.get("task") or {}
    return str(task.get("id", "unknown"))


def _task_description(trajectory: dict[str, Any]) -> Any:
    if trajectory.get("goal"):
        return trajectory["goal"]
    task = trajectory.get("task") or {}
    return task.get("instruction") or task


def _trajectory_success(trajectory: dict[str, Any]) -> bool:
    if "success" in trajectory:
        return bool(trajectory["success"])
    return bool((trajectory.get("evaluation") or {}).get("success"))


def _step_observation(step: dict[str, Any]) -> Any:
    return step.get("observation", "")


def _observation_excerpt(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        head = max_chars // 3
        tail = max_chars - head
        return value[:head] + "\n...[observation middle omitted]...\n" + value[-tail:]
    if isinstance(value, dict):
        compact = deepcopy(value)
        if isinstance(compact.get("text"), str):
            compact["text"] = _observation_excerpt(compact["text"], max_chars)
        return compact
    return value


def trajectory_for_controller(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": _task_id(trajectory),
        "task": _task_description(trajectory),
        "success": _trajectory_success(trajectory),
        "reward": trajectory.get("reward"),
        "final_answer": trajectory.get("final_answer"),
        "error": trajectory.get("error"),
        "steps": [
            {
                "index": step.get("index"),
                "observation": _observation_excerpt(_step_observation(step), 1_200),
                "action": step.get("action"),
                "action_error": step.get("action_error", step.get("execution_error")),
                "reward": step.get("reward"),
            }
            for step in trajectory.get("steps", [])
        ],
    }


def trajectory_for_tool(
    trajectory: dict[str, Any],
    *,
    max_total_observation_chars: int = 100_000,
    max_per_observation_chars: int = 42_000,
) -> dict[str, Any]:
    """Keep every step and feedback while fitting the server's 65k-token context."""
    steps = trajectory.get("steps", [])
    per_observation_chars = min(
        max_per_observation_chars,
        max(1_000, max_total_observation_chars // max(1, len(steps))),
    )
    return {
        "task_id": _task_id(trajectory),
        "task": _task_description(trajectory),
        "success": _trajectory_success(trajectory),
        "reward": trajectory.get("reward"),
        "final_answer": trajectory.get("final_answer"),
        "evaluation": trajectory.get("evaluation"),
        "error": trajectory.get("error"),
        "context_before": trajectory.get("context_before", []),
        "steps": [
            {
                "index": step.get("index"),
                "url": step.get("url")
                or (
                    (_step_observation(step) or {}).get("url")
                    if isinstance(_step_observation(step), dict)
                    else None
                ),
                "observation": _observation_excerpt(
                    _step_observation(step), per_observation_chars
                ),
                "action": step.get("action"),
                "execution": step.get("execution", step.get("action_code")),
                "action_error": step.get("action_error", step.get("execution_error")),
                "reward": step.get("reward"),
            }
            for step in steps
        ],
    }


def normalize_controller_decision(value: dict[str, Any]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_calls = value.get("tool_calls")
    if not isinstance(raw_calls, list):
        raw_calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if name not in {MEMORY_TOOL, SFT_TOOL} or name in seen:
            continue
        arguments = (
            raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
        )
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
        calls.append(
            {
                "name": name,
                "arguments": {
                    "reason": _text(arguments.get("reason"))[:500],
                    "evidence_steps": evidence_steps,
                },
            }
        )
        seen.add(name)
    return {"tool_calls": calls}


def normalize_memory_bank(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            continue
        item = deepcopy(raw)
        memory_id = _text(item.get("id")) or f"mem_{index + 1:06d}"
        if memory_id in used_ids:
            suffix = 2
            while f"{memory_id}_{suffix}" in used_ids:
                suffix += 1
            memory_id = f"{memory_id}_{suffix}"
        used_ids.add(memory_id)
        item["id"] = memory_id
        item["content"] = _text(item.get("content"))
        item["scope"] = _text(item.get("scope"))
        item["status"] = _text(item.get("status")) or "active"
        item["version"] = (
            item.get("version") if isinstance(item.get("version"), int) else 1
        )
        item["evidence"] = (
            item.get("evidence") if isinstance(item.get("evidence"), list) else []
        )
        item["history"] = (
            item.get("history") if isinstance(item.get("history"), list) else []
        )
        normalized.append(item)
    return normalized


def memory_for_agent(
    entries: list[dict[str, Any]], max_chars: int = 18_000
) -> list[dict[str, Any]]:
    rendered = [
        {key: entry[key] for key in ("id", "content", "scope") if entry.get(key)}
        for entry in normalize_memory_bank(entries)
        if entry.get("status") == "active" and entry.get("content")
    ]
    selected: list[dict[str, Any]] = []
    used = 0
    for entry in reversed(rendered):
        size = len(json.dumps(entry, ensure_ascii=False))
        if selected and used + size > max_chars:
            break
        selected.append(entry)
        used += size
    return list(reversed(selected))


def normalize_memory_operations(value: dict[str, Any]) -> list[dict[str, Any]]:
    raw_operations = value.get("operations")
    if not isinstance(raw_operations, list):
        raw_operations = [value] if value.get("operation") or value.get("op") else []
    operations: list[dict[str, Any]] = []
    for raw in raw_operations[:8]:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op", raw.get("operation", ""))).strip().lower()
        if op == "noop":
            continue
        if op not in _MEMORY_OPS:
            continue
        memory = raw.get("memory") if isinstance(raw.get("memory"), dict) else {}
        content = _text(memory.get("content"))
        if not content:
            continue
        raw_steps = memory.get("evidence_steps")
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
            )
        confidence = memory.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None
        elif not 0 <= float(confidence) <= 1:
            confidence = None
        operation = {
            "op": op,
            "memory": {
                "content": content,
                "scope": _text(memory.get("scope")),
                "evidence_steps": evidence_steps,
                "confidence": confidence,
            },
        }
        target = _text(raw.get("target_memory_id"))
        if op in {"refine", "replace"}:
            operation["target_memory_id"] = target
        operations.append(operation)
    return operations


def _next_memory_id(entries: list[dict[str, Any]]) -> str:
    numbers = []
    for entry in entries:
        match = re.fullmatch(r"mem_(\d+)", str(entry.get("id", "")))
        if match:
            numbers.append(int(match.group(1)))
    return f"mem_{max(numbers, default=0) + 1:06d}"


def apply_memory_operations(
    entries: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    *,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    bank = normalize_memory_bank(entries)
    valid_steps = {step.get("index") for step in trajectory.get("steps", [])}
    task_id = _task_id(trajectory)
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for operation in operations:
        op = operation["op"]
        memory = operation["memory"]
        evidence_steps = memory.get("evidence_steps") or []
        reason = None
        if not evidence_steps:
            reason = "missing_evidence_steps"
        elif any(step not in valid_steps for step in evidence_steps):
            reason = "invalid_evidence_step"
        active = [entry for entry in bank if entry.get("status") == "active"]
        if op == "add" and any(
            entry.get("content", "").casefold() == memory["content"].casefold()
            for entry in active
        ):
            reason = "duplicate_active_memory"
        target = None
        if op in {"refine", "replace"}:
            target_id = operation.get("target_memory_id")
            target = next(
                (
                    entry
                    for entry in bank
                    if entry.get("id") == target_id and entry.get("status") == "active"
                ),
                None,
            )
            if target is None:
                reason = "unknown_or_inactive_target"
            elif target.get("content") == memory["content"] and target.get(
                "scope", ""
            ) == memory.get("scope", ""):
                reason = "no_change"
        if reason:
            rejected.append({"operation": operation, "reason": reason})
            continue

        evidence = [
            {"source_task_id": task_id, "step": step} for step in evidence_steps
        ]
        timestamp = _now()
        if op == "add":
            new_entry = {
                "id": _next_memory_id(bank),
                "content": memory["content"],
                "scope": memory.get("scope", ""),
                "confidence": memory.get("confidence"),
                "status": "active",
                "version": 1,
                "evidence": evidence,
                "history": [],
                "source_task_id": task_id,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            bank.append(new_entry)
            applied.append({"op": op, "memory_id": new_entry["id"], "version": 1})
            continue

        assert target is not None
        target.setdefault("history", []).append(
            {
                "version": target.get("version", 1),
                "content": target.get("content", ""),
                "scope": target.get("scope", ""),
                "confidence": target.get("confidence"),
                "evidence": target.get("evidence", []),
                "changed_by": op,
                "changed_at": timestamp,
            }
        )
        target["version"] = int(target.get("version", 1)) + 1
        target["content"] = memory["content"]
        target["scope"] = memory.get("scope", "")
        target["confidence"] = memory.get("confidence")
        if op == "refine":
            # A refinement preserves the central claim, so keep the evidence
            # supporting that inherited claim and append the new evidence used
            # to tighten its scope or wording. A replace changes the central
            # claim and therefore intentionally starts a fresh evidence set.
            combined_evidence = list(target.get("evidence", [])) + evidence
            seen_evidence = set()
            target["evidence"] = []
            for item in combined_evidence:
                key = (item.get("source_task_id"), item.get("step"))
                if key not in seen_evidence:
                    target["evidence"].append(item)
                    seen_evidence.add(key)
        else:
            target["evidence"] = evidence
        target["source_task_id"] = task_id
        target["updated_at"] = timestamp
        applied.append(
            {"op": op, "memory_id": target["id"], "version": target["version"]}
        )

    entries[:] = bank
    return {"applied": applied, "rejected": rejected}


def normalize_sft_episodes(value: dict[str, Any]) -> list[dict[str, Any]]:
    raw_episode = value.get("episode")
    if not isinstance(raw_episode, dict):
        return []
    mode = _text(raw_episode.get("mode"))
    if mode == "preserve_recorded":
        return [
            {
                "mode": "preserve_recorded",
                "rationale": _text(raw_episode.get("rationale")),
            }
        ]
    if mode not in {"", "corrected"}:
        return []
    raw_steps = raw_episode.get("steps")
    if not isinstance(raw_steps, list):
        return []
    episode_steps: list[dict[str, Any]] = []
    for raw in raw_steps[:64]:
        if not isinstance(raw, dict):
            continue
        source_step = raw.get("source_step", raw.get("index"))
        action = raw.get("target_action", raw.get("action", raw.get("a")))
        if (
            not isinstance(source_step, int)
            or isinstance(source_step, bool)
            or not isinstance(action, dict)
        ):
            continue
        supporting_steps = raw.get("supporting_steps")
        if not isinstance(supporting_steps, list):
            supporting_steps = []
        episode_steps.append(
            {
                "source_step": source_step,
                "target_action": action,
                "label_type": "corrected"
                if raw.get("label_type") == "corrected"
                else "recorded",
                "supporting_steps": sorted(
                    {
                        step
                        for step in supporting_steps
                        if isinstance(step, int)
                        and not isinstance(step, bool)
                        and step >= 0
                    }
                ),
                "rationale": _text(raw.get("rationale")),
            }
        )
    if not episode_steps:
        return []
    return [
        {
            "mode": "corrected",
            "steps": episode_steps,
            "rationale": _text(raw_episode.get("rationale")),
        }
    ]


def _available_element_ids(observation: Any) -> set[str]:
    if isinstance(observation, str):
        return set(re.findall(r"\[([^\]\n]+)\]", observation))
    if isinstance(observation, dict):
        return {
            str(element.get("ref"))
            for element in observation.get("elements", [])
            if isinstance(element, dict) and element.get("ref") is not None
        }
    return set()


def _actions_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    ignored = {"note", "rationale"}
    return {k: v for k, v in left.items() if k not in ignored} == {
        k: v for k, v in right.items() if k not in ignored
    }


def _validate_action(
    action: dict[str, Any], observation: Any, allowed_actions: set[str]
) -> str | None:
    kind = str(action.get("action", "")).lower()
    if kind not in allowed_actions:
        return "unsupported_action"
    ids = _available_element_ids(observation)
    identifier = action.get("bid", action.get("ref"))
    interactive_kinds = {"click", "fill", "select_option", "hover"}
    # The generic action space permits {"action":"press","ref":null};
    # TimeWarp requires a bid and its action_validator enforces that variant.
    if kind in interactive_kinds or (kind == "press" and identifier is not None):
        if identifier is None:
            return "missing_element_identifier"
        if str(identifier) not in ids:
            return "element_not_in_source_observation"
    if kind == "press" and identifier is None and "ref" not in action:
        return "missing_element_identifier"
    if kind == "fill" and not isinstance(action.get("value", action.get("text")), str):
        return "missing_fill_value"
    if kind == "select_option" and not isinstance(action.get("value"), str):
        return "missing_option_value"
    if kind == "goto" and not _text(action.get("url")):
        return "missing_url"
    if kind == "finish" and not _text(action.get("answer")):
        return "missing_answer"
    return None


def validate_sft_episodes(
    episodes: list[dict[str, Any]],
    *,
    trajectory: dict[str, Any],
    allowed_actions: set[str],
    action_validator: Callable[[dict[str, Any]], Any] | None = None,
    require_action_note: bool = False,
) -> dict[str, Any]:
    steps = trajectory.get("steps", [])
    by_index = {step.get("index"): step for step in steps}
    expected_indexes = [step.get("index") for step in steps]
    valid_step_indexes = set(expected_indexes)
    accepted: list[dict[str, Any]] = []
    needs_replay: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for episode in episodes:
        mode = episode.get("mode", "corrected")
        if mode == "preserve_recorded":
            episode_steps = [
                {
                    "source_step": step.get("index"),
                    "target_action": deepcopy(step.get("action"))
                    if isinstance(step.get("action"), dict)
                    else {},
                    "label_type": "recorded",
                    "supporting_steps": [step.get("index")],
                    "rationale": "selected from complete recorded trajectory",
                }
                for step in steps
            ]
        else:
            episode_steps = episode.get("steps", [])
        source_indexes = [item.get("source_step") for item in episode_steps]
        reason = None
        detail: dict[str, Any] = {}
        if not steps:
            reason = "empty_source_trajectory"
        elif source_indexes != expected_indexes:
            reason = "incomplete_or_nonconsecutive_episode"
            detail = {
                "expected_source_steps": expected_indexes,
                "actual_source_steps": source_indexes,
            }
        elif str(episode_steps[-1]["target_action"].get("action", "")).lower() != "finish":
            reason = "missing_terminal_finish"
        elif any(
            str(item["target_action"].get("action", "")).lower() == "finish"
            for item in episode_steps[:-1]
        ):
            reason = "finish_before_terminal_step"

        checked_steps: list[dict[str, Any]] = []
        all_equivalent = True
        has_source_error = False
        if reason is None:
            for episode_step in episode_steps:
                source_step = episode_step["source_step"]
                source = by_index[source_step]
                action = episode_step["target_action"]
                step_reason = None
                if any(
                    step not in valid_step_indexes
                    for step in episode_step.get("supporting_steps", [])
                ):
                    step_reason = "invalid_supporting_step"
                if step_reason is None:
                    step_reason = _validate_action(
                        action, _step_observation(source), allowed_actions
                    )
                if (
                    step_reason is None
                    and require_action_note
                    and not _text(action.get("note"))
                ):
                    step_reason = "missing_required_progress_note"
                if step_reason is None and action_validator is not None:
                    try:
                        action_validator(action)
                    except Exception as exc:
                        step_reason = f"action_validator:{type(exc).__name__}:{exc}"
                if step_reason is not None:
                    reason = step_reason
                    detail = {"source_step": source_step}
                    break
                recorded_action = (
                    source.get("action")
                    if isinstance(source.get("action"), dict)
                    else {}
                )
                equivalent = _actions_equivalent(recorded_action, action)
                all_equivalent = all_equivalent and equivalent
                has_source_error = has_source_error or bool(
                    source.get("action_error", source.get("execution_error"))
                )
                checked_steps.append(
                    {
                        **episode_step,
                        "recorded_action": recorded_action,
                        "label_type": "recorded" if equivalent else "corrected",
                    }
                )

        if (
            reason is None
            and mode == "preserve_recorded"
            and (not _trajectory_success(trajectory) or has_source_error)
        ):
            reason = "preserve_recorded_requires_successful_error_free_trajectory"

        key = json.dumps(episode_steps, ensure_ascii=False, sort_keys=True)
        if reason is None and key in seen:
            reason = "duplicate_episode"
        if reason:
            rejected.append({"episode": episode, "reason": reason, **detail})
            continue
        seen.add(key)
        item = {
            **episode,
            "steps": checked_steps,
            "label_type": "recorded" if all_equivalent else "corrected",
        }
        if all_equivalent and _trajectory_success(trajectory) and not has_source_error:
            item["validation_status"] = "accepted"
            item["validation_basis"] = "complete_successful_recorded_episode"
            accepted.append(item)
        else:
            item["validation_status"] = "needs_replay"
            item["validation_basis"] = "complete_corrected_or_unverified_episode"
            needs_replay.append(item)
    return {"accepted": accepted, "needs_replay": needs_replay, "rejected": rejected}


def _json_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _failure_record(exc: Exception, *, stage: str) -> dict[str, str]:
    """Serialize a model/tool failure so retention can fail closed per call."""
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def run_retention_tools(
    model: ModelClient,
    *,
    trajectory: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    agent_system: str,
    allowed_actions: set[str],
    action_validator: Callable[[dict[str, Any]], Any] | None = None,
    audit: bool = True,
) -> dict[str, Any]:
    """Run the controller and its independently grounded retention tools."""
    memory_bank[:] = normalize_memory_bank(memory_bank)
    memory_snapshot = deepcopy(memory_bank)
    try:
        controller_reply = model.json_chat(
            system=RETENTION_CONTROLLER_SYSTEM,
            user=_json_prompt(
                {
                    "trajectory": trajectory_for_controller(trajectory),
                    "current_memory": memory_for_agent(memory_snapshot),
                }
            ),
        )
    except Exception as exc:
        return {
            "protocol": PROTOCOL_VERSION,
            "controller": {
                "decision": {"tool_calls": []},
                "error": _failure_record(exc, stage="controller"),
            },
            "tools": {},
            "artifact_choice": "neither",
            "controller_choice": "neither",
        }
    controller = normalize_controller_decision(controller_reply.parsed)
    result: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "controller": {"decision": controller, **_reply_record(controller_reply)},
        "tools": {},
    }
    full_trajectory = trajectory_for_tool(trajectory)

    for call in controller["tool_calls"]:
        name = call["name"]
        common = {
            "controller_suggested_evidence_steps": call["arguments"][
                "evidence_steps"
            ],
            "controller_suggested_steps_are_not_evidence": True,
            "current_memory": memory_snapshot,
            "trajectory": full_trajectory,
        }
        stage = "draft"
        try:
            if name == MEMORY_TOOL:
                draft_reply = model.json_chat(
                    system=MEMORY_EDITOR_SYSTEM, user=_json_prompt(common)
                )
                draft = normalize_memory_operations(draft_reply.parsed)
                audit_record = None
                final = draft
                if audit and draft:
                    stage = "audit"
                    audit_reply = model.json_chat(
                        system=MEMORY_AUDIT_SYSTEM,
                        user=_json_prompt({**common, "proposed_operations": draft}),
                    )
                    final = normalize_memory_operations(audit_reply.parsed)
                    audit_record = _reply_record(audit_reply)
                stage = "application"
                application = apply_memory_operations(
                    memory_bank, final, trajectory=trajectory
                )
                result["tools"][name] = {
                    "candidate": {
                        "operations": draft,
                        **_reply_record(draft_reply),
                    },
                    "audit": audit_record,
                    "final_operations": final,
                    "application": application,
                }
            elif name == SFT_TOOL:
                sft_payload = {
                    **common,
                    "browser_agent_action_specification": agent_system,
                }
                draft_reply = model.json_chat(
                    system=SFT_BUILDER_SYSTEM, user=_json_prompt(sft_payload)
                )
                draft = normalize_sft_episodes(draft_reply.parsed)
                audit_record = None
                final = draft
                require_action_note = "MUST include a short `note` field" in agent_system
                draft_validation = validate_sft_episodes(
                    draft,
                    trajectory=trajectory,
                    allowed_actions=allowed_actions,
                    action_validator=action_validator,
                    require_action_note=require_action_note,
                )
                # A builder-selected episode that exactly reproduces a successful,
                # error-free source trajectory is already grounded by the
                # environment verifier and deterministic action validation. A
                # second model pass tends to re-transcribe or second-guess it.
                # Reserve model auditing for genuinely corrected episodes.
                if audit and draft and not draft_validation["accepted"]:
                    stage = "audit"
                    audit_reply = model.json_chat(
                        system=SFT_AUDIT_SYSTEM,
                        user=_json_prompt({**sft_payload, "proposed_episodes": draft}),
                    )
                    final = normalize_sft_episodes(audit_reply.parsed)
                    audit_record = _reply_record(audit_reply)
                stage = "validation"
                validation = validate_sft_episodes(
                    final,
                    trajectory=trajectory,
                    allowed_actions=allowed_actions,
                    action_validator=action_validator,
                    require_action_note=require_action_note,
                )
                result["tools"][name] = {
                    "candidate": {
                        "episodes": draft,
                        **_reply_record(draft_reply),
                    },
                    "audit": audit_record,
                    "audit_skipped_reason": (
                        "complete_successful_recorded_episode_hard_validated"
                        if audit and draft_validation["accepted"]
                        else None
                    ),
                    "final_episodes": final,
                    "validation": validation,
                }
        except Exception as exc:
            # A malformed or truncated model response must not terminate the
            # source collection. Fail closed: record it and retain no artifact
            # from this tool call.
            result["tools"][name] = {
                "error": _failure_record(exc, stage=stage),
                "final_operations": [] if name == MEMORY_TOOL else None,
                "application": {"applied": [], "rejected": []}
                if name == MEMORY_TOOL
                else None,
                "final_episodes": [] if name == SFT_TOOL else None,
                "validation": {
                    "accepted": [],
                    "needs_replay": [],
                    "rejected": [],
                }
                if name == SFT_TOOL
                else None,
            }

    memory_changed = bool(
        (result["tools"].get(MEMORY_TOOL) or {}).get("application", {}).get("applied")
    )
    sft_validation = (result["tools"].get(SFT_TOOL) or {}).get("validation", {})
    has_sft = bool(sft_validation.get("accepted") or sft_validation.get("needs_replay"))
    result["artifact_choice"] = (
        "both"
        if memory_changed and has_sft
        else "context_only"
        if memory_changed
        else "sft_only"
        if has_sft
        else "neither"
    )
    called = {call["name"] for call in controller["tool_calls"]}
    result["controller_choice"] = (
        "both"
        if called == {MEMORY_TOOL, SFT_TOOL}
        else "context_only"
        if MEMORY_TOOL in called
        else "sft_only"
        if SFT_TOOL in called
        else "neither"
    )
    return result
