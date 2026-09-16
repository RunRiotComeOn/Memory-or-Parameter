"""Version 2 retention harness: the writer routes before it writes.

v1 forced every trajectory through both writers and only varied writing style.
Here the model returns one allocation decision -- memory / sft / both / neither --
together with whatever artifacts that decision requires, and the bank is built
sequentially so that duplicate topics, refine/replace, and a real `neither` all
exist.
"""

from __future__ import annotations

from typing import Any

from .memory_writer_harness import _strings, _task_identifiers, _text, _tokens
import json


ROUTES = ("memory", "sft", "both", "neither")
# A refine/replace whose new text shares less than this fraction of vocabulary
# with its target is not editing that topic -- the writer is simply targeting
# whatever happens to be in the bank, which collapses it to one entry.
REFINE_TOPIC_OVERLAP_MIN = 0.25
MEMORY_OPERATIONS = ("add", "refine", "replace")
GAP_TYPES = ("knowledge", "procedure", "both", "none")


def writes_memory(route: str) -> bool:
    return route in {"memory", "both"}


def writes_sft(route: str) -> bool:
    return route in {"sft", "both"}


def _evidence_steps(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            step
            for step in value
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0
        }
    )[:24]


def normalize_alloc_decision(value: Any) -> dict[str, Any]:
    """Normalize one writer response to the closed route + artifact action space."""
    if not isinstance(value, dict):
        return {"route": "invalid", "route_rationale": "non_object_response"}
    route = _text(value.get("route"), 32).lower()
    if route not in ROUTES:
        return {
            "route": "invalid",
            "route_rationale": _text(value.get("route_rationale"), 1_000),
        }
    gap_type = _text(value.get("gap_type"), 32).lower()
    decision: dict[str, Any] = {
        "route": route,
        "gap_type": gap_type if gap_type in GAP_TYPES else None,
        "route_rationale": _text(value.get("route_rationale"), 1_000),
        "memory_operation": None,
        "target_memory_id": None,
        "memory": None,
        "sft_plan": None,
    }
    if writes_memory(route):
        operation = _text(value.get("memory_operation"), 32).lower()
        decision["memory_operation"] = (
            operation if operation in MEMORY_OPERATIONS else None
        )
        decision["target_memory_id"] = _text(value.get("target_memory_id"), 128) or None
        raw = value.get("memory")
        if isinstance(raw, dict):
            confidence = raw.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                confidence = None
            elif not 0 <= float(confidence) <= 1:
                confidence = None
            else:
                confidence = float(confidence)
            decision["memory"] = {
                "content": _text(raw.get("content")),
                "scope": _text(raw.get("scope"), 1_000),
                "conditions": _strings(raw.get("conditions")),
                "exceptions": _strings(raw.get("exceptions")),
                "evidence_steps": _evidence_steps(raw.get("evidence_steps")),
                "confidence": confidence,
            }
    if writes_sft(route):
        raw = value.get("sft_plan")
        if isinstance(raw, dict):
            decision["sft_plan"] = {
                "repair_target": _text(raw.get("repair_target"), 2_000),
                "evidence_steps": _evidence_steps(raw.get("evidence_steps")),
            }
    return decision


def validate_alloc_decision(
    decision: dict[str, Any],
    trajectory: dict[str, Any],
    active_bank: list[dict[str, Any]],
) -> dict[str, Any]:
    """Gate each artifact branch independently.

    A `both` decision with an unusable SFT plan still has a usable memory, so
    the branches are accepted or rejected separately; the caller commits only
    the branches that passed.
    """
    route = decision.get("route")
    if route not in ROUTES:
        return {
            "route_valid": False,
            "memory": {"required": False, "accepted": False, "reasons": ["invalid_route"]},
            "sft": {"required": False, "accepted": False, "reasons": ["invalid_route"]},
        }
    steps = {
        step.get("index"): step
        for step in trajectory.get("steps", [])
        if isinstance(step, dict)
    }

    def evidence_reasons(evidence: list[int], *, authoritative: bool) -> list[str]:
        if not evidence:
            return ["missing_evidence_steps"]
        if any(index not in steps for index in evidence):
            return ["invalid_evidence_step"]
        # Memory must rest on authoritative evidence (tool results, user facts).
        # An SFT repair points at the assistant turns that went wrong, so the
        # same gate would reject exactly the right citations.
        if authoritative and not any(
            steps[index].get("role") in {"user", "tool"} for index in evidence
        ):
            return ["no_authoritative_message_in_evidence"]
        return []

    memory_reasons: list[str] = []
    if writes_memory(route):
        memory = decision.get("memory")
        operation = decision.get("memory_operation")
        if not isinstance(memory, dict):
            memory_reasons.append("missing")
        else:
            if not memory.get("content"):
                memory_reasons.append("missing_content")
            if not memory.get("scope"):
                memory_reasons.append("missing_scope")
            if memory.get("confidence") is None:
                memory_reasons.append("invalid_confidence")
            memory_reasons.extend(
                evidence_reasons(memory.get("evidence_steps") or [], authoritative=True)
            )
            rendered = json.dumps(
                {
                    "content": memory.get("content"),
                    "scope": memory.get("scope"),
                    "conditions": memory.get("conditions"),
                    "exceptions": memory.get("exceptions"),
                },
                ensure_ascii=False,
            ).casefold()
            leaked = sorted(
                identifier
                for identifier in _task_identifiers(trajectory)
                if identifier in rendered
            )
            if leaked:
                memory_reasons.append("task_identifier_leak:" + ",".join(leaked[:5]))
        if operation not in MEMORY_OPERATIONS:
            memory_reasons.append("invalid_operation")
        elif operation in {"refine", "replace"}:
            target = decision.get("target_memory_id")
            active_ids = {entry["id"] for entry in active_bank}
            if not target:
                memory_reasons.append("missing_target")
            elif target not in active_ids:
                memory_reasons.append("unknown_target")
        elif decision.get("target_memory_id"):
            memory_reasons.append("add_with_target")

    sft_reasons: list[str] = []
    if writes_sft(route):
        plan = decision.get("sft_plan")
        if not isinstance(plan, dict) or not plan.get("repair_target"):
            sft_reasons.append("missing_plan")
        else:
            sft_reasons.extend(
                evidence_reasons(plan.get("evidence_steps") or [], authoritative=False)
            )

    return {
        "route_valid": True,
        "memory": {
            "required": writes_memory(route),
            "accepted": writes_memory(route) and not memory_reasons,
            "reasons": memory_reasons,
        },
        "sft": {
            "required": writes_sft(route),
            "accepted": writes_sft(route) and not sft_reasons,
            "reasons": sft_reasons,
        },
    }


def apply_memory_operation(
    bank: list[dict[str, Any]],
    decision: dict[str, Any],
    *,
    entry_id: str,
    source_task_id: str,
    rubric_id: str,
) -> dict[str, Any]:
    """Commit one validated memory operation to the running bank."""
    memory = decision["memory"]
    operation = decision["memory_operation"]
    target_id = decision.get("target_memory_id")
    entry = {
        "id": entry_id,
        "scope": memory["scope"],
        "content": memory["content"],
        "conditions": memory.get("conditions") or [],
        "exceptions": memory.get("exceptions") or [],
        "status": "active",
        "operation": operation,
        "supersedes": target_id if operation in {"refine", "replace"} else None,
        "source_task_id": source_task_id,
        "rubric_id": rubric_id,
    }
    if operation in {"refine", "replace"}:
        for existing in bank:
            if existing["id"] == target_id:
                existing["status"] = "superseded"
                existing["superseded_by"] = entry_id
    bank.append(entry)
    return entry


def active_entries(bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in bank if entry.get("status") == "active"]


def render_bank(bank: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The view the writer sees: exactly what retrieval would render, plus ids."""
    return [
        {"id": entry["id"], "scope": entry["scope"], "content": entry["content"]}
        for entry in active_entries(bank)
    ]


def topic_overlap(memory: dict[str, Any], entry: dict[str, Any]) -> float:
    """Jaccard vocabulary overlap between a proposed edit and its target."""
    new = set(_tokens(f"{memory.get('scope', '')} {memory.get('content', '')}"))
    old = set(_tokens(f"{entry.get('scope', '')} {entry.get('content', '')}"))
    if not new or not old:
        return 0.0
    return len(new & old) / len(new | old)
