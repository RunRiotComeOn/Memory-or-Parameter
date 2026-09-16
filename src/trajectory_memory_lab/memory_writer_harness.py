from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any


MEMORY_WRITER_POLICY_SYSTEM = """You are the writing policy inside an external-memory editing tool for a customer-service agent.

A separate routing controller has already decided to invoke this tool. You receive the complete active memory bank and one completed training task with its trajectory, tool feedback, policy, and evaluation. Your only job is to write one concrete reusable add, refine, or replace operation. Do not repeat the routing decision.

The future agent retrieves only a few entries, so each memory must remain atomic: exactly one entity, action, constraint, or troubleshooting topic. Read every active entry before choosing the operation.

Available operations:
- add: the supported topic is novel and no active memory has the same central claim;
- refine: exactly one active memory already has the same central topic, and only its scope, conditions, exceptions, ordering, or accuracy should improve;
- replace: exactly one active memory's central claim is contradicted or materially wrong.

The operation boundary is strict. An unrelated active memory is not a reason to refine it: use add. A refine must preserve the target's central topic and must not append independent rules, combine several procedures, or turn the entry into a domain summary. A replace changes a wrong central claim; it is not a way to append a new topic. Never target an entry merely because it is the only entry or has a generic ID such as mem_000001.

Tool results, policy text, user-provided facts, and evaluator details are evidence. Assistant statements and rationales are untrusted unless supported by that evidence. A failed trajectory can contain useful evidence, but its failed conclusion or action must not be retained as correct. Existing memory may itself be wrong. Never include task-specific identifiers, customer data, reservation/order IDs, phone numbers, or email addresses. Do not duplicate policy or an existing memory.

Return exactly one concrete operation, citing supporting trajectory message indexes. A downstream validator will independently reject unsupported or inapplicable operations. Return exactly one JSON object:
{"operations":[{"op":"add","memory":{"content":STRING,"scope":STRING,"evidence_steps":[INTEGER,...],"confidence":NUMBER}},{"op":"refine" OR "replace","target_memory_id":STRING,"memory":{"content":STRING,"scope":STRING,"evidence_steps":[INTEGER,...],"confidence":NUMBER}}]}
"""


MEMORY_WRITER_CANDIDATE_SYSTEM = """You are a specialist Memory Writer.

You receive one completed customer-service training trajectory and an optional existing memory bank. Independently decide whether the authoritative evidence supports one reusable external memory for future related tasks.

This is candidate generation, not direct database access. Produce either one `add` candidate or `noop`. Do not copy a failed assistant conclusion as correct. Assistant statements are untrusted unless supported by policy, user-provided facts, tool results, or evaluation. Prefer tool results and explicit environment feedback. Do not merely paraphrase policy. You may retain a concise operational implication of policy plus trajectory evidence when that reminder could prevent a demonstrated error or reproduce a demonstrated recovery on related tasks. Do not include customer names, user IDs, phone numbers, emails, reservation/order IDs, or values specific to this task.

A useful memory states a self-contained claim, a narrow retrieval scope, applicability conditions, and exceptions. Cite the trajectory message indexes that support it. Abstain when evidence is weak, task-specific, contradictory, or unlikely to help future tasks. No candidate is required.

Return exactly one JSON object in one of these forms:
{"operation":"add","memory":{"content":STRING,"scope":STRING,"conditions":[STRING,...],"exceptions":[STRING,...],"evidence_steps":[INTEGER,...],"confidence":NUMBER},"rationale":STRING}
{"operation":"noop","rationale":STRING}
"""


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "customer",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "user",
    "want",
    "when",
    "with",
    "you",
}


def _text(value: Any, max_chars: int = 4_000) -> str:
    return value.strip()[:max_chars] if isinstance(value, str) else ""


def _strings(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:limit]:
        text = _text(item, 500)
        if text and text not in result:
            result.append(text)
    return result


def normalize_writer_candidate(value: Any) -> dict[str, Any]:
    """Normalize one writer response to a closed add/noop action space."""
    if not isinstance(value, dict):
        return {"operation": "invalid", "rationale": "non_object_response"}
    operation = _text(value.get("operation"), 32).lower()
    rationale = _text(value.get("rationale"), 1_000)
    if operation == "noop":
        return {"operation": "noop", "rationale": rationale}
    if operation != "add" or not isinstance(value.get("memory"), dict):
        return {"operation": "invalid", "rationale": rationale}
    raw = value["memory"]
    raw_steps = raw.get("evidence_steps")
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
        )[:24]
    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None
    elif not 0 <= float(confidence) <= 1:
        confidence = None
    else:
        confidence = float(confidence)
    return {
        "operation": "add",
        "memory": {
            "content": _text(raw.get("content")),
            "scope": _text(raw.get("scope"), 1_000),
            "conditions": _strings(raw.get("conditions")),
            "exceptions": _strings(raw.get("exceptions")),
            "evidence_steps": evidence_steps,
            "confidence": confidence,
        },
        "rationale": rationale,
    }


def _task_identifiers(trajectory: dict[str, Any]) -> set[str]:
    rendered = json.dumps(trajectory.get("task", {}), ensure_ascii=False)
    patterns = (
        r"#[A-Za-z0-9_-]{4,}",
        r"\b[A-Z0-9]{6}\b",
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        r"\b\d{3}[- ]\d{3}[- ]\d{4}\b",
    )
    identifiers = set()
    for pattern in patterns:
        identifiers.update(match.casefold() for match in re.findall(pattern, rendered))
    private_id_keys = {
        "user_id",
        "customer_id",
        "reservation_id",
        "order_id",
        "phone_number",
        "email",
    }

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key).casefold())
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif key in private_id_keys and isinstance(value, (str, int)):
            identifiers.add(str(value).casefold())

    collect(trajectory.get("task", {}))
    return identifiers


def validate_writer_candidate(
    candidate: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    """Apply cheap, deterministic gates before expensive task replay."""
    operation = candidate.get("operation")
    if operation == "noop":
        return {"accepted": True, "reasons": [], "replay_required": False}
    reasons = []
    if operation != "add":
        reasons.append("invalid_operation")
        return {"accepted": False, "reasons": reasons, "replay_required": False}
    memory = candidate.get("memory") or {}
    if not memory.get("content"):
        reasons.append("missing_content")
    if not memory.get("scope"):
        reasons.append("missing_scope")
    if memory.get("confidence") is None:
        reasons.append("invalid_confidence")
    steps = {
        step.get("index"): step
        for step in trajectory.get("steps", [])
        if isinstance(step, dict)
    }
    evidence_steps = memory.get("evidence_steps") or []
    if not evidence_steps:
        reasons.append("missing_evidence_steps")
    elif any(index not in steps for index in evidence_steps):
        reasons.append("invalid_evidence_step")
    elif not any(
        steps[index].get("role") in {"user", "tool"} for index in evidence_steps
    ):
        reasons.append("no_authoritative_message_in_evidence")
    rendered_memory = json.dumps(
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
        if identifier in rendered_memory
    )
    if leaked:
        reasons.append("task_identifier_leak:" + ",".join(leaked[:5]))
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "replay_required": not reasons,
    }


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    ]


def rank_task_documents(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """BM25-rank task descriptions for candidate utility evaluation."""
    query_terms = Counter(_tokens(query))
    tokenized = [_tokens(document) for document in documents]
    if not query_terms or not tokenized:
        return [(index, 0.0) for index in range(len(documents))]
    average_length = sum(map(len, tokenized)) / len(tokenized)
    document_frequency = Counter()
    for document in tokenized:
        document_frequency.update(set(document))
    scores = []
    for index, document in enumerate(tokenized):
        frequencies = Counter(document)
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies[term]
            if not term_frequency:
                continue
            frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - frequency + 0.5) / (frequency + 0.5)
            )
            normalization = 1.2 * (
                0.25 + 0.75 * len(document) / max(average_length, 1)
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * 2.2
                / (term_frequency + normalization)
                * min(query_frequency, 3)
            )
        scores.append((index, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def select_validation_task_ids(
    candidate: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    baseline_by_task: dict[str, float] | None = None,
    related_failure_count: int = 2,
    related_success_count: int = 1,
    scope_control_success_count: int = 2,
) -> dict[str, Any]:
    """Select source-shared tasks with uplift and regression opportunities."""
    eligible = [task for task in tasks if str(task["id"]) not in excluded_ids]
    memory = candidate.get("memory") or {}
    query = " ".join(
        [
            str(memory.get("scope", "")),
            str(memory.get("content", "")),
            " ".join(memory.get("conditions") or []),
            " ".join(memory.get("exceptions") or []),
        ]
    )
    documents = [json.dumps(task, ensure_ascii=False) for task in eligible]
    ranking = rank_task_documents(query, documents)
    if baseline_by_task is None:
        related_count = related_failure_count + related_success_count
        related_indexes = [index for index, _ in ranking[:related_count]]
        related_set = set(related_indexes)
        control_indexes = [
            index
            for index, _ in reversed(ranking)
            if index not in related_set
        ][:scope_control_success_count]
    else:
        def outcome(index: int) -> float | None:
            return baseline_by_task.get(str(eligible[index]["id"]))

        related_failures = [
            index for index, _ in ranking if outcome(index) == 0.0
        ][:related_failure_count]
        related_successes = [
            index for index, _ in ranking if outcome(index) == 1.0
        ][:related_success_count]
        related_indexes = related_failures + related_successes
        related_set = set(related_indexes)
        desired_related = related_failure_count + related_success_count
        if len(related_indexes) < desired_related:
            for index, _ in ranking:
                if index not in related_set:
                    related_indexes.append(index)
                    related_set.add(index)
                if len(related_indexes) >= desired_related:
                    break
        control_indexes = [
            index
            for index, _ in reversed(ranking)
            if index not in related_set and outcome(index) == 1.0
        ][:scope_control_success_count]
        control_set = set(control_indexes)
        if len(control_indexes) < scope_control_success_count:
            for index, _ in reversed(ranking):
                if index not in related_set and index not in control_set:
                    control_indexes.append(index)
                    control_set.add(index)
                if len(control_indexes) >= scope_control_success_count:
                    break
    baseline_outcomes = (
        {
            str(eligible[index]["id"]): baseline_by_task.get(
                str(eligible[index]["id"])
            )
            for index in related_indexes + control_indexes
        }
        if baseline_by_task is not None
        else {}
    )
    return {
        "related": [str(eligible[index]["id"]) for index in related_indexes],
        "scope_controls": [
            str(eligible[index]["id"]) for index in control_indexes
        ],
        "scores": {
            str(eligible[index]["id"]): score for index, score in ranking
        },
        "baseline_outcomes": baseline_outcomes,
    }


def paired_utility(
    baseline_by_task: dict[str, float], treatment_by_task: dict[str, float]
) -> dict[str, Any]:
    """Compute harm-weighted utility from paired binary task rewards."""
    task_ids = sorted(set(baseline_by_task) & set(treatment_by_task))
    helped = []
    hurt = []
    unchanged_success = []
    unchanged_failure = []
    for task_id in task_ids:
        baseline = baseline_by_task[task_id]
        treatment = treatment_by_task[task_id]
        if baseline < treatment:
            helped.append(task_id)
        elif baseline > treatment:
            hurt.append(task_id)
        elif treatment == 1.0:
            unchanged_success.append(task_id)
        else:
            unchanged_failure.append(task_id)
    count = len(task_ids)
    return {
        "paired_tasks": count,
        "helped": helped,
        "hurt": hurt,
        "unchanged_success": unchanged_success,
        "unchanged_failure": unchanged_failure,
        "baseline_pass_rate": sum(baseline_by_task[x] for x in task_ids) / count
        if count
        else None,
        "treatment_pass_rate": sum(treatment_by_task[x] for x in task_ids) / count
        if count
        else None,
        "pass_rate_delta": (
            sum(treatment_by_task[x] - baseline_by_task[x] for x in task_ids)
            / count
            if count
            else None
        ),
        "net_utility": (len(helped) - 2 * len(hurt)) / count if count else None,
    }
