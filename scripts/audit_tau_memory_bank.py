#!/usr/bin/env python3
"""Audit cumulative tau memory banks against sources and replay outcomes."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DOMAINS = ("airline", "retail", "telecom")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "customer", "do", "for", "from", "has", "have", "if", "in",
    "is", "it", "must", "of", "on", "or", "should", "that", "the",
    "their", "them", "they", "this", "to", "user", "when", "with", "you",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if token not in STOPWORDS
    ]


def cosine(left: str, right: str) -> float:
    a, b = Counter(tokens(left)), Counter(tokens(right))
    numerator = sum(a[key] * b[key] for key in a.keys() & b.keys())
    denominator = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values())
    )
    return numerator / denominator if denominator else 0.0


def duplicate_clusters(
    memories: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent = list(range(len(memories)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    pairs = []
    for left, first in enumerate(memories):
        for right in range(left + 1, len(memories)):
            second = memories[right]
            score = cosine(first.get("content", ""), second.get("content", ""))
            if score >= threshold:
                union(left, right)
                pairs.append(
                    {
                        "left": first.get("id"),
                        "right": second.get("id"),
                        "cosine": round(score, 4),
                    }
                )
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, memory in enumerate(memories):
        groups[find(index)].append(memory)
    clusters = [
        {
            "memory_ids": [memory.get("id") for memory in group],
            "scopes": sorted({memory.get("scope", "") for memory in group}),
            "size": len(group),
        }
        for group in groups.values()
        if len(group) > 1
    ]
    clusters.sort(key=lambda item: (-item["size"], item["memory_ids"]))
    pairs.sort(key=lambda item: (-item["cosine"], item["left"], item["right"]))
    return clusters, pairs


def retrieve(
    entries: list[dict[str, Any]], query: str, top_k: int = 3
) -> list[tuple[str, float]]:
    query_terms = Counter(tokens(query))
    if not entries or not query_terms:
        return []
    documents = [
        tokens(f"{entry.get('scope', '')} {entry.get('scope', '')} {entry.get('content', '')}")
        for entry in entries
    ]
    average_length = sum(map(len, documents)) / len(documents)
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    scored = []
    for entry, document in zip(entries, documents):
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
            length_normalization = 1.2 * (
                1 - 0.75 + 0.75 * len(document) / max(average_length, 1)
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * 2.2
                / (term_frequency + length_normalization)
                * min(query_frequency, 3)
            )
        if score > 0:
            scored.append((str(entry.get("id")), score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


def simulation_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(simulation["task_id"]): simulation
        for simulation in read_json(path).get("simulations", [])
    }


def retrieved_by_task(
    memories: list[dict[str, Any]], simulations: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for task_id, simulation in simulations.items():
        user_messages = []
        events = []
        for message in simulation.get("messages") or []:
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            user_messages.append(message["content"])
            ranked = retrieve(memories, "\n".join(user_messages), top_k=3)
            events.append(
                {
                    "turn_idx": message.get("turn_idx"),
                    "memory_ids": [memory_id for memory_id, _ in ranked],
                    "scores": [round(score, 4) for _, score in ranked],
                }
            )
        output[task_id] = {
            "events": events,
            "memory_ids": sorted(
                {memory_id for event in events for memory_id in event["memory_ids"]}
            ),
        }
    return output


def source_index(results_root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for domain in DOMAINS:
        result = read_json(results_root / f"qwen35_base_{domain}_full_v1/results.json")
        simulations = {
            str(item["task_id"]): item for item in result.get("simulations", [])
        }
        for task_id, simulation in simulations.items():
            output[f"tau2.{domain}.{task_id}"] = simulation
    return output


def evidence_audit(
    memory: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    roles = Counter()
    invalid = []
    source_rewards = {}
    excerpts = []
    for item in memory.get("evidence") or []:
        source_id = str(item.get("source_task_id"))
        step = item.get("step")
        simulation = sources.get(source_id)
        if simulation is None:
            invalid.append({"source_task_id": source_id, "step": step, "reason": "missing_source"})
            continue
        reward = (simulation.get("reward_info") or {}).get("reward")
        source_rewards[source_id] = reward
        messages = simulation.get("messages") or []
        if not isinstance(step, int) or step < 0 or step >= len(messages):
            invalid.append({"source_task_id": source_id, "step": step, "reason": "missing_step"})
            continue
        message = messages[step]
        role = str(message.get("role"))
        roles[role] += 1
        content = str(message.get("content") or "")
        excerpts.append(
            {
                "source_task_id": source_id,
                "step": step,
                "role": role,
                "content_excerpt": content[:700],
                "tool_calls": message.get("tool_calls"),
            }
        )
    return {
        "evidence_count": len(memory.get("evidence") or []),
        "roles": dict(roles),
        "invalid": invalid,
        "source_rewards": source_rewards,
        "all_sources_failed": bool(source_rewards)
        and all(reward != 1 for reward in source_rewards.values()),
        "excerpts": excerpts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-retention",
        type=Path,
        default=Path("tau_experiment/memory_writer_sft_cumulative_20260817"),
    )
    parser.add_argument(
        "--old-retention", type=Path, default=Path("tau_experiment/v1/retention")
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("tau_experiment/memory_writer_cumulative_replay_20260817"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tau_experiment/memory_writer_bank_audit_20260817")
    )
    parser.add_argument("--duplicate-threshold", type=float, default=0.72)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = read_json(args.replay / "summary.json")
    comparison = summary["comparisons"]["sft_writer_vs_base_writer"]
    sources = source_index(args.results_root)
    report: dict[str, Any] = {
        "protocol": "tau_memory_bank_static_outcome_audit_v1",
        "duplicate_threshold": args.duplicate_threshold,
        "replay_pass_rates": summary["arm_pass_rates"],
        "domains": {},
    }
    all_memory_records = []
    for domain in DOMAINS:
        new_memories = read_json(args.new_retention / f"memory_{domain}.json")
        old_memories = read_json(args.old_retention / f"memory_{domain}.json")
        clusters, pairs = duplicate_clusters(new_memories, args.duplicate_threshold)
        simulations = simulation_map(
            args.results_root
            / f"qwen35_baseagent_sft_writer_memory_{domain}_test_cumwriter_v1/results.json"
        )
        retrieval = retrieved_by_task(new_memories, simulations)
        helped = set(comparison["domains"][domain]["helped"])
        hurt = set(comparison["domains"][domain]["hurt"])
        exposure = defaultdict(lambda: {"helped_tasks": [], "hurt_tasks": [], "all_tasks": []})
        for task_id, selected in retrieval.items():
            for memory_id in selected["memory_ids"]:
                exposure[memory_id]["all_tasks"].append(task_id)
                if task_id in helped:
                    exposure[memory_id]["helped_tasks"].append(task_id)
                if task_id in hurt:
                    exposure[memory_id]["hurt_tasks"].append(task_id)
        memory_records = []
        clustered_ids = {
            memory_id for cluster in clusters for memory_id in cluster["memory_ids"]
        }
        for memory in new_memories:
            audit = evidence_audit(memory, sources)
            item = {
                "domain": domain,
                "id": memory.get("id"),
                "scope": memory.get("scope"),
                "content": memory.get("content"),
                "confidence": memory.get("confidence"),
                "version": memory.get("version"),
                "history_versions": len(memory.get("history") or []),
                "source_task_id": memory.get("source_task_id"),
                "content_tokens": len(tokens(memory.get("content", ""))),
                "in_duplicate_cluster": memory.get("id") in clustered_ids,
                "evidence_audit": audit,
                "retrieval_exposure": dict(exposure[memory.get("id")]),
            }
            item["risk_flags"] = [
                flag
                for flag, condition in (
                    ("semantic_duplicate", item["in_duplicate_cluster"]),
                    ("failed_source_only", audit["all_sources_failed"]),
                    ("invalid_evidence_reference", bool(audit["invalid"])),
                    ("retrieved_on_hurt_task", bool(item["retrieval_exposure"]["hurt_tasks"])),
                    ("very_long", item["content_tokens"] > 180),
                )
                if condition
            ]
            memory_records.append(item)
            all_memory_records.append(item)
        report["domains"][domain] = {
            "old_memory_count": len(old_memories),
            "new_memory_count": len(new_memories),
            "growth_multiple": round(len(new_memories) / max(len(old_memories), 1), 3),
            "duplicate_clusters": clusters,
            "duplicate_pairs": pairs,
            "memories_in_duplicate_clusters": len(clustered_ids),
            "helped_tasks": sorted(helped),
            "hurt_tasks": sorted(hurt),
            "retrieval_by_task": retrieval,
            "memories": memory_records,
        }
    report["headline"] = {
        "old_memory_count": sum(
            domain["old_memory_count"] for domain in report["domains"].values()
        ),
        "new_memory_count": len(all_memory_records),
        "duplicate_pairs": sum(
            len(domain["duplicate_pairs"]) for domain in report["domains"].values()
        ),
        "memories_in_duplicate_clusters": sum(
            domain["memories_in_duplicate_clusters"]
            for domain in report["domains"].values()
        ),
        "failed_source_only": sum(
            "failed_source_only" in item["risk_flags"] for item in all_memory_records
        ),
        "retrieved_on_hurt_task": sum(
            "retrieved_on_hurt_task" in item["risk_flags"] for item in all_memory_records
        ),
    }
    write_json(args.output / "audit.json", report)

    lines = [
        "# Tau cumulative memory-bank audit",
        "",
        f"- Old bank: {report['headline']['old_memory_count']} entries",
        f"- New SFT bank: {report['headline']['new_memory_count']} entries",
        f"- Similarity pairs >= {args.duplicate_threshold}: {report['headline']['duplicate_pairs']}",
        f"- Entries participating in a duplicate cluster: {report['headline']['memories_in_duplicate_clusters']}",
        f"- Entries derived only from failed source trajectories: {report['headline']['failed_source_only']}",
        f"- Entries retrieved on at least one SFT-vs-Base hurt task: {report['headline']['retrieved_on_hurt_task']}",
        "",
    ]
    for domain in DOMAINS:
        item = report["domains"][domain]
        lines.extend(
            [
                f"## {domain}",
                "",
                f"- Bank growth: {item['old_memory_count']} -> {item['new_memory_count']} ({item['growth_multiple']}x)",
                f"- Duplicate clusters: {len(item['duplicate_clusters'])}; entries in clusters: {item['memories_in_duplicate_clusters']}",
                f"- Replay helped/hurt vs Base writer: {len(item['helped_tasks'])}/{len(item['hurt_tasks'])}",
                "",
            ]
        )
        for cluster in item["duplicate_clusters"]:
            lines.append("- duplicate cluster: " + ", ".join(cluster["memory_ids"]))
        lines.append("")
    (args.output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["headline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
