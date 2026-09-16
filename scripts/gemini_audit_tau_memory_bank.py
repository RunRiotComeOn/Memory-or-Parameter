#!/usr/bin/env python3
"""Use Gemini 3 Flash to semantically audit a tau cumulative memory bank."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


DOMAINS = ("airline", "retail", "telecom")
ISSUES = (
    "unsupported",
    "duplicate",
    "overgeneralized",
    "task_specific",
    "procedural_mistake",
    "low_value",
    "none",
)

SYSTEM = """You are auditing an external memory bank used by a customer-service agent.

Judge each memory against the supplied domain policy and its cited source evidence. Assistant
claims are not authoritative. Tool results, user-provided facts, policy, and evaluator details
are authoritative. A failed source trajectory may still support a tool fact, but it does not
support the failed assistant's conclusion.

The memory bank is retrieved with lexical BM25 top-3. Redundant, broad, and low-value entries
can displace useful entries, so compactness is part of correctness. A good entry is atomic,
reusable, evidence-grounded, narrowly scoped, and adds marginal information not already covered.

Verdicts:
- keep: useful and accurate as written;
- refine: central idea is useful but wording, scope, conditions, or exceptions need repair;
- merge: substantially duplicates another entry; name the best target memory ID;
- remove: unsupported, incorrect, task-specific, or too low-value to retain.

Audit every ID listed in audit_memory_ids exactly once. Use the complete memory bank to find
duplicates and merge targets, but do not emit audits for IDs outside audit_memory_ids. Do not
assume high confidence means correct."""


SCHEMA = {
    "type": "object",
    "properties": {
        "audits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["keep", "refine", "merge", "remove"],
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ISSUES)},
                    },
                    "reason": {"type": "string"},
                    "merge_target_id": {"type": ["string", "null"]},
                    "suggested_content": {"type": ["string", "null"]},
                    "suggested_scope": {"type": ["string", "null"]},
                },
                "required": [
                    "memory_id",
                    "verdict",
                    "issues",
                    "reason",
                    "merge_target_id",
                    "suggested_content",
                    "suggested_scope",
                ],
            },
        }
    },
    "required": ["audits"],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def generate(client: genai.Client, model: str, prompt: str) -> tuple[dict[str, Any], Any]:
    error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA,
                ),
            )
            return json.loads(response.text), response.usage_metadata
        except Exception as exc:  # API failures need bounded retry.
            error = exc
            if attempt == 4:
                break
            time.sleep(min(30, 2 ** attempt))
    assert error is not None
    raise error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-audit",
        type=Path,
        default=Path("tau_experiment/memory_writer_bank_audit_20260817/audit.json"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tau_experiment/memory_writer_bank_audit_20260817/gemini"),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/nas04/yixuh/.config/continual-memory/gemini_api_key"),
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Gemini API key file is empty")
    client = genai.Client(api_key=key)
    static = read_json(args.static_audit)
    manifest = {
        "protocol": "gemini_semantic_memory_audit_v1",
        "model": args.model,
        "static_audit": str(args.static_audit.resolve()),
        "domains": {},
    }
    all_audits = []
    for domain in DOMAINS:
        output_path = args.output / f"audit_{domain}.json"
        if output_path.exists():
            result = read_json(output_path)
            print(f"[{domain}] cached", flush=True)
        else:
            source_result = read_json(
                args.results_root / f"qwen35_base_{domain}_full_v1/results.json"
            )
            memories = []
            for memory in static["domains"][domain]["memories"]:
                evidence = memory["evidence_audit"]
                memories.append(
                    {
                        "memory_id": memory["id"],
                        "scope": memory["scope"],
                        "content": memory["content"],
                        "confidence": memory["confidence"],
                        "source_rewards": evidence["source_rewards"],
                        "cited_evidence": evidence["excerpts"],
                        "static_duplicate_cluster": memory["in_duplicate_cluster"],
                    }
                )
            expected = {memory["memory_id"] for memory in memories}
            combined = []
            usages = []
            for start in range(0, len(memories), args.batch_size):
                batch = memories[start : start + args.batch_size]
                batch_number = start // args.batch_size
                part_path = args.output / f"audit_{domain}_part_{batch_number:02d}.json"
                if part_path.exists():
                    part = read_json(part_path)
                    print(f"[{domain} part {batch_number + 1}] cached", flush=True)
                else:
                    prompt = json.dumps(
                        {
                            "domain": domain,
                            "domain_policy": source_result["info"]["environment_info"]["policy"],
                            "complete_memory_bank": memories,
                            "audit_memory_ids": [item["memory_id"] for item in batch],
                        },
                        ensure_ascii=False,
                    )
                    parsed, usage = generate(client, args.model, prompt)
                    part = {"audits": parsed["audits"], "usage": str(usage)}
                    write_json(part_path, part)
                    print(
                        f"[{domain} part {batch_number + 1}] complete: "
                        f"{len(part['audits'])}",
                        flush=True,
                    )
                batch_expected = {item["memory_id"] for item in batch}
                received = {item.get("memory_id") for item in part.get("audits", [])}
                if received != batch_expected or len(part.get("audits", [])) != len(batch_expected):
                    raise RuntimeError(
                        f"{domain} part {batch_number}: incomplete Gemini audit; "
                        f"missing={sorted(batch_expected - received)} "
                        f"extra={sorted(received - batch_expected)}"
                    )
                combined.extend(part["audits"])
                usages.append(part.get("usage"))
            received = {item.get("memory_id") for item in combined}
            if received != expected or len(combined) != len(expected):
                raise RuntimeError(f"{domain}: combined audit is incomplete")
            result = {
                "domain": domain,
                "model": args.model,
                "usage": usages,
                "audits": combined,
            }
            write_json(output_path, result)
            print(f"[{domain}] complete: {len(result['audits'])}", flush=True)
        all_audits.extend({"domain": domain, **item} for item in result["audits"])
        manifest["domains"][domain] = {
            "entries": len(result["audits"]),
            "verdicts": dict(Counter(item["verdict"] for item in result["audits"])),
            "issues": dict(
                Counter(issue for item in result["audits"] for issue in item["issues"])
            ),
        }
    manifest["total"] = {
        "entries": len(all_audits),
        "verdicts": dict(Counter(item["verdict"] for item in all_audits)),
        "issues": dict(Counter(issue for item in all_audits for issue in item["issues"])),
    }
    write_json(args.output / "summary.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
