#!/usr/bin/env python3
"""Build cumulative tau memory edits with `codex exec` and GPT-5.6."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from run_gemini_tau_memory_teacher import (
    DOMAINS,
    TEACHER_AUDIT_SYSTEM,
    TEACHER_SYSTEM,
    compact_trajectory,
    task_manifest,
    validate_operations,
)
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    memory_for_agent,
    normalize_memory_bank,
    normalize_memory_operations,
)
from trajectory_memory_lab.storage import write_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/memory_writer_response.schema.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def codex_generate(
    *,
    model: str,
    reasoning_effort: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "Do not inspect the workspace and do not invoke any tools. All authoritative inputs "
        "are included below. Internally check the proposed edit against every rule before "
        "returning the final answer. Return only the JSON object required by the output schema.\n\n"
        "<writer_policy>\n"
        + TEACHER_SYSTEM
        + "\n</writer_policy>\n\n<audit_policy>\n"
        + TEACHER_AUDIT_SYSTEM
        + "\n</audit_policy>\n\n<input>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</input>"
    )
    with tempfile.TemporaryDirectory(prefix="codex-memory-writer-") as temp_dir:
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
                f"stdout_tail={completed.stdout[-4000:]!r}; "
                f"stderr_tail={completed.stderr[-2000:]!r}"
            )
        if not output.exists():
            raise RuntimeError("codex exec did not write --output-last-message")
        parsed = json.loads(output.read_text(encoding="utf-8"))
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        usage = [event for event in events if event.get("type") == "turn.completed"]
        return parsed, {
            "turn_completed": usage[-1] if usage else None,
            "stderr_tail": completed.stderr[-1000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("third_party/tau2-bench/data/simulations"),
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--limit-per-domain", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, metadata = task_manifest(
        args.results_root, args.seed, args.limit_per_domain
    )
    if args.limit > 0:
        manifest = manifest[: args.limit]
    write_json(
        args.output / "manifest.json",
        {
            "protocol": "tau_codex_memory_teacher_v1",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "seed": args.seed,
            "split": "train",
            "domains": metadata,
            "total": len(manifest),
            "prompt": TEACHER_SYSTEM,
            "audit_prompt": TEACHER_AUDIT_SYSTEM,
        },
    )
    banks = {}
    for domain in DOMAINS:
        path = args.output / f"memory_{domain}.json"
        banks[domain] = normalize_memory_bank(read_json(path)) if path.exists() else []

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
        payload = {
            "current_memory": memory_for_agent(banks[domain], max_chars=1_000_000),
            "trajectory": trajectory,
        }
        try:
            parsed, usage = codex_generate(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                payload=payload,
                timeout=args.timeout,
            )
            candidate = normalize_memory_operations(parsed)
            validated, deterministic_rejections = validate_operations(
                candidate, trajectory
            )
            application = apply_memory_operations(
                banks[domain], validated, trajectory=trajectory
            )
            record = {
                "protocol": "tau_codex_memory_teacher_v1",
                "source_task_id": trajectory["source_task_id"],
                "controller": {
                    "tool_calls": [{"name": "edit_memory", "arguments": {}}],
                    "teacher_dataset_routing_only": True,
                },
                "tools": {
                    "edit_memory": {
                        "candidate": candidate,
                        "audit": {"performed_within_codex_turn": True},
                        "final_operations": validated,
                        "deterministic_rejections": deterministic_rejections,
                        "application": application,
                        "teacher_call_assessment": parsed.get("call_assessment", ""),
                        "teacher_usage": usage,
                    }
                },
                "controller_choice": "context_only",
                "artifact_choice": "context_only" if application["applied"] else "neither",
            }
        except Exception as exc:
            record = {
                "protocol": "tau_codex_memory_teacher_v1",
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
        for operation in decision.get("tools", {})
        .get("edit_memory", {})
        .get("final_operations", [])
    )
    summary = {
        "protocol": "tau_codex_memory_teacher_v1",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "tasks": len(decisions),
        "errors": sum(
            "error" in decision.get("tools", {}).get("edit_memory", {})
            for decision in decisions
        ),
        "calls_with_edits": sum(decision["artifact_choice"] == "context_only" for decision in decisions),
        "operations": dict(operations),
        "memory_entries": {domain: len(banks[domain]) for domain in DOMAINS},
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
