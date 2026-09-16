#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_memory_lab.experiment import (
    _normalize_decision,
    _normalize_optional_string,
    _sft_agent_prompt,
)
from trajectory_memory_lab.model_client import _extract_json
from trajectory_memory_lab.prompts import AGENT_SYSTEM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir

    rows = []
    sft_records = []
    legacy_run = False
    legacy_qa_run = False
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for task_dir in sorted((run_dir / "tasks").iterdir()):
        trajectory = json.loads((task_dir / "trajectory.json").read_text())
        artifact = json.loads((task_dir / "artifact_decision.json").read_text())
        raw = _extract_json(artifact["model_content"])
        legacy = "sft_workflow" in raw and "sft" not in raw
        legacy_qa = (
            isinstance(raw.get("sft"), dict)
            and "a" in raw["sft"]
            and "examples" not in raw["sft"]
        )
        legacy_run = legacy_run or legacy
        legacy_qa_run = legacy_qa_run or legacy_qa
        if legacy:
            context = _normalize_optional_string(raw.get("context"))
            legacy_workflow = _normalize_optional_string(raw.get("sft_workflow"))
            decision = {"context": context, "sft_workflow": legacy_workflow}
            sft_selected = legacy_workflow is not None
        elif legacy_qa:
            context = _normalize_optional_string(raw.get("context"))
            answer = _normalize_optional_string(raw["sft"].get("a"))
            decision = {"context": context, "sft": {"a": answer} if answer else None}
            sft_selected = answer is not None
        else:
            decision = _normalize_decision(raw)
            sft_selected = decision["sft"] is not None
        choice = (
            "both"
            if decision["context"] and sft_selected
            else "context_only"
            if decision["context"]
            else "sft_only"
            if sft_selected
            else "neither"
        )
        malformed_literal_null = [
            key
            for key in ("context", "sft_workflow", "sft")
            if isinstance(raw.get(key), str) and raw[key].strip().lower() == "null"
        ]
        rows.append(
            {
                "task_id": trajectory["task"]["id"],
                "success": trajectory["evaluation"]["success"],
                "choice": choice,
                "decision": decision,
                "malformed_literal_null_fields": malformed_literal_null,
            }
        )
        if sft_selected and (legacy or legacy_qa):
            record = {
                "source_task_id": trajectory["task"]["id"],
                "context_before": trajectory["context_before"],
                "trajectory_path": str(
                    (task_dir / "trajectory.json").relative_to(run_dir)
                ),
                "model": "Qwen/Qwen3.5-35B-A3B",
            }
            if legacy:
                record.update(
                    {
                        "instruction": trajectory["task"]["instruction"],
                        "workflow": decision["sft_workflow"],
                    }
                )
            elif legacy_qa:
                record.update(
                    {
                        "q": trajectory["task"]["instruction"],
                        "a": decision["sft"]["a"],
                    }
                )
            sft_records.append(record)
        elif sft_selected:
            for example in decision["sft"]["examples"]:
                source_step = example["source_step"]
                if not 0 <= source_step < len(trajectory["steps"]):
                    continue
                source = trajectory["steps"][source_step]
                sft_records.append(
                    {
                        "messages": [
                            {"role": "system", "content": AGENT_SYSTEM},
                            {
                                "role": "user",
                                "content": _sft_agent_prompt(trajectory, source_step),
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    example["a"], ensure_ascii=False, separators=(",", ":")
                                ),
                            },
                        ],
                        "source_task_id": trajectory["task"]["id"],
                        "source_step": source_step,
                        "recorded_action": source["action"],
                        "target_action": example["a"],
                        "context_before": trajectory["context_before"],
                        "trajectory_path": str(
                            (task_dir / "trajectory.json").relative_to(run_dir)
                        ),
                        "model": "Qwen/Qwen3.5-35B-A3B",
                    }
                )
        all_usage = [step["model_usage"] for step in trajectory["steps"]]
        all_usage.append(artifact["model_usage"])
        for item in all_usage:
            for key in usage:
                usage[key] += item.get(key) or 0

    result = {
        "tasks": rows,
        "choice_counts": {
            name: sum(row["choice"] == name for row in rows)
            for name in ("both", "context_only", "sft_only", "neither")
        },
        "successes": sum(row["success"] for row in rows),
        "total": len(rows),
        "model_usage": usage,
    }
    (run_dir / "audited_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    output_name = (
        "sft_workflow_candidates.cleaned.jsonl"
        if legacy_run
        else "sft_examples.cleaned.jsonl"
        if legacy_qa_run
        else "sft_action_examples.cleaned.jsonl"
    )
    with (run_dir / output_name).open("w") as handle:
        for record in sft_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
