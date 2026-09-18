#!/usr/bin/env python3
"""Compare base and SFT writers on held-out router-positive inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from trajectory_memory_lab.memory_writer_harness import validate_writer_candidate
from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    normalize_memory_bank,
    normalize_memory_operations,
)
from trajectory_memory_lab.storage import write_json


def _class(operations: list[dict]) -> str:
    kinds = {operation["op"] for operation in operations}
    if not kinds:
        return "noop"
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _operation_signature(value: Any) -> list[tuple[str, str | None]]:
    """Compare operation choice without requiring identical generated prose."""
    if not isinstance(value, dict):
        return []
    return [
        (
            operation["op"],
            operation.get("target_memory_id")
            if operation["op"] in {"refine", "replace"}
            else None,
        )
        for operation in normalize_memory_operations(value)
    ]


def _schema_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"operations"}:
        return False
    operations = value["operations"]
    if not isinstance(operations, list) or not 1 <= len(operations) <= 8:
        return False
    for operation in operations:
        if not isinstance(operation, dict):
            return False
        op = operation.get("op")
        if op not in {"add", "refine", "replace"}:
            return False
        if op in {"refine", "replace"} and not isinstance(
            operation.get("target_memory_id"), str
        ):
            return False
        memory = operation.get("memory")
        if not isinstance(memory, dict):
            return False
        if not isinstance(memory.get("content"), str) or not memory["content"].strip():
            return False
        if not isinstance(memory.get("scope"), str) or not memory["scope"].strip():
            return False
        steps = memory.get("evidence_steps")
        if not isinstance(steps, list) or not steps or any(
            not isinstance(step, int) or isinstance(step, bool) or step < 0
            for step in steps
        ):
            return False
        confidence = memory.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            return False
    return True


def _executable(value: dict, writer_input: dict) -> tuple[bool, list[str]]:
    operations = normalize_memory_operations(value)
    if not operations:
        return False, ["writer_returned_no_operation"]
    reasons = []
    for operation in operations:
        candidate = {
            "operation": "add",
            "memory": {
                **operation["memory"],
                "conditions": [],
                "exceptions": [],
            },
        }
        validation = validate_writer_candidate(candidate, writer_input["trajectory"])
        reasons.extend(validation["reasons"])
    if reasons:
        return False, reasons
    bank = normalize_memory_bank(deepcopy(writer_input["current_memory"]))
    application = apply_memory_operations(
        bank, operations, trajectory=writer_input["trajectory"]
    )
    reasons.extend(item["reason"] for item in application["rejected"])
    return len(application["applied"]) == len(operations), reasons


def _run_one(model: str, row: dict, args: argparse.Namespace) -> dict:
    client = ModelClient(
        base_url=args.base_url,
        api_key="EMPTY",
        model=model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=False,
        timeout=args.timeout,
    )
    target = json.loads(row["messages"][2]["content"])
    writer_input = json.loads(row["messages"][1]["content"])
    result = {
        "model": model,
        "source_task_id": row["source_task_id"],
        "domain": row["domain"],
        "target": target,
        "target_class": _class(normalize_memory_operations(target)),
    }
    try:
        reply = client.json_chat(
            system=row["messages"][0]["content"],
            user=row["messages"][1]["content"],
        )
        schema_valid = _schema_valid(reply.parsed)
        operations = normalize_memory_operations(reply.parsed) if schema_valid else []
        executable, reasons = (
            _executable(reply.parsed, writer_input)
            if schema_valid
            else (False, ["invalid_schema"])
        )
        prediction_class = _class(operations) if schema_valid else "invalid"
        result.update(
            {
                "content": reply.content,
                "parsed": reply.parsed,
                "usage": reply.usage,
                "json_valid": True,
                "schema_valid": schema_valid,
                "executable": executable,
                "validation_reasons": reasons,
                "prediction_class": prediction_class,
                "class_correct": prediction_class == result["target_class"],
                "exact_match": reply.parsed == target,
            }
        )
    except Exception as exc:
        result.update(
            {
                "error": repr(exc),
                "json_valid": False,
                "schema_valid": False,
                "executable": False,
                "validation_reasons": ["request_or_json_error"],
                "prediction_class": "invalid",
                "class_correct": False,
                "exact_match": False,
            }
        )
    return result


def _summary(records: list[dict], models: list[str]) -> dict:
    result = {"examples": len(records) // len(models), "models": {}}
    for model in models:
        items = [record for record in records if record["model"] == model]
        count = len(items)
        target_classes = Counter(item["target_class"] for item in items)
        prediction_classes = Counter(item["prediction_class"] for item in items)
        class_recall = {
            target_class: sum(
                item["prediction_class"] == target_class
                for item in items
                if item["target_class"] == target_class
            )
            / target_count
            for target_class, target_count in sorted(target_classes.items())
        }
        confusion = {
            target_class: dict(
                Counter(
                    item["prediction_class"]
                    for item in items
                    if item["target_class"] == target_class
                )
            )
            for target_class in sorted(target_classes)
        }
        positive_items = [item for item in items if item["target_class"] != "noop"]
        result["models"][model] = {
            "count": count,
            "json_valid": sum(item["json_valid"] for item in items) / count,
            "schema_valid": sum(item["schema_valid"] for item in items) / count,
            "executable": sum(item["executable"] for item in items) / count,
            "class_accuracy": sum(item["class_correct"] for item in items) / count,
            "operation_signature_accuracy": sum(
                _operation_signature(item.get("parsed"))
                == _operation_signature(item["target"])
                for item in items
            )
            / count,
            "positive_class_recall": sum(
                item["class_correct"] for item in positive_items
            )
            / len(positive_items),
            "exact_match": sum(item["exact_match"] for item in items) / count,
            "class_recall": class_recall,
            "confusion_matrix": confusion,
            "prediction_classes": dict(prediction_classes),
            "target_classes": dict(target_classes),
            "errors": sum("error" in item for item in items),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training/tau_memory_writer_sft_v2/validation.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tau_experiment/memory_writer_sft_eval_v2"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen35-tau", "qwen35-tau-writer-sft-v2"],
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prediction_path = args.output / "predictions.jsonl"
    cached = {}
    if prediction_path.exists():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cached[(record["model"], record["source_task_id"])] = record
    work = [
        (model, row)
        for model in args.models
        for row in rows
        if (model, row["source_task_id"]) not in cached
    ]
    with prediction_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = {
                executor.submit(_run_one, model, row, args): (model, row)
                for model, row in work
            }
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                cached[(record["model"], record["source_task_id"])] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(work)}] {record['model']} "
                    f"{record['source_task_id']} pred={record['prediction_class']} "
                    f"target={record['target_class']} executable={record['executable']}",
                    flush=True,
                )
    records = [cached[(model, row["source_task_id"])] for model in args.models for row in rows]
    summary = _summary(records, args.models)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
