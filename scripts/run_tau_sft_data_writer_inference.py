#!/usr/bin/env python3
"""Generate complete trajectory candidates with a trained SFT-data writer."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.tau_sft_data_writer import validate_writer_output


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=1301)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in args.inputs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def run_one(ordinal_row: tuple[int, dict[str, Any]]) -> tuple[int, str]:
        ordinal, row = ordinal_row
        task_dir = args.output / "tasks" / f"{ordinal:03d}_{row['domain']}"
        path = task_dir / "candidate.json"
        if path.exists():
            existing = read_json(path)
            if existing.get("status") == "prediction_ready":
                return ordinal, f"resume-skip {row['domain']}.{row['source_task_id']}"
        client = ModelClient(
            base_url=args.base_url,
            api_key="EMPTY",
            model=args.model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed + ordinal,
            enable_thinking=False,
            timeout=args.timeout,
        )
        reply = None
        try:
            reply = client.json_chat(
                system=row["system"],
                user=json.dumps(
                    row["student_input"], ensure_ascii=False, separators=(",", ":")
                ),
            )
            allowed = {
                schema["function"]["name"]
                for schema in row["student_input"]["tool_schemas"]
            }
            prediction = validate_writer_output(reply.parsed, allowed)
            record = {
                "protocol": "tau_sft_data_writer_prediction_v1",
                "domain": row["domain"],
                "source_task_id": row["source_task_id"],
                "source_reward": row.get("source_reward"),
                "student_input": row["student_input"],
                "candidate": prediction,
                "prediction": prediction,
                "usage": reply.usage,
                "status": "prediction_ready",
            }
        except Exception as exc:
            record = {
                "protocol": "tau_sft_data_writer_prediction_v1",
                "domain": row["domain"],
                "source_task_id": row["source_task_id"],
                "source_reward": row.get("source_reward"),
                "error": repr(exc),
                "raw_prediction": reply.parsed if reply is not None else None,
                "usage": reply.usage if reply is not None else None,
                "status": "error",
            }
        write_json(path, record)
        return ordinal, f"{row['domain']}.{row['source_task_id']} status={record['status']}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [pool.submit(run_one, item) for item in enumerate(rows)]
        for future in concurrent.futures.as_completed(futures):
            ordinal, message = future.result()
            print(f"[{ordinal + 1}/{len(rows)}] {message}", flush=True)

    records = [
        read_json(path)
        for path in sorted((args.output / "tasks").glob("*/candidate.json"))
    ]
    summary = {
        "protocol": "tau_sft_data_writer_prediction_v1",
        "model": args.model,
        "inputs": len(rows),
        "records": len(records),
        "prediction_ready": sum(item.get("status") == "prediction_ready" for item in records),
        "errors": sum(item.get("status") == "error" for item in records),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
