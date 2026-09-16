#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert model-selected chat SFT JSONL into VERL parquet input."
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_parquet", type=Path)
    parser.add_argument(
        "--repeat-to",
        type=int,
        default=0,
        help="Repeat the selected records cyclically to at least this many rows.",
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("input JSONL contains no records")
    original_records = len(records)
    if args.repeat_to > len(records):
        records = [records[index % len(records)] for index in range(args.repeat_to)]
    for index, record in enumerate(records):
        validation_status = record.get("validation_status")
        if validation_status is not None and validation_status != "accepted":
            raise ValueError(
                f"record {index} has validation_status={validation_status!r}; "
                "only accepted retention-tool examples may be trained"
            )
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"record {index} has no usable messages list")
        expected_roles = ["system"] + [
            role
            for _ in range((len(messages) - 1) // 2)
            for role in ("user", "assistant")
        ]
        if len(messages) % 2 == 0 or [
            message.get("role") for message in messages
        ] != expected_roles:
            raise ValueError(
                f"record {index} must contain system/(user/assistant)+ messages"
            )
        targets = []
        for message in messages[2::2]:
            try:
                target = json.loads(message["content"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"record {index} assistant target is not JSON"
                ) from exc
            if not isinstance(target, dict) or "action" not in target:
                raise ValueError(
                    f"record {index} assistant target is not an action object"
                )
            targets.append(target)
        if str(targets[-1].get("action", "")).lower() != "finish":
            raise ValueError(f"record {index} episode does not end with finish")
        if any(
            str(target.get("action", "")).lower() == "finish"
            for target in targets[:-1]
        ):
            raise ValueError(f"record {index} episode finishes before its final turn")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(args.output_parquet))

    unique_pairs = {
        tuple((message["role"], message["content"]) for message in record["messages"])
        for record in records
    }
    print(
        json.dumps(
            {
                "records": len(records),
                "original_records": original_records,
                "unique_pairs": len(unique_pairs),
                "output": str(args.output_parquet),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
