#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--repeat", type=int, default=32)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")

    examples = [
        json.loads(line)
        for line in args.input_jsonl.read_text().splitlines()
        if line.strip()
    ]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for _ in range(args.repeat):
            for example in examples:
                record = {
                    "messages": [
                        {"role": "user", "content": example["q"]},
                        {"role": "assistant", "content": example["a"]},
                    ]
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "unique_examples": len(examples),
                "repeat": args.repeat,
                "training_records": len(examples) * args.repeat,
                "output": str(args.output_jsonl),
            }
        )
    )


if __name__ == "__main__":
    main()
