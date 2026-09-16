#!/usr/bin/env python3
"""Select an exact number of unique accepted SFT episodes from JSONL inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    selected: list[dict] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("validation_status") != "accepted":
                continue
            key = (
                str(item.get("source_task_id")),
                tuple(item.get("source_steps", [])),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) == args.count:
                break
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError(
            f"requested {args.count} unique accepted episodes, found {len(selected)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "count": len(selected),
                "tasks": [item["source_task_id"] for item in selected],
                "turns": [len(item["source_steps"]) for item in selected],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
