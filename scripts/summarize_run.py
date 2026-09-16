#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted(Path("runs").glob("*/run_result.json"))
        if not candidates:
            raise SystemExit("No completed runs found")
        run_dir = candidates[-1].parent

    result = json.loads((run_dir / "run_result.json").read_text())
    print(f"run: {run_dir}")
    print(f"success: {result['successes']}/{result['total']}")
    print(f"choices: {json.dumps(result['choice_counts'], sort_keys=True)}")
    if "controller_choice_counts" in result:
        print(
            "controller choices: "
            f"{json.dumps(result['controller_choice_counts'], sort_keys=True)}"
        )
    print()
    for item in result["tasks"]:
        print(
            f"{item['task_id']}: success={item['success']} steps={item['steps']} "
            f"controller={item.get('controller_choice', 'legacy')} "
            f"choice={item['choice']} memory_after={item['memory_entries_after']}"
        )


if __name__ == "__main__":
    main()
