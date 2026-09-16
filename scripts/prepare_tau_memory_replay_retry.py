#!/usr/bin/env python3
"""Redirect only incomplete/infra-error memory treatments to a fresh seeded run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SIMULATIONS = ROOT / "third_party/tau2-bench/data/simulations"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def valid(item: dict[str, Any]) -> bool:
    path = SIMULATIONS / item["save_name"] / "results.json"
    if not path.exists():
        return False
    wanted = set(map(str, item["task_ids"]))
    simulations = {
        str(simulation["task_id"]): simulation
        for simulation in read_json(path)["simulations"]
        if str(simulation["task_id"]) in wanted
    }
    return len(simulations) == len(wanted) and all(
        simulation.get("termination_reason") != "infrastructure_error"
        and (simulation.get("reward_info") or {}).get("reward") is not None
        for simulation in simulations.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    path = args.experiment / "replay_manifest.json"
    manifest = read_json(path)
    original_manifest = json.loads(json.dumps(manifest))
    redirected = []
    for item in manifest:
        if valid(item):
            continue
        item["save_name"] = f"{item['save_name']}_retry{args.attempt}"
        item["seed"] = 20260820 + args.attempt
        redirected.append(item["candidate_id"])
    backup = args.experiment / f"replay_manifest_before_retry{args.attempt}.json"
    backup.write_text(json.dumps(original_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attempt": args.attempt, "redirected": redirected}, indent=2))


if __name__ == "__main__":
    main()
