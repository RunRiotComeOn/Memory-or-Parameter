#!/usr/bin/env python3
"""Wait for tau-bench repair runs, validate them, and merge them atomically."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/nas04/yixuh/memory")
SIMULATIONS = WORKSPACE / "third_party/tau2-bench/data/simulations"
LOG_DIR = WORKSPACE / "runtime_logs"
SPECS = {
    "airline": {
        "expected": 1,
        "base": "qwen35_base_airline_full_v1",
        "repair": "qwen35_base_airline_repair_v1",
    },
    "retail": {
        "expected": 39,
        "base": "qwen35_base_retail_full_v1",
        "repair": "qwen35_base_retail_repair_v1",
    },
}


def load_results(name: str) -> dict:
    path = SIMULATIONS / name / "results.json"
    return json.loads(path.read_text())


def reward(simulation: dict) -> float | None:
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, dict):
        return None
    value = reward_info.get("reward")
    return None if value is None else float(value)


def summarize(simulations: list[dict]) -> dict:
    rewards = [value for sim in simulations if (value := reward(sim)) is not None]
    terminations = Counter(sim.get("termination_reason") for sim in simulations)
    return {
        "total": len(simulations),
        "scored": len(rewards),
        "passed": sum(value == 1.0 for value in rewards),
        "pass_rate_scored": (
            sum(value == 1.0 for value in rewards) / len(rewards)
            if rewards
            else None
        ),
        "infrastructure_errors": terminations["infrastructure_error"],
        "terminations": dict(terminations),
    }


def atomic_dump(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=".repair_", dir=path.parent, delete=False
    ) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def repairs_ready() -> tuple[bool, dict[str, int]]:
    counts = {}
    for domain, spec in SPECS.items():
        try:
            counts[domain] = len(load_results(spec["repair"])["simulations"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            counts[domain] = 0
    return all(counts[d] >= SPECS[d]["expected"] for d in SPECS), counts


def validate_and_merge(domain: str, spec: dict) -> dict:
    base_name = spec["base"]
    repair_name = spec["repair"]
    base_results = load_results(base_name)
    repair_results = load_results(repair_name)
    base_simulations = base_results["simulations"]
    repair_simulations = repair_results["simulations"]

    if len(repair_simulations) != spec["expected"]:
        raise RuntimeError(
            f"{domain}: expected {spec['expected']} repairs, "
            f"found {len(repair_simulations)}"
        )

    repair_by_task = {sim["task_id"]: sim for sim in repair_simulations}
    if len(repair_by_task) != len(repair_simulations):
        raise RuntimeError(f"{domain}: duplicate task IDs in repair results")

    original_invalid = {
        sim["task_id"]
        for sim in base_simulations
        if sim.get("termination_reason") == "infrastructure_error"
    }
    if set(repair_by_task) != original_invalid:
        raise RuntimeError(
            f"{domain}: repair IDs do not match original infra-error IDs"
        )

    invalid_repairs = [
        sim["task_id"]
        for sim in repair_simulations
        if sim.get("termination_reason") == "infrastructure_error"
        or reward(sim) is None
    ]
    if invalid_repairs:
        raise RuntimeError(f"{domain}: invalid repairs: {invalid_repairs}")

    before = summarize(base_simulations)
    base_results["simulations"] = [
        repair_by_task.get(sim["task_id"], sim) for sim in base_simulations
    ]
    after = summarize(base_results["simulations"])
    if after["total"] != before["total"] or after["infrastructure_errors"]:
        raise RuntimeError(f"{domain}: merged result failed integrity checks")

    result_path = SIMULATIONS / base_name / "results.json"
    atomic_dump(result_path, base_results)
    return {
        "base_result": str(result_path),
        "repair_result": str(SIMULATIONS / repair_name / "results.json"),
        "replaced_task_ids": sorted(repair_by_task),
        "before": before,
        "after": after,
    }


def main() -> None:
    while True:
        ready, counts = repairs_ready()
        print(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "repair_counts": counts,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        if ready:
            break
        time.sleep(60)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domains": {
            domain: validate_and_merge(domain, spec)
            for domain, spec in SPECS.items()
        },
    }
    summary_path = LOG_DIR / "tau_repair_summary.json"
    atomic_dump(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
