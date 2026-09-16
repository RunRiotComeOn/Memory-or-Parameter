#!/usr/bin/env python3
"""Compare held-out tau baseline and optional reference with a replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/nas04/yixuh/memory")
DATA = ROOT / "third_party/tau2-bench/data"


def score(simulations: list[dict]) -> dict:
    values = [
        float(sim["reward_info"]["reward"])
        for sim in simulations
        if isinstance(sim.get("reward_info"), dict)
        and sim["reward_info"].get("reward") is not None
    ]
    return {
        "tasks": len(simulations),
        "scored": len(values),
        "passed": sum(value == 1.0 for value in values),
        "pass_rate": sum(value == 1.0 for value in values) / len(values)
        if values
        else None,
        "infrastructure_errors": sum(
            sim.get("termination_reason") == "infrastructure_error"
            for sim in simulations
        ),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--replay-version", default="v1")
parser.add_argument("--reference-version")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
out = args.output or ROOT / f"tau_experiment/{args.replay_version}/replay_comparison.json"

comparison = {"domains": {}}
for domain in ("airline", "retail", "telecom"):
    split = json.loads(
        (DATA / f"tau2/domains/{domain}/split_tasks.json").read_text()
    )
    test_ids = set(split["test"])
    baseline = json.loads(
        (
            DATA
            / f"simulations/qwen35_base_{domain}_full_v1/results.json"
        ).read_text()
    )
    replay = json.loads(
        (
            DATA
            / f"simulations/qwen35_tau_sft_memory_{domain}_test_{args.replay_version}/results.json"
        ).read_text()
    )
    base_sims = [sim for sim in baseline["simulations"] if sim["task_id"] in test_ids]
    replay_sims = replay["simulations"]
    base = score(base_sims)
    trained_memory = score(replay_sims)
    comparison["domains"][domain] = {
        "baseline": base,
        "sft_memory": trained_memory,
        "absolute_pass_rate_change": trained_memory["pass_rate"]
        - base["pass_rate"],
    }
    if args.reference_version:
        reference = json.loads(
            (
                DATA
                / "simulations"
                / f"qwen35_tau_sft_memory_{domain}_test_"
                f"{args.reference_version}/results.json"
            ).read_text()
        )
        reference_trained_memory = score(reference["simulations"])
        comparison["domains"][domain]["reference_sft_memory"] = (
            reference_trained_memory
        )
        comparison["domains"][domain][
            "absolute_pass_rate_change_vs_reference"
        ] = trained_memory["pass_rate"] - reference_trained_memory["pass_rate"]

base_pass = sum(x["baseline"]["passed"] for x in comparison["domains"].values())
base_n = sum(x["baseline"]["scored"] for x in comparison["domains"].values())
replay_pass = sum(x["sft_memory"]["passed"] for x in comparison["domains"].values())
replay_n = sum(x["sft_memory"]["scored"] for x in comparison["domains"].values())
comparison["overall"] = {
    "baseline": {"passed": base_pass, "scored": base_n, "pass_rate": base_pass / base_n},
    "sft_memory": {
        "passed": replay_pass,
        "scored": replay_n,
        "pass_rate": replay_pass / replay_n,
    },
    "absolute_pass_rate_change": replay_pass / replay_n - base_pass / base_n,
}
if args.reference_version:
    reference_pass = sum(
        x["reference_sft_memory"]["passed"]
        for x in comparison["domains"].values()
    )
    reference_n = sum(
        x["reference_sft_memory"]["scored"]
        for x in comparison["domains"].values()
    )
    comparison["overall"]["reference_sft_memory"] = {
        "passed": reference_pass,
        "scored": reference_n,
        "pass_rate": reference_pass / reference_n,
    }
    comparison["overall"]["absolute_pass_rate_change_vs_reference"] = (
        replay_pass / replay_n - reference_pass / reference_n
    )
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
