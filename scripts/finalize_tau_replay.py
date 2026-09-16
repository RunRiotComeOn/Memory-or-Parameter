#!/usr/bin/env python3
"""Monitor the resumed tau replay and build the baseline comparison on completion."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/nas04/yixuh/memory")
EXPERIMENT = ROOT / "tau_experiment/v1"
SIMULATIONS = ROOT / "third_party/tau2-bench/data/simulations"
TOTALS = {"airline": 20, "retail": 40, "telecom": 40}
POLL_SECONDS = 60


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_exists(domain: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"tau_replay_{domain}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def status(domain: str) -> dict[str, int]:
    path = SIMULATIONS / f"qwen35_tau_sft_memory_{domain}_test_v1/results.json"
    try:
        simulations = json.loads(path.read_text())["simulations"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        simulations = []
    return {
        "completed": len(simulations),
        "scored": sum(
            isinstance(sim.get("reward_info"), dict)
            and sim["reward_info"].get("reward") is not None
            for sim in simulations
        ),
        "infrastructure_errors": sum(
            sim.get("termination_reason") == "infrastructure_error"
            for sim in simulations
        ),
    }


EXPERIMENT.mkdir(parents=True, exist_ok=True)
with (EXPERIMENT / "stages.log").open("a") as handle:
    handle.write(f"{timestamp()} lora_server:ready\n")
    handle.write(f"{timestamp()} replay:start\n")

while True:
    statuses = {domain: status(domain) for domain in TOTALS}
    record = {"timestamp": timestamp(), "replay": statuses}
    print(json.dumps(record, separators=(",", ":")), flush=True)
    with (EXPERIMENT / "replay_progress.log").open("a") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    for domain, total in TOTALS.items():
        current = statuses[domain]
        if current["infrastructure_errors"]:
            raise SystemExit(
                f"{domain} has {current['infrastructure_errors']} infrastructure errors"
            )
        if current["completed"] < total and not session_exists(domain):
            raise SystemExit(
                f"{domain} replay exited at {current['completed']}/{total}"
            )

    if all(statuses[domain]["completed"] == total for domain, total in TOTALS.items()):
        if not all(statuses[domain]["scored"] == total for domain, total in TOTALS.items()):
            raise SystemExit(f"replay completed with missing scores: {statuses}")
        break
    time.sleep(POLL_SECONDS)

with (EXPERIMENT / "stages.log").open("a") as handle:
    handle.write(f"{timestamp()} replay:complete\n")

comparison_log = ROOT / "runtime_logs/tau_pipeline/replay_comparison.log"
with comparison_log.open("w") as handle:
    subprocess.run(
        ["python", str(ROOT / "scripts/summarize_tau_replay.py")],
        stdout=handle,
        stderr=subprocess.STDOUT,
        check=True,
    )

with (EXPERIMENT / "stages.log").open("a") as handle:
    handle.write(f"{timestamp()} pipeline:complete\n")
print((EXPERIMENT / "replay_comparison.json").read_text(), flush=True)
