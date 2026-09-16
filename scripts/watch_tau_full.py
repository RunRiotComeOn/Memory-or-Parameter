#!/usr/bin/env python3
"""Write compact progress records for the three tau-bench full runs."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/nas04/yixuh/memory")
SIMULATIONS = WORKSPACE / "third_party/tau2-bench/data/simulations"
SPECS = {
    "airline": (50, "qwen35_base_airline_full_v1", "tau_full_airline"),
    "retail": (114, "qwen35_base_retail_full_v1", "tau_full_retail"),
    "telecom": (114, "qwen35_base_telecom_full_v1", "tau_full_telecom"),
}


def session_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def domain_status(domain: str, total: int, save_name: str, session: str) -> dict:
    path = SIMULATIONS / save_name / "results.json"
    simulations = []
    read_error = None
    try:
        simulations = json.loads(path.read_text())["simulations"]
    except Exception as exc:  # The file can be between atomic checkpoint writes.
        read_error = f"{type(exc).__name__}: {exc}"
    reward_infos = [item.get("reward_info") for item in simulations]
    rewards = [
        float(reward_info.get("reward") or 0)
        for reward_info in reward_infos
        if isinstance(reward_info, dict)
    ]
    infrastructure_errors = sum(
        item.get("termination_reason") == "infrastructure_error"
        for item in simulations
    )
    return {
        "domain": domain,
        "completed": len(simulations),
        "total": total,
        "passed": sum(reward == 1 for reward in rewards),
        "average_reward": sum(rewards) / len(rewards) if rewards else None,
        "scored": len(rewards),
        "infrastructure_errors": infrastructure_errors,
        "session_running": session_exists(session),
        "read_error": read_error,
    }


while True:
    statuses = [
        domain_status(domain, total, save_name, session)
        for domain, (total, save_name, session) in SPECS.items()
    ]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domains": statuses,
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)
    if all(item["completed"] >= item["total"] for item in statuses):
        break
    if any(
        not item["session_running"] and item["completed"] < item["total"]
        for item in statuses
    ):
        raise SystemExit(2)
    time.sleep(60)
