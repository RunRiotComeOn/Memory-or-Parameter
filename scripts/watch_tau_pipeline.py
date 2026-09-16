#!/usr/bin/env python3
"""Record compact progress and fail if the tau pipeline dies early."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/nas04/yixuh/memory")
EXPERIMENT = ROOT / "tau_experiment/v1"
SESSION = "tau_train_replay_pipeline"


def session_exists() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


while True:
    stages_path = EXPERIMENT / "stages.log"
    stages = stages_path.read_text().splitlines() if stages_path.exists() else []
    retention = len(list((EXPERIMENT / "retention/tasks").glob("*/retention_decision.json")))
    sft = len(list((EXPERIMENT / "retention/tasks").glob("*/sft_episode.json")))
    replay = {}
    for domain in ("airline", "retail", "telecom"):
        path = (
            ROOT
            / "third_party/tau2-bench/data/simulations"
            / f"qwen35_tau_sft_memory_{domain}_test_v1/results.json"
        )
        try:
            replay[domain] = len(json.loads(path.read_text())["simulations"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            replay[domain] = 0
    running = session_exists()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_running": running,
        "latest_stage": stages[-1] if stages else None,
        "retention": {"completed": retention, "total": 178, "sft": sft},
        "replay": replay,
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)
    if stages and stages[-1].endswith("pipeline:complete"):
        break
    if not running:
        raise SystemExit(2)
    time.sleep(60)
