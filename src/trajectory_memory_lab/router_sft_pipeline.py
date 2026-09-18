"""Turn committed sft/both decisions into real training and actually train
them in (DESIGN.md section 15).

Before this module, `sft_plan` was recorded and never consumed -- route=sft
and route=neither were reward-equivalent no-ops, since nothing downstream of
a committed sft_plan ever changed what the evaluated agent could do. This
closes that gap: every committed repair plan is replayed live in AppWorld
(`run_appworld_guided_replay.py`), and only a replay AppWorld itself scores
success=True contributes its own real transcript -- not the writer's plan
text -- as one training example. Once TRAIN_TRIGGER_SIZE examples have
accumulated, `router_sft_lora_update.sh` retrains and reloads both replicas.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from .appworld_agent import AGENT_SYSTEM
from .appworld_sft_writer import training_messages

ROOT = Path(__file__).resolve().parents[2]
APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root"

TRAIN_TRIGGER_SIZE = 8  # accumulate this many newly VERIFIED examples, then retrain once


def sft_candidates_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Committed sft/both decisions from one candidate's records: the ones
    with an accepted plan worth replaying and possibly training on.

    v6 carries `previous_success` along so the replay knows whether the plan
    repairs a failed attempt or consolidates a successful one -- since the
    guard that made this list failures-only is gone, both kinds occur."""
    out = []
    for record in records:
        if record.get("sft_status") != "selected":
            continue
        sft_plan = (record.get("decision") or {}).get("sft_plan") or {}
        plan_text = sft_plan.get("plan")
        if not plan_text:
            continue
        out.append({
            "task_id": record["source_task_id"],
            "plan": plan_text,
            "previous_success": bool(record.get("base_agent_success")),
        })
    return out


def replay_and_verify(
    candidate: dict[str, Any], output_dir: Path, model: str, base_url: str, seed: int,
) -> dict[str, Any] | None:
    """One guided replay; returns a training-ready example iff AppWorld
    itself scores the fresh live attempt success=True."""
    task_id = candidate["task_id"]
    out_path = output_dir / f"{task_id}.json"
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_guided_replay.py"),
        "--task-id", task_id, "--guidance", candidate["plan"], "--output", str(out_path),
        "--experiment-name", f"router_sft_replay_{task_id}", "--model", model, "--base-url", base_url,
        "--seed", str(seed),
    ]
    if candidate.get("previous_success"):
        cmd.append("--previous-success")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0 or not out_path.exists():
        return None
    record = json.loads(out_path.read_text())
    trajectory = record.get("trajectory") or {}
    if not trajectory.get("success"):
        return None
    return {"task_id": task_id, "messages": training_messages(AGENT_SYSTEM, trajectory["steps"])}


def collect_batch_sft_examples(
    chosen_records: list[dict[str, Any]], output_dir: Path, model: str, base_url: str, seed: int,
) -> list[dict[str, Any]]:
    """Replays every committed sft/both decision from the batch's CHOSEN
    candidate (not all K -- replaying every candidate's sft picks would
    multiply AppWorld-eval cost by K for no reward benefit, since only the
    chosen candidate's bank continues into the next batch)."""
    examples = []
    for candidate in sft_candidates_from_records(chosen_records):
        example = replay_and_verify(candidate, output_dir, model, base_url, seed)
        if example is not None:
            examples.append(example)
    return examples


def append_to_pool(pool_path: Path, examples: list[dict[str, Any]]) -> int:
    """Appends verified examples to the running pool; returns pool size."""
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("a", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"messages": example["messages"]}, ensure_ascii=False) + "\n")
    if not pool_path.exists():
        return 0
    with pool_path.open() as handle:
        return sum(1 for _ in handle)


BASE_MODEL_PATH = "/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
_TRAINING_TIMEOUT_SECONDS = 7200


def _restore_base_servers() -> None:
    """Best-effort recovery: if a training/merge/reload attempt failed or hung
    partway through, router_sft_lora_update.sh may have already killed one or
    both replicas. Rather than guess which state it left them in, force both
    back onto the known-good, untrained base checkpoint -- serving something
    correct beats leaving the training loop's only inference backends down.

    Session names/ports/GPUs mirror router_sft_lora_update.sh's own
    SERVER_A_*/SERVER_B_* env var overrides, for the same reason: this
    machine's det_server_a/b may not be at the 8000/8001, 0,1/2,3 defaults
    (e.g. those ports already taken by someone else on a shared box)."""
    for name, cuda, port, cache in (
        (
            os.environ.get("SERVER_A_NAME", "det_server_a"),
            os.environ.get("SERVER_A_GPUS", "0,1"),
            os.environ.get("SERVER_A_PORT", "8000"),
            "/tmp/appworld-det-server-cache-a",
        ),
        (
            os.environ.get("SERVER_B_NAME", "det_server_b"),
            os.environ.get("SERVER_B_GPUS", "2,3"),
            os.environ.get("SERVER_B_PORT", "8001"),
            "/tmp/appworld-det-server-cache-b",
        ),
    ):
        subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
        subprocess.run([
            "tmux", "new-session", "-d", "-s", name,
            f"CUDA_VISIBLE_DEVICES={cuda} MODEL_PATH={BASE_MODEL_PATH} PORT={port} "
            f"TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=0.85 TRITON_CACHE_DIR={cache} "
            f"{ROOT}/scripts/serve_appworld_deterministic.sh 2>&1 | tee -a {ROOT}/appworld_experiment/{name}.log",
        ])
    print("[router_sft_pipeline] both replicas relaunched on the base (untrained) checkpoint", flush=True)


def maybe_trigger_training(pool_path: Path, work_dir: Path, pool_size_before: int, pool_size_after: int) -> bool:
    """Fires router_sft_lora_update.sh exactly once per TRAIN_TRIGGER_SIZE
    boundary crossed -- e.g. going from 6 to 9 fires once (crossed 8), going
    from 9 to 17 fires once (crossed 16), not once per example added.

    Runs the updater in its own process group so a hang can be killed
    entirely (a bare subprocess.run timeout only kills the shell script
    itself, leaving any swift/deepspeed worker processes it spawned as
    orphans still holding GPU memory) and always leaves both replicas
    serving SOMETHING afterward -- the fine-tuned checkpoint on success, the
    base checkpoint on any failure or timeout.
    """
    crossed = pool_size_after // TRAIN_TRIGGER_SIZE > pool_size_before // TRAIN_TRIGGER_SIZE
    if not crossed:
        return False
    print(
        f"[router_sft_pipeline] pool reached {pool_size_after} verified examples "
        f"(crossed a multiple of {TRAIN_TRIGGER_SIZE}) -- triggering LoRA retrain + replica reload",
        flush=True,
    )
    proc = subprocess.Popen(
        [str(ROOT / "scripts/router_sft_lora_update.sh"), str(pool_path), str(work_dir)],
        cwd=str(ROOT), start_new_session=True,
    )
    try:
        returncode = proc.wait(timeout=_TRAINING_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(
            f"[router_sft_pipeline] router_sft_lora_update.sh exceeded {_TRAINING_TIMEOUT_SECONDS}s -- "
            "killing its whole process group and restoring the base model on both replicas",
            flush=True,
        )
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        _restore_base_servers()
        return False
    if returncode != 0:
        print(
            f"[router_sft_pipeline] router_sft_lora_update.sh FAILED (exit {returncode}) -- "
            "restoring the base model on both replicas so serving isn't left down",
            flush=True,
        )
        _restore_base_servers()
        return False
    return True
