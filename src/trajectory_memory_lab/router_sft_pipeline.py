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

# Per-domain replay config -- what interpreter, script, and agent system
# prompt turn a committed sft/both decision into a live replayed trajectory.
# AppWorld stays the literal defaults on every existing call so
# train_router_selfreward.py's calls are unaffected; ALFWorld is here so
# scripts/run_alfworld_router_llm_probe.py (and any future ALFWorld GRPO
# trainer) can reuse this same replay/verify machinery instead of
# duplicating it -- everything in this module except this dict and the
# literal AppWorld defaults below is already domain-agnostic (it only reads
# generic trajectory/record shapes).
_REPLAY_CONFIG = {
    "appworld": {
        "python": APPWORLD_PYTHON,
        "script": "scripts/run_appworld_guided_replay.py",
        "agent_system": AGENT_SYSTEM,
        "env_var": "APPWORLD_ROOT",
        "env_default": APPWORLD_ROOT_DEFAULT,
        "extra_args": [],
    },
    "alfworld": {
        "python": "/nas04/yixuh/alfworld_venv310/bin/python",
        "script": "scripts/run_alfworld_guided_replay.py",
        "agent_system": None,  # resolved lazily below to avoid importing alfworld_agent under the main .venv
        "env_var": "ALFWORLD_DATA",
        "env_default": "/nas04/yixuh/alfworld_data",
        "extra_args": ["--split", "train"],
    },
    "babyai": {
        # Runs under the repo's .venv: like webshop and tau2, the replay is
        # only an HTTP client of a long-lived env server (AgentGym's
        # agentenv-babyai on :36001), which is the one process needing
        # babyai_venv.
        "python": str(ROOT / ".venv/bin/python"),
        "script": "scripts/run_babyai_guided_replay.py",
        "agent_system": None,  # resolved lazily, same reason as the others
        "env_var": None,       # the server address comes from --env-url/$BABYAI_ENV_URL
        "env_default": None,
        "extra_args": [],
    },
    "textcraft": {
        # Same shape as babyai: the env is a long-lived AgentGym server
        # (agentenv-textcraft on :36002, the one process needing
        # textcraft_venv), so the replay is just an HTTP client under .venv.
        "python": str(ROOT / ".venv/bin/python"),
        "script": "scripts/run_textcraft_guided_replay.py",
        "agent_system": None,  # resolved lazily, same reason as the others
        "env_var": None,       # server address comes from --env-url/$TEXTCRAFT_ENV_URL
        "env_default": None,
        "extra_args": [],
    },
    "sqlgym": {
        # sqlgym is a LIBRARY, not a server: the replay opens the SQLite
        # databases in-process, so it needs sqlgym_venv rather than the
        # repo .venv that the AgentGym-backed domains use.
        "python": "/nas04/yixuh/sqlgym_venv/bin/python",
        "script": "scripts/run_sqlgym_guided_replay.py",
        "agent_system": None,  # resolved lazily, same reason as the others
        "env_var": None,       # BIRD path comes from --bird-path
        "env_default": None,
        "extra_args": [],
    },
    "scienceworld": {
        "python": "/nas04/yixuh/scienceworld_venv/bin/python",
        "script": "scripts/run_scienceworld_guided_replay.py",
        "agent_system": None,  # resolved lazily below, same reason as alfworld
        "env_var": None,  # the scienceworld package carries its own data
        "env_default": None,
        # No --split: a ScienceWorld task_id ("<task_name>::<variation_id>")
        # identifies a task outright, because train/dev/test are disjoint
        # variation-id ranges of the same task names.
        "extra_args": [],
    },
    "webshop": {
        # Runs under the repo's .venv: the replay is only an HTTP client of
        # the long-lived env server (scripts/webshop_env_server.py), which is
        # the one process that needs webshop_venv. It must be the same server
        # (same world seed) the plan's source trajectory ran against.
        "python": str(ROOT / ".venv/bin/python"),
        "script": "scripts/run_webshop_guided_replay.py",
        "agent_system": None,  # resolved lazily below, like the others
        "env_var": "WEBSHOP_ENV_URL",
        "env_default": "http://127.0.0.1:3100",
        # No --split: `goal_<index>` names one goal outright.
        "extra_args": [],
    },
    "tau2": {
        # tau2's own venv (Python 3.12); the Gemini key is read from its file
        # by `tau2_agent.configure_gemini`, so no env var is needed.
        "python": str(ROOT / "third_party/tau2-bench/.venv/bin/python"),
        "script": "scripts/run_tau2_guided_replay.py",
        "agent_system": None,  # unused: tau2 replays carry their own `sft_example`
        "env_var": None,
        "env_default": None,
        # No --split: `<domain>::<tau2 id>` names one task outright.
        "extra_args": [],
    },
}


def _agent_system_for(domain: str) -> str:
    if domain == "alfworld":
        from .alfworld_agent import AGENT_SYSTEM as ALFWORLD_AGENT_SYSTEM

        return ALFWORLD_AGENT_SYSTEM
    if domain == "scienceworld":
        from .scienceworld_agent import AGENT_SYSTEM as SCIENCEWORLD_AGENT_SYSTEM

        return SCIENCEWORLD_AGENT_SYSTEM
    if domain == "babyai":
        from .babyai_agent import AGENT_SYSTEM as BABYAI_AGENT_SYSTEM

        return BABYAI_AGENT_SYSTEM
    if domain == "textcraft":
        from .textcraft_agent import AGENT_SYSTEM as TEXTCRAFT_AGENT_SYSTEM

        return TEXTCRAFT_AGENT_SYSTEM
    if domain == "sqlgym":
        from .sqlgym_agent import AGENT_SYSTEM as SQLGYM_AGENT_SYSTEM

        return SQLGYM_AGENT_SYSTEM
    if domain == "webshop":
        from .webshop_agent import AGENT_SYSTEM as WEBSHOP_AGENT_SYSTEM

        return WEBSHOP_AGENT_SYSTEM
    return AGENT_SYSTEM

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
    *, domain: str = "appworld",
) -> dict[str, Any] | None:
    """One guided replay; returns a training-ready example iff the
    environment itself scores the fresh live attempt success=True.

    `domain` selects the interpreter/script/env var/agent prompt via
    `_REPLAY_CONFIG` -- AppWorld's script takes a bare task_id and resolves
    it against its own installed task DB regardless of split, while
    ALFWorld's needs an explicit `--split` since task_ids only disambiguate
    within one split directory (hence "train" being baked into
    `_REPLAY_CONFIG["alfworld"]["extra_args"]`: guided replay only ever
    targets committed decisions from a train-split rollout)."""
    cfg = _REPLAY_CONFIG[domain]
    task_id = candidate["task_id"]
    # ALFWorld ids contain "/", ScienceWorld ids contain "::" -- both are
    # flattened the same way the rollout runners flatten them, so a replay
    # file can be matched back to its task by name. tau2 telecom ids carry
    # "[", "|" and spaces as well, so tau2 uses its own safe stem.
    if domain == "tau2":
        from .tau2_agent import task_file_stem

        out_path = output_dir / f"{task_file_stem(task_id)}.json"
    else:
        out_path = output_dir / f"{task_id.replace('/', '__').replace('::', '__')}.json"
    env = dict(os.environ)
    # ScienceWorld carries its data inside the installed package, so it has no
    # env var to point at a data directory the way AppWorld and ALFWorld do.
    if cfg["env_var"]:
        env[cfg["env_var"]] = env.get(cfg["env_var"], cfg["env_default"])
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        cfg["python"], "-u", str(ROOT / cfg["script"]),
        "--task-id", task_id, "--guidance", candidate["plan"], "--output", str(out_path),
        *cfg["extra_args"],
        "--model", model, "--base-url", base_url, "--seed", str(seed),
    ]
    if domain == "appworld":
        cmd.extend(["--experiment-name", f"router_sft_replay_{task_id}"])
    if candidate.get("previous_success"):
        cmd.append("--previous-success")
    # A single slow replay must not kill the whole arm. Measured: tau2's
    # telecom tasks are long multi-turn dialogues (200-step cap, a live Gemini
    # user on every turn) and one of them blew the 20-minute budget -- the
    # TimeoutExpired propagated out of `collect_batch_sft_examples`, ending
    # that arm and discarding the 56 replays already verified, because the
    # pool is only written once at the end. One un-replayable candidate is a
    # candidate that yields no training example, which is exactly what a
    # `None` return already means everywhere else here.
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"  replay timed out after 1200s, skipping: {task_id}", flush=True)
        return None
    if result.returncode != 0 or not out_path.exists():
        return None
    record = json.loads(out_path.read_text())
    trajectory = record.get("trajectory") or {}
    if not trajectory.get("success"):
        return None
    if trajectory.get("sft_example"):
        # Tool-calling benchmarks (tau2) record the agent's own view in
        # OpenAI chat format with tool schemas; flattening it into
        # `training_messages`' text shape would train on a prompt the served
        # model never sees (see tau2_agent's module docstring).
        return {"task_id": task_id, **trajectory["sft_example"]}
    return {
        "task_id": task_id,
        "messages": training_messages(_agent_system_for(domain), trajectory["steps"]),
    }


def collect_batch_sft_examples(
    chosen_records: list[dict[str, Any]], output_dir: Path, model: str, base_url: str, seed: int,
    *, domain: str = "appworld",
) -> list[dict[str, Any]]:
    """Replays every committed sft/both decision from the batch's CHOSEN
    candidate (not all K -- replaying every candidate's sft picks would
    multiply eval cost by K for no reward benefit, since only the chosen
    candidate's bank continues into the next batch)."""
    examples = []
    for candidate in sft_candidates_from_records(chosen_records):
        example = replay_and_verify(candidate, output_dir, model, base_url, seed, domain=domain)
        if example is not None:
            examples.append(example)
    return examples


def append_to_pool(pool_path: Path, examples: list[dict[str, Any]]) -> int:
    """Appends verified examples to the running pool; returns pool size."""
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("a", encoding="utf-8") as handle:
        for example in examples:
            row = {"messages": example["messages"]}
            if example.get("tools"):
                row["tools"] = example["tools"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
