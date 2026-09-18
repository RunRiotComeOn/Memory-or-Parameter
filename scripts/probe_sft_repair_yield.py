#!/usr/bin/env python3
"""Measure the SFT route's repair yield directly, outside the training loop.

The whole `sft` route bets on one number nobody has ever measured: given a
task the base agent FAILED, how often does a writer-generated repair plan,
injected into a fresh live attempt, actually turn it into a success? Inside
the training loop that number is buried -- it only surfaces for whichever
failed tasks the router happened to route to sft/both in the chosen
candidate, which through v5 was a handful per run (the only evidence on
record is `2 replayed, 0 verified`, and both of those were variants of the
SAME task template, 22cc237). That is not a sample, it is an anecdote.

This probe takes all 33 failed tasks from the recorded train split (18
distinct templates, difficulty 1/2/3 roughly evenly) and measures the
conversion rate head-on.

TWO ARMS, both run today against the same servers:

  control -- re-run the task with no memory and no guidance.
  repair  -- re-run it with the writer's repair plan injected as guidance.

The control is not optional. The recorded failures come from
`base_train_v2`, rolled out 2026-08-28, well before the two-replica
deterministic serving was set up (DESIGN.md section 11, 2026-09-12).
Section 13 already established that absolute scores are not comparable
across replicas/configs -- gaps of 10-20pp -- and section 7's correction
exists precisely because a number from an old config got compared against a
new one. Scoring a guided replay run TODAY against a failure recorded five
weeks ago would repeat that mistake. The control gives the denominator that
is actually comparable, and it doubles as a check on how reproducible those
failures even are.

What comes out: overall repair yield, split by difficulty and by the
original termination reason -- the last one matters because 27 of the 33
failures ended in `task_completed`, i.e. the agent believed it was done and
silently wasn't, which is the failure mode a "verify X before calling
complete_task" plan is aimed at.

Resumable: per-task outputs are skipped if already present, so a killed run
picks up where it stopped.

Run (needs both replicas up):
  PYTHONPATH=src python3 -u scripts/probe_sft_repair_yield.py \
      --output router_reward_v1/sft_repair_probe_v1 \
      --base-url http://127.0.0.1:8010/v1 \
      --base-url-control http://127.0.0.1:8011/v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root_nofeat"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_failed_tasks() -> list[dict[str, Any]]:
    protocol = read_json(TRAIN_ROLLOUT / "protocol.json")
    if protocol.get("split") != "train":
        raise ValueError(f"expected split=train, got {protocol.get('split')!r}")
    failed = []
    for path in sorted((TRAIN_ROLLOUT / "trajectories").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        trajectory = record["trajectory"]
        if trajectory.get("success"):
            continue
        failed.append({
            "task_id": record["task_id"],
            "difficulty": (trajectory.get("evaluation") or {}).get("difficulty"),
            "termination_reason": trajectory.get("termination_reason"),
            "trajectory": trajectory,
        })
    return failed


def generate_plans(tasks: list[dict[str, Any]], plans_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """One writer call per failed task. Cached, so a restart does not re-pay."""
    from trajectory_memory_lab.appworld_sft_writer import (
        APPWORLD_SFT_WRITER_SYSTEM,
        build_writer_payload,
        validate_writer_output,
    )
    from trajectory_memory_lab.model_client import ModelClient

    plans: dict[str, Any] = read_json(plans_path) if plans_path.exists() else {}
    # A cached None means the writer FAILED for that task. Retrying it is only
    # worth anything once the CAUSE is fixed: the observed failure is the model
    # writing chain-of-thought inside a JSON string until it burns max_tokens,
    # which is deterministic per task -- three fresh seeds all did it. So a
    # plain resume must NOT retry them, or every restart silently re-pays
    # 3 x max_tokens per failed task for a guaranteed failure. Opt in with
    # --retry-failed-plans once the writer is actually constrained.
    if args.retry_failed_plans:
        todo = [t for t in tasks if plans.get(t["task_id"]) is None]
    else:
        todo = [t for t in tasks if t["task_id"] not in plans]
    cached = sum(1 for v in plans.values() if v is not None)
    known_bad = sum(1 for v in plans.values() if v is None)
    print(f"[plans] {cached} cached, {known_bad} known-bad "
          f"({'retrying' if args.retry_failed_plans else 'skipped, pass --retry-failed-plans to retry'}), "
          f"{len(todo)} to generate", flush=True)

    for index, task in enumerate(todo, 1):
        client = ModelClient(
            base_url=args.base_url, api_key="EMPTY", model=args.model,
            temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
            seed=args.seed, enable_thinking=False, timeout=args.timeout,
        )
        try:
            reply = client.json_chat(
                system=APPWORLD_SFT_WRITER_SYSTEM,
                user=json.dumps(
                    build_writer_payload(task["trajectory"]), ensure_ascii=False, separators=(",", ":")
                ),
            )
            output = validate_writer_output(reply.parsed)
        except Exception as exc:  # noqa: BLE001
            output = None
            print(f"  [{index}/{len(todo)}] {task['task_id']}: writer FAILED {exc!r}", flush=True)
        plans[task["task_id"]] = output
        if output is not None:
            print(
                f"  [{index}/{len(todo)}] {task['task_id']}: plan {len(output['plan'])} chars, "
                f"mistake={output['mistake_summary'][:60]!r}, evidence={output['evidence_steps'][:6]}",
                flush=True,
            )
        write_json(plans_path, plans)
    return plans


def launch_control(task_ids: list[str], out_dir: Path, args: argparse.Namespace) -> subprocess.Popen:
    """No memory, no guidance -- today's honest baseline for these tasks."""
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_rollout.py"),
        "--split", "train", "--output", str(out_dir),
        "--experiment-name", "sft_repair_probe_control",
        "--max-parallel", "1", "--seed", str(args.seed),
        "--model", args.model, "--base-url", args.base_url_control,
        "--task-ids", *task_ids,
    ]
    print(f"[control] launching {len(task_ids)} tasks on {args.base_url_control}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def launch_repair(task: dict[str, Any], plan: dict[str, Any], out_dir: Path, base_url: str, args: argparse.Namespace) -> subprocess.Popen:
    task_id = task["task_id"]
    out_path = out_dir / f"{task_id}.json"
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_guided_replay.py"),
        "--task-id", task_id, "--guidance", plan["plan"], "--output", str(out_path),
        "--experiment-name", f"sft_repair_probe_{task_id}",
        "--model", args.model, "--base-url", base_url, "--seed", str(args.seed),
    ]
    # The attempt being repaired failed, so this is the repair arm of the
    # writer -- leave --previous-success off deliberately.
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)


def collect_repair(task: dict[str, Any], out_dir: Path, proc: subprocess.Popen, args: argparse.Namespace) -> dict[str, Any] | None:
    task_id = task["task_id"]
    out_path = out_dir / f"{task_id}.json"
    try:
        _, stderr = proc.communicate(timeout=args.replay_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        print(f"  {task_id}: replay TIMED OUT after {args.replay_timeout}s", flush=True)
        return None
    if proc.returncode != 0 or not out_path.exists():
        print(f"  {task_id}: replay process failed (exit {proc.returncode}) {(stderr or '')[-300:]}", flush=True)
        return None
    return read_json(out_path)


def summarize(tasks, plans, repair_dir: Path, control_dir: Path, out_path: Path) -> None:
    rows = []
    for task in tasks:
        task_id = task["task_id"]
        repair_rec = repair_dir / f"{task_id}.json"
        control_rec = control_dir / "trajectories" / f"{task_id}.json"
        repaired = control_ok = None
        if repair_rec.exists():
            repaired = bool(((read_json(repair_rec).get("trajectory")) or {}).get("success"))
        if control_rec.exists():
            control_ok = bool(((read_json(control_rec).get("trajectory")) or {}).get("success"))
        rows.append({
            "task_id": task_id,
            "difficulty": task["difficulty"],
            "original_termination": task["termination_reason"],
            "plan_generated": plans.get(task_id) is not None,
            "control_success": control_ok,
            "repair_success": repaired,
        })

    scored = [r for r in rows if r["control_success"] is not None and r["repair_success"] is not None]
    still_fails = [r for r in scored if not r["control_success"]]
    rescued = [r for r in still_fails if r["repair_success"]]
    self_healed = [r for r in scored if r["control_success"]]

    print("\n" + "=" * 72)
    print(f"scored {len(scored)}/{len(rows)} tasks")
    print(f"  self-healed WITHOUT any plan (control passed): {len(self_healed)}/{len(scored)}")
    print(f"  still failing in control (the real denominator): {len(still_fails)}")
    if still_fails:
        print(f"  RESCUED by a repair plan: {len(rescued)}/{len(still_fails)} "
              f"= {len(rescued) / len(still_fails) * 100:.1f}%")
    broken = [r for r in scored if r["control_success"] and not r["repair_success"]]
    print(f"  BROKEN by the plan (control passed, repair failed): {len(broken)}")

    for key in ("difficulty", "original_termination"):
        buckets: dict[Any, list] = defaultdict(list)
        for r in still_fails:
            buckets[r[key]].append(r)
        print(f"\n  by {key} (denominator = still failing in control):")
        for value in sorted(buckets, key=lambda v: str(v)):
            group = buckets[value]
            got = sum(1 for r in group if r["repair_success"])
            print(f"    {value}: {got}/{len(group)}")

    no_plan = [r for r in rows if not r["plan_generated"]]
    if no_plan:
        print(f"\n  writer produced no usable plan for {len(no_plan)}: {[r['task_id'] for r in no_plan]}")
    print("=" * 72)

    write_json(out_path, {
        "protocol": "sft_repair_probe_v1",
        "rows": rows,
        "counts": {
            "scored": len(scored),
            "self_healed": len(self_healed),
            "still_failing_in_control": len(still_fails),
            "rescued": len(rescued),
            "broken_by_plan": len(broken),
        },
    })
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "router_reward_v1/sft_repair_probe_v1")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1", help="repair arm + writer calls")
    parser.add_argument("--base-url-control", default="http://127.0.0.1:8011/v1", help="control arm, run in parallel")
    parser.add_argument("--seed", type=int, default=20260822)
    # Must match RouterBuilderConfig.max_tokens, which is what the real
    # pipeline gives the writer -- now 4096 repo-wide. The probe first ran at
    # 2048 and lost two plans to truncated, unparseable JSON: measuring a
    # method under a tighter budget than it actually runs with biases against
    # exactly the tasks whose repair plans are longest.
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--replay-timeout", type=float, default=1800)
    parser.add_argument(
        "--retry-failed-plans", action="store_true",
        help="re-run the writer for tasks whose plan generation failed previously; "
             "pointless until the runaway-output cause is fixed (see generate_plans)",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    out_root: Path = args.output
    repair_dir = out_root / "repair"
    control_dir = out_root / "control"
    plans_path = out_root / "plans.json"
    repair_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_failed_tasks()
    print(f"failed tasks in train split: {len(tasks)} "
          f"({len({t['task_id'].split('_')[0] for t in tasks})} distinct templates)", flush=True)
    print(f"  difficulty: {dict(Counter(t['difficulty'] for t in tasks))}", flush=True)
    print(f"  termination: {dict(Counter(t['termination_reason'] for t in tasks))}", flush=True)

    if args.summarize_only:
        plans = read_json(plans_path) if plans_path.exists() else {}
        summarize(tasks, plans, repair_dir, control_dir, out_root / "summary.json")
        return

    # Control first: it needs no plans, so it can occupy the second replica
    # while the writer is still generating plans on the first. Launching it
    # after plan generation would leave that replica idle for the ~1h of
    # serial writer calls, for nothing.
    control = launch_control([t["task_id"] for t in tasks], control_dir, args)

    plans = generate_plans(tasks, plans_path, args)

    # The repair arm is the bottleneck (one live rollout per task, ~5 min), so
    # spread it across BOTH replicas instead of leaving the second one idle
    # once the control arm finishes -- the same k-parity trick the training
    # loop uses in `score_candidates_self_only`. Determinism is unaffected:
    # `--max-num-seqs 1` means a replica still runs exactly one sequence at a
    # time, so two concurrent requests to one replica queue rather than batch,
    # and batch composition stays fixed at 1 (DESIGN.md section 11).
    replicas = [args.base_url, args.base_url_control]
    todo: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        if plans.get(task_id) is None:
            print(f"[repair {index}/{len(tasks)}] {task_id}: no plan, skipped", flush=True)
            continue
        if (repair_dir / f"{task_id}.json").exists():
            record = read_json(repair_dir / f"{task_id}.json")
            success = bool((record.get("trajectory") or {}).get("success"))
            print(f"[repair {index}/{len(tasks)}] {task_id}: resume, success={success}", flush=True)
            continue
        todo.append(task)

    done = 0
    for start in range(0, len(todo), len(replicas)):
        group = todo[start : start + len(replicas)]
        procs = [
            (task, launch_repair(task, plans[task["task_id"]], repair_dir, base_url, args), base_url)
            for task, base_url in zip(group, replicas)
        ]
        for task, proc, base_url in procs:
            record = collect_repair(task, repair_dir, proc, args)
            success = bool(((record or {}).get("trajectory") or {}).get("success"))
            done += 1
            print(f"[repair {done}/{len(todo)}] {task['task_id']} "
                  f"(difficulty={task['difficulty']}, was={task['termination_reason']}, "
                  f"on={base_url.rsplit('/', 2)[-2]}): success={success}", flush=True)

    print("[control] waiting for the control arm to finish...", flush=True)
    control.wait()
    summarize(tasks, plans, repair_dir, control_dir, out_root / "summary.json")


if __name__ == "__main__":
    main()
