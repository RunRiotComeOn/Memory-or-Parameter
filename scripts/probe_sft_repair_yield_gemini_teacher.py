#!/usr/bin/env python3
"""Same probe as probe_sft_repair_yield.py, ONE variable changed: the repair
plan is written by Gemini (external teacher) instead of the same base model
that plays the task agent (self-writer).

Rationale: probe_sft_repair_yield.py's writer is qwen35-tau grading its own
mistakes -- the model that failed the task is also the one deciding what it
did wrong and how to fix it. A stronger, independent teacher model reading
the same failed trajectory may name the actual mistake more reliably. This
script tests exactly that, holding everything else fixed: same 33 failed
train tasks, same guided-replay mechanism (the plan is injected into a fresh
attempt via `memory_block`, a live qwen35-tau agent executes it against the
real environment -- the TEACHER never touches AppWorld directly, it only
writes the plan), same control arm, same summarize() logic, same output
schema. Only `generate_plans()` changes: Gemini via the `google-genai` SDK
instead of a local ModelClient call to APPWORLD_SFT_WRITER_SYSTEM.

Run (needs both replicas up for the repair/control arms; Gemini plan
generation needs outbound network, not a local server):
  PYTHONPATH=src python3 -u scripts/probe_sft_repair_yield_gemini_teacher.py \
      --output router_reward_v1/sft_repair_probe_gemini_v1 \
      --base-url http://127.0.0.1:8000/v1 \
      --base-url-control http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
APPWORLD_PYTHON = "/nas04/yixuh/appworld_venv/bin/python"
APPWORLD_ROOT_DEFAULT = "/nas04/yixuh/appworld_root"

# Same JSON shape as appworld_sft_writer.validate_writer_output, so every
# downstream consumer (run_appworld_guided_replay.py, summarize()) is
# unchanged regardless of which model produced it.
TEACHER_SYSTEM = """You are an expert teacher reviewing a coding agent's failed attempt at an AppWorld task. AppWorld is not a tool-call benchmark: the agent writes Python code that calls `apis.<app>.<api>(...)` against a set of app APIs and reads back whatever the environment prints.

You receive the task instruction, the full failed attempt (its code turns and the environment's real outputs), and the evaluator's verdict. Your job is to write a CORRECTED PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Name the specific mistake the attempt made (wrong argument, missing login, unchecked pagination, premature complete_task, wrong app, misread API doc, etc.), put a short summary of it in `mistake_summary`, and cite in `evidence_steps` the step indexes where it went wrong.

The plan is followed by a live student agent that writes its own code and reads real API responses; it is NOT executed verbatim. So name the concrete APIs to call and in what order (use exact names you saw in the attempt's API-discovery calls, e.g. `apis.spotify.login`, not vague descriptions), and what to verify before calling `apis.supervisor.complete_task`. Do not invent API names or arguments you did not see evidence for in the attempt or its discovery calls -- if the attempt never discovered the right API, say what to search for (`apis.api_docs.search_api_docs`), not what the answer is.

Return exactly one JSON object matching the schema."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "mistake_summary": {"type": "string"},
        "evidence_steps": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["plan", "mistake_summary", "evidence_steps"],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_failed_tasks() -> list[dict[str, Any]]:
    """Identical to probe_sft_repair_yield.py's -- same 33 tasks, so the two
    probes' `still_failing_in_control` denominators are directly comparable."""
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


def _generate(client: Any, model: str, payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from google.genai import types

    error: Exception | None = None
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=TEACHER_SYSTEM,
                    temperature=args.temperature,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                ),
            )
            return json.loads(response.text)
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt == 5:
                break
            time.sleep(min(30, 2 ** attempt))
    assert error is not None
    raise error


def generate_plans(tasks: list[dict[str, Any]], plans_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """One Gemini call per failed task. Cached, so a restart does not re-pay."""
    from google import genai

    from trajectory_memory_lab.appworld_sft_writer import build_writer_payload, validate_writer_output

    key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Gemini API key file is empty")
    client = genai.Client(api_key=key)

    plans: dict[str, Any] = read_json(plans_path) if plans_path.exists() else {}
    todo = [t for t in tasks if t["task_id"] not in plans]
    print(f"[plans] {len(plans)} cached, {len(todo)} to generate via {args.teacher_model}", flush=True)

    for index, task in enumerate(todo, 1):
        try:
            parsed = _generate(client, args.teacher_model, build_writer_payload(task["trajectory"]), args)
            output = validate_writer_output(parsed)
        except Exception as exc:  # noqa: BLE001
            output = None
            print(f"  [{index}/{len(todo)}] {task['task_id']}: teacher FAILED {exc!r}", flush=True)
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
    """No memory, no guidance -- identical to probe_sft_repair_yield.py's control."""
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_rollout.py"),
        "--split", "train", "--output", str(out_dir),
        "--experiment-name", "sft_repair_probe_gemini_control",
        "--max-parallel", "1", "--seed", str(args.seed),
        "--model", args.model, "--base-url", args.base_url_control,
        "--task-ids", *task_ids,
    ]
    print(f"[control] launching {len(task_ids)} tasks on {args.base_url_control}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def run_repair(task: dict[str, Any], plan: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    """Same student agent (qwen35-tau, args.model) executes under the
    teacher's plan -- only who WROTE the plan differs from the sibling probe."""
    task_id = task["task_id"]
    out_path = out_dir / f"{task_id}.json"
    if out_path.exists():
        return read_json(out_path)
    env = dict(os.environ)
    env["APPWORLD_ROOT"] = env.get("APPWORLD_ROOT", APPWORLD_ROOT_DEFAULT)
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        APPWORLD_PYTHON, "-u", str(ROOT / "scripts/run_appworld_guided_replay.py"),
        "--task-id", task_id, "--guidance", plan["plan"], "--output", str(out_path),
        "--experiment-name", f"sft_repair_probe_gemini_{task_id}",
        "--model", args.model, "--base-url", args.base_url, "--seed", str(args.seed),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=args.replay_timeout)
    if result.returncode != 0 or not out_path.exists():
        print(f"  {task_id}: replay process failed (exit {result.returncode}) {result.stderr[-300:]}", flush=True)
        return None
    return read_json(out_path)


def summarize(tasks, plans, repair_dir: Path, control_dir: Path, out_path: Path) -> None:
    """Identical logic/output schema to probe_sft_repair_yield.py's summarize()."""
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
        print(f"  RESCUED by a Gemini-teacher repair plan: {len(rescued)}/{len(still_fails)} "
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
        print(f"\n  teacher produced no usable plan for {len(no_plan)}: {[r['task_id'] for r in no_plan]}")
    print("=" * 72)

    write_json(out_path, {
        "protocol": "sft_repair_probe_gemini_v1",
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
    parser.add_argument("--output", type=Path, default=ROOT / "router_reward_v1/sft_repair_probe_gemini_v1")
    parser.add_argument("--model", default="qwen35-tau", help="student/task agent for control + repair execution")
    parser.add_argument(
        "--teacher-model", default="gemini-3.1-pro-preview",
        help="Gemini model that WRITES the plan. gemini-3-pro-preview (initial guess) returned "
             "404 NOT_FOUND on first run, with the API itself naming gemini-3.1-pro-preview as "
             "its replacement -- that is now the default. The codebase's other Gemini scripts "
             "default to the cheaper gemini-3-flash-preview; override if this drifts again.",
    )
    parser.add_argument("--api-key-file", type=Path, default=Path("/nas04/yixuh/.config/continual-memory/gemini_api_key"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="repair arm (student agent)")
    parser.add_argument("--base-url-control", default="http://127.0.0.1:8001/v1", help="control arm, run in parallel")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--replay-timeout", type=float, default=1800)
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

    # Control first: it needs no plans and runs entirely against the local
    # replica, so it can proceed while Gemini calls (a separate, outbound
    # network path) are still generating plans.
    control = launch_control([t["task_id"] for t in tasks], control_dir, args)

    plans = generate_plans(tasks, plans_path, args)

    for index, task in enumerate(tasks, 1):
        plan = plans.get(task["task_id"])
        if plan is None:
            print(f"[repair {index}/{len(tasks)}] {task['task_id']}: no plan, skipped", flush=True)
            continue
        record = run_repair(task, plan, repair_dir, args)
        success = bool(((record or {}).get("trajectory") or {}).get("success"))
        print(f"[repair {index}/{len(tasks)}] {task['task_id']} "
              f"(difficulty={task['difficulty']}, was={task['termination_reason']}): "
              f"success={success}", flush=True)

    print("[control] waiting for the control arm to finish...", flush=True)
    control.wait()
    summarize(tasks, plans, repair_dir, control_dir, out_root / "summary.json")


if __name__ == "__main__":
    main()
