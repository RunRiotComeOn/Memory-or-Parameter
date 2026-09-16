#!/usr/bin/env python3
"""Re-run retention tools over saved TimeWarp trajectories without re-running actors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_timewarp_pilot import (
    ALLOWED_ACTIONS,
    MULTI_TURN_AGENT_SYSTEM,
    _action_code,
)

from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.retention import normalize_memory_bank, run_retention_tools
from trajectory_memory_lab.storage import append_jsonl, write_json


def _write_episode(
    *,
    output_path: Path,
    run_dir: Path,
    task_dir: Path,
    trajectory: dict,
    episode: dict,
    model: str,
    protocol: str,
) -> None:
    steps = trajectory.get("steps", [])
    episode_steps = episode.get("steps", [])
    source_steps = [item["source_step"] for item in episode_steps]
    if source_steps != list(range(len(steps))):
        return
    messages = [{"role": "system", "content": MULTI_TURN_AGENT_SYSTEM}]
    for episode_step in episode_steps:
        source = steps[episode_step["source_step"]]
        messages.extend(
            [
                {"role": "user", "content": source["agent_prompt"]},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        episode_step["target_action"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
    append_jsonl(
        output_path,
        {
            "messages": messages,
            "source_task_id": trajectory.get("source_task_id")
            or f"timewarp.{trajectory.get('task_id')}",
            "source_steps": source_steps,
            "recorded_actions": [item["recorded_action"] for item in episode_steps],
            "target_actions": [item["target_action"] for item in episode_steps],
            "label_type": episode["label_type"],
            "validation_status": episode["validation_status"],
            "validation_basis": episode["validation_basis"],
            "selection_rationale": episode.get("rationale", ""),
            "trajectory_path": str(
                (task_dir / "trajectory.json").relative_to(run_dir)
            ),
            "model": model,
            "retention_protocol": protocol,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--initial-memory", type=Path)
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Only re-extract trajectories whose environment verifier succeeded.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    source_tasks = sorted(args.source_run.glob("tasks/*/trajectory.json"))
    if args.successful_only:
        source_tasks = [
            path
            for path in source_tasks
            if json.loads(path.read_text(encoding="utf-8")).get("success")
        ]
    write_json(
        run_dir / "manifest.json",
        {
            "source_run": str(args.source_run),
            "model": args.model,
            "protocol": "retention_reextract",
            "tasks": len(source_tasks),
        },
    )

    model = ModelClient(
        base_url=args.base_url,
        api_key="EMPTY",
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=not args.disable_thinking,
    )
    memory: list[dict] = []
    if args.initial_memory:
        memory = normalize_memory_bank(
            json.loads(args.initial_memory.read_text(encoding="utf-8"))
        )
        write_json(run_dir / "initial_memory_bank.json", memory)
    summary = []
    for ordinal, source_path in enumerate(source_tasks):
        trajectory = json.loads(source_path.read_text(encoding="utf-8"))
        task_id = trajectory.get("task_id")
        task_dir = run_dir / "tasks" / f"{ordinal:02d}_timewarp_{task_id}"
        task_dir.mkdir(parents=True)
        write_json(task_dir / "trajectory.json", trajectory)
        print(
            f"[{ordinal + 1}/{len(source_tasks)}] timewarp.{task_id}",
            flush=True,
        )
        retention = run_retention_tools(
            model,
            trajectory=trajectory,
            memory_bank=memory,
            agent_system=MULTI_TURN_AGENT_SYSTEM,
            allowed_actions=ALLOWED_ACTIONS,
            action_validator=_action_code,
            audit=True,
        )
        write_json(task_dir / "retention_decision.json", retention)
        write_json(run_dir / "memory_bank.json", memory)
        validation = (retention["tools"].get("build_sft_examples") or {}).get(
            "validation", {}
        )
        for status, filename in (
            ("accepted", "sft_examples.jsonl"),
            ("needs_replay", "sft_candidates.jsonl"),
        ):
            for episode in validation.get(status, []):
                _write_episode(
                    output_path=run_dir / filename,
                    run_dir=run_dir,
                    task_dir=task_dir,
                    trajectory=trajectory,
                    episode=episode,
                    model=args.model,
                    protocol=retention["protocol"],
                )
        item = {
            "task_id": task_id,
            "success": bool(trajectory.get("success")),
            "steps": len(trajectory.get("steps", [])),
            "choice": retention["artifact_choice"],
            "controller_choice": retention["controller_choice"],
            "memory_entries_after": len(memory),
            "accepted_episodes": len(validation.get("accepted", [])),
            "needs_replay": len(validation.get("needs_replay", [])),
        }
        summary.append(item)
        write_json(run_dir / "summary.json", summary)
        print(
            f"  choice={item['choice']} accepted={item['accepted_episodes']} "
            f"candidate={item['needs_replay']} memories={len(memory)}",
            flush=True,
        )
    write_json(
        run_dir / "run_result.json",
        {
            "total": len(summary),
            "accepted_episodes": sum(x["accepted_episodes"] for x in summary),
            "needs_replay": sum(x["needs_replay"] for x in summary),
            "memory_entries": len(memory),
            "tasks": summary,
        },
    )
    print(f"Run written to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
