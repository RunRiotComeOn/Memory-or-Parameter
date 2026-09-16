from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .browser import BrowserSession, Observation
from .model_client import ModelClient
from .prompts import (
    AGENT_SYSTEM,
    agent_user_prompt,
)
from .retention import (
    memory_for_agent,
    normalize_memory_bank,
    run_retention_tools,
)
from .storage import append_jsonl, write_json


@dataclass
class ExperimentConfig:
    task_file: Path
    output_root: Path
    model: str
    base_url: str
    api_key: str
    limit: int
    max_steps: int
    headless: bool
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    initial_memory_file: Path | None = None
    review_artifacts: bool = True
    audit_retention_tools: bool = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_observation(observation: Observation) -> dict[str, Any]:
    return {
        "url": observation.url,
        "title": observation.title,
        "text": observation.text[:6000],
        "elements": observation.elements[:100],
    }


def _trajectory_for_review(trajectory: dict[str, Any]) -> dict[str, Any]:
    task = trajectory["task"]
    return {
        "task": {
            "id": task["id"],
            "instruction": task["instruction"],
            "start_url": task["start_url"],
        },
        "final_answer": trajectory["final_answer"],
        "success": trajectory["evaluation"]["success"],
        "error": trajectory["error"],
        "steps": [
            {
                "index": step["index"],
                "observation": {
                    "url": step["observation"]["url"],
                    "title": step["observation"]["title"],
                    "text": step["observation"]["text"][:4000],
                    "elements": step["observation"]["elements"][:60],
                },
                "action": step["action"],
                "execution": step.get("execution"),
                "execution_error": step.get("execution_error"),
                "model_reasoning": (step.get("model_reasoning") or "")[:2500],
            }
            for step in trajectory["steps"]
        ],
    }


def _steps_for_agent_prompt(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain decision-relevant history without repeating entire page bodies."""
    compact: list[dict[str, Any]] = []
    for step in steps:
        observation = step.get("observation") or {}
        item = {
            "index": step.get("index"),
            "observation": {
                "url": observation.get("url", ""),
                "title": observation.get("title", ""),
                "elements": (observation.get("elements") or [])[:100],
            },
            "action": step.get("action"),
        }
        for key in ("execution", "execution_error"):
            if step.get(key) is not None:
                item[key] = step[key]
        compact.append(item)
    return compact


def _saved_observation_prompt(
    observation: dict[str, Any], *, max_text_chars: int = 6_000
) -> str:
    element_lines = []
    for item in (observation.get("elements") or [])[:180]:
        attrs = []
        if item.get("role"):
            attrs.append(f"role={item['role']!r}")
        if item.get("name"):
            attrs.append(f"name={item['name']!r}")
        if item.get("placeholder"):
            attrs.append(f"placeholder={item['placeholder']!r}")
        element_lines.append(f"[{item['ref']}] <{item['tag']}> " + " ".join(attrs))
    return (
        f"URL: {observation.get('url', '')}\nTITLE: {observation.get('title', '')}\n\n"
        "INTERACTIVE ELEMENTS:\n"
        + "\n".join(element_lines)
        + "\n\nVISIBLE PAGE TEXT:\n"
        + str(observation.get("text", ""))[:max_text_chars]
    )


def _sft_agent_prompt(trajectory: dict[str, Any], source_step: int) -> str:
    """Reconstruct the state for SFT while removing duplicated raw page text."""
    source = trajectory["steps"][source_step]
    return agent_user_prompt(
        instruction=trajectory["task"]["instruction"],
        memory_bank=_memory_for_prompt(trajectory["context_before"]),
        observation=_saved_observation_prompt(source["observation"]),
        recent_steps=_steps_for_agent_prompt(trajectory["steps"][:source_step]),
    )


def _memory_for_prompt(memory_bank: list[dict[str, Any]], max_chars: int = 18_000):
    return memory_for_agent(memory_bank, max_chars=max_chars)


def _evaluate(task: dict[str, Any], answer: str) -> dict[str, Any]:
    patterns = task.get("answer_patterns", [])
    matches = [bool(re.search(pattern, answer, re.I)) for pattern in patterns]
    return {
        "success": bool(patterns) and all(matches),
        "patterns": patterns,
        "matches": matches,
        "answer": answer,
    }


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    stripped = value.strip()
    return None if not stripped or stripped.lower() == "null" else stripped


def _normalize_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    context = _normalize_optional_string(parsed.get("context"))
    raw_sft = parsed.get("sft")
    episode_steps: list[dict[str, Any]] = []
    raw_episode = raw_sft.get("episode") if isinstance(raw_sft, dict) else None
    if isinstance(raw_episode, dict) and isinstance(raw_episode.get("steps"), list):
        for raw_step in raw_episode["steps"]:
            if not isinstance(raw_step, dict):
                continue
            source_step = raw_step.get("source_step")
            action = raw_step.get("target_action")
            if (
                isinstance(source_step, int)
                and not isinstance(source_step, bool)
                and isinstance(action, dict)
                and str(action.get("action", "")).lower()
                in {"click", "fill", "press", "scroll", "back", "goto", "finish"}
            ):
                episode_steps.append(
                    {"source_step": source_step, "target_action": action}
                )
    return {
        "context": context,
        "sft": {"episode": {"steps": episode_steps}} if episode_steps else None,
    }


def run_experiment(config: ExperimentConfig) -> Path:
    tasks = json.loads(config.task_file.read_text())[: config.limit]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_config = asdict(config)
    manifest_config["task_file"] = str(config.task_file)
    manifest_config["output_root"] = str(config.output_root)
    manifest_config["api_key"] = "EMPTY" if config.api_key == "EMPTY" else "[redacted]"
    manifest_config["initial_memory_file"] = (
        str(config.initial_memory_file) if config.initial_memory_file else None
    )
    write_json(
        run_dir / "manifest.json",
        {"started_at": _now(), **manifest_config},
    )

    model = ModelClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )
    memory_bank: list[dict[str, Any]] = normalize_memory_bank(
        json.loads(config.initial_memory_file.read_text())
        if config.initial_memory_file
        else []
    )
    if memory_bank:
        write_json(run_dir / "memory_bank.json", memory_bank)
    summary: list[dict[str, Any]] = []

    for task_index, task in enumerate(tasks):
        task_id = str(task["id"])
        task_dir = run_dir / "tasks" / f"{task_index:02d}_{task_id}"
        context_before = list(memory_bank)
        trajectory_steps: list[dict[str, Any]] = []
        answer = ""
        error: str | None = None
        started = time.monotonic()

        print(
            f"[{task_index + 1}/{len(tasks)}] {task_id}: {task['instruction']}",
            flush=True,
        )
        try:
            with BrowserSession(
                headless=config.headless,
                screenshot_dir=task_dir / "screenshots",
            ) as browser:
                browser.goto(task["start_url"])
                for step_index in range(config.max_steps):
                    observation = browser.observe(step_index)
                    prompt = agent_user_prompt(
                        instruction=task["instruction"],
                        memory_bank=_memory_for_prompt(memory_bank),
                        observation=observation.prompt_text(),
                        recent_steps=_steps_for_agent_prompt(trajectory_steps),
                    )
                    reply = model.json_chat(system=AGENT_SYSTEM, user=prompt)
                    action = reply.parsed
                    step: dict[str, Any] = {
                        "index": step_index,
                        "observation": _compact_observation(observation),
                        "agent_prompt": prompt,
                        "action": action,
                        "model_content": reply.content,
                        "model_reasoning": reply.reasoning,
                        "model_usage": reply.usage,
                    }
                    try:
                        step["execution"] = browser.execute(action)
                    except Exception as exc:
                        step["execution_error"] = f"{type(exc).__name__}: {exc}"
                    trajectory_steps.append(step)
                    if str(action.get("action", "")).lower() == "finish":
                        answer = str(action.get("answer", ""))
                        break
                else:
                    error = f"max_steps_exceeded:{config.max_steps}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        evaluation = _evaluate(task, answer)
        trajectory = {
            "task": task,
            "context_before": context_before,
            "steps": trajectory_steps,
            "final_answer": answer,
            "evaluation": evaluation,
            "error": error,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(task_dir / "trajectory.json", trajectory)

        if config.review_artifacts:
            retention = run_retention_tools(
                model,
                trajectory=trajectory,
                memory_bank=memory_bank,
                agent_system=AGENT_SYSTEM,
                allowed_actions={
                    "click",
                    "fill",
                    "press",
                    "scroll",
                    "back",
                    "goto",
                    "finish",
                },
                audit=config.audit_retention_tools,
            )
            write_json(task_dir / "retention_decision.json", retention)
            memory_application = (retention["tools"].get("edit_memory") or {}).get(
                "application", {}
            )
            if memory_application.get("applied"):
                write_json(run_dir / "memory_bank.json", memory_bank)

            validation = (retention["tools"].get("build_sft_examples") or {}).get(
                "validation", {}
            )
            for status, output_name in (
                ("accepted", "sft_examples.jsonl"),
                ("needs_replay", "sft_candidates.jsonl"),
            ):
                for episode in validation.get(status, []):
                    episode_steps = episode.get("steps", [])
                    source_steps = [item["source_step"] for item in episode_steps]
                    if source_steps != list(range(len(trajectory_steps))):
                        continue
                    messages: list[dict[str, str]] = [
                        {"role": "system", "content": AGENT_SYSTEM}
                    ]
                    for episode_step in episode_steps:
                        source_step = episode_step["source_step"]
                        messages.extend(
                            [
                                {
                                    "role": "user",
                                    "content": _sft_agent_prompt(
                                        trajectory, source_step
                                    ),
                                },
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
                        run_dir / output_name,
                        {
                            "messages": messages,
                            "source_task_id": task_id,
                            "source_steps": source_steps,
                            "recorded_actions": [
                                item["recorded_action"] for item in episode_steps
                            ],
                            "target_actions": [
                                item["target_action"] for item in episode_steps
                            ],
                            "label_type": episode["label_type"],
                            "validation_status": episode["validation_status"],
                            "validation_basis": episode["validation_basis"],
                            "selection_rationale": episode["rationale"],
                            "context_before": context_before,
                            "trajectory_path": str(
                                (task_dir / "trajectory.json").relative_to(run_dir)
                            ),
                            "model": config.model,
                            "retention_protocol": retention["protocol"],
                        },
                    )

            choice = retention["artifact_choice"]
            controller_choice = retention["controller_choice"]
        else:
            choice = "not_reviewed"
            controller_choice = "not_reviewed"
        item = {
            "task_id": task_id,
            "success": evaluation["success"],
            "answer": answer,
            "choice": choice,
            "controller_choice": controller_choice,
            "memory_entries_after": len(memory_bank),
            "steps": len(trajectory_steps),
            "error": error,
        }
        summary.append(item)
        write_json(run_dir / "summary.json", summary)
        print(
            f"  success={item['success']} steps={item['steps']} choice={choice} "
            f"memory={item['memory_entries_after']}",
            flush=True,
        )

    write_json(
        run_dir / "run_result.json",
        {
            "finished_at": _now(),
            "tasks": summary,
            "choice_counts": {
                name: sum(item["choice"] == name for item in summary)
                for name in (
                    "both",
                    "context_only",
                    "sft_only",
                    "neither",
                    "not_reviewed",
                )
            },
            "controller_choice_counts": {
                name: sum(item["controller_choice"] == name for item in summary)
                for name in (
                    "both",
                    "context_only",
                    "sft_only",
                    "neither",
                    "not_reviewed",
                )
            },
            "successes": sum(item["success"] for item in summary),
            "total": len(summary),
        },
    )
    return run_dir
