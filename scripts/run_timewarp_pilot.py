#!/usr/bin/env python3
"""Run Qwen trajectory/memory/action-SFT experiments on local TimeWarp tasks."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gymnasium as gym
from PIL import Image
from browsergym.utils.obs import flatten_axtree_to_str

import browsergym.timewarp  # noqa: F401 - registers the Gym environments

from trajectory_memory_lab.model_client import ModelClient
from trajectory_memory_lab.retention import (
    memory_for_agent,
    normalize_memory_bank,
    run_retention_tools,
)
from trajectory_memory_lab.storage import append_jsonl, write_json


AGENT_SYSTEM = """You are a browser agent operating a local TimeWarp website. Complete the task using the current accessibility tree. Element identifiers are strings shown as `bid` attributes and are valid only for the current observation.

Return exactly one JSON object and no prose. Every object MUST include a short `note` field that cumulatively preserves task-local findings (including which required pages/items have and have not been checked) for later turns. For example: {"note":"Biology checked: X; Physics not checked yet","action":"goto","url":"http://127.0.0.1:5000/wiki/Physics"}. Choose one action:
- {"action":"click","bid":"..."}
- {"action":"fill","bid":"...","value":"..."}
- {"action":"press","bid":"...","key":"Enter"}
- {"action":"select_option","bid":"...","value":"..."}
- {"action":"hover","bid":"..."}
- {"action":"scroll","delta_y":INTEGER}
- {"action":"goto","url":"http://127.0.0.1:PORT/local/path"}
- {"action":"go_back"}
- {"action":"tab_focus","index":INTEGER}
- {"action":"new_tab"}
- {"action":"tab_close"}
- {"action":"noop"}
- {"action":"finish","answer":"..."}

Use only one action per turn. Positive scroll delta moves down; negative moves up. Never invent a bid: only bracketed identifiers actually shown in the observation are valid. If a desired element (especially the bottom search box) has no identifier, scroll until it is visible or use goto for a predictable local article URL. On this local Wiki, article URLs always use `/wiki/<article title>` (with URL encoding as needed); do not invent `/search` or root-level article paths. You may use goto only for a URL on 127.0.0.1. Finish only when you have gathered all information required by the task. Do not revisit already checked pages without a concrete reason, and do not alternate between the same pages. Do not navigate to non-local websites."""

MULTI_TURN_AGENT_SYSTEM = AGENT_SYSTEM.replace(
    "Every object MUST include a short `note` field that cumulatively preserves "
    "task-local findings (including which required pages/items have and have not "
    "been checked) for later turns. For example: "
    '{"note":"Biology checked: X; Physics not checked yet","action":"goto",'
    '"url":"http://127.0.0.1:5000/wiki/Physics"}. ',
    "The complete conversation is retained across turns, so a `note` field is optional. ",
)


ALLOWED_ACTIONS = {
    "click",
    "fill",
    "press",
    "select_option",
    "hover",
    "scroll",
    "goto",
    "go_back",
    "tab_focus",
    "new_tab",
    "tab_close",
    "noop",
    "finish",
}


def _memory_for_prompt(memory: list[dict[str, Any]], max_chars: int = 18_000):
    return memory_for_agent(memory, max_chars=max_chars)


def _observation_text(obs: dict[str, Any], max_chars: int = 48_000) -> str:
    tree = flatten_axtree_to_str(
        obs["axtree_object"],
        extra_properties=obs.get("extra_element_properties"),
        with_visible=True,
        with_clickable=True,
        # Keep identifiers for below-the-fold controls. Playwright will scroll
        # them into view when acted on, and old TimeWarp themes put Search last.
        hide_bid_if_invisible=False,
    )
    pages = [
        {"index": i, "url": url, "title": obs["open_pages_titles"][i]}
        for i, url in enumerate(obs["open_pages_urls"])
    ]
    rendered = (
        f"ACTIVE PAGE INDEX: {int(obs['active_page_index'][0])}\n"
        f"OPEN PAGES: {json.dumps(pages, ensure_ascii=False)}\n"
        f"CURRENT URL: {obs['url']}\n"
        f"LAST ACTION ERROR: {obs.get('last_action_error', '')}\n\n"
        f"ACCESSIBILITY TREE:\n{tree}"
    )
    if len(rendered) <= max_chars:
        return rendered
    # Long article pages often put exactly the useful lists/related links at the
    # bottom. Preserve both ends instead of silently dropping all tail evidence.
    head_chars = max_chars * 3 // 5
    tail_chars = max_chars - head_chars
    omitted = len(rendered) - max_chars
    return (
        rendered[:head_chars]
        + f"\n\n...[{omitted} characters omitted from the middle]...\n\n"
        + rendered[-tail_chars:]
    )


def _agent_prompt(
    goal: str,
    memory: list[dict[str, Any]],
    observation: str,
    recent_steps: list[dict[str, Any]],
) -> str:
    def history_excerpt(text: str, max_chars: int = 4_000) -> str:
        if len(text) <= max_chars:
            return text
        return (
            text[:1_000]
            + "\n...[earlier observation middle omitted]...\n"
            + text[-3_000:]
        )

    compact_history = [
        {
            "index": step["index"],
            "url": step["url"],
            "action": step["action"],
            "action_code": step.get("action_code"),
            "action_error": step.get("action_error"),
            "reward": step.get("reward"),
            "observation_excerpt": history_excerpt(step["observation"]),
        }
        for step in recent_steps[-3:]
    ]
    return (
        f"TASK:\n{goal}\n\n"
        "CONTEXT ACCUMULATED FROM EARLIER TRAJECTORIES "
        "(it may be empty; use it at your own discretion):\n"
        f"{json.dumps(_memory_for_prompt(memory), ensure_ascii=False, indent=2)}\n\n"
        f"RECENT STEPS:\n{json.dumps(compact_history, ensure_ascii=False, indent=2)}\n\n"
        f"CURRENT OBSERVATION:\n{observation}\n\n"
        f"TASK REMINDER:\n{goal}"
    )


def _agent_turn_prompt(
    goal: str,
    memory: list[dict[str, Any]],
    observation: str,
    previous_step: dict[str, Any] | None,
) -> str:
    """Build one incremental user turn for persistent multi-turn acting."""
    if previous_step is None:
        return (
            f"TASK:\n{goal}\n\n"
            "CONTEXT ACCUMULATED FROM EARLIER TRAJECTORIES "
            "(it may be empty; use it at your own discretion):\n"
            f"{json.dumps(_memory_for_prompt(memory), ensure_ascii=False, indent=2)}\n\n"
            f"CURRENT OBSERVATION:\n{observation}\n\n"
            f"TASK REMINDER:\n{goal}"
        )
    feedback = {
        "previous_action_error": previous_step.get("action_error"),
        "previous_reward": previous_step.get("reward"),
        "current_url": previous_step.get("next_url"),
    }
    return (
        "ENVIRONMENT FEEDBACK AFTER YOUR PREVIOUS ACTION:\n"
        f"{json.dumps(feedback, ensure_ascii=False, indent=2)}\n\n"
        f"CURRENT OBSERVATION:\n{observation}\n\n"
        f"TASK REMINDER:\n{goal}"
    )


def _action_code(action: dict[str, Any]) -> str:
    kind = str(action.get("action", "")).lower()
    if kind not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action: {kind!r}")
    if kind == "click":
        return f"click({str(action['bid'])!r})"
    if kind == "fill":
        return f"fill({str(action['bid'])!r}, {str(action['value'])!r})"
    if kind == "press":
        return f"press({str(action['bid'])!r}, {str(action.get('key', 'Enter'))!r})"
    if kind == "select_option":
        return f"select_option({str(action['bid'])!r}, {str(action['value'])!r})"
    if kind == "hover":
        return f"hover({str(action['bid'])!r})"
    if kind == "scroll":
        return f"scroll(0, {int(action.get('delta_y', 600))})"
    if kind == "goto":
        url = str(action["url"])
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname != "127.0.0.1":
            raise ValueError(f"goto only permits local TimeWarp URLs: {url!r}")
        return f"goto({url!r})"
    if kind == "go_back":
        return "go_back()"
    if kind == "tab_focus":
        return f"tab_focus({int(action['index'])})"
    if kind in {"new_tab", "tab_close", "noop"}:
        return f"{kind}()"
    return f"send_msg_to_user({str(action['answer'])!r})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", default="1")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("timewarp_runs"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--review-max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument(
        "--no-tool-audit",
        action="store_true",
        help="Skip the second 35B pass that audits each retention tool proposal.",
    )
    parser.add_argument(
        "--no-memory-during-run",
        action="store_true",
        help="Collect reviewer context but keep the acting agent memory empty for every task.",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--multi-turn-agent",
        action="store_true",
        help="Keep one persistent system/user/assistant conversation per task.",
    )
    parser.add_argument(
        "--multi-turn-observation-chars",
        type=int,
        default=8_000,
        help="Per-turn observation cap used to keep a complete episode within context.",
    )
    parser.add_argument("--initial-memory", type=Path)
    args = parser.parse_args()
    agent_system = MULTI_TURN_AGENT_SYSTEM if args.multi_turn_agent else AGENT_SYSTEM

    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True)
    memory = normalize_memory_bank(
        json.loads(args.initial_memory.read_text()) if args.initial_memory else []
    )
    client = ModelClient(
        base_url=args.base_url,
        api_key="EMPTY",
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=not args.disable_thinking,
    )
    review_client = ModelClient(
        base_url=args.base_url,
        api_key="EMPTY",
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.review_max_tokens,
        seed=args.seed,
        enable_thinking=False,
    )
    manifest = {**vars(args), "task_ids": task_ids}
    manifest["output_root"] = str(args.output_root)
    manifest["initial_memory"] = (
        str(args.initial_memory) if args.initial_memory else None
    )
    write_json(run_dir / "manifest.json", manifest)
    if memory:
        write_json(run_dir / "memory_bank.json", memory)
    summary = []

    for ordinal, task_id in enumerate(task_ids):
        task_dir = run_dir / "tasks" / f"{ordinal:02d}_timewarp_{task_id}"
        (task_dir / "screenshots").mkdir(parents=True)
        env = gym.make(f"browsergym/timewarp.{task_id}", headless=True)
        steps, error, final_reward = [], None, 0.0
        started = time.monotonic()
        try:
            obs, info = env.reset(seed=args.seed)
            goal = info["task_info"]["goal"]
            conversation: list[dict[str, str]] = [
                {"role": "system", "content": agent_system}
            ]
            print(
                f"[{ordinal + 1}/{len(task_ids)}] timewarp.{task_id}: {goal}",
                flush=True,
            )
            for step_index in range(args.max_steps):
                Image.fromarray(obs["screenshot"]).save(
                    task_dir / "screenshots" / f"step_{step_index:02d}.png"
                )
                observation = _observation_text(
                    obs,
                    max_chars=(
                        args.multi_turn_observation_chars
                        if args.multi_turn_agent
                        else 48_000
                    ),
                )
                acting_memory = [] if args.no_memory_during_run else memory
                if args.multi_turn_agent:
                    prompt = _agent_turn_prompt(
                        goal,
                        acting_memory,
                        observation,
                        steps[-1] if steps else None,
                    )
                    conversation.append({"role": "user", "content": prompt})
                    reply = client.json_chat_messages(conversation)
                else:
                    prompt = _agent_prompt(goal, acting_memory, observation, steps)
                    reply = client.json_chat(system=agent_system, user=prompt)
                action = reply.parsed
                if args.multi_turn_agent:
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                action,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
                try:
                    code = _action_code(action)
                    next_obs, reward, terminated, truncated, env_info = env.step(code)
                    action_error = str(next_obs.get("last_action_error", "")) or None
                    final_reward = float(reward)
                except Exception as exc:
                    code = None
                    action_error = f"{type(exc).__name__}: {exc}"
                    next_obs, terminated, truncated, env_info = obs, False, False, {}
                    reward = 0.0
                steps.append(
                    {
                        "index": step_index,
                        "url": obs["url"],
                        "observation": observation,
                        "agent_prompt": prompt,
                        "action": action,
                        "action_code": code,
                        "action_error": action_error,
                        "reward": float(reward),
                        "next_url": next_obs.get("url"),
                        "model_content": reply.content,
                        "model_reasoning": reply.reasoning,
                        "model_usage": reply.usage,
                    }
                )
                write_json(
                    task_dir / "trajectory.partial.json",
                    {
                        "task_id": task_id,
                        "goal": goal,
                        "context_before": list(memory),
                        "steps": steps,
                    },
                )
                print(
                    f"    step={step_index} action={json.dumps(action, ensure_ascii=False)} "
                    f"reward={float(reward)} error={action_error}",
                    flush=True,
                )
                obs = next_obs
                if terminated or truncated:
                    break
            else:
                error = f"max_steps_exceeded:{args.max_steps}"
            success = final_reward > 0
        except Exception as exc:
            goal = locals().get("goal", "")
            success = False
            error = f"{type(exc).__name__}: {exc}"
        finally:
            env.close()

        trajectory = {
            "task_id": task_id,
            "source_task_id": f"timewarp.{task_id}",
            "goal": goal,
            "context_before": [] if args.no_memory_during_run else list(memory),
            "steps": steps,
            "reward": final_reward,
            "success": success,
            "error": error,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(task_dir / "trajectory.json", trajectory)

        choice = "not_reviewed"
        controller_choice = "not_reviewed"
        if not args.no_review:
            retention = run_retention_tools(
                review_client,
                trajectory=trajectory,
                memory_bank=memory,
                agent_system=agent_system,
                allowed_actions=ALLOWED_ACTIONS,
                action_validator=_action_code,
                audit=not args.no_tool_audit,
            )
            write_json(task_dir / "retention_decision.json", retention)
            memory_application = (retention["tools"].get("edit_memory") or {}).get(
                "application", {}
            )
            if memory_application.get("applied"):
                write_json(run_dir / "memory_bank.json", memory)

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
                    if source_steps != list(range(len(steps))):
                        continue
                    messages: list[dict[str, str]] = [
                        {"role": "system", "content": agent_system}
                    ]
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
                        run_dir / output_name,
                        {
                            "messages": messages,
                            "source_task_id": f"timewarp.{task_id}",
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
                            "trajectory_path": str(
                                (task_dir / "trajectory.json").relative_to(run_dir)
                            ),
                            "model": args.model,
                            "retention_protocol": retention["protocol"],
                        },
                    )
            choice = retention["artifact_choice"]
            controller_choice = retention["controller_choice"]

        item = {
            "task_id": task_id,
            "success": success,
            "reward": final_reward,
            "steps": len(steps),
            "choice": choice,
            "controller_choice": controller_choice,
            "memory_entries_after": sum(
                entry.get("status") == "active" for entry in memory
            ),
            "error": error,
        }
        summary.append(item)
        write_json(run_dir / "summary.json", summary)
        print(
            f"  success={success} reward={final_reward} steps={len(steps)} choice={choice} error={error}",
            flush=True,
        )

    write_json(
        run_dir / "run_result.json",
        {
            "successes": sum(x["success"] for x in summary),
            "total": len(summary),
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
            "tasks": summary,
        },
    )
    print(f"Run written to {run_dir}")


if __name__ == "__main__":
    main()
