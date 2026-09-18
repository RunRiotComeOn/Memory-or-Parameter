"""AppWorld counterpart of tau_sft_data_writer.py (DESIGN.md section 15).

Same base model, no separately trained teacher -- this is a system prompt,
not a fine-tuned component. The tau2-bench SFT-data writer produces a full
scripted `assistant_turns` candidate because tau2's replay harness needs an
exact tool-call script to feed a guided agent. AppWorld has no simulated user
turn-taking to script around (`appworld_agent.run_task` is a plain code-write
-> execute -> observe loop against a real sandbox), so a literal scripted
replay would break the moment any API call returns something the writer
didn't predict verbatim (an id, a timestamp, a page of results).

Instead the writer produces a natural-language corrected PLAN -- what to call,
in what order, what the original attempt got wrong -- injected into a fresh
attempt the exact same way `memory_block` already is (see
`appworld_agent.build_initial_user_message`). A live agent then executes it
against the real environment and adapts to whatever actually comes back.
Only a replay that AppWorld itself scores as `success=True` is kept as SFT
training data (`router_sft_pool.append_if_verified`), and the TRAINING
EXAMPLE is that replay's own real transcript, not the writer's plan text --
the plan is a hint, never the label.
"""

from __future__ import annotations

import json
from typing import Any

APPWORLD_SFT_WRITER_SYSTEM = """You are the repair-planning policy inside an SFT-data tool for an AppWorld coding agent.

A separate routing controller has already decided this failed task is worth repairing. You receive the task instruction, the full failed attempt (its code turns and the environment's real outputs), and the evaluator's verdict. Your only job is to write a corrected PLAN for a fresh attempt at the SAME task -- not a summary of what went wrong, a plan for what to do right.

The plan is followed by a live agent that writes its own code and reads real API responses; it is not executed verbatim. So: name the concrete APIs to call and in what order (use exact names you saw in the failed attempt's API-discovery calls, e.g. `apis.spotify.login`, not vague descriptions), the specific mistake the failed attempt made (wrong argument, missing login, unchecked pagination, premature complete_task, wrong app), and what to verify before calling `apis.supervisor.complete_task`. Do not invent API names or arguments you did not see evidence for in the failed attempt or its discovery calls. Cite the step indexes that show the mistake.

Return exactly one JSON object:
{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}"""


def build_writer_payload(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_instruction": (trajectory.get("task") or {}).get("instruction"),
        "success": trajectory.get("success"),
        "termination_reason": trajectory.get("termination_reason"),
        "evaluation": trajectory.get("evaluation"),
        "steps": trajectory.get("steps", []),
    }


def validate_writer_output(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    plan = str(value.get("plan") or "").strip()
    if not plan:
        return None
    evidence = [
        step
        for step in (value.get("evidence_steps") or [])
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0
    ]
    return {
        "plan": plan,
        "mistake_summary": str(value.get("mistake_summary") or "").strip(),
        "evidence_steps": sorted(set(evidence))[:24],
    }


GUIDANCE_MARKER = "Guidance from a repair plan for a previous failed attempt at this exact task:"


def guidance_block(writer_output: dict[str, Any]) -> str:
    """Formats exactly like a memory entry -- `run_task`'s `memory_block` is a
    plain text block appended to the initial user message, no special
    handling for where the text came from."""
    return f"\n{GUIDANCE_MARKER}\n{writer_output['plan']}"


def training_messages(agent_system: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Real replay transcript -> one SFT training example.

    The guidance text must NOT end up in the training example: it was
    injected into the replay's initial user message so the live agent could
    use it (`guidance_block`, baked into steps[0]'s content by
    `appworld_agent.build_initial_user_message`), but a normal task at
    inference time never carries it. Training on the guidance-augmented
    prompt would teach the model to expect a hint it will never see again --
    a real train/inference mismatch, not a hypothetical one -- so it is
    stripped back out of steps[0] before this example is kept.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": agent_system}]
    for index, step in enumerate(steps):
        role = "user" if step["role"] in ("user", "tool") else "assistant"
        content = step["content"]
        if index == 0 and GUIDANCE_MARKER in content:
            content = content.split(GUIDANCE_MARKER, 1)[0].rstrip("\n")
        messages.append({"role": role, "content": content})
    return messages
