"""AppWorld counterpart of tau_sft_data_writer.py (DESIGN.md section 15).

Two interchangeable writers produce the exact same output shape
(`validate_writer_output`'s `{"plan", "mistake_summary", "evidence_steps"}`),
so nothing downstream (guidance injection, guided replay, router feature
hashing) needs to know or care which one ran:

  self  -- `APPWORLD_SFT_WRITER_SYSTEM` via the SAME base model that plays
           the task agent, through the local `ModelClient`. No separately
           trained component, just a different system prompt on qwen35-tau.
  teacher (DEFAULT) -- `generate_plan_with_teacher`, an external, generally
           stronger model (Gemini) via the `google-genai` SDK. Probed head
           -to-head against `self` in `probe_sft_repair_yield.py` /
           `probe_sft_repair_yield_gemini_teacher.py` on the same 33 failed
           training tasks: the self-writer probe never finished a comparable
           run, but the teacher probe scored 15/24 = 62.5% rescue yield on
           tasks that genuinely still failed on a fresh, guidance-free
           attempt (running_log.md section 11) -- good enough to make
           teacher the default rather than an opt-in experiment.

The tau2-bench SFT-data writer produces a full scripted `assistant_turns`
candidate because tau2's replay harness needs an exact tool-call script to
feed a guided agent. AppWorld has no simulated user turn-taking to script
around (`appworld_agent.run_task` is a plain code-write -> execute -> observe
loop against a real sandbox), so a literal scripted replay would break the
moment any API call returns something the writer didn't predict verbatim (an
id, a timestamp, a page of results).

Instead both writers produce a natural-language PLAN -- what to call, in what
order, what the previous attempt got wrong (or, when it got the task right,
how to do it without that attempt's dead ends) -- injected into a fresh
attempt the exact same way `memory_block` already is (see
`appworld_agent.build_initial_user_message`). A live agent then executes it
against the real environment and adapts to whatever actually comes back.
Only a replay that AppWorld itself scores as `success=True` is kept as SFT
training data (`router_sft_pool.append_if_verified`), and the TRAINING
EXAMPLE is that replay's own real transcript, not the writer's plan text --
the plan is a hint, never the label.

v6: this runs on successful trajectories too. Through v5 the caller only
invoked it when the previous attempt had failed, which quietly made the
router's `sft` route a no-op on every task the agent already solved (see
`router_bank_builder.run_router_chain`). The success case is a different
job, not the repair job with a missing mistake -- consolidating a verified
correct solution rather than fixing a broken one -- so both writers branch
on `success` instead of asserting failure, and `mistake_summary` is allowed
to be empty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

APPWORLD_SFT_WRITER_SYSTEM = """You are the plan-writing policy inside an SFT-data tool for an AppWorld coding agent.

A separate routing controller has already decided this task is worth training on. You receive the task instruction, the full previous attempt (its code turns and the environment's real outputs), and the evaluator's verdict -- including whether that attempt SUCCEEDED. Your job is to write a PLAN for a fresh attempt at the SAME task.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake it made (wrong argument, missing login, unchecked pagination, premature complete_task, wrong app), put it in `mistake_summary`, and cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same task done cleanly and directly, with the dead ends, redundant discovery calls and abandoned branches of the original attempt left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the steps that actually did the work. Say plainly if the successful route was already direct and there is nothing to trim.

In both cases the plan is followed by a live agent that writes its own code and reads real API responses; it is not executed verbatim. So name the concrete APIs to call and in what order (use exact names you saw in the attempt's API-discovery calls, e.g. `apis.spotify.login`, not vague descriptions) and what to verify before calling `apis.supervisor.complete_task`. Do not invent API names or arguments you did not see evidence for in the attempt or its discovery calls.

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


GEMINI_TEACHER_SYSTEM = """You are an expert teacher reviewing an AppWorld coding agent's previous attempt at a task. AppWorld is not a tool-call benchmark: the agent writes Python code that calls `apis.<app>.<api>(...)` against a set of app APIs and reads back whatever the environment prints.

You receive the task instruction, the full previous attempt (its code turns and the environment's real outputs), and the evaluator's verdict -- including whether that attempt SUCCEEDED. Your job is to write a PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake it made (wrong argument, missing login, unchecked pagination, premature complete_task, wrong app, misread API doc), put it in `mistake_summary`, and cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same task done cleanly and directly, with the dead ends, redundant discovery calls and abandoned branches of the original attempt left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the steps that actually did the work.

In both cases the plan is followed by a live student agent that writes its own code and reads real API responses; it is NOT executed verbatim. So name the concrete APIs to call and in what order (use exact names you saw in the attempt's API-discovery calls, e.g. `apis.spotify.login`, not vague descriptions), and what to verify before calling `apis.supervisor.complete_task`. Do not invent API names or arguments you did not see evidence for in the attempt or its discovery calls -- if the attempt never discovered the right API, say what to search for (`apis.api_docs.search_api_docs`), not what the answer is.

Return exactly one JSON object matching the schema."""

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "mistake_summary": {"type": "string"},
        "evidence_steps": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["plan", "mistake_summary", "evidence_steps"],
}

DEFAULT_TEACHER_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TEACHER_API_KEY_FILE = Path("/nas04/yixuh/.config/continual-memory/gemini_api_key")


def generate_plan_with_teacher(
    trajectory: dict[str, Any],
    *,
    model: str = DEFAULT_TEACHER_MODEL,
    api_key_file: Path = DEFAULT_TEACHER_API_KEY_FILE,
    temperature: float = 0.1,
    max_retries: int = 6,
) -> dict[str, Any] | None:
    """External-teacher counterpart of a local `client.json_chat(system=
    APPWORLD_SFT_WRITER_SYSTEM, ...)` call -- same input (`build_writer_payload`),
    same output shape (this always runs its result through
    `validate_writer_output`, so callers cannot tell which writer produced a
    given plan). Returns None on any failure (bad/empty key, network error,
    exhausted retries, or a response that fails validation) so callers can
    treat it exactly like the self-writer's `except Exception: writer_output
    = None` path -- a probe/router call is not fatal just because Gemini is
    briefly unreachable.

    `DEFAULT_TEACHER_MODEL` names the model that worked as of 2026-09-18 --
    an earlier guess (`gemini-3-pro-preview`) had already been retired, with
    the API's own 404 naming this one as the replacement. Verify with
    `client.models.list()` if this starts 404ing again.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None

    key = api_key_file.read_text(encoding="utf-8").strip() if api_key_file.exists() else ""
    if not key:
        return None
    client = genai.Client(api_key=key)
    payload = build_writer_payload(trajectory)

    error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=GEMINI_TEACHER_SYSTEM,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_json_schema=GEMINI_RESPONSE_SCHEMA,
                ),
            )
            return validate_writer_output(json.loads(response.text))
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(30, 2 ** attempt))
    del error  # exhausted retries -- caller treats None like any other writer failure
    return None


# The repair marker is the ORIGINAL string and must stay byte-identical: it is
# written by the guided-replay subprocess and stripped by the training-example
# builder in the parent process, and a long-running v5 parent holds the old
# module in memory while launching subprocesses that load this file fresh. A
# rename would leave that parent unable to strip what its own subprocess wrote,
# leaking the guidance into its training data. New wording therefore arrives as
# an ADDITIONAL marker, and stripping accepts either.
GUIDANCE_MARKER = "Guidance from a repair plan for a previous failed attempt at this exact task:"
GUIDANCE_MARKER_CONSOLIDATE = (
    "Guidance from a plan that consolidates a previous SUCCESSFUL attempt at this exact task:"
)
GUIDANCE_MARKERS = (GUIDANCE_MARKER, GUIDANCE_MARKER_CONSOLIDATE)


def guidance_block(writer_output: dict[str, Any], *, previous_success: bool | None = None) -> str:
    """Formats exactly like a memory entry -- `run_task`'s `memory_block` is a
    plain text block appended to the initial user message, no special
    handling for where the text came from.

    v6: `previous_success` picks the wording, because the plan is no longer
    always a repair. Telling the agent its previous attempt failed when it
    actually succeeded is a false premise in the prompt it replays under.
    `None` keeps the legacy repair wording, so a caller that predates the flag
    (a v5 process still launching this subprocess) is unaffected.
    """
    marker = GUIDANCE_MARKER_CONSOLIDATE if previous_success else GUIDANCE_MARKER
    return f"\n{marker}\n{writer_output['plan']}"


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
        if index == 0:
            for marker in GUIDANCE_MARKERS:
                if marker in content:
                    content = content.split(marker, 1)[0].rstrip("\n")
                    break
        messages.append({"role": role, "content": content})
    return messages
