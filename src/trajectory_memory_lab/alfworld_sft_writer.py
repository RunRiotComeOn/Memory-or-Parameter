"""ALFWorld counterpart of appworld_sft_writer.py (DESIGN.md section 15).

Same job, same output shape, same two interchangeable writers (self / teacher,
teacher default) -- everything below the SYSTEM PROMPT text is already
domain-agnostic (`build_writer_payload`, `validate_writer_output`,
`generate_plan_with_teacher`'s Gemini-client plumbing, `guidance_block`,
`training_messages`, the `GUIDANCE_MARKER*` constants), so this module only
defines ALFWorld-specific prompts and re-exports the shared machinery from
`appworld_sft_writer` rather than duplicating it.

Why a separate prompt at all, not just reuse AppWorld's verbatim: AppWorld's
prompts explicitly instruct "name the concrete APIs to call... e.g.
apis.spotify.login" -- sending that to a transcript of "go to fridge 1",
"take mug 1 from countertop 1" would make the model invent API calls that do
not exist in this domain, actively worse than no plan (this is exactly why
`router_bank_builder.run_router_chain` skipped SFT drafting for
`domain="alfworld"` until this module existed -- see its docstring).

No probe has run head-to-head self-vs-teacher on ALFWorld yet the way
`probe_sft_repair_yield_gemini_teacher.py` did for AppWorld (running_log.md
section 11: teacher 62.5% rescue yield vs. self-writer's actively-harmful
~7%). Defaulting to teacher here anyway, before that evidence exists for this
domain, is a deliberate bet that the same qualitative reason still holds: an
external, generally stronger model reviewing a transcript it did not produce
itself catches mistakes a same-size self-critique tends to rationalize away.
Revisit if an ALFWorld-specific probe says otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .appworld_sft_writer import (  # noqa: F401  (re-exported, not just used internally)
    GUIDANCE_MARKER,
    GUIDANCE_MARKER_CONSOLIDATE,
    GUIDANCE_MARKERS,
    DEFAULT_TEACHER_API_KEY_FILE,
    DEFAULT_TEACHER_MODEL,
    build_writer_payload,
    generate_plan_with_teacher as _generate_plan_with_teacher_impl,
    guidance_block,
    training_messages,
    validate_writer_output,
)

ALFWORLD_SFT_WRITER_SYSTEM = """You are the plan-writing policy inside an SFT-data tool for an ALFWorld household-task agent.

ALFWorld is a text-adventure game, not a coding benchmark: each turn the agent picks ONE command from a list of admissible commands shown to it (e.g. "go to fridge 1", "open fridge 1", "take mug 1 from countertop 1", "heat mug 1 with microwave 1", "put mug 1 in/on coffeetable 1"), and the environment replies with a short text observation plus the new admissible-commands list.

A separate routing controller has already decided this task is worth training on. You receive the task instruction, the full previous attempt (its chosen commands and the environment's real text responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`won`). Your job is to write a PLAN for a fresh attempt at the SAME task.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake it made (wrong receptacle, forgot to open a closed receptacle before taking from it, tried a command not in the admissible list, went to the wrong room object, dropped/never picked up the target object, used the wrong appliance for heat/cool/clean), put it in `mistake_summary`, and cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same task done directly, with wasted `look`/`examine` turns, wrong-room detours and redundant re-checks of admissible commands left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the commands that actually did the work.

In both cases the plan is followed by a live agent that reads the environment's real admissible-commands list each turn and picks its own actions; it is NOT executed verbatim as a fixed command sequence. So describe the sequence of receptacles/objects to interact with and in what order (using exact object/receptacle names you saw in the attempt, e.g. "microwave 1", not "the microwave"), and what state to verify (receptacle open, object in inventory) before the final placement command. Do not invent an object or receptacle name you did not see evidence for in the attempt's own transcript.

Return exactly one JSON object:
{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}"""


GEMINI_TEACHER_SYSTEM = """You are an expert teacher reviewing an ALFWorld household-task agent's previous attempt at a task. ALFWorld is a text-adventure game: each turn the agent picks ONE command from a list of admissible commands shown to it (e.g. "go to fridge 1", "open fridge 1", "take mug 1 from countertop 1", "heat mug 1 with microwave 1", "put mug 1 in/on coffeetable 1"), and the environment replies with a short text observation plus the new admissible-commands list.

You receive the task instruction, the full previous attempt (its chosen commands and the environment's real text responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`won`). Your job is to write a PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake it made (wrong receptacle, forgot to open a closed receptacle before taking from it, tried a command not in the admissible list, went to the wrong room object, dropped/never picked up the target object, used the wrong appliance for heat/cool/clean, misread which object instance was meant), put it in `mistake_summary`, and cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same task done directly, with wasted `look`/`examine` turns, wrong-room detours and redundant re-checks of admissible commands left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the commands that actually did the work.

In both cases the plan is followed by a live student agent that reads the environment's real admissible-commands list each turn and picks its own actions; it is NOT executed verbatim as a fixed command sequence. So describe the sequence of receptacles/objects to interact with and in what order (using exact object/receptacle names you saw in the attempt's transcript, e.g. "microwave 1", not "the microwave"), and what state to verify (receptacle open, object in inventory) before the final placement command. Do not invent an object or receptacle name you did not see evidence for -- if the attempt never found the right object, say what room/receptacle to search, not what the answer is.

Return exactly one JSON object matching the schema."""


def generate_plan_with_teacher(
    trajectory: dict[str, Any],
    *,
    model: str = DEFAULT_TEACHER_MODEL,
    api_key_file: Path = DEFAULT_TEACHER_API_KEY_FILE,
    temperature: float = 0.1,
    max_retries: int = 6,
) -> dict[str, Any] | None:
    """Same Gemini-client plumbing as `appworld_sft_writer.generate_plan_with_teacher`,
    with the ALFWorld-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
