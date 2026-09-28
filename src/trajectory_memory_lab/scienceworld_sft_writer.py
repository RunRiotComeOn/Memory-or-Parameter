"""ScienceWorld counterpart of `alfworld_sft_writer` / `appworld_sft_writer`.

Only the two system prompts are domain-specific. The Gemini client, its
retry/validation plumbing, `build_writer_payload`, `validate_writer_output`,
`guidance_block` and `training_messages` are imported unchanged from
`appworld_sft_writer` -- the payload is built from the canonical trajectory
shape that all three agent modules already produce, so nothing about the
mechanics depends on the benchmark.

What the prompts have to say differently from ALFWorld's, and why:

- The action space is not a short admissible-command list but a verb x object
  product over the current room graph, and only a sample of it is shown when
  a room is object-rich (see `scienceworld_agent.MAX_ACTION_LIST_CHARS`). A
  plan therefore cannot assume the student will be shown any particular
  command; it has to name rooms, objects and devices so the student can find
  them itself.
- `focus on <object>` is a SCORING action, not an inspection: aiming it at the
  wrong object ends the episode immediately with a large penalty (measured:
  `focus on air` on a boil task -> score -100 at step 1). It is the single
  most common way an attempt dies here, so both prompts are told to treat it
  as the decisive commitment it is and to say exactly what to focus on.
- Object names are matched exactly by the environment, so a plan that says
  "the pea plant" when the environment calls it "adult pea plant" sends the
  student into the same near-miss loop the harness already measured. Plans
  must quote names verbatim from the attempt's own transcript.
- Scoring is incremental across sub-goals and the episode ends when the task
  is fully solved; there is no explicit finish command. A plan should say
  which sub-goals exist and in what order, not just the final action.
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

_DOMAIN_BRIEF = """ScienceWorld is a text-adventure environment built around the elementary science curriculum (states of matter, electrical circuits, plant and animal life cycles, mixtures, forces). Each turn the agent is shown the current room description or the result of its last action plus a list of VALID ACTIONS, and replies with exactly one command copied from that list (e.g. "go to kitchen", "open door to greenhouse", "pick up thermometer", "move adult pea plant to red box", "activate stove", "connect battery anode to cathode of red light bulb"). Rooms are connected by doors that must be opened before moving through them, and the agent only sees the room it is in.

Three properties decide whether a plan works here:

- `focus on <object>` is a SCORING COMMITMENT, not a look. Pointing it at the wrong object ends the episode immediately with a large score penalty -- it does not fail quietly. Most tasks require exactly one `focus on`, aimed at the substance or organism the instruction names.
- Object names must match the environment's own wording exactly. "adult pea plant" and "pea plant" are different strings and the second is rejected.
- Score accrues over sub-goals and the episode ends when the task is fully solved; there is no finish command. Getting the ordering of sub-goals right matters as much as the final action."""


SCIENCEWORLD_SFT_WRITER_SYSTEM = f"""You are the plan-writing policy inside an SFT-data tool for a ScienceWorld science-experiment agent.

{_DOMAIN_BRIEF}

A separate routing controller has already decided this task is worth training on. You receive the task instruction, the full previous attempt (its chosen commands and the environment's real text responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME task.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- focused on the wrong object and ended the episode, never opened the door to the room holding the target, used the wrong device (stove vs. freezer vs. sink), left the substance in the wrong container, never picked the target up, repeated a command the environment does not accept instead of reading the valid-actions list, or ran out of steps exploring rooms that could not contain the target. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same task done directly, with wasted `look around` turns, wrong-room detours and redundant re-reads of the valid-actions list left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the commands that actually did the work.

In both cases the plan is followed by a live agent that reads the environment's real valid-actions list each turn and picks its own commands; it is NOT executed verbatim as a fixed command sequence, and the list it sees may be only a sample of what is legal. So describe which rooms to go to and in what order, which object to `focus on` and at what point, which device performs the state change, and what to verify before the final action -- using object, room and device names quoted exactly from the attempt's own transcript. Do not invent a name you did not see evidence for; if the attempt never found the target, say which rooms and containers to search rather than asserting where it is.

Return exactly one JSON object:
{{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}}"""


GEMINI_TEACHER_SYSTEM = f"""You are an expert teacher reviewing a ScienceWorld agent's previous attempt at a science-experiment task.

{_DOMAIN_BRIEF}

You receive the task instruction, the full previous attempt (its chosen commands and the environment's real text responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- focused on the wrong object and ended the episode, never opened the door to the room holding the target, used the wrong device, left the substance in the wrong container, never picked the target up, repeated a command the environment does not accept, or ran out of steps searching rooms that could not contain the target. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same task done directly, with wasted `look around` turns, wrong-room detours and redundant re-reads of the valid-actions list left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the commands that actually did the work.

In both cases the plan is followed by a live student agent that reads the environment's real valid-actions list each turn and picks its own commands; it is NOT executed verbatim as a fixed command sequence, and the list it sees may be only a sample of what is legal. So describe which rooms to go to and in what order, which object to `focus on` and at what point, which device performs the state change, and what to verify before the final action -- using object, room and device names quoted exactly from the attempt's transcript. Do not invent a name you did not see evidence for: if the attempt never found the right object, say which rooms and containers to search, not what the answer is.

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
    with the ScienceWorld-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
