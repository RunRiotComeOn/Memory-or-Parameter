"""BabyAI counterpart of `alfworld_sft_writer` / `scienceworld_sft_writer`.

Only the two system prompts are domain-specific; the Gemini client, retry
and validation plumbing, `build_writer_payload`, `validate_writer_output`,
`guidance_block` and `training_messages` come unchanged from
`appworld_sft_writer`.

What a BabyAI plan has to get right, and why it differs from the other
text-game domains:

- **The action list is enumerated every turn and names objects by index**
  (`pickup grey key 2`, `go to red ball 1`). A plan that says "the grey key"
  when three grey keys are visible is not actionable, and a near-miss string
  is rejected outright -- the same exact-name discipline ScienceWorld needed.
- **High-level actions exist and are strictly better.** `go to red ball 1`
  walks there in one turn; doing it with `turn left`/`move forward` costs
  many primitive steps, and reward is DISCOUNTED by primitive steps consumed
  (measured: solved episodes score 0.83-0.99, never 1.0). So a plan that
  spells out a turn-by-turn route is worse than one that names the
  high-level action, even when both reach the goal.
- **Failure is usually "never found it" or "acted on the wrong object",**
  not a wrong multi-step procedure. Rooms are small; what generalizes is
  search order and disambiguation, not long recipes.
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

_DOMAIN_BRIEF = """BabyAI is a gridworld navigation and object-manipulation environment presented entirely as text. Each turn the agent sees its goal, an egocentric description of the room ("There is a grey key 1 1 steps in front of you and 1 steps to your left"), and a list of VALID ACTIONS it must copy from verbatim.

Three properties decide whether a plan is useful here:

- Actions name objects by colour, type AND index (`pickup grey key 2`, `go to red ball 1`). With several same-coloured objects in view, a plan that says "the grey key" does not identify an action; the exact string is required and a near-miss is rejected.
- The list mixes low-level moves (`turn left`, `move forward`) with high-level ones (`go to red ball 1`) that cover the whole approach in one turn. Reward is discounted by how many primitive steps were consumed, so the high-level action is both faster and better scored. A turn-by-turn route is a worse plan than one naming the right high-level action.
- The room is small and the usual failure is not finding the target, or acting on a similar-looking wrong object -- not executing a long procedure incorrectly."""


BABYAI_SFT_WRITER_SYSTEM = f"""You are the plan-writing policy inside an SFT-data tool for a BabyAI gridworld agent.

{_DOMAIN_BRIEF}

A separate routing controller has already decided this task is worth training on. You receive the task instruction, the full previous attempt (its chosen actions and the environment's real text responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME task.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- went to or picked up a same-coloured object of the wrong type or index, wandered with `turn`/`move forward` when a high-level `go to ...` was already offered, never explored the direction the target was in, or repeated a command the environment does not accept. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same goal reached directly, with wandering and redundant looks removed, and with the high-level action named instead of the primitive route. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the actions that actually did the work.

In both cases the plan is followed by a live agent that reads the environment's real action list each turn and picks its own actions; it is NOT executed verbatim as a fixed action sequence. So say which object to target and how to recognize it among look-alikes, which high-level action to prefer, and where to explore first if it is not visible at the start -- using object names quoted exactly from the attempt's own transcript. Do not invent an object or index you did not see evidence for.

Return exactly one JSON object:
{{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}}"""


GEMINI_TEACHER_SYSTEM = f"""You are an expert teacher reviewing a BabyAI gridworld agent's previous attempt at a task.

{_DOMAIN_BRIEF}

You receive the task instruction, the full previous attempt (its chosen actions and the environment's real text responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- targeted a same-coloured object of the wrong type or index, wandered with primitive moves when a high-level `go to ...` was offered, never explored where the target was, or repeated an unaccepted command. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the goal reached directly, wandering removed, the high-level action named rather than the primitive route. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the actions that did the work.

In both cases the plan is followed by a live student agent that reads the real action list each turn and picks its own actions; it is NOT executed verbatim. So say which object to target and how to tell it from look-alikes, which high-level action to prefer, and where to search first -- using object names quoted exactly from the attempt's transcript. Do not invent a name or index you did not see evidence for: if the attempt never found the target, say where to search, not where it is.

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
    with the BabyAI-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
