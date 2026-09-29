"""TextCraft counterpart of `babyai_sft_writer` / `scienceworld_sft_writer`.

Only the two system prompts are domain-specific; the Gemini client, retry
and validation plumbing, `build_writer_payload`, `validate_writer_output`,
`guidance_block` and `training_messages` come unchanged from
`appworld_sft_writer`.

What a TextCraft plan has to get right, and why it differs from the other
text-game domains:

- **The useful unit is the subgoal tree, not the action sequence.** The goal
  decomposes into intermediates that must exist before the recipes above
  them can fire. A plan that names the chain -- "crimson stems -> crimson
  planks -> crimson button" -- transfers; a plan that lists turns does not,
  because the agent must also satisfy quantities it discovers along the way.
- **Counts are where correct plans fail.** A recipe yielding 4 planks feeds
  several later steps, while one consuming 6 planks needs the producing
  recipe run twice. This is the single most common repairable mistake and a
  plan that ignores arithmetic is not actionable.
- **`get` only works on items with no recipe.** Trying to fetch a craftable
  item fails with the same "Could not find X" message an invalid name
  produces, so plans must say explicitly which items are base materials.
- **Up to 10 distractor recipes are mixed in.** Naming the relevant subset
  is part of the plan's value.
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

_DOMAIN_BRIEF = """TextCraft is a Minecraft crafting game presented entirely as text. Each episode shows a list of crafting recipes and one goal item; the agent must decompose the goal into subgoals, fetch base materials, and craft upward until the goal item is in its inventory.

The agent has exactly three actions: `craft <output> using <inputs>` (copied from a recipe in the list), `get <count> <item>` (base materials only -- anything with a recipe must be crafted), and `inventory`.

Four properties decide whether a plan is useful here:

- What transfers is the SUBGOAL CHAIN, not a turn-by-turn script. "crimson stems -> crimson planks -> crimson button" is the content; the agent works out the rest from the recipe list it can see.
- QUANTITIES are where otherwise-correct plans fail. A recipe producing 4 planks may cover several later steps, while a step consuming 6 planks requires running the producing recipe twice. A plan that does not do this arithmetic is not actionable.
- `get` fails on anything craftable, with the same "Could not find X" message an invalid item name produces. So a plan must say which items are base materials to be fetched and which must be crafted.
- Up to 10 recipes in the list are DISTRACTORS with no bearing on the goal. Identifying the relevant subset is part of the work."""


TEXTCRAFT_SFT_WRITER_SYSTEM = f"""You are the plan-writing policy inside an SFT-data tool for a TextCraft agent.

{_DOMAIN_BRIEF}

A separate routing controller has already decided this task is worth training on. You receive the goal, the recipe list, the full previous attempt (its chosen actions and the environment's real responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME task.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- tried to `get` an item that has a recipe, crafted in an order that consumed an intermediate it still needed, ran out of a material because it did not account for how many a recipe produces or consumes, pursued a distractor recipe, or repeated an action the environment had already rejected. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach that worked -- the same subgoal chain with dead ends and redundant `inventory` checks removed, and the quantities stated. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the actions that did the work.

In both cases the plan is followed by a live agent that reads the real recipe list itself and chooses its own actions; it is NOT executed verbatim. So state the subgoal chain from base materials up to the goal, say how many of each intermediate are needed and why, and name which items are fetched with `get` -- quoting item names exactly as the recipe list writes them. Do not invent a recipe or an item that is not in the list you were shown.

Return exactly one JSON object:
{{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}}"""


GEMINI_TEACHER_SYSTEM = f"""You are an expert teacher reviewing a TextCraft agent's previous attempt at a crafting task.

{_DOMAIN_BRIEF}

You receive the goal, the recipe list, the full previous attempt (its chosen actions and the environment's real responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- tried to `get` a craftable item, crafted in an order that destroyed an intermediate it still needed, miscounted how many units a recipe produces or consumes, chased a distractor recipe, or repeated a rejected action. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES what worked -- the subgoal chain with dead ends removed and the quantities made explicit. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the actions that did the work.

In both cases the plan is followed by a live student agent that reads the recipe list itself and picks its own actions; it is NOT executed verbatim. So give the subgoal chain from base materials up to the goal, the quantity of each intermediate and the arithmetic behind it, and which items are base materials fetched with `get` -- quoting names exactly as the recipe list writes them. Do not invent a recipe or item you were not shown: if the attempt never found a route, say which intermediate to build first, not a recipe you assume exists.

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
    with the TextCraft-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
