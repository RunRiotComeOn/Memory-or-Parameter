"""SQLGym / BIRD counterpart of `textcraft_sft_writer` / `babyai_sft_writer`.

Only the two system prompts are domain-specific; the Gemini client, retry
and validation plumbing, `build_writer_payload`, `validate_writer_output`,
`guidance_block` and `training_messages` come unchanged from
`appworld_sft_writer`.

What a BIRD plan has to get right, and why it differs from the other
domains:

- **The teacher is partly blind, by construction.** It reads the schema
  description and the transcript, but NOT the database. Whatever the agent
  did not surface with an EXPLORE, the teacher cannot see either. This is
  the property that makes the benchmark a test of
  `textcraft_summary.md` conclusion 4 (rescue rate tracks how much of the
  task is visible to the teacher), so the prompt must NOT invite the
  teacher to assert values it has no way to know.
- **Column names are the dominant failure.** The schema description
  lowercases and expands names for readability, but the real ones often
  carry spaces, capitals or punctuation and need backticks. A plan naming a
  column that does not exist is worse than no plan.
- **What transfers across schemas is the QUERY SHAPE** -- which joins,
  which aggregate, where the filter belongs, whether a subquery or window
  is needed -- not this database's particulars. The cross-schema split is
  the main evaluation line, so plans that only restate one schema's
  specifics are exactly the ones expected not to generalize.
- **Exploration strategy is part of the plan.** "Check the distinct values
  of X before filtering on it" is advice a fresh attempt can act on; a bare
  final query is not, because the agent still has to verify the names.
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

_DOMAIN_BRIEF = """BIRD is text-to-SQL over real SQLite databases. The agent sees a natural-language description of the schema and a question, and must produce one SQL query whose result set matches a reference query's. Correctness is all-or-nothing: the two result sets are compared as sets.

The agent works in turns and has exactly two commands: `EXPLORE: <sql>` runs a read-only query and shows up to a few hundred rows, and `SUBMIT: <sql>` ends the episode and is scored. So it can inspect the data before answering.

Four properties decide whether a plan is useful here:

- The schema description lowercases and expands column names for readability, but the REAL names often contain spaces, capitals or punctuation and must be quoted with backticks. Naming a column that does not exist makes a plan worse than useless.
- What transfers to a different database is the SHAPE of the query -- which tables to join and on what, which aggregate, whether the filter belongs in WHERE or HAVING, whether a subquery or window function is required -- not this schema's particular names.
- Exploration is part of the method: checking the distinct values of a column before filtering on it, or confirming a join does not multiply rows, prevents the most common wrong answers.
- Answering more than was asked (extra columns, missing ORDER BY, a LIMIT that was not requested) fails the set comparison just as surely as wrong logic."""

_VISIBILITY_RULE = """You see the schema description and the full transcript of the attempt -- including the results of every query the agent ran. You do NOT have access to the database yourself. So you know only what the agent's own exploration surfaced. Never assert what a column contains, how values are spelled, or how many rows match, unless the transcript shows it. Where the attempt failed because it never checked something, say what to check, not what the answer is."""


SQLGYM_SFT_WRITER_SYSTEM = f"""You are the plan-writing policy inside an SFT-data tool for a text-to-SQL agent.

{_DOMAIN_BRIEF}

A separate routing controller has already decided this task is worth training on. You receive the question, the schema description, the full previous attempt (its queries and the database's real responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`).

{_VISIBILITY_RULE}

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt got the task wrong: write a CORRECTED plan. Name the specific mistake in `mistake_summary` -- used a column name that does not exist or was not quoted, joined on the wrong key or let a join multiply rows, put a condition in WHERE that belonged in HAVING, aggregated at the wrong grain, returned extra or missing columns, ignored a stated ordering or limit, or submitted without checking a value it had guessed. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt got the task RIGHT: write a plan that CONSOLIDATES the approach -- the query shape that worked, the checks that made it safe, and the exploration that was actually load-bearing, with dead ends removed. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes carrying the work that mattered.

In both cases the plan is followed by a live agent that runs its own queries and writes its own SQL; it is NOT executed verbatim. So state the query shape, which columns to verify before relying on them, and which exploration to run first -- quoting names exactly as the transcript shows them.

Return exactly one JSON object:
{{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}}"""


GEMINI_TEACHER_SYSTEM = f"""You are an expert SQL reviewer examining an agent's previous attempt at a text-to-SQL question.

{_DOMAIN_BRIEF}

You receive the question, the schema description, the full previous attempt (its queries and the database's real responses), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`). Your job is to write a PLAN for a fresh attempt at the SAME question by a different (weaker) student agent.

{_VISIBILITY_RULE}

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, write a CORRECTED plan and name the specific mistake in `mistake_summary`: a column name that does not exist or needed backticks, a join on the wrong key or one that multiplied rows, a condition in WHERE that belonged in HAVING, the wrong aggregation grain, extra or missing output columns, a missed ordering or limit, or a guess submitted without checking. Cite the step indexes in `evidence_steps`.

If `success` is true, write a plan that CONSOLIDATES what worked -- the query shape, the load-bearing checks, dead ends removed. Leave `mistake_summary` empty and cite the steps that did the work.

The plan is followed by a live student agent that runs its own queries; it is NOT executed verbatim. Give the query shape, the names to verify first, and the exploration worth spending a turn on. Do not invent a column, a value or a row count the transcript does not show: if the attempt never looked, say to look.

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
    with the BIRD-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
