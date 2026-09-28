"""tau2-bench counterpart of `webshop_sft_writer` / `alfworld_sft_writer`.

Only the two system prompts are domain-specific; everything else is
re-exported unchanged from `appworld_sft_writer`, as for the other domains.

What the prompts have to say differently, and why:

- The agent's conduct is governed by a written POLICY, and most failures
  are policy failures rather than tool failures: acting without explicit
  confirmation, granting something the policy forbids (or refusing something
  it allows), skipping identity verification, or not transferring when the
  policy says to. The writer is told to ground its diagnosis in the policy
  text visible in the transcript's own behavior, not in general
  customer-service instinct.
- The customer is a simulator with a private goal the agent never sees
  directly. A plan must say what to ASK and VERIFY, not assert the customer's
  details; ids and values may be named only as the transcript showed them.
- `evaluation` reports only which scoring components passed (database state,
  information communicated, NL / environment assertions) -- never the
  expected actions or target state (see `tau2_agent.summarize_reward`). The
  writer diagnoses from the transcript, and must not pretend to know the
  expected outcome.
- Telecom is dual-control: the customer performs device actions with their
  own tools on the agent's instruction. A plan there is largely about which
  troubleshooting step to walk the customer through, in what order, and what
  to check after each.
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

_DOMAIN_BRIEF = """tau2-bench is customer service over tool APIs, in one of three domains: airline (reservations, changes, cancellations, baggage, compensation), retail (orders, returns, exchanges, address and payment changes), or telecom (mobile service troubleshooting, where the CUSTOMER has device tools -- toggling airplane mode, mobile data, network settings, re-seating the SIM -- and the agent must walk them through it). The agent follows a written domain policy, calls domain tools against a real database, and talks with a simulated customer who has a private goal and details the agent only learns by asking.

How an episode is scored (1 = full pass): depending on the domain, whether the final DATABASE state matches the expected one, whether the agent COMMUNICATED the required information, whether natural-language ASSERTIONS about its conduct hold, and (telecom) whether ENVIRONMENT assertions about the device hold. `evaluation` says which of these passed -- never what was expected.

The usual ways an attempt fails:
- acting on the database (cancel, modify, refund, exchange) without the explicit confirmation the policy requires, or with the wrong item, reservation, payment method or amount;
- doing something the policy forbids (e.g. a refund or change outside the allowed conditions) because the customer pushed -- or refusing something the policy allows;
- not verifying the customer's identity first, or skipping a lookup and guessing an id;
- transferring to a human when the task was doable, or not transferring when the policy says to;
- in telecom: skipping a diagnostic step, instructing the wrong device action, or not re-checking the device state after a fix."""


TAU2_SFT_WRITER_SYSTEM = f"""You are the plan-writing policy inside an SFT-data tool for a tau2-bench customer-service agent.

{_DOMAIN_BRIEF}

A separate routing controller has already decided this task is worth training on. You receive the task instruction (the customer's opening message), the full previous attempt (the conversation, the agent's tool calls and their real results), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`) and which scoring components passed. Your job is to write a PLAN for a fresh attempt at the SAME task.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt failed: write a CORRECTED plan. Using the failed scoring components and the transcript, name the most likely specific mistake in `mistake_summary`. You are not told the expected outcome, so do not assert one: say what to ask, look up, verify or refuse, grounded in the policy behavior the transcript shows. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt passed: write a plan that CONSOLIDATES what worked -- the lookups, confirmations and actions in the order that passed, with detours left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes that actually did the work.

In both cases the plan is followed by a live agent talking to a live customer; it is NOT executed verbatim, and the customer may phrase things differently. So say which details to ask for, which tools to call and in what order (tool names exactly as in the transcript), where explicit confirmation is required before acting, and what the policy allows or forbids at the decisive step. Name ids and values only as the transcript showed them.

Return exactly one JSON object:
{{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}}"""


GEMINI_TEACHER_SYSTEM = f"""You are an expert teacher reviewing a tau2-bench customer-service agent's previous attempt at a task.

{_DOMAIN_BRIEF}

You receive the task instruction (the customer's opening message), the full previous attempt (the conversation, the agent's tool calls and their real results), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`) and which scoring components passed. Your job is to write a PLAN for a fresh attempt at the SAME task by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt failed: write a CORRECTED plan. Using the failed scoring components and the transcript, name the most likely specific mistake in `mistake_summary`. You are not told the expected outcome, so do not assert one as fact: say what to ask, look up, verify or refuse, grounded in the policy behavior the transcript shows. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt passed: write a plan that CONSOLIDATES what worked -- the lookups, confirmations and actions in the order that passed, with detours left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes that actually did the work.

In both cases the plan is followed by a live student agent talking to a live customer; it is NOT executed verbatim, and the customer may phrase things differently. So say which details to ask for, which tools to call and in what order (tool names exactly as in the transcript), where explicit confirmation is required before acting, and what the policy allows or forbids at the decisive step. Name ids and values only as the transcript showed them.

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
    with the tau2-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
