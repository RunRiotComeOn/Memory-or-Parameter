"""WebShop counterpart of `scienceworld_sft_writer` / `alfworld_sft_writer`.

Only the two system prompts are domain-specific; everything else is
re-exported unchanged from `appworld_sft_writer`, as for the other domains.

What the prompts have to say differently, and why:

- The grader is partial-credit and multi-part: product type, the request's
  attributes, the SELECTED OPTIONS on the product page, and the price bound.
  `evaluation.score_components` shows which parts fell short (`r_type`,
  `r_att`, `r_option`, `r_price`), and the teacher should read them rather
  than guess -- an `r_option` below 1 means the right product was bought with
  the wrong or missing size/color, a completely different repair from a low
  `r_type` (wrong kind of product).
- The target product is never shown (see `webshop_agent`'s module docstring),
  and in a 1.18M-product catalog a plan cannot promise that one specific
  item exists or will be found. Plans must describe HOW to search and verify
  -- query wording, which option buttons to click, what to check on the page
  -- and may name a product id only if the transcript itself showed it.
- A successful attempt usually has little to trim (episodes are short); the
  useful consolidation is the query that worked and the option clicks that
  earned full credit.
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

_DOMAIN_BRIEF = """WebShop is a simulated online store over about 1.18 million real Amazon products. The agent gets a shopping request (product type, attributes, options such as size/color, and a price limit) and each turn replies with one action: `search[<query>]` on a page with a search bar, or `click[<item>]` for a listed clickable -- a product id from the results, an option value on a product page, "buy now", "next >", "< prev", "back to search", or a product sub-page ("description", "features", "reviews", "attributes").

How a purchase is scored (0 to 1): the bought product is compared with the hidden target on product TYPE, the request's ATTRIBUTES (matched against the product's attributes, title, bullet points and description), the OPTIONS the agent selected on the product page, and the PRICE limit. Only a full match scores 1. `evaluation.score_components` reports the parts: `r_type` (1.0 = right kind of product), `r_att` (fraction of attributes matched), `r_option` (fraction of requested options selected correctly), `r_price` (true = within the limit). The target product itself is never revealed.

Common ways an attempt loses credit: buying without clicking the requested size/color/flavor options first (low `r_option`); settling on the first search result that is the wrong kind of product (low `r_type`); ignoring the price limit, which the search engine does not apply; long, sentence-like search queries that return poor results where a short query of the product type plus key attributes works better; running out of steps paging through results or re-reading sub-pages and never buying."""


WEBSHOP_SFT_WRITER_SYSTEM = f"""You are the plan-writing policy inside an SFT-data tool for a WebShop shopping agent.

{_DOMAIN_BRIEF}

A separate routing controller has already decided this task is worth training on. You receive the task instruction, the full previous attempt (its actions and the real pages it saw), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`, i.e. score 1) and its score components. Your job is to write a PLAN for a fresh attempt at the SAME request.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt fell short: write a CORRECTED plan. Use the score components to name the specific shortfall in `mistake_summary` -- wrong product type, attributes not matched, options not selected or wrong, over the price limit, or never bought. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt scored full marks: write a plan that CONSOLIDATES what worked -- the query that surfaced the right product, the options clicked, the checks made -- with any detours left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes that actually did the work.

In both cases the plan is followed by a live agent that sees real pages and picks its own actions; it is NOT executed verbatim. So give the search query to use (short: product type plus the key attributes), what to check on a result or product page before choosing it (type, attributes, price), and exactly which option values to click before buy now, quoted as the request states them. Name a product id only if the attempt's own pages showed it matching; you are not told the target, so do not assert one.

Return exactly one JSON object:
{{"plan": STRING, "mistake_summary": STRING, "evidence_steps": [INTEGER,...]}}"""


GEMINI_TEACHER_SYSTEM = f"""You are an expert teacher reviewing a WebShop shopping agent's previous attempt at a shopping request.

{_DOMAIN_BRIEF}

You receive the task instruction, the full previous attempt (its actions and the real pages it saw), and the evaluator's verdict -- including whether that attempt SUCCEEDED (`success`, i.e. score 1) and its score components. Your job is to write a PLAN for a fresh attempt at the SAME request by a different (weaker) student agent.

Read `success` in the payload first; it decides which plan you are writing.

If `success` is false, the attempt fell short: write a CORRECTED plan. Use the score components to name the specific shortfall in `mistake_summary` -- wrong product type, attributes not matched, options not selected or wrong, over the price limit, or never bought. Cite in `evidence_steps` the step indexes where it went wrong.

If `success` is true, the attempt scored full marks: write a plan that CONSOLIDATES what worked -- the query that surfaced the right product, the options clicked, the checks made -- with any detours left out. Do not invent a mistake; leave `mistake_summary` as an empty string. Cite in `evidence_steps` the step indexes that actually did the work.

In both cases the plan is followed by a live student agent that sees real pages and picks its own actions; it is NOT executed verbatim. So give the search query to use (short: product type plus the key attributes), what to check on a result or product page before choosing it (type, attributes, price), and exactly which option values to click before buy now, quoted as the request states them. Name a product id only if the attempt's own pages showed it matching; you are not told the target, so do not assert one as the answer.

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
    with the WebShop-specific system prompt substituted in."""
    return _generate_plan_with_teacher_impl(
        trajectory,
        model=model,
        api_key_file=api_key_file,
        temperature=temperature,
        max_retries=max_retries,
        system_prompt=GEMINI_TEACHER_SYSTEM,
    )
