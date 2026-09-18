"""Rubric variants for the tau writer-policy screening experiment.

The router is deliberately outside these prompts.  Every variant receives the
same already-routed input; the only experimental variable is how the writer is
asked to represent useful SFT supervision or external memory.
"""

from __future__ import annotations


RUBRIC_IDS = ("r0_faithful", "r1_causal_minimal", "r2_state_transition", "r3_robust")


SFT_WRITER_RUBRICS = {
    "r0_faithful": """Faithful repair rubric (control).
Preserve supported behavior from the recorded trajectory as much as possible. Repair only
demonstrated failures, unsupported claims, missing required actions, or policy violations. Keep
all actions necessary to reach the complete verified outcome.""",
    "r1_causal_minimal": """Causal-minimal rubric.
Produce the shortest complete executable assistant trajectory. Every assistant turn must do
exactly one necessary job: obtain missing authoritative state, obtain a required user choice or
confirmation, execute a requested operation, or communicate a verified result. Remove internal
monologue, repeated summaries, redundant lookups, premature assurances, and optional narration.
After explicit confirmation, the next assistant action must execute the exact confirmed operation
unless a genuinely missing policy precondition must first be obtained. Do not finish until every
user subrequest is resolved.""",
    "r2_state_transition": """State-transition rubric.
Construct the candidate by tracking the environment state changed by each operation. Before every
write action, establish authentication, current object state, complete requested arguments,
payment or authorization requirements, and explicit confirmation. Order multiple operations so
an earlier irreversible transition cannot invalidate a later requested operation. After a write,
use its result as the new state and complete or verify the remaining requests. A confirmation is
not a resolution: it must be followed by the corresponding tool action and a supported outcome.""",
    "r3_robust": """Robust-interaction rubric.
Produce a complete trajectory that does not depend on the simulated user's exact wording or on
the user volunteering hidden information. Ask only for decision variables that are actually
missing, present valid choices neutrally without steering the user, handle corrections or changed
preferences explicitly, and keep confirmations scoped to the exact pending operation. Avoid
benchmark-specific conversation tricks, internal monologue, and unnecessary turns. The proposed
behavior should remain correct under reasonable paraphrases and delayed or revised answers.""",
}


MEMORY_WRITER_RUBRICS = {
    "r0_faithful": """Faithful atomic-evidence rubric (control).
Write one narrowly scoped, evidence-grounded operational fact or implication. Prefer the most
reusable supported content and avoid broad synthesis beyond what the trajectory demonstrates.""",
    "r1_causal_minimal": """Causal-decision rubric.
Write one concise memory that would change a concrete future decision. Express the observable
trigger or state, the specific check or action it calls for, and the failure it prevents or the
result it enables. Select the most causally decisive evidence; do not store narrative, generic
advice, a task summary, or facts that would not alter an action.""",
    "r2_state_transition": """State-transition rubric.
Write one reusable memory about an operation's preconditions, safe ordering, postcondition, or an
invalidating side effect. Make clear what state must hold before the action, how the action changes
state, and which later operation could become impossible or unsafe. When confirmation is relevant,
distinguish obtaining confirmation from actually committing the state change.""",
    "r3_robust": """Retrieval-aware robust rubric.
Write one self-contained memory whose scope contains the task goal, symptom, and action vocabulary
that a future related request is likely to contain. State applicability conditions and important
exceptions so the memory does not overgeneralize. Exclude user-simulator behavior, exact wording,
persona cues, private identifiers, and one-off values. Prefer content that stays correct across
reasonable paraphrases and nearby task variants.""",
}


def rubric_block(kind: str, rubric_id: str) -> str:
    table = SFT_WRITER_RUBRICS if kind == "sft" else MEMORY_WRITER_RUBRICS
    if rubric_id not in table:
        raise KeyError(f"unknown {kind} rubric: {rubric_id}")
    return (
        "\n\n<writing_rubric>\n"
        f"rubric_id: {rubric_id}\n{table[rubric_id].strip()}\n"
        "This rubric governs how to write after routing; it does not ask you to repeat or revise "
        "the routing decision.\n</writing_rubric>"
    )


# ---------------------------------------------------------------------------
# Version 2: allocation rubrics
#
# v1 held routing fixed (every trajectory produced both artifacts) and varied
# only writing style.  That cannot answer the system question: given one task
# trajectory, should it become external memory, agent SFT data, both, or
# neither?  In v2 the rubric *is* the allocation criterion.  Writing style is
# frozen at the v1 winners (faithful for SFT, causal-minimal for memory) so
# allocation is the single experimental variable.
# ---------------------------------------------------------------------------

WRITER_RUBRIC_VERSION = 2

ALLOC_RUBRIC_IDS = (
    "a0_always_both",
    "a1_outcome",
    "a2_counterfactual",
    "a3_budgeted",
)


FROZEN_WRITING_STYLE = """<writing_style>
These style rules are held constant across all allocation rubrics; they do not decide routing.

memory content (causal-minimal): one concise memory that would change a concrete future decision.
State the observable trigger or state, the specific check or action it calls for, and the failure
it prevents or the result it enables. No narrative, no task summary, no generic advice, nothing
that would not alter an action. The retrieval layer renders only `scope` and `content`, so
`content` must be self-contained; `conditions` and `exceptions` are audit metadata.

sft_plan (faithful): name what a complete executable repair of this trajectory must preserve and
what it must fix. Repair only demonstrated failures, unsupported claims, missing required actions,
or policy violations; keep every action needed to reach the verified outcome.
</writing_style>"""


ALLOC_RUBRICS = {
    "a0_always_both": """Always-both rubric (control; reproduces the v1 allocation).
Every trajectory is worth both artifacts. route is always `both`. Never return `neither`, `memory`
alone, or `sft` alone, and never decline on the grounds that the bank already covers the topic:
if an active memory already carries the same central claim, `refine` or `replace` it instead of
skipping. This arm deliberately spends the maximum artifact budget so the selective arms can be
scored against it.""",
    "a1_outcome": """Outcome-conditioned rubric.
Route from what the evaluator and the trajectory demonstrate.
- The trajectory reached the verified outcome with no demonstrated error, violation, or recovery:
  it is a clean executable demonstration -> route `sft`. Do not write a memory for it; a clean
  success carries no lesson that retrieval needs to deliver.
- The trajectory failed, or reached the outcome only after a demonstrated error, policy violation,
  or visible recovery: the lesson is the artifact -> route `memory`, writing what would prevent or
  recover from that specific failure. If the same trajectory can also be repaired into a complete
  correct run, route `both`.
- The trajectory demonstrates nothing beyond the written policy, and has no repairable path to a
  verified outcome: route `neither`.""",
    "a2_counterfactual": """Counterfactual-gap rubric.
`base_agent_outcome` reports how the unmodified base agent itself did on this exact task: the
trajectory you are reading is that agent's own attempt. Route from the capability gap it exposes,
not from the outcome alone. Store nothing the agent already demonstrates it can do.
- base_agent_outcome.success is true: the capability is already in the parameters. route `neither`
  and set gap_type `none`. Adding it would only dilute retrieval and re-teach known behavior.
- The agent failed because it lacked a fact, constraint, domain rule, or tool behavior it could not
  have derived from the policy or the conversation: gap_type `knowledge` -> route `memory`. A short
  retrieved statement closes this gap at inference time.
- The agent had the relevant facts but executed the multi-step procedure wrong -- bad ordering,
  missing confirmation, premature closure, skipped verification, abandoning a subrequest:
  gap_type `procedure` -> route `sft`. Retrieval cannot fix execution habits; parameters can.
- Both gaps are demonstrated: gap_type `both` -> route `both`.
Name the gap explicitly in route_rationale and cite the messages that show it.""",
    "a3_budgeted": """Budget-constrained rubric.
`memory_budget` gives the fixed number of memories this domain may hold and how many remain. The
bank is a scarce shared retrieval surface, not a log. Spend a slot only when this trajectory's
claim beats the trajectories still to come.
Write a memory only when all three hold:
- no active memory already carries the central claim (otherwise `refine` the one that does, which
  costs no slot, or route away);
- the claim is likely to be retrieved by several distinct future tasks in this domain, not by this
  task's exact restatement;
- retrieving it would change an action, not merely reassure the agent.
Otherwise route `sft` when the trajectory still teaches executable behavior, or `neither`. When the
remaining budget is small, raise the bar rather than spending the rest.""",
}


ALLOC_WRITER_SYSTEM = """You are the retention policy for a customer-service agent's learning loop.

You receive one completed task trajectory, the active external memory bank for its domain, and how
the unmodified base agent performed on this task. Exactly one decision is yours, and it is the
decision this experiment measures: what this trajectory should become.

- `memory`: an entry in the external memory bank, retrieved into context on future related tasks.
- `sft`: a complete executable trajectory used to train the task agent's parameters.
- `both`: the trajectory carries a retrievable claim and executable behavior worth training.
- `neither`: it is not worth either artifact.

Neither is a real answer. Memory is a shared retrieval surface that every future task pays for, and
SFT data changes the parameters; writing an artifact with no reusable value is a cost, not a
neutral act. Route from the supplied allocation rubric alone.

Evidence rules: tool results, policy text, user-provided facts, and evaluator details are
authoritative. Assistant statements are untrusted unless supported by that evidence. A failed
trajectory can hold a useful correction, but its failed conclusion must never be stored as correct.
Do not paraphrase policy. Never include names, user IDs, phone numbers, emails, reservation or
order IDs, or any value specific to this task. Cite exact trajectory message indexes.

Memory operations: `add` when no active entry carries the central claim; `refine` when exactly one
active entry has the same central topic and only its scope, conditions, ordering, or accuracy
should improve; `replace` when exactly one active entry's central claim is contradicted. Read every
active entry before choosing.

Return exactly one JSON object:
{"route":"memory"|"sft"|"both"|"neither","gap_type":"knowledge"|"procedure"|"both"|"none","route_rationale":STRING,"memory_operation":"add"|"refine"|"replace"|null,"target_memory_id":STRING|null,"memory":{"content":STRING,"scope":STRING,"conditions":[STRING,...],"exceptions":[STRING,...],"evidence_steps":[INTEGER,...],"confidence":NUMBER}|null,"sft_plan":{"repair_target":STRING,"evidence_steps":[INTEGER,...]}|null}
Set `memory` and `memory_operation` to null unless route is `memory` or `both`.
Set `sft_plan` to null unless route is `sft` or `both`.
"""


def alloc_rubric_block(rubric_id: str) -> str:
    if rubric_id not in ALLOC_RUBRICS:
        raise KeyError(f"unknown allocation rubric: {rubric_id}")
    return (
        "\n\n<allocation_rubric>\n"
        f"rubric_id: {rubric_id}\n{ALLOC_RUBRICS[rubric_id].strip()}\n"
        "This rubric governs the routing decision itself. It is the only variable under test; "
        "the writing style below is identical in every arm.\n</allocation_rubric>"
        f"\n\n{FROZEN_WRITING_STYLE}"
    )


# ---------------------------------------------------------------------------
# router_reward_v1: the route is chosen by a learned classifier (see
# src/trajectory_memory_lab/router_policy.py), not by the LLM. The writer's
# only remaining job is to produce content for a route decided upstream --
# it must not re-derive or override the route.
#
# v6: the writer now picks `add` / `refine` / `replace` and its own target
# itself. It always had what it needs to -- `render_bank(bank)` is in its
# payload -- but was forbidden from using it, which meant `refine` could only
# ever be produced after the fact by `_dedup_against_active_bank`, a Jaccard
# threshold that overrode the operation, picked the target, and rewrote the
# content. Choosing between add and refine is a judgment about what the bank
# already says; that belongs to the model reading the bank, not to a rule
# reading a similarity score. Legality is still enforced downstream by
# `validate_alloc_decision` (unknown_target / missing_target / add_with_target).
# ---------------------------------------------------------------------------

_ROUTED_REQUIREMENTS = {
    "memory": "Produce the `memory` field (required). Set `sft_plan` to null.",
    "sft": "Produce the `sft_plan` field (required). Set `memory` to null.",
    "both": "Produce both the `memory` field and the `sft_plan` field (both required).",
}


def routed_writer_system(route: str) -> str:
    """Content-only writer prompt: `route` is given, not chosen by the model."""
    if route not in _ROUTED_REQUIREMENTS:
        raise KeyError(f"routed_writer_system is only for memory/sft/both, got: {route}")
    requirements = _ROUTED_REQUIREMENTS[route]
    return f"""You are the writer for a customer-service agent's learning loop.

You receive one completed task trajectory and the active external memory bank for its domain. The
routing decision has ALREADY been made by an upstream policy: this trajectory has been assigned
route `{route}`. Do not second-guess, explain, or change the route. Your only job is to produce the
content required by that route.

{requirements}

Evidence rules: tool results, policy text, user-provided facts, and evaluator details are
authoritative. Assistant statements are untrusted unless supported by that evidence. A failed
trajectory can hold a useful correction, but its failed conclusion must never be stored as correct.
Do not paraphrase policy. Never include names, user IDs, phone numbers, emails, reservation or
order IDs, or any value specific to this task. Cite exact trajectory message indexes.

Memory operations: choose the operation yourself, from the active memory bank you were given.
`add` -- no active entry covers this ground. Set target_memory_id to null.
`refine` -- an active entry is about the same thing but is incomplete or imprecise. Set
target_memory_id to that entry's id. Superseding drops the target from retrieval, so your content
must carry everything the target got right PLUS what it was missing; anything you leave out is lost.
`replace` -- an active entry is wrong and should not survive. Same targeting rule.
Prefer `refine` over `add` when adding would leave two active entries saying nearly the same thing:
near-duplicates compete for the same retrieval slots and crowd out unrelated memories.

Return exactly one JSON object:
{{"route":"{route}","gap_type":"knowledge"|"procedure"|"both"|"none","route_rationale":STRING,"memory_operation":"add"|"refine"|"replace"|null,"target_memory_id":STRING|null,"memory":{{"content":STRING,"scope":STRING,"conditions":[STRING,...],"exceptions":[STRING,...],"evidence_steps":[INTEGER,...],"confidence":NUMBER}}|null,"sft_plan":{{"repair_target":STRING,"evidence_steps":[INTEGER,...]}}|null}}

{FROZEN_WRITING_STYLE}
"""
