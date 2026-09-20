"""LLM-based router (system default): the SAME model that plays the task
agent (qwen35-tau, served by det_server_a/b) DECIDES the route by prompted
judgment, not the trained linear classifier in `router_policy.py`.

Why: `router_policy.RouterPolicy` is a 144-parameter linear layer over hashed
n-gram features -- it can only ever act on a lossy numeric summary of the
actually-drafted content (`router_bank_builder._draft_content_text`), and it
has no way to "know" anything about what each route actually does beyond
whatever a GRPO update manages to encode into 4x35 weights. An LLM router
reads the FULL task instruction and trajectory as real text (same shape
`appworld_sft_writer.build_writer_payload` already hands the sft teacher --
see `build_router_payload` below), not just an outcome summary, and can be
told directly, in its own prompt, what committing to each route actually
does (including the fact that `sft` is now teacher-model-authored and only
becomes real training data if a live guided replay succeeds) -- properties
that had no way to reach the linear router except as an uninterpretable
constant folded into its bias term.

Model choice: deliberately the SAME model/size as the task agent, not a
smaller dedicated router model -- a router judging whether a trajectory is
worth extracting from should be at least as capable as the agent that
produced it. This also means no separate server: `decide_route` is just
another call against `config.model`/`config.base_url`, the same det_server_a
/b replicas everything else in this pipeline already uses.

Not trained (yet, per explicit instruction): this module has no logprob,
entropy, or gradient anywhere in it. `router_bank_builder.run_router_chain`
fills `live_decisions[i]["logprob"]`/`["entropy"]` with zero tensors in this
mode purely so existing downstream code that expects those keys does not
break; nothing here is meant to be backpropped through yet. Real GRPO
training of this LLM router's weights needs infrastructure that doesn't
exist in this repo yet (vLLM-side logprob extraction + a local trainable
LoRA copy to recompute a differentiable logprob + a weight-sync step back
into the serving replicas) -- see router_reward_v1/handoff_prompt_llm_router_grpo.md.
The router_policy GRPO path is still available (`RouterBuilderConfig(
router_mode="trained", ...)`) and is what `train_router_selfreward.py` uses
explicitly, for exactly this reason -- an untrained LLM router judgment is a
first look, not (yet) a replacement for the trained one.
"""

from __future__ import annotations

import json
from typing import Any

from .model_client import ModelClient

ROUTES = ("memory", "sft", "both", "neither")

ROUTER_LLM_SYSTEM = """You are the routing policy for a coding agent's continual-learning loop over AppWorld tasks.

Exactly one decision is yours for this one completed task: what happens to it. There are four options, and you are told exactly what each one costs and produces -- decide from that, not from a fixed rule:

- `memory`: commit the drafted memory entry into a shared retrieval bank. Every future related task in this domain retrieves the top-3 matching entries from this bank and pays their context cost -- a redundant, overly narrow, or wrong entry actively hurts OTHER tasks, not just this one. Worth it only if the entry states something a future task could not already infer and will plausibly need.
- `sft`: commit the drafted plan as training data -- but only CONDITIONALLY. The plan (written by a teacher model, either correcting a specific mistake or consolidating an already-successful approach) is handed to a live student agent that reattempts this exact task from scratch; it becomes a real training example ONLY IF that live replay actually succeeds. A plan that names a clear, mechanical fix (wrong argument, missing login call, unchecked pagination) is far more likely to pay off on replay than a plan attempting something structurally difficult the agent may fail again regardless.
- `both`: commit both artifacts independently -- their fates are separate (the memory is committed unconditionally, the sft plan is still gated on replay success).
- `neither`: commit nothing. This is a genuine, often-correct answer, not a fallback: a clean success with nothing generalizable to extract, or a failure too messy, task-specific, or ambiguous to trust either artifact, should route here.

You will see the task's actual instruction, its FULL recorded trajectory (every code turn and environment response, not just a summary), how large the active memory bank already is, what changed in it over the last two batches, and the CONTENT that has already been drafted for you -- a candidate memory entry and a candidate plan. These are the real artifacts `memory`/`sft`/`both` would commit verbatim, not a preview to be rewritten. Read the trajectory yourself to judge whether each artifact is worth its stated cost given everything shown, using your own judgment about this specific case -- do not just defer to the reported outcome.

Return exactly one JSON object:
{"route": "memory"|"sft"|"both"|"neither", "rationale": STRING}"""


# Character budget for `trajectory_steps`, the one field big enough to matter.
#
# Measured on the 90 base_train_v2 transcripts: the steps alone are a median
# of 11,536 tokens and up to 30,713, while everything else the router reads is
# small -- the drafted memory candidate is a median of 184 tokens and the
# drafted sft plan 285, about 3% of the prompt between them. So the transcript
# is the only thing worth cutting, and cutting the drafts instead would remove
# exactly the content DESIGN.md section 14.3 put in front of the router to
# judge while leaving the bulk untouched.
#
# The cap exists because the 35B MoE router has to BACKWARD through this
# prompt on two 49GB cards. Measured with gradient checkpointing on: 12,288
# tokens peaks at 40.2GiB, 16,384 at 42.8GiB, 24,576 OOMs. 12,000 tokens of
# transcript leaves room for the system prompt (~530), the drafts (~620 worst
# case) and the instruction under a 16,384 guard.
#
# Budgeted in CHARACTERS, not tokens, because this function is shared by the
# prompted router (`decide_route`, no tokenizer in reach) and the trainable
# one, and `router_llm_trainable`'s whole untrained-probe-equals-step-0
# argument depends on both seeing byte-identical text. 2.646 chars/token was
# the LOWEST ratio across all 90 transcripts, so a character budget converts
# to a token bound that holds for every one of them rather than on average.
ROUTER_MAX_STEP_CHARS = 31_000


def _elide_steps(steps: list[Any], max_chars: int) -> tuple[list[Any], int]:
    """Keep whole steps from both ends, drop the middle, say so in the text.

    Returns (steps, dropped_count). Two deliberate choices:

    - Whole steps, never a cut inside one: half a tool call reads as a
      malformed transcript, and the router would be judging an artifact of the
      truncation rather than the episode.
    - The TAIL grows first. The end of an AppWorld episode is where the
      failure surfaces and where the drafted memory and repair plan point; the
      opening is mostly task setup. When the budget only fits one side, the
      ending is the side worth keeping.

    The elision is written into the transcript as a visible marker rather than
    left implicit, so the router can tell "the agent did nothing here" apart
    from "this was cut".
    """
    if not steps:
        return steps, 0
    encoded = json.dumps(steps, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_chars:
        return steps, 0

    sizes = [len(json.dumps(s, ensure_ascii=False, separators=(",", ":"))) + 1 for s in steps]
    head = tail = used = 0
    while head + tail < len(steps):
        take_tail = tail <= head
        index = len(steps) - 1 - tail if take_tail else head
        if used + sizes[index] > max_chars:
            break
        used += sizes[index]
        if take_tail:
            tail += 1
        else:
            head += 1

    dropped = len(steps) - head - tail
    if dropped <= 0:
        return steps, 0
    marker = {
        "role": "system",
        "index": None,
        "content": f"[... {dropped} of {len(steps)} steps elided to fit the router's context ...]",
    }
    return steps[:head] + [marker] + (steps[len(steps) - tail:] if tail else []), dropped


def build_router_payload(
    trajectory: dict[str, Any],
    active_memory_count: int,
    recent_changes_text: str,
    draft_memory: dict[str, Any] | None,
    draft_sft_plan: dict[str, Any] | None,
    *,
    max_step_chars: int = ROUTER_MAX_STEP_CHARS,
) -> dict[str, Any]:
    steps, dropped = _elide_steps(trajectory.get("steps", []), max_step_chars)
    return {
        "task_instruction": (trajectory.get("task") or {}).get("instruction"),
        "trajectory_steps": steps,
        "trajectory_steps_elided": dropped or None,
        "base_agent_outcome": {
            "success": trajectory.get("success"),
            "reward": trajectory.get("reward"),
            "termination_reason": trajectory.get("termination_reason"),
        },
        "active_memory_count": active_memory_count,
        "recent_bank_changes": recent_changes_text or None,
        "drafted_memory_candidate": (
            {
                "content": draft_memory.get("content"),
                "scope": draft_memory.get("scope"),
                "conditions": draft_memory.get("conditions"),
                "exceptions": draft_memory.get("exceptions"),
            }
            if isinstance(draft_memory, dict) else None
        ),
        "drafted_sft_plan_candidate": (
            {"plan": draft_sft_plan.get("plan"), "targets_a_failure": bool(not trajectory.get("success"))}
            if isinstance(draft_sft_plan, dict) else None
        ),
    }


def decide_route(
    trajectory: dict[str, Any],
    active_memory_count: int,
    recent_changes_text: str,
    draft_memory: dict[str, Any] | None,
    draft_sft_plan: dict[str, Any] | None,
    *,
    model: str,
    base_url: str,
    seed: int,
    max_tokens: int = 2048,
    timeout: float = 300,
) -> tuple[str, str]:
    """Returns (route, rationale). Falls back to ("neither", <reason>) on any
    failure (bad JSON, unlisted route, network error, timeout) -- the same
    "not fatal, just no artifact this task" posture every writer in this
    pipeline already takes, just landing on a route instead of a plan."""
    client = ModelClient(
        base_url=base_url, api_key="EMPTY", model=model,
        temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=seed,
        enable_thinking=False, timeout=timeout,
    )
    payload = build_router_payload(trajectory, active_memory_count, recent_changes_text, draft_memory, draft_sft_plan)
    try:
        reply = client.json_chat(
            system=ROUTER_LLM_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        parsed = reply.parsed if isinstance(reply.parsed, dict) else {}
        route = str(parsed.get("route") or "").strip().lower()
        rationale = str(parsed.get("rationale") or "").strip()
        if route not in ROUTES:
            return "neither", f"invalid_route:{route!r}"
        return route, rationale
    except Exception as exc:  # noqa: BLE001
        return "neither", f"router_llm_error:{exc!r}"
