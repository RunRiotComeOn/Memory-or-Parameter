"""Router-driven sibling of alloc_bank_builder.run_chain.

Same chain structure (sequential per group, bank state carried forward), but
the route (memory/sft/both/neither) comes from a learned `RouterPolicy`
(src/trajectory_memory_lab/router_policy.py) instead of a hand-written
rubric. Content for whatever route is chosen still comes from the LLM, via
`writer_rubrics.routed_writer_system`, which takes the route as given and
only fills in `memory`/`sft_plan` text -- see DESIGN.md section 7.

Every routed decision's live logprob tensor (with grad) is returned alongside
the JSON-serializable record, so the caller can run a GRPO backward pass
after collecting a full group of rollouts. Tensors are never written to disk.

Dedup (v6): the forced add->refine rule is GONE. Through v5 it ran after
content generation and rewrote the decision -- any draft whose best match
against the active bank cleared `REFINE_TOPIC_OVERLAP_MIN` had its operation
overwritten to `refine`, its target picked by the rule, and its content
rewritten -- "regardless of what the router or the writer LLM said". On v5
batch 0 that fired on 18 of 42 memory writes and rewrote content in 13, so
nearly half of all committed memories were neither what the writer wrote nor
what the router chose, while the reward could only attribute the outcome to
the route. Choosing between add and refine is a judgment about what the bank
already says, so it now belongs to the writer, which has had `render_bank`
in its payload all along and was merely forbidden from acting on it (see
`writer_rubrics.routed_writer_system`); `validate_alloc_decision` still
enforces that a chosen target is a real active entry.

What the rule was defending against is real and does not go away: the
near-duplicate crowding found in router_reward_v1/pilot_inbatch_v1 (same
-template variants like `07b42fd_1/_2/_3` each ADDing near-identical
memories and monopolizing BM25 top-k), and the v4 failure where a careless
refine destroyed whatever the superseded entry said and the new text did
not. Both are now the writer's responsibility to avoid and ours to MEASURE:
`_duplicate_diagnostics` records the overlap the old rule would have fired
on and the retention of every writer-chosen refine, without gating either.

Content-before-route (DESIGN.md section 13): earlier versions decided the
route first and only then asked the writer LLM for content matching that
route -- so the router's features could never reflect what would actually
get written, only numeric proxies (bank size, position in the chain). Now
every task is drafted with `routed_writer_system("both")` FIRST (before the
router sees anything), unconditionally producing a candidate memory note and
sft plan regardless of what route ends up chosen. The router's features then
include a hashed bag-of-words of that draft's actual text, plus of whatever
changed in the bank over the last two batches (`_recent_changes_text`) --
see `router_policy.TEXT_HASH_DIM`. Once the router picks a route, the
already-drafted content is filtered down to whatever that route requires;
nothing is re-generated. The cost is a writer LLM call on every task
including ones that end up `neither` (previously free) -- accepted
deliberately in exchange for the router seeing real content instead of a
numeric summary of it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from .alloc_writer_harness import (
    REFINE_TOPIC_OVERLAP_MIN,
    active_entries,
    apply_memory_operation,
    normalize_alloc_decision,
    render_bank,
    topic_overlap,
    validate_alloc_decision,
    writes_memory,
    writes_sft,
)
from .memory_writer_harness import _tokens
from .model_client import ModelClient
from .router_policy import (
    RouterPolicy,
    action_distribution,
    features_of,
    greedy_action,
    sample_action,
)
from .writer_rubrics import routed_writer_system


def _retention(old_text: str, new_text: str) -> float:
    """Fraction of the old text's distinctive tokens that survive in the new one."""
    old = set(_tokens(old_text))
    if not old:
        return 1.0
    return len(old & set(_tokens(new_text))) / len(old)


def _duplicate_diagnostics(decision: dict[str, Any], bank: list[dict[str, Any]]) -> None:
    """Measure only -- never mutates the decision. See the module docstring.

    Two numbers, both recorded on every memory-writing decision and both
    gating nothing:

    `dup_best_overlap` / `dup_would_have_forced_refine` say what the deleted
    v5 rule would have done, so "did removing it bring the near-duplicate
    crowding back" is answerable from the records rather than from a pass-rate
    drop 40 hours later.

    `refine_retention` is the early warning for the OTHER failure mode. When
    the writer chooses `refine`/`replace`, `apply_memory_operation` marks the
    target superseded and drops it from retrieval, so anything the target said
    and the new content does not is lost for good. That is exactly what made
    v4 worse than writing no memory at all: forced refines kept only 72% of
    the superseded entry on average and as little as 17%, and retention
    correlated +0.69 with the candidate's own pass rate. v5 defended against
    that by splicing the old text in mechanically; v6 asks the writer to carry
    it over (`routed_writer_system`) and measures whether it actually did.
    """
    memory = decision.get("memory")
    if not memory:
        return
    candidates = active_entries(bank)
    if not candidates:
        return

    best = max(candidates, key=lambda entry: topic_overlap(memory, entry))
    overlap = topic_overlap(memory, best)
    decision["dup_best_id"] = best["id"]
    decision["dup_best_overlap"] = overlap
    decision["dup_would_have_forced_refine"] = overlap >= REFINE_TOPIC_OVERLAP_MIN

    target_id = decision.get("target_memory_id")
    if decision.get("memory_operation") in {"refine", "replace"} and target_id:
        target = next((entry for entry in candidates if entry["id"] == target_id), None)
        if target is not None:
            decision["refine_retention"] = _retention(
                (target.get("content") or "").strip(),
                (memory.get("content") or "").strip(),
            )

def _draft_content_text(memory: dict[str, Any] | None, sft_plan: dict[str, Any] | None) -> str:
    """The text a router feature is hashed from -- see DESIGN.md section 13.
    Scope+content for the candidate memory, repair_target for the candidate
    sft plan; whichever half the eventual route doesn't need is simply never
    read out of the draft, but both still inform the routing decision."""
    parts: list[str] = []
    if isinstance(memory, dict):
        parts.append(str(memory.get("scope") or ""))
        parts.append(str(memory.get("content") or ""))
    if isinstance(sft_plan, dict):
        parts.append(str(sft_plan.get("repair_target") or ""))
    return " ".join(part for part in parts if part)


def _recent_changes_text(bank: list[dict[str, Any]], position: int, window: int) -> str:
    """Text of entries added/changed strictly before `position`, within the
    last `window` task-positions -- "the last two batches" at the default
    batch_size=10, window=20. Only entries the router could not have caused
    itself (they were committed by earlier tasks) so there is no leakage from
    this task's own not-yet-decided write."""
    parts: list[str] = []
    for entry in bank:
        created = entry.get("created_position")
        if created is not None and position - window <= created < position:
            parts.append(f"{entry.get('scope', '')} {entry.get('content', '')}")
    return " ".join(parts)


RESUMABLE_STATUSES = {"committed", "memory_rejected", "no_write", "error"}


@dataclass
class RouterBuilderConfig:
    output: Path
    record_protocol: str
    model: str = "qwen35-tau"
    base_url: str = "http://127.0.0.1:8000/v1"
    max_tokens: int = 4096
    timeout: float = 1200
    seed: int = 20260822
    # Who writes the sft repair/consolidation plan -- "teacher" (default, an
    # external Gemini model) or "self" (the same base model that plays the
    # task agent). See appworld_sft_writer.py's module docstring for the
    # probe results behind defaulting to teacher.
    sft_writer: str = "teacher"
    teacher_model: str = "gemini-3.1-pro-preview"
    teacher_api_key_file: Path = Path("/nas04/yixuh/.config/continual-memory/gemini_api_key")
    # System default (2026-09-18): route comes from a prompted small LLM
    # (router_llm_policy.decide_route), not the trained linear RouterPolicy.
    # "trained" keeps the exact pre-existing GRPO path -- train_router_
    # selfreward.py passes this explicitly so its behavior is unaffected by
    # this default changing. See router_llm_policy.py's module docstring.
    # "trained_llm" (router_llm_trainable.TrainableLLMRouter) is the
    # GRPO-trainable version of "llm": the same prompt and payload, but the
    # route comes from a differentiable 4-way categorical over LoRA-adapted
    # logits instead of an untrained temperature-0 API call. `router_model`
    # is then a TrainableLLMRouter rather than a RouterPolicy, and there is
    # no server involved -- see router_llm_trainable.py's module docstring.
    router_mode: str = "llm"
    # Which benchmark produced `trajectories` -- "appworld" (default) or
    # "alfworld". Changes two things: the one-sentence framing passed to
    # `routed_writer_system` (a coding agent vs. a household-task agent), and
    # which sft writer module (`appworld_sft_writer` / `alfworld_sft_writer`)
    # supplies the self/teacher system prompts and `generate_plan_with_teacher`
    # -- AppWorld's are written specifically for a code-execution transcript
    # ("apis.spotify.login", "code turns"); sending them an ALFWorld
    # transcript of room navigation and admissible commands would produce a
    # plan hallucinating APIs that do not exist, actively worse than no plan,
    # which is why `alfworld_sft_writer.py` exists as a parallel module
    # instead of just reusing AppWorld's prompts. Guided replay
    # (turning a committed sft/both decision into verified SFT training data)
    # is a separate step, done by `run_alfworld_guided_replay.py` /
    # `run_appworld_guided_replay.py` outside this function -- this function
    # only drafts and records the decision.
    domain: str = "appworld"


@dataclass
class RouterChainResult:
    summary: dict[str, Any]
    # {"logprob": Tensor, "route": str, "entropy": Tensor, "probs": Tensor}
    decisions: list[dict[str, Any]] = field(default_factory=list)
    bank: list[dict[str, Any]] = field(default_factory=list)  # full bank incl. superseded, for chaining batches


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Per-benchmark wiring, as tables rather than the `x if domain == "alfworld"
# else y` pairs this started as: those read fine with two domains and become
# wrong-by-default with three, since an unrecognized domain silently gets the
# AppWorld branch. `.get(domain, ...["appworld"])` keeps that same default
# explicit and in one place.
_DOMAIN_WRITER_DESCRIPTION = {
    "appworld": "a customer-service agent",
    "alfworld": "a household-task agent operating in a text-adventure environment",
    "scienceworld": "a science-experiment agent operating in a text-adventure environment",
    "babyai": "a gridworld navigation agent operating in a text-described environment",
    "textcraft": "a crafting agent decomposing a goal item into recipe subgoals in a text-described Minecraft world",
    "sqlgym": "a text-to-SQL agent answering questions against a real relational database it can query before answering",
    "webshop": "a shopping agent operating a simulated web store",
    "tau2": "a customer-service agent following a written policy and calling tool APIs",
}

# domain -> (module holding the sft writer, name of its self-writer system prompt)
_DOMAIN_SFT_WRITER = {
    "appworld": ("appworld_sft_writer", "APPWORLD_SFT_WRITER_SYSTEM"),
    "alfworld": ("alfworld_sft_writer", "ALFWORLD_SFT_WRITER_SYSTEM"),
    "scienceworld": ("scienceworld_sft_writer", "SCIENCEWORLD_SFT_WRITER_SYSTEM"),
    "babyai": ("babyai_sft_writer", "BABYAI_SFT_WRITER_SYSTEM"),
    "textcraft": ("textcraft_sft_writer", "TEXTCRAFT_SFT_WRITER_SYSTEM"),
    "sqlgym": ("sqlgym_sft_writer", "SQLGYM_SFT_WRITER_SYSTEM"),
    "webshop": ("webshop_sft_writer", "WEBSHOP_SFT_WRITER_SYSTEM"),
    "tau2": ("tau2_sft_writer", "TAU2_SFT_WRITER_SYSTEM"),
}


def run_router_chain(
    router_model: RouterPolicy | None,
    group: str,
    task_ids: list[str],
    trajectories: dict[str, dict[str, Any]],
    config: RouterBuilderConfig,
    log: Callable[[str], None] = print,
    *,
    initial_bank: list[dict[str, Any]] | None = None,
    start_position: int = 0,
    total_task_count: int | None = None,
    greedy: bool = False,
    recent_window_tasks: int = 20,
) -> RouterChainResult:
    """Process `task_ids` against a running bank.

    `greedy=True` uses argmax route selection (no sampling, no gradient) --
    for building the deterministic validation bank that reflects what the
    current parameters would actually do in deployment, not an exploratory
    training rollout. `decisions[i]["logprob"]` is a non-grad zero tensor in
    this mode; callers must not use it for a backward pass.

    `initial_bank`/`start_position`/`total_task_count` let a caller split one
    logical domain chain into sequential batches (see DESIGN.md section 9):
    each batch continues the previous batch's bank state, and the router's
    "progress through the domain" features stay continuous across the split
    rather than resetting per batch.
    """
    records_dir = config.output / "records" / group
    bank: list[dict[str, Any]] = list(initial_bank) if initial_bank else []
    records: list[dict[str, Any]] = []
    live_decisions: list[dict[str, Any]] = []
    domain_length = total_task_count if total_task_count is not None else len(task_ids)

    for offset, task_id in enumerate(task_ids):
        position = start_position + offset
        trajectory = trajectories[task_id]
        record_path = records_dir / f"{position:03d}_{task_id}.json"

        # Draft candidate content FIRST, unconditionally, before the route is
        # decided -- see the module docstring's "content-before-route" note.
        # Memory and sft are drafted by TWO SEPARATE writers, not one "both"
        # call: sft repair plans need a different tool (DESIGN.md section 15)
        # -- the generic content writer only ever produced a one-line
        # `repair_target` guess, not something a live replay could act on.
        draft_payload = {
            "allocation_state": {
                "active_memory_count": len(active_entries(bank)),
                "trajectories_seen": position,
            },
            "active_memory": render_bank(bank),
            "base_agent_outcome": {
                "success": trajectory["success"],
                "reward": trajectory["reward"],
                "termination_reason": trajectory["termination_reason"],
            },
            "trajectory": trajectory,
        }
        client = ModelClient(
            base_url=config.base_url,
            api_key="EMPTY",
            model=config.model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=config.max_tokens,
            seed=config.seed + position,
            enable_thinking=False,
            timeout=config.timeout,
        )
        writer_domain_description = _DOMAIN_WRITER_DESCRIPTION.get(
            config.domain, _DOMAIN_WRITER_DESCRIPTION["appworld"])
        draft_reply = None
        try:
            draft_reply = client.json_chat(
                system=routed_writer_system("memory", writer_domain_description),
                user=json.dumps(draft_payload, ensure_ascii=False, separators=(",", ":")),
            )
            draft_decision = normalize_alloc_decision(draft_reply.parsed)
        except Exception as exc:
            # Couldn't even get a draft -- no route-independent content exists
            # for this task, so there is nothing for a chosen route to commit.
            # Record and move on, same failure semantics as before v5.
            record = {
                "protocol": config.record_protocol,
                "domain": group,
                "position": position,
                "source_task_id": task_id,
                "base_agent_success": trajectory["success"],
                "status": "error",
                "error": repr(exc),
                "raw_prediction": draft_reply.parsed if draft_reply is not None else None,
            }
            write_json(record_path, record)
            records.append(record)
            continue

        draft_memory = draft_decision.get("memory")

        # v6: drafted for EVERY task, not just failed ones. Through v5 this was
        # guarded by `if not trajectory.get("success")`, which meant a route of
        # `sft` landing on an already-successful task could never produce a
        # plan, was rejected as `missing_plan`, and committed nothing -- making
        # `sft` bit-for-bit equivalent to `neither`, and `both` to `memory`, on
        # every successful task. With a base success rate of 0.8 that silently
        # collapsed a 4-way action space to 2-way on 80% of decisions, and the
        # reward could never separate the collapsed pairs, so the router had no
        # gradient by which to learn the difference. Whether a task is worth
        # training on is a judgment the router should make and be scored on,
        # not one a caller-side guard should make for it; the writer is handed
        # `success` and the evaluator verdict (`build_writer_payload`) and
        # writes the appropriate kind of plan.
        sft_writer_module, self_writer_system_name = _DOMAIN_SFT_WRITER.get(
            config.domain, _DOMAIN_SFT_WRITER["appworld"])
        draft_sft_plan: dict[str, Any] | None = None
        if config.sft_writer == "none":
            # Skip sft drafting entirely -- for runs (e.g. router_mode=
            # "force_memory") where the route can never be sft/both anyway,
            # this avoids paying for 200 wasted teacher-model calls (each a
            # real, sequential external API round-trip) that would never be
            # used regardless of what they returned.
            writer_output = None
        elif config.sft_writer == "teacher":
            # External model (default: Gemini) writes the plan; the base
            # model still executes it later via guided replay. See
            # appworld_sft_writer.py's module docstring for why this is the
            # default (probe_sft_repair_yield_gemini_teacher.py: 62.5% rescue
            # yield on genuinely-still-failing tasks, running_log.md section
            # 11 -- ALFWorld has no equivalent probe yet, see
            # alfworld_sft_writer.py's module docstring for the bet being
            # made by defaulting to teacher here too). Any failure here
            # (network, bad key, retries exhausted) returns None exactly like
            # the self-writer's except-clause below -- not fatal to this
            # task's decision, just no sft plan for it.
            import importlib

            generate_plan_with_teacher = importlib.import_module(
                f".{sft_writer_module}", __package__
            ).generate_plan_with_teacher

            writer_output = generate_plan_with_teacher(
                trajectory, model=config.teacher_model, api_key_file=config.teacher_api_key_file,
            )
        else:
            try:
                import importlib

                writer_mod = importlib.import_module(f".{sft_writer_module}", __package__)
                self_writer_system = getattr(writer_mod, self_writer_system_name)

                sft_reply = client.json_chat(
                    system=self_writer_system,
                    user=json.dumps(
                        writer_mod.build_writer_payload(trajectory), ensure_ascii=False, separators=(",", ":")
                    ),
                )
                writer_output = writer_mod.validate_writer_output(sft_reply.parsed)
            except Exception:
                writer_output = None
        if writer_output is not None:
            draft_sft_plan = {
                "repair_target": (
                    f"{writer_output['plan']}\n(mistake: {writer_output['mistake_summary']})"
                    if writer_output["mistake_summary"]
                    else writer_output["plan"]
                ),
                "evidence_steps": writer_output["evidence_steps"],
                "plan": writer_output["plan"],  # kept raw for guided replay's memory_block
            }

        # ROUTER_DISABLE_CONTENT_FEATURES (DESIGN.md section 14.3 ablation):
        # content is still drafted and committed exactly as above -- this only
        # blanks what the ROUTER sees, isolating "does seeing real content
        # help the routing decision" from everything else, which stays byte
        # -for-byte identical between the two arms. Empty text hashes to the
        # zero vector (`_hash_bag_of_words`), which is informationally the
        # same as the feature not existing for a linear model: weight * 0 is
        # 0 regardless of what the model learns for that dimension.
        if os.environ.get("ROUTER_DISABLE_CONTENT_FEATURES"):
            draft_text = ""
            recent_changes_text = ""
        else:
            draft_text = _draft_content_text(draft_memory, draft_sft_plan)
            recent_changes_text = _recent_changes_text(bank, position, recent_window_tasks)

        llm_rationale: str | None = None
        sampled_decision = None
        if config.router_mode == "force_sft":
            # Mirror ablation of "force_memory": every task commits its
            # drafted sft plan and NOTHING is ever written to the memory
            # bank, so the frozen end state is "empty bank + an agent trained
            # on every plan" -- the exact complement of force_memory's
            # "fat unfiltered bank + untrained agent". Running both against
            # the same 200-task pool and the same no-memory baseline is what
            # separates "the router's filtering earns its keep" from "one of
            # the two artifacts is carrying the whole effect".
            #
            # A plan may still be missing (the teacher call returned None),
            # in which case validate_alloc_decision rejects the sft side and
            # the task simply contributes nothing -- same as an unroutable
            # task under force_memory, and recorded the same way.
            route, llm_rationale = "sft", "force_sft_mode"
            logprob, entropy, probs = torch.zeros(()), torch.zeros(()), None
        elif config.router_mode == "force_memory":
            # Ablation: no routing decision at all -- every task commits its
            # drafted memory, unconditionally. Used to measure the "write
            # everything, let dedup/refine be the only filter" ceiling
            # against the router-filtered bank, on the exact same trajectory
            # pool -- see DESIGN.md/running_log.md for the comparison this
            # was built for.
            route, llm_rationale = "memory", "force_memory_mode"
            logprob, entropy, probs = torch.zeros(()), torch.zeros(()), None
        elif config.router_mode == "trained_llm":
            # GRPO-trainable LLM router. Unlike the "llm" mode below this has
            # a real, differentiable logprob -- but the graph is NOT retained
            # here: 80 live forward graphs of an 8B model per batch does not
            # fit, so what is recorded is the prompt token ids plus the chosen
            # index, and the training loop replays them with grad at update
            # time. The weights do not change in between, so the replayed
            # logprob is the sampled one exactly (asserted by
            # router_llm_trainable.verify_recompute_matches_sample).
            from .router_llm_trainable import SampledDecision

            prompt_ids = router_model.build_prompt_ids(
                trajectory, len(active_entries(bank)), recent_changes_text, draft_memory, draft_sft_plan,
            )
            pick = router_model.greedy_action if greedy else router_model.sample_action
            route, route_index, probs, entropy = pick(prompt_ids)
            logprob = torch.zeros(())  # placeholder; the real one is recomputed at update time
            sampled_decision = SampledDecision(
                prompt_ids=prompt_ids, index=route_index, route=route,
                task_id=task_id, group=group, probs=probs, entropy=entropy,
            )
        elif config.router_mode == "llm":
            # System default: a prompted small model decides, reading the
            # SAME drafted content as the router features below would
            # otherwise only see hashed. No logprob/entropy/gradient here --
            # this mode is not trained (see router_llm_policy.py). Zero
            # tensors keep every downstream consumer that expects these keys
            # (entropy logging, GRPO's pg_term sum) working unchanged; a
            # zero-entropy, zero-logprob decision simply contributes nothing
            # if a caller mistakenly tries to train through this mode.
            from .router_llm_policy import decide_route

            route, llm_rationale = decide_route(
                trajectory, len(active_entries(bank)), recent_changes_text, draft_memory, draft_sft_plan,
                model=config.model, base_url=config.base_url, seed=config.seed + position,
            )
            logprob, entropy, probs = torch.zeros(()), torch.zeros(()), None
        else:
            features = features_of(trajectory, bank, recent_changes_text, draft_text)
            if greedy:
                route, logprob = greedy_action(router_model, features), torch.zeros(())
                # No gradient here, but the distribution is still worth recording:
                # it is how we see whether the *policy* has collapsed at validation
                # time, which argmax alone cannot show.
                with torch.no_grad():
                    dist = action_distribution(router_model, features)
                    entropy, probs = dist.entropy(), dist.probs
            else:
                route, logprob, dist = sample_action(router_model, features)
                entropy, probs = dist.entropy(), dist.probs
        live_decisions.append({
            "logprob": logprob, "route": route, "task_id": task_id, "group": group,
            "entropy": entropy, "probs": probs, "sampled_decision": sampled_decision,
        })

        record: dict[str, Any]

        if route == "neither":
            decision = {
                "route": "neither",
                "gap_type": draft_decision.get("gap_type"),
                "route_rationale": llm_rationale if llm_rationale is not None else "router",
                "memory_operation": None,
                "target_memory_id": None,
                "memory": None,
                "sft_plan": None,
            }
            record = {
                "protocol": config.record_protocol,
                "domain": group,
                "position": position,
                "source_task_id": task_id,
                "base_agent_success": trajectory["success"],
                "router_route": route,
                "decision": decision,
                "draft_usage": draft_reply.usage,
                # What the router DECLINED. `decision` deliberately nulls the
                # content out -- `neither` commits nothing, and the bank and
                # the sft pool must not see it -- but the drafts were paid
                # for (one writer call, one teacher call) and thrown away,
                # which makes the largest route in most runs the one we can
                # say the least about. Keeping them here costs nothing at
                # eval time (nothing reads this field) and makes the obvious
                # follow-ups answerable offline: was the router right to
                # decline, and how do declined drafts differ from committed
                # ones? Stored under its own key so no consumer can mistake
                # it for something that was committed.
                "declined_draft": {
                    "memory": draft_memory,
                    "sft_plan": draft_sft_plan,
                },
            }
            # Whether the router HAD an sft plan to choose, independent of
            # which route it took. Recorded on EVERY record, including the
            # ones that commit nothing: `sft_status` is derived from the route
            # that was taken and so cannot answer this, and the missing
            # distinction is exactly what made DESIGN.md section 16.3's first
            # exploration measurement read as policy collapse -- p(sft)=0 is
            # correct judgment when there is no plan to commit, and a bug only
            # when there is one. (First written only in the commit branch,
            # which left it absent on precisely the `neither` records the
            # question is about.)
            record["drafted_sft_plan_available"] = draft_sft_plan is not None
        else:
            # Content already exists from the draft above -- just keep the
            # parts this route actually needs and force the route field, same
            # as the old post-hoc override of whatever the writer LLM echoed.
            #
            # v6: the operation and target are the WRITER's, not hardcoded.
            # Through v5 this pinned `memory_operation` to "add" and the target
            # to None, which is why unbanning refine in the writer prompt alone
            # would have changed nothing: whatever the writer chose was
            # overwritten one function later, and the only `refine` that could
            # ever reach the bank was the one the dedup rule forced. Fall back
            # to "add" when the writer left it unset, since a memory-writing
            # route with no operation is rejected outright by
            # `validate_alloc_decision` as `invalid_operation`.
            draft_operation = draft_decision.get("memory_operation") or "add"
            decision = {
                "route": route,
                "gap_type": draft_decision.get("gap_type"),
                "route_rationale": llm_rationale if llm_rationale is not None else draft_decision.get("route_rationale"),
                "memory_operation": draft_operation if writes_memory(route) else None,
                "target_memory_id": (
                    draft_decision.get("target_memory_id")
                    if writes_memory(route) and draft_operation in {"refine", "replace"}
                    else None
                ),
                "memory": draft_memory if writes_memory(route) else None,
                "sft_plan": draft_sft_plan if writes_sft(route) else None,
            }
            if writes_memory(route):
                _duplicate_diagnostics(decision, bank)
            record = {
                "protocol": config.record_protocol,
                "domain": group,
                "position": position,
                "source_task_id": task_id,
                "base_agent_success": trajectory["success"],
                "router_route": route,
                "decision": decision,
                "usage": draft_reply.usage,
            }
            # Whether the router HAD an sft plan to choose, independent of
            # which route it took. Recorded on EVERY record, including the
            # ones that commit nothing: `sft_status` is derived from the route
            # that was taken and so cannot answer this, and the missing
            # distinction is exactly what made DESIGN.md section 16.3's first
            # exploration measurement read as policy collapse -- p(sft)=0 is
            # correct judgment when there is no plan to commit, and a bug only
            # when there is one. (First written only in the commit branch,
            # which left it absent on precisely the `neither` records the
            # question is about.)
            record["drafted_sft_plan_available"] = draft_sft_plan is not None

            validation = validate_alloc_decision(decision, trajectory, active_entries(bank))
            record["hard_validation"] = validation
            if not validation["memory"]["required"]:
                record["status"] = "no_write"
            elif validation["memory"]["accepted"]:
                record["status"] = "committed"
            else:
                record["status"] = "memory_rejected"
            if not validation["sft"]["required"]:
                record["sft_status"] = "not_selected"
            elif validation["sft"]["accepted"]:
                record["sft_status"] = "selected"
            else:
                record["sft_status"] = "sft_rejected"

        decision = record.get("decision") or {}
        if record.get("status") == "committed" and writes_memory(decision.get("route", "")):
            entry = apply_memory_operation(
                bank,
                decision,
                entry_id=f"router_{group}_{len(bank):03d}",
                source_task_id=task_id,
                rubric_id="router_v1",
                created_position=position,
            )
            record["committed_entry_id"] = entry["id"]

        write_json(record_path, record)
        records.append(record)
        log(
            f"  router/{group} [{position + 1}/{domain_length}] task={task_id} "
            f"route={route} status={record.get('status')} active={len(active_entries(bank))}"
        )

    write_json(config.output / "banks" / f"memory_{group}.json", active_entries(bank))
    write_json(config.output / "banks" / f"full_{group}.json", bank)

    summary = {
        "domain": group,
        "records": records,
        "active_entries": len(active_entries(bank)),
        "total_entries": len(bank),
    }
    return RouterChainResult(summary=summary, decisions=live_decisions, bank=bank)
