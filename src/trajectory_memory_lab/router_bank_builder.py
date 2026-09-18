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

Dedup (DESIGN.md section 10): the router only learns `route`, never which
existing bank entry to touch -- choosing among however many entries are
currently active is a variable-size action, and RL over that was exactly
what v1 punted on. But deciding "is this new content basically the same
claim as something already active" needs no learning at all: it is settled
by `alloc_writer_harness.topic_overlap`, the same vocabulary-overlap check
the old LLM-rubric system already used to validate a *chosen* refine. Here
it runs the other way -- after content is generated, any candidate whose
best match against the active bank clears `REFINE_TOPIC_OVERLAP_MIN` is
force-converted from `add` to `refine` against that entry, regardless of
what the router or the writer LLM said. This is what stops same-template
task variants (e.g. `07b42fd_1/_2/_3`) from each independently ADDing a
near-duplicate memory and crowding out other topics' BM25 top-k slots --
the concrete failure mode found in router_reward_v1/pilot_inbatch_v1.

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


# A forced refine whose new text already carries this much of the superseded
# entry's distinctive vocabulary is a genuine refinement -- the old wording adds
# nothing and is dropped. Below it, the two entries carry different information
# and the old text is kept (see the v5 note in DESIGN.md section 13).
CONTENT_SUBSUMED_MIN = 0.9


def _retention(old_text: str, new_text: str) -> float:
    """Fraction of the old text's distinctive tokens that survive in the new one."""
    old = set(_tokens(old_text))
    if not old:
        return 1.0
    return len(old & set(_tokens(new_text))) / len(old)


def _union(first: list[Any], second: list[Any]) -> list[Any]:
    """Order-preserving union, so merged conditions/exceptions keep both sides."""
    merged = list(first)
    seen = {str(item) for item in first}
    for item in second:
        if str(item) not in seen:
            merged.append(item)
            seen.add(str(item))
    return merged


def _dedup_against_active_bank(decision: dict[str, Any], bank: list[dict[str, Any]]) -> None:
    """Mutates `decision` in place: add -> refine when content nearly duplicates
    an already-active entry, regardless of what route/operation was chosen.

    The refine is NON-DESTRUCTIVE. `apply_memory_operation` marks the target
    superseded and drops it from retrieval, so whatever the target said and the
    new text does not is lost forever. That is what made v4 so much worse than
    no memory at all: `REFINE_TOPIC_OVERLAP_MIN` is a Jaccard of 0.25, which two
    entries share merely by being about the same app, so genuinely complementary
    memories were being collapsed into whichever was written last. Measured over
    v4 batches 0-1, forced refines kept only 72% of the superseded entry on
    average and as little as 17%; retention correlated +0.69 with the candidate's
    own pass rate. So the old text is carried into the merged entry unless the
    new text already subsumes it.
    """
    memory = decision.get("memory")
    if not memory:
        return
    candidates = active_entries(bank)
    if not candidates:
        return
    best = max(candidates, key=lambda entry: topic_overlap(memory, entry))
    if topic_overlap(memory, best) < REFINE_TOPIC_OVERLAP_MIN:
        return
    decision["memory_operation"] = "refine"
    decision["target_memory_id"] = best["id"]

    old_text = (best.get("content") or "").strip()
    new_text = (memory.get("content") or "").strip()
    retention = _retention(old_text, new_text)
    decision["dedup_retention"] = retention
    if not old_text or old_text in new_text or retention >= CONTENT_SUBSUMED_MIN:
        decision["dedup_merged"] = False
        return
    memory["content"] = f"{old_text}\n{new_text}"
    memory["conditions"] = _union(best.get("conditions") or [], memory.get("conditions") or [])
    memory["exceptions"] = _union(best.get("exceptions") or [], memory.get("exceptions") or [])
    decision["dedup_merged"] = True

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
    max_tokens: int = 3072
    timeout: float = 1200
    seed: int = 20260822


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


def run_router_chain(
    router_model: RouterPolicy,
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
        draft_reply = None
        try:
            draft_reply = client.json_chat(
                system=routed_writer_system("memory"),
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

        # Only worth drafting a repair plan for a task that actually failed --
        # a clean success has no mistake to plan around, and calling the sft
        # writer on it would just invent one.
        draft_sft_plan: dict[str, Any] | None = None
        if not trajectory.get("success"):
            try:
                from .appworld_sft_writer import (
                    APPWORLD_SFT_WRITER_SYSTEM,
                    build_writer_payload,
                    validate_writer_output,
                )

                sft_reply = client.json_chat(
                    system=APPWORLD_SFT_WRITER_SYSTEM,
                    user=json.dumps(
                        build_writer_payload(trajectory), ensure_ascii=False, separators=(",", ":")
                    ),
                )
                writer_output = validate_writer_output(sft_reply.parsed)
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
            "entropy": entropy, "probs": probs,
        })

        record: dict[str, Any]

        if route == "neither":
            decision = {
                "route": "neither",
                "gap_type": draft_decision.get("gap_type"),
                "route_rationale": "router",
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
            }
        else:
            # Content already exists from the draft above -- just keep the
            # parts this route actually needs and force the route field, same
            # as the old post-hoc override of whatever the writer LLM echoed.
            decision = {
                "route": route,
                "gap_type": draft_decision.get("gap_type"),
                "route_rationale": draft_decision.get("route_rationale"),
                "memory_operation": "add" if writes_memory(route) else None,
                "target_memory_id": None,
                "memory": draft_memory if writes_memory(route) else None,
                "sft_plan": draft_sft_plan if writes_sft(route) else None,
            }
            if writes_memory(route):
                _dedup_against_active_bank(decision, bank)
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
