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
"""

from __future__ import annotations

import json
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
from .model_client import ModelClient
from .router_policy import (
    RouterPolicy,
    action_distribution,
    features_of,
    greedy_action,
    sample_action,
)
from .writer_rubrics import routed_writer_system


def _dedup_against_active_bank(decision: dict[str, Any], bank: list[dict[str, Any]]) -> None:
    """Mutates `decision` in place: add -> refine when content nearly duplicates
    an already-active entry, regardless of what route/operation was chosen."""
    memory = decision.get("memory")
    if not memory:
        return
    candidates = active_entries(bank)
    if not candidates:
        return
    best = max(candidates, key=lambda entry: topic_overlap(memory, entry))
    if topic_overlap(memory, best) >= REFINE_TOPIC_OVERLAP_MIN:
        decision["memory_operation"] = "refine"
        decision["target_memory_id"] = best["id"]

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

        features = features_of(trajectory, bank, position, domain_length)
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

        record_path = records_dir / f"{position:03d}_{task_id}.json"
        record: dict[str, Any]

        if route == "neither":
            decision = {
                "route": "neither",
                "gap_type": None,
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
            }
        else:
            payload = {
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
            reply = None
            try:
                reply = client.json_chat(
                    system=routed_writer_system(route),
                    user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                )
                decision = normalize_alloc_decision(reply.parsed)
                # The route is the router's, not the LLM's -- the LLM was only asked
                # for content. Force it even if the model echoed something else.
                decision["route"] = route
                decision["memory_operation"] = "add" if writes_memory(route) else None
                decision["target_memory_id"] = None
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
                    "usage": reply.usage,
                }
            except Exception as exc:
                decision = {"route": route, "memory_operation": None, "memory": None, "sft_plan": None}
                record = {
                    "protocol": config.record_protocol,
                    "domain": group,
                    "position": position,
                    "source_task_id": task_id,
                    "base_agent_success": trajectory["success"],
                    "router_route": route,
                    "status": "error",
                    "error": repr(exc),
                    "raw_prediction": reply.parsed if reply is not None else None,
                }
                write_json(record_path, record)
                records.append(record)
                continue

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
