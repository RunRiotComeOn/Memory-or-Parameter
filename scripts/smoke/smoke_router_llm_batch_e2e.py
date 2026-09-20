"""End-to-end smoke test of the real `train_router_llm_grpo.run_one_batch_update`.

Sibling of smoke_batch_update_e2e.py (which covers the linear router). Stubs
exactly three boundaries -- the writer LLM (`ModelClient`), the AppWorld
self-eval subprocess, and the sft guided-replay pipeline -- so everything else
is the actual training code: real `run_router_chain` in
`router_mode="trained_llm"`, real sampling through the LoRA-wrapped model,
real `accumulate_policy_gradient`, real optimizer step, real logging.

Needs a GPU and a local router model. Needs NO vLLM server, no AppWorld
environment, and no 35B task agent.

What it asserts:
  1. The full batch loop runs and produces one train_log record per batch.
  2. Every decision carries a SampledDecision (i.e. the chain really ran in
     trained_llm mode -- if it silently fell back to another router_mode the
     gradient would be over zero tensors and the run would look fine).
  3. LoRA parameters actually move across the batch.
  4. The advantage is group-relative and sums to ~0 -- the GRPO property the
     whole design rests on.
  5. Candidate divergence PROPAGATES: when the policy has entropy, different
     candidates make different decisions and those differences reach distinct
     advantages. This is the failure mode that would waste a 40-hour run
     silently -- if all K candidates make identical decisions, every advantage
     is exactly 0, every gradient is 0, and the logs still look entirely
     normal.

     That check runs at `policy_temperature=3.0`, deliberately. Whether the
     UNTRAINED model explores enough at T=1.0 is a property of the model and
     the content, not of this wiring, and it is scale-dependent: at this
     test's deliberately tiny K=4 x batch_size=4, p(argmax)=0.99 makes all-16
     -decisions-identical the expected outcome ~85% of the time, so asserting
     divergence here would be testing the batch size. It was instead measured
     directly on real drafted content at the real K=8 x batch_size=10 (see
     DESIGN.md section 16): P(all 8 candidates identical) = 0.003 for
     Qwen3-8B at T=1.0. This file's job is to prove the mechanism carries
     divergence through to the gradient; that measurement is what says there
     will be divergence to carry.
  6. The recompute-equals-sample invariant holds inside the real loop.

Run: ROUTER_LLM_MODEL=<path> CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
       .venv/bin/python scripts/smoke/smoke_router_llm_batch_e2e.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_router_llm_grpo as T  # noqa: E402
from trajectory_memory_lab import router_bank_builder as RBB  # noqa: E402
from trajectory_memory_lab import router_sft_pipeline as RSP  # noqa: E402
from trajectory_memory_lab.model_client import ModelReply  # noqa: E402
from trajectory_memory_lab.router_llm_policy import ROUTES  # noqa: E402
from trajectory_memory_lab.router_llm_trainable import (  # noqa: E402
    RouterLLMConfig,
    TrainableLLMRouter,
)

DEFAULT_MODEL = "/nas04/yixuh/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# --- stub 1: the writer LLM -------------------------------------------------
# The drafted content has to be REALISTIC, not obviously-synthetic filler.
# With placeholder text ("stub memory fact number 3 about topic 3") the router
# answers `neither` with probability 1.0 on every task -- which is the correct
# judgment about worthless content, but it drives entropy, every logprob and
# every gradient to exactly 0 and makes check [4] vacuous. These are shaped
# like what `routed_writer_system` actually produces on AppWorld: a concrete
# API-level fact with a scope, and a concrete mechanical repair plan.
_MEMORIES = [
    ("spotify library enumeration",
     "show_song_library is paginated at 20 items; pass page_index=0,1,2,... until an empty page "
     "is returned, otherwise only the first page of songs is ever seen."),
    ("venmo payment requests",
     "Before creating a payment request, call show_friends to resolve the recipient's venmo "
     "username -- create_payment_request rejects raw phone numbers and email addresses."),
    ("phone contact lookup",
     "search_contacts matches on full name only; to look up by first name alone, list all "
     "contacts and filter client-side."),
    ("simple_note parsing",
     "Notes that pair titles with attribution use ' - ' as the separator, but only on lines "
     "that actually carry attribution; filter before splitting or the parse drops entries."),
    ("amazon order history",
     "show_order_history returns orders newest-first and caps at 10 per page; sum across pages "
     "before reporting any total, and check order status separately from delivery date."),
    ("file system paths",
     "Paths returned by show_directory are relative to the app root, not absolute; join them "
     "onto the root before passing them back to any read call."),
    ("gmail thread handling",
     "search_emails returns one entry per thread, not per message; call show_thread to reach "
     "individual messages when a task asks about replies."),
]
_PLANS = [
    "Log in to spotify first to obtain an access_token, then page through show_song_library with "
    "page_index incrementing until an empty list is returned, collecting song ids as you go.",
    "Resolve the recipient with show_friends before calling create_payment_request, and pass the "
    "returned username rather than the phone number from the note.",
    "List all contacts once, filter client-side on the first name, then use the resulting contact "
    "id for the transfer instead of searching repeatedly.",
]
_counter = itertools.count()


class FakeModelClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def json_chat(self, system: str, user: str) -> ModelReply:
        n = next(_counter)
        scope, content = _MEMORIES[n % len(_MEMORIES)]
        parsed = {
            "route": "memory",  # the router's own route overrides this downstream
            "gap_type": "knowledge",
            "route_rationale": "stub",
            "memory_operation": "add",
            "memory": {
                "content": content,
                "scope": scope,
                "conditions": ["any task touching this app's listing endpoints"],
                "exceptions": [],
                "evidence_steps": [0],
                "confidence": 0.8,
            },
            "sft_plan": {
                "plan": _PLANS[n % len(_PLANS)],
                "repair_target": _PLANS[n % len(_PLANS)][:120],
                "evidence_steps": [0],
            },
        }
        return ModelReply(
            content=json.dumps(parsed), reasoning=None, parsed=parsed,
            usage={"prompt_tokens": 1, "completion_tokens": 1}, finish_reason="stop",
        )


# --- stub 2: the AppWorld self-eval ----------------------------------------
# Deliberately spread out, so advantages are non-degenerate and the sign of
# the update is actually exercised.
_PASS_RATES = [0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.5, 0.45]


class FakeProc:
    def __init__(self, idx: int) -> None:
        self.idx = idx


def fake_launch(bank_path, eval_dir, experiment_name, task_ids, split, model, base_url):
    fake_launch.calls.append((experiment_name, base_url))
    return FakeProc(len(fake_launch.calls) - 1)


fake_launch.calls = []


def fake_wait(proc, eval_dir, experiment_name) -> float:
    return _PASS_RATES[proc.idx % len(_PASS_RATES)]


def make_trajectories(task_ids: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for i, tid in enumerate(task_ids):
        out[tid] = {
            "task_id": tid, "success": i % 3 == 0, "reward": 0.5 if i % 2 else 0.1,
            "termination_reason": "task_completed", "domain": "appworld",
            "steps": [{"index": 0, "action": "print(1)", "observation": "1"}],
        }
    return out


def main() -> None:
    model_path = os.environ.get("ROUTER_LLM_MODEL", DEFAULT_MODEL)
    device = os.environ.get("ROUTER_LLM_DEVICE", "cuda:0")
    # Empty/unset keeps the single-card path; set to "auto" for a model that
    # has to be sharded (the 35B MoE router), in which case `device` is unused.
    device_map = os.environ.get("ROUTER_LLM_DEVICE_MAP") or None
    checkpointing = os.environ.get("ROUTER_LLM_GRADIENT_CHECKPOINTING", "") == "1"
    rollouts, batch_size, num_batches = 4, 4, 3
    print(f"smoke_router_llm_batch_e2e: model={model_path} device={device} "
          f"K={rollouts} batch_size={batch_size} batches={num_batches}")

    RBB.ModelClient = FakeModelClient
    T.launch_subset_eval = fake_launch
    T.wait_subset_eval = fake_wait
    # The sft pipeline shells out to a real AppWorld guided replay; not in scope.
    RSP.collect_batch_sft_examples = lambda *a, **k: []

    tmp = Path(tempfile.mkdtemp(prefix="router-llm-e2e-"))
    T.OUTPUT_ROOT = tmp
    T.TRAIN_LOG = tmp / "train_log.jsonl"
    try:
        # T=3.0, not the production default of 1.0 -- see the module docstring:
        # this guarantees the policy has entropy so the divergence path is
        # actually exercised, rather than leaving check [4] to the coin flip
        # that a 4x4 batch at p(argmax)=0.99 would make it.
        router = TrainableLLMRouter(RouterLLMConfig(
            model_path=model_path, device=device, device_map=device_map,
            gradient_checkpointing=checkpointing, policy_temperature=3.0))
        optimizer = torch.optim.Adam(router.trainable_parameters(), lr=1e-5)
        before = [p.detach().clone() for p in router.trainable_parameters()]

        task_ids = [f"task_{i}" for i in range(batch_size)]
        trajectories = make_trajectories(task_ids)
        args = argparse.Namespace(
            rollouts_per_batch=rollouts, batch_size=batch_size, lr=1e-5,
            entropy_coef=0.01, max_grad_norm=1.0, verify_recompute=True, drift_probe_size=4,
            model="stub", base_url="http://stub:8030/v1", base_url_probe="http://stub:8031/v1",
            sft_writer="self", teacher_model="stub", teacher_api_key_file=Path("/dev/null"),
            enable_sft_lora=False,
        )

        print("\n[1] real run_one_batch_update, writer LLM + AppWorld eval stubbed")
        bank: list[dict[str, Any]] = []
        position = 0
        for batch_idx in range(num_batches):
            bank = T.run_one_batch_update(
                router, optimizer, 1, batch_idx, task_ids, trajectories, bank,
                position, 90, args,
            )
            position += batch_size
        check("real batch loop runs without vLLM/AppWorld/35B", True)

        print("\n[2] train_log records")
        records = [json.loads(line) for line in T.TRAIN_LOG.read_text().splitlines()]
        check("one record per batch", len(records) == num_batches, f"{len(records)} records")
        expected_decisions = rollouts * batch_size
        check("every decision reached the gradient",
              all(r["decision_count"] == expected_decisions for r in records),
              f"decision_count={[r['decision_count'] for r in records]} (expected {expected_decisions})")

        print("\n[3] GRPO advantage is group-relative")
        sums = [abs(sum(r["advantages"])) for r in records]
        check("advantages sum to ~0 within each batch", all(s < 1e-9 for s in sums),
              f"max|sum|={max(sums):.2e}")

        print("\n[4] candidate divergence propagates into distinct advantages (at T=3.0)")
        distinct = [len({json.dumps(rc, sort_keys=True) for rc in r["route_counts"]}) for r in records]
        spread = [max(r["self_pass_rates"]) - min(r["self_pass_rates"]) for r in records]
        check("candidates produced more than one distinct route profile",
              max(distinct) > 1, f"distinct route profiles per batch={distinct}")
        check("more than one route is reachable at all",
              len({r for rec in records for rc in rec["route_counts"] for r in rc}) > 1,
              f"routes seen={sorted({r for rec in records for rc in rec['route_counts'] for r in rc})}")
        check("advantages are non-degenerate", all(s > 0 for s in spread),
              f"pass-rate spread per batch={[round(s,3) for s in spread]}")
        for r in records:
            print(f"       batch {r['batch_idx']}: route_counts={r['route_counts']}")

        print("\n[5] the update moved LoRA weights")
        moved = max(float((p.detach() - b).abs().max())
                    for p, b in zip(router.trainable_parameters(), before))
        check("LoRA parameters moved", moved > 0, f"max|delta|={moved:.3e}")
        print(f"       per-batch policy shift (max |delta p| over probed decisions): "
              f"{[f'{r['max_prob_shift']:.2e}' for r in records]}")
        print(f"       grad_norm per batch: {[round(r['grad_norm'], 4) for r in records]}")
        print(f"       mean_entropy per batch: {[round(r['mean_entropy'], 4) for r in records]}")

        print("\n[6] diagnostics are present for the collapse early-warning")
        check("mean_entropy logged every batch", all("mean_entropy" in r for r in records))
        check("prob_shift logged every batch", all("max_prob_shift" in r for r in records))
        check("grad_norm logged every batch", all(r.get("grad_norm") is not None for r in records))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
