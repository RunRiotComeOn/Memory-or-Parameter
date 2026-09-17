"""End-to-end CPU smoke test of the real `run_one_batch_update` (v4).

Stubs exactly two boundaries -- the writer LLM (`ModelClient`) and the
AppWorld self-eval subprocess (`launch_subset_eval`/`wait_subset_eval`) -- so
the actual training code runs: real `run_router_chain`, real decision
recording, real loss, real optimizer step. No GPU, no network.

Asserts that the entropy term reaches the real loss and that a deliberately
collapsed router recovers entropy through the real update path.

Run: PYTHONPATH=src .venv/bin/python scripts/smoke/smoke_batch_update_e2e.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_router_selfreward as T  # noqa: E402
from trajectory_memory_lab import router_bank_builder as RBB  # noqa: E402
from trajectory_memory_lab.model_client import ModelReply  # noqa: E402
from trajectory_memory_lab.router_policy import (  # noqa: E402
    ROUTES,
    RouterPolicy,
    action_distribution,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# --- stub 1: the writer LLM -------------------------------------------------
_counter = itertools.count()


class FakeModelClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def json_chat(self, system: str, user: str) -> ModelReply:
        n = next(_counter)
        parsed = {
            "route": "memory",  # overwritten by the router's own route downstream
            "gap_type": "knowledge",
            "route_rationale": "stub",
            "memory_operation": "add",
            "memory": {
                # distinct content per call so the dedup path does not collapse
                # everything into one entry
                "content": f"stub memory fact number {n} about topic {n % 7}",
                "scope": f"appworld stub scope {n % 7}",
                "conditions": [f"condition {n}"],
                "exceptions": [],
                "evidence_steps": [0],
                "confidence": 0.8,
            },
            "sft_plan": {"repair_target": f"stub repair {n}", "evidence_steps": [0]},
        }
        return ModelReply(
            content=json.dumps(parsed), reasoning=None, parsed=parsed,
            usage={"prompt_tokens": 1, "completion_tokens": 1}, finish_reason="stop",
        )


# --- stub 2: the AppWorld self-eval ----------------------------------------
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
            "termination_reason": "task_completed",
            "steps": [{"index": 0, "action": "print(1)", "observation": "1"}],
        }
    return out


def run_batch(entropy_coef: float, tmp: Path, seed_model: RouterPolicy, lr: float = 0.01):
    RBB.ModelClient = FakeModelClient
    T.launch_subset_eval = fake_launch
    T.wait_subset_eval = fake_wait
    T.OUTPUT_ROOT = tmp / f"coef{entropy_coef}"
    T.TRAIN_LOG = T.OUTPUT_ROOT / "train_log.jsonl"

    model = RouterPolicy()
    model.load_state_dict(seed_model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    task_ids = [f"task_{i}" for i in range(10)]
    trajectories = make_trajectories(task_ids)
    args = argparse.Namespace(
        rollouts_per_batch=8, batch_size=10, lr=lr, entropy_coef=entropy_coef,
        model="stub", base_url="http://stub:8010/v1", base_url_probe="http://stub:8011/v1",
    )
    before = [p.detach().clone() for p in model.parameters()]
    bank = []
    for batch_idx in range(6):  # 6 sequential updates, same batch of tasks
        bank = T.run_one_batch_update(
            model, opt, 1, batch_idx, task_ids, trajectories, bank, 0, 90, args,
        )
    moved = max(float((p.detach() - b).abs().max()) for p, b in zip(model.parameters(), before))
    feats = torch.tensor([1.0, 0.5, 0.3, 0.4, 0.6])
    dist = action_distribution(model, feats)
    return moved, float(dist.entropy().detach()), dist.probs.detach(), T.TRAIN_LOG


tmp = Path(tempfile.mkdtemp(prefix="v4-smoke-"))
try:
    # Start both arms from the SAME collapsed router, so the only difference
    # between them is the entropy term.
    seed = RouterPolicy()
    with torch.no_grad():
        seed.linear.weight.zero_()
        seed.linear.bias.zero_()
        seed.linear.bias[ROUTES.index("both")] = 6.0
    H_start = float(action_distribution(seed, torch.tensor([1.0, 0.5, 0.3, 0.4, 0.6])).entropy())

    print("\n1. real run_one_batch_update, LLM + AppWorld eval stubbed")
    print(f"  starting router: collapsed on 'both', H={H_start:.4f}")

    torch.manual_seed(20260822)
    moved_off, H_off, probs_off, log_off = run_batch(0.0, tmp, seed)
    torch.manual_seed(20260822)
    moved_on, H_on, probs_on, log_on = run_batch(0.01, tmp, seed)

    print(f"  entropy_coef=0.00 -> H={H_off:.4f} probs={ {r: round(float(p),4) for r,p in zip(ROUTES, probs_off)} }")
    print(f"  entropy_coef=0.01 -> H={H_on:.4f} probs={ {r: round(float(p),4) for r,p in zip(ROUTES, probs_on)} }")

    check("real update path runs without a GPU/LLM/AppWorld", True)
    check("params update (coef=0)", moved_off > 1e-6, f"max|delta|={moved_off:.6f}")
    check("params update (coef=0.01)", moved_on > 1e-6, f"max|delta|={moved_on:.6f}")
    check("entropy term raises entropy vs identical run without it",
          H_on > H_off, f"{H_off:.4f} (off) vs {H_on:.4f} (on)")
    # Recovery from a *fully* collapsed router (bias 6.0) is inherently slow at
    # lr=0.01: Adam caps movement at ~lr per step, so undoing a 6-unit bias needs
    # hundreds of steps, not the 6 here. What this asserts is the sign -- entropy
    # moving up rather than further down. Prevention from a realistic init is the
    # regime that matters, and is checked against the coef=0 arm above and in the
    # entropy_coef sweep recorded in DESIGN.md section 12.
    check("entropy moves up rather than further down", H_on > H_start,
          f"{H_start:.4f} -> {H_on:.4f} (coef=0 arm went to {H_off:.4f})")
    # No absolute claim about the coef=0 arm here: this stub's pass rates are a
    # fixed list unrelated to the sampled routes, so there is no systematic
    # pro-`both` pressure for the entropy term to fight. The collapse pressure
    # itself is modeled in the entropy_coef sweep (DESIGN.md section 12); what
    # this file pins down is that the term is wired into the real loss and moves
    # the real parameters in the right direction.

    print("\n2. train_log.jsonl records the new diagnostics")
    records = [json.loads(l) for l in log_on.read_text().splitlines()]
    check("one record per batch", len(records) == 6, f"{len(records)} records")
    r = records[-1]
    check("record has mean_entropy", "mean_entropy" in r, str(r.get("mean_entropy")))
    check("record has pg_term", "pg_term" in r, str(r.get("pg_term")))
    check("record has entropy_coef", r.get("entropy_coef") == 0.01, str(r.get("entropy_coef")))
    check("loss != pg_term when the entropy term is on",
          abs(r["loss"] - r["pg_term"]) > 1e-9, f"loss={r['loss']:.4f} pg={r['pg_term']:.4f}")
    off_last = [json.loads(l) for l in log_off.read_text().splitlines()][-1]
    check("loss == pg_term when entropy_coef=0",
          abs(off_last["loss"] - off_last["pg_term"]) < 1e-9)

    ent_series = [round(x["mean_entropy"], 4) for x in records]
    ent_series_off = [round(json.loads(l)["mean_entropy"], 4) for l in log_off.read_text().splitlines()]
    print(f"  mean_entropy per batch, coef=0.01: {ent_series}")
    print(f"  mean_entropy per batch, coef=0.00: {ent_series_off}")
    check("entropy trends up with the term on", ent_series[-1] > ent_series[0],
          f"{ent_series[0]} -> {ent_series[-1]}")

    print("\n3. eval parallelism: candidates are spread across both replicas")
    urls = [u for _, u in fake_launch.calls]
    check("both replica URLs used", len(set(urls)) == 2, str(sorted(set(urls))))
    check("8 rollouts/batch -> 8 evals per batch",
          len(fake_launch.calls) == 8 * 6 * 2, f"{len(fake_launch.calls)} total across both arms")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
