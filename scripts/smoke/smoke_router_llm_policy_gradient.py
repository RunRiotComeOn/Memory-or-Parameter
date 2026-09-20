"""Smoke test for the trainable LLM router's GRPO step (step 2 of 2).

Runs the REAL `accumulate_policy_gradient` -- the same function the training
loop calls -- on synthetic advantages. No AppWorld, no reward pipeline, no
35B task agent: the only thing under test is that a policy-gradient step
moves the LoRA weights in the direction the advantage asked for.

What it checks:
  1. A POSITIVE advantage on a sampled route raises that route's probability;
     a NEGATIVE advantage lowers it. (Loss sign. Getting this backwards
     trains the router to do the opposite of what the reward says, and
     nothing downstream would notice -- the numbers all still look plausible.)
  2. LoRA weights actually change, and base weights do not.
  3. The entropy bonus alone (zero advantage) pushes toward uniform, so the
     v4 collapse defense still works through an LLM.
  4. A learning-rate sweep reporting how far one step actually moves the route
     probability. This exists because the update BUDGET here is tiny -- 9
     gradient steps per iteration, set by a reward that costs ~70 min per
     candidate -- so "is this lr large enough to matter in 9 steps" is a real
     question, and `train_router_selfreward.py`'s lr=0.01 was tuned for a
     144-parameter linear model, not for LoRA on an LLM.

The sweep is also how the default lr was picked, and it found the opposite of
the expected problem: 1e-4 (this repo's SFT LoRA rate, `train_qwen_lora.sh`)
is far too LARGE here, not too small -- a single step moved p(route) from
0.00000 to 0.949, and 10 entropy-bonus steps at that rate diverged into a
fully collapsed distribution (H 0.113 -> 0.0001 nats) rather than a uniform
one. Entropy rises correctly at every rate from 1e-7 through 3e-5, so the
collapse was step size, not a sign error. The dynamics are strongly
nonlinear: LoRA B initializes to zero, so the first steps move almost
nothing and later ones move a great deal.

Run: ROUTER_LLM_MODEL=<path> CUDA_VISIBLE_DEVICES=6 \
       .venv/bin/python scripts/smoke/smoke_router_llm_policy_gradient.py
"""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_llm_policy import ROUTES  # noqa: E402
from trajectory_memory_lab.router_llm_trainable import (  # noqa: E402
    RouterLLMConfig,
    SampledDecision,
    TrainableLLMRouter,
    accumulate_policy_gradient,
)

DEFAULT_MODEL = "/nas04/yixuh/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def make_decision(router: TrainableLLMRouter, index: int) -> SampledDecision:
    """A single routing decision on a synthetic-but-realistically-shaped task
    where all four routes are live (a failed task with both a drafted memory
    and a drafted repair plan)."""
    trajectory = {"success": False, "reward": 0.0, "termination_reason": "max_steps", "domain": "appworld"}
    draft_memory = {
        "content": "show_song_library is paginated; page until an empty page returns.",
        "scope": "spotify library enumeration", "conditions": "full-library enumeration", "exceptions": "",
    }
    draft_sft_plan = {
        "plan": "Log in to spotify for an access_token, then page show_song_library until empty.",
        "repair_target": "missing pagination", "evidence_steps": [12],
    }
    prompt_ids = router.build_prompt_ids(trajectory, 7, "", draft_memory, draft_sft_plan)
    _, _, probs, entropy = router.sample_action(prompt_ids)
    return SampledDecision(
        prompt_ids=prompt_ids, index=index, route=ROUTES[index],
        task_id="smoke_0", group="appworld", probs=probs, entropy=entropy,
    )


def route_probs(router: TrainableLLMRouter, prompt_ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(router.route_logits(prompt_ids), dim=-1).cpu()


def fresh_adapter_state(router: TrainableLLMRouter) -> dict:
    return copy.deepcopy({n: p.detach().clone() for n, p in router.model.named_parameters() if p.requires_grad})


def restore_adapter(router: TrainableLLMRouter, state: dict) -> None:
    with torch.no_grad():
        for n, p in router.model.named_parameters():
            if n in state:
                p.copy_(state[n])


def main() -> None:
    model_path = os.environ.get("ROUTER_LLM_MODEL", DEFAULT_MODEL)
    device = os.environ.get("ROUTER_LLM_DEVICE", "cuda:0")
    # Empty/unset keeps the single-card path; set to "auto" for a model that
    # has to be sharded (the 35B MoE router), in which case `device` is unused.
    device_map = os.environ.get("ROUTER_LLM_DEVICE_MAP") or None
    checkpointing = os.environ.get("ROUTER_LLM_GRADIENT_CHECKPOINTING", "") == "1"
    lr = float(os.environ.get("ROUTER_LLM_LR", "1e-5"))
    print(f"smoke_router_llm_policy_gradient: model={model_path} device={device} lr={lr}")

    router = TrainableLLMRouter(RouterLLMConfig(
        model_path=model_path, device=device, device_map=device_map,
        gradient_checkpointing=checkpointing))
    pristine = fresh_adapter_state(router)

    # Target the LEAST likely route, so "did the update move it" is not
    # confounded by the route already sitting near probability 1.
    probe = make_decision(router, 0)
    base_probs = route_probs(router, probe.prompt_ids)
    target_index = int(torch.argmin(base_probs).item())
    print(f"       base probs={dict(zip(ROUTES, [round(float(p),4) for p in base_probs]))}")
    print(f"       targeting least-likely route {ROUTES[target_index]!r} (p={float(base_probs[target_index]):.4f})")

    print("\n[1] loss sign: a positive advantage raises the chosen route's probability")
    decision = make_decision(router, target_index)
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=lr)
    optimizer.zero_grad()
    stats = accumulate_policy_gradient(router, [decision], [1.0], entropy_coef=0.0)
    optimizer.step()
    up_probs = route_probs(router, decision.prompt_ids)
    delta_up = float(up_probs[target_index] - base_probs[target_index])
    check("positive advantage raises p(route)", delta_up > 0,
          f"p({ROUTES[target_index]}) {float(base_probs[target_index]):.5f} -> {float(up_probs[target_index]):.5f} (delta={delta_up:+.2e})")
    check("pg_term sign is -advantage*logprob", stats["pg_term"] > 0,
          f"pg_term={stats['pg_term']:.4f} (logprob<0, advantage>0 => -a*lp>0)")

    print("\n[2] loss sign: a negative advantage lowers it")
    restore_adapter(router, pristine)
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=lr)
    optimizer.zero_grad()
    accumulate_policy_gradient(router, [decision], [-1.0], entropy_coef=0.0)
    optimizer.step()
    down_probs = route_probs(router, decision.prompt_ids)
    delta_down = float(down_probs[target_index] - base_probs[target_index])
    check("negative advantage lowers p(route)", delta_down < 0,
          f"p({ROUTES[target_index]}) {float(base_probs[target_index]):.5f} -> {float(down_probs[target_index]):.5f} (delta={delta_down:+.2e})")

    print("\n[3] the update touches LoRA weights only")
    restore_adapter(router, pristine)
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=lr)
    optimizer.zero_grad()
    accumulate_policy_gradient(router, [decision], [1.0], entropy_coef=0.0)
    optimizer.step()
    moved = sum(1 for n, p in router.model.named_parameters()
                if n in pristine and not torch.equal(p.detach(), pristine[n]))
    check("LoRA parameters moved", moved > 0, f"{moved}/{len(pristine)} adapter tensors changed")
    base_grads = [n for n, p in router.model.named_parameters()
                  if "lora" not in n.lower() and p.grad is not None]
    check("base weights untouched (no grad)", not base_grads, f"{len(base_grads)} base tensors carry a grad")

    print("\n[4] entropy bonus alone pushes toward uniform (v4 collapse defense)")
    restore_adapter(router, pristine)
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=lr)
    start_entropy = float(-(base_probs * base_probs.clamp_min(1e-12).log()).sum())
    for _ in range(10):
        optimizer.zero_grad()
        accumulate_policy_gradient(router, [decision], [0.0], entropy_coef=1.0)
        optimizer.step()
    ent_probs = route_probs(router, decision.prompt_ids)
    end_entropy = float(-(ent_probs * ent_probs.clamp_min(1e-12).log()).sum())
    check("entropy bonus raises entropy", end_entropy > start_entropy,
          f"H {start_entropy:.4f} -> {end_entropy:.4f} nats (ln4={math.log(4):.4f}), 10 steps at coef=1.0")

    print("\n[5] learning-rate sweep: how far does ONE step actually move p(route)?")
    print("     (the iteration budget is ~9 steps, so a step that moves nothing is the real risk)")
    print(f"     {'lr':>8} {'delta p(route)':>16} {'p after 9 steps':>17}")
    for sweep_lr in [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        restore_adapter(router, pristine)
        optimizer = torch.optim.Adam(router.trainable_parameters(), lr=sweep_lr)
        optimizer.zero_grad()
        accumulate_policy_gradient(router, [decision], [1.0], entropy_coef=0.0)
        optimizer.step()
        one = float(route_probs(router, decision.prompt_ids)[target_index])
        for _ in range(8):
            optimizer.zero_grad()
            accumulate_policy_gradient(router, [decision], [1.0], entropy_coef=0.0)
            optimizer.step()
        nine = float(route_probs(router, decision.prompt_ids)[target_index])
        print(f"     {sweep_lr:>8.0e} {one - float(base_probs[target_index]):>+16.2e} {nine:>17.5f}")
    restore_adapter(router, pristine)

    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
