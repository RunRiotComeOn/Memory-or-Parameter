"""Smoke test for the trainable LLM router's core mechanism (step 1 of 2).

What is actually under test -- the claims the whole GRPO design rests on, each
of which would otherwise only fail silently, deep inside a 40-hour run:

  1. The four routes have DISTINCT first tokens after `{"route": "`, so a
     single position addresses the whole action space.
  2. A forward pass produces a well-formed 4-way distribution (probs sum to 1,
     entropy within [0, ln4]).
  3. The recompute-with-grad path returns the SAME logprob as the no-grad
     sampling path. This replaces the "does vLLM's logprob match a local
     teacher-forced pass" check the vLLM design would have needed: same
     weights, same process, so the tolerance can be ~1e-3 instead of "close
     enough", and any real drift is a bug rather than kernel noise.
  4. That logprob carries a gradient into the LoRA parameters and ONLY into
     them -- the base model must stay frozen.
  5. Greedy and sampling agree about which route is most likely (argmax of the
     same distribution), and the payload the trained router sees is
     byte-identical to what the untrained `decide_route` probe sends.

Needs a GPU and a local model; it does NOT need any vLLM server, any AppWorld
environment, or the 35B task agent.

Run: ROUTER_LLM_MODEL=<path> CUDA_VISIBLE_DEVICES=6 \
       .venv/bin/python scripts/smoke/smoke_router_llm_logprob.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_llm_policy import (  # noqa: E402
    ROUTER_LLM_SYSTEM,
    ROUTES,
    build_router_payload,
)
from trajectory_memory_lab.router_llm_trainable import (  # noqa: E402
    ROUTE_PREFIX,
    RouterLLMConfig,
    TrainableLLMRouter,
    verify_recompute_matches_sample,
)

DEFAULT_MODEL = "/nas04/yixuh/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def fake_decision_inputs() -> tuple[dict, int, str, dict, dict]:
    """A realistic-shaped routing decision: a FAILED task with both a drafted
    memory and a drafted repair plan, i.e. the case where all four routes are
    genuinely live. Content is synthetic on purpose -- this test is about the
    mechanism, not about whether the judgment is any good."""
    trajectory = {
        "success": False,
        "reward": 0.0,
        "termination_reason": "task_completed",
        "domain": "appworld",
    }
    draft_memory = {
        "content": "Spotify's show_song_library is paginated; pass page_index until an empty page is returned, otherwise only the first 20 songs are ever seen.",
        "scope": "spotify library enumeration",
        "conditions": "any task that must enumerate a full song or album library",
        "exceptions": "not needed when the task names a specific song id",
    }
    draft_sft_plan = {
        "plan": "Log in to spotify first to obtain an access_token, then page through show_song_library with page_index=0,1,2,... until an empty list comes back, collecting song ids as you go.",
        "repair_target": "missing pagination on show_song_library",
        "evidence_steps": [12, 14],
    }
    return trajectory, 7, "phone: contacts must be looked up by name before venmo transfers", draft_memory, draft_sft_plan


def main() -> None:
    model_path = os.environ.get("ROUTER_LLM_MODEL", DEFAULT_MODEL)
    device = os.environ.get("ROUTER_LLM_DEVICE", "cuda:0")
    # Empty/unset keeps the single-card path; set to "auto" for a model that
    # has to be sharded (the 35B MoE router), in which case `device` is unused.
    device_map = os.environ.get("ROUTER_LLM_DEVICE_MAP") or None
    checkpointing = os.environ.get("ROUTER_LLM_GRADIENT_CHECKPOINTING", "") == "1"
    print(f"smoke_router_llm_logprob: model={model_path} device={device}")

    print("\n[1] tokenizer: routes distinguishable at one position")
    router = TrainableLLMRouter(RouterLLMConfig(
        model_path=model_path, device=device, device_map=device_map,
        gradient_checkpointing=checkpointing))
    ids = router.route_token_ids
    check("four distinct route first-tokens", len(set(ids)) == 4, str(dict(zip(ROUTES, ids))))
    decoded = [router.tokenizer.decode([i]) for i in ids]
    check("route tokens decode to route prefixes",
          all(r.startswith(d) for r, d in zip(ROUTES, decoded)), str(decoded))
    check("LoRA is the only trainable part",
          router.trainable_parameter_count() > 0,
          f"{router.trainable_parameter_count():,} trainable params")
    base_trainable = [n for n, p in router.model.named_parameters()
                      if p.requires_grad and "lora" not in n.lower()]
    check("no base weight is trainable", not base_trainable, f"{len(base_trainable)} non-LoRA trainable")

    print("\n[2] prompt: identical payload to the untrained decide_route probe")
    trajectory, mem_count, recent, draft_memory, draft_sft = fake_decision_inputs()
    prompt_ids = router.build_prompt_ids(trajectory, mem_count, recent, draft_memory, draft_sft)
    text = router.tokenizer.decode(prompt_ids)
    check("prompt ends with the forced route prefix", text.endswith(ROUTE_PREFIX), repr(text[-30:]))
    check("system prompt is embedded verbatim", ROUTER_LLM_SYSTEM[:120] in text)
    payload = build_router_payload(trajectory, mem_count, recent, draft_memory, draft_sft)
    check("drafted content reaches the prompt",
          json.dumps(payload, ensure_ascii=False, separators=(",", ":"))[:200] in text)
    print(f"       prompt is {len(prompt_ids)} tokens")

    print("\n[3] forward pass: a well-formed 4-way policy")
    route, index, probs, entropy = router.sample_action(prompt_ids)
    check("sampled route is a real route", route in ROUTES, f"route={route}")
    check("probs sum to 1", abs(float(probs.sum()) - 1.0) < 1e-4, f"sum={float(probs.sum()):.6f}")
    check("entropy within [0, ln4]", 0.0 <= float(entropy) <= math.log(4) + 1e-4,
          f"H={float(entropy):.4f} (ln4={math.log(4):.4f})")
    print(f"       probs={dict(zip(ROUTES, [round(float(p), 4) for p in probs]))}")
    greedy_route, greedy_index, _, _ = router.greedy_action(prompt_ids)
    check("greedy == argmax of the sampled distribution",
          greedy_index == int(torch.argmax(probs).item()), f"greedy={greedy_route}")

    print("\n[4] recompute-with-grad reproduces the sampled logprob")
    sampled, recomputed, difference = verify_recompute_matches_sample(router, prompt_ids, index)
    check("sampled logprob == recomputed logprob", difference < 1e-3,
          f"sampled={sampled:.6f} recomputed={recomputed:.6f} |diff|={difference:.2e}")

    print("\n[5] the recomputed logprob is differentiable into LoRA only")
    logprob, ent = router.logprob_and_entropy(prompt_ids, index)
    check("logprob requires grad", logprob.requires_grad)
    check("entropy requires grad", ent.requires_grad)
    for p in router.trainable_parameters():
        p.grad = None
    logprob.backward()
    with_grad = [p for p in router.trainable_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    check("gradient reaches LoRA parameters", len(with_grad) > 0,
          f"{len(with_grad)}/{len(router.trainable_parameters())} LoRA tensors got a nonzero grad")
    frozen_with_grad = [n for n, p in router.model.named_parameters()
                        if "lora" not in n.lower() and p.grad is not None]
    check("no gradient on frozen base weights", not frozen_with_grad,
          f"{len(frozen_with_grad)} base tensors carry a grad")

    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
