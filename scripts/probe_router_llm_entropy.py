#!/usr/bin/env python3
"""Re-measure DESIGN.md section 16.3's exploration table for a given router model.

Section 16.3 is the measurement that decides whether GRPO can work at all: if
the policy is too confident, all K candidates in a batch make identical
decisions, every advantage is exactly 0, the gradient is 0, and the run spins
for 40 hours with a log that looks completely normal. It also records the
mistake that nearly inverted the conclusion -- measuring on records where
`drafted_sft_plan_candidate` was null, so `sft`/`both` had nothing to commit
and a 0 probability for them was CORRECT judgment rather than collapse.

So this measures on REAL records with all four routes genuinely available:
it runs the actual `run_router_chain` drafting path against the live task-agent
replica, exactly as a training batch would, and reads the distribution off the
decisions that come back. It never evaluates anything -- no AppWorld replay, no
reward -- so it costs only the drafting time (~40s/task).

Temperatures other than the sampled one are derived exactly rather than
re-run: `probs = softmax(logits / T0)`, so `log(probs)` recovers the logits up
to an additive constant and `softmax(log(probs) * T0 / T)` is the tempered
distribution the policy would actually have used.

Run:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/nas04/yixuh/hf_cache \
  /nas04/yixuh/router_venv/bin/python -u scripts/probe_router_llm_entropy.py \
      --limit 20 --base-url http://127.0.0.1:8030/v1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_memory_lab.router_bank_builder import RouterBuilderConfig, run_router_chain  # noqa: E402
from trajectory_memory_lab.router_llm_policy import ROUTES  # noqa: E402
from trajectory_memory_lab.router_llm_trainable import RouterLLMConfig, TrainableLLMRouter  # noqa: E402

GROUP = "appworld"
TRAIN_ROLLOUT = ROOT / "appworld_experiment/base_train_v2"
DEFAULT_ROUTER_MODEL = (
    "/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "59d61f3ce65a6d9863b86d2e96597125219dc754"
)


def load_train_trajectories() -> dict[str, dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted((TRAIN_ROLLOUT / "trajectories").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        trajectory = record["trajectory"]
        trajectory["domain"] = GROUP
        trajectories[record["task_id"]] = trajectory
    return trajectories


def temper(probs: torch.Tensor, from_t: float, to_t: float) -> torch.Tensor:
    return torch.softmax(torch.log(probs.double().clamp_min(1e-30)) * (from_t / to_t), dim=-1)


def report(rows: list[torch.Tensor], sampled_t: float, target_t: float, k: int) -> dict[str, Any]:
    tempered = [temper(p, sampled_t, target_t) for p in rows]
    entropies = [float(-(p * p.clamp_min(1e-30).log()).sum()) for p in tempered]
    p_argmax = [float(p.max()) for p in tempered]
    # P(all K candidates decide identically on EVERY decision in the batch):
    # per decision, sum_r p_r^K is the chance all K draws land on the same
    # route; the decisions are independent draws, so the batch is the product.
    p_same_batch = math.prod(float((p.double() ** k).sum()) for p in tempered)
    mean_probs = torch.stack(tempered).mean(dim=0)
    argmax_counts = Counter(ROUTES[int(p.argmax())] for p in tempered)
    return {
        "temperature": target_t,
        "n": len(rows),
        "mean_entropy": statistics.mean(entropies),
        "mean_p_argmax": statistics.mean(p_argmax),
        "p_all_k_identical_per_decision": statistics.mean(
            [float((p.double() ** k).sum()) for p in tempered]),
        "p_all_k_identical_whole_batch": p_same_batch,
        "mean_route_probs": {r: round(float(v), 4) for r, v in zip(ROUTES, mean_probs)},
        "argmax_counts": dict(argmax_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "router_reward_v1/router_llm_entropy_probe")
    parser.add_argument("--router-model-path", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-device", default="cuda:0")
    parser.add_argument("--router-device-map", default="auto")
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8030/v1")
    parser.add_argument("--sft-writer", choices=("teacher", "self"), default="teacher")
    parser.add_argument("--teacher-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--teacher-api-key-file", type=Path,
                        default=Path("/nas04/yixuh/.config/continual-memory/gemini_api_key"))
    parser.add_argument("--limit", type=int, default=20, help="tasks to draft and route; 16.3 used 20")
    parser.add_argument("--rollouts-per-batch", type=int, default=8, help="the K the P(identical) column assumes")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    args = parser.parse_args()

    trajectories = load_train_trajectories()
    task_ids = sorted(trajectories)[: args.limit]

    router = TrainableLLMRouter(RouterLLMConfig(
        model_path=args.router_model_path, device=args.router_device,
        device_map=args.router_device_map or None,
        gradient_checkpointing=False,  # no backward here; sampling only
        policy_temperature=args.policy_temperature,
    ))
    print(f"router: {args.router_model_path}\n"
          f"        input_device={router.input_device} logit_device={router.device} "
          f"trainable={router.trainable_parameter_count():,}", flush=True)

    result = run_router_chain(
        router, GROUP, task_ids, trajectories,
        RouterBuilderConfig(
            output=args.output, record_protocol="router_llm_entropy_probe",
            model=args.model, base_url=args.base_url, seed=20260822,
            sft_writer=args.sft_writer, teacher_model=args.teacher_model,
            teacher_api_key_file=args.teacher_api_key_file,
            router_mode="trained_llm",
        ),
    )

    probs = [d["probs"] for d in result.decisions]
    prompt_lengths = [len(d["sampled_decision"].prompt_ids) for d in result.decisions]
    # `drafted_sft_plan_available` is the direct signal; records written
    # before it existed only reveal availability when the route actually took
    # the plan, so fall back to that and say which reading this is.
    records = result.summary["records"]
    direct = [r for r in records if "drafted_sft_plan_available" in r]
    if direct:
        sft_available = sum(1 for r in direct if r["drafted_sft_plan_available"])
        availability_basis = "drafted_sft_plan_available"
    else:
        sft_available = sum(1 for r in records if (r.get("decision") or {}).get("sft_plan"))
        availability_basis = "inferred from committed sft_plan (LOWER BOUND)"
    print(f"\nprompt tokens: min={min(prompt_lengths)} median={statistics.median(prompt_lengths)} "
          f"max={max(prompt_lengths)}", flush=True)
    print(f"records with an sft plan actually available: {sft_available}/{len(result.decisions)} "
          f"[{availability_basis}]\n"
          "  (16.3's trap: 0 here would make p(sft)=0 correct judgment, not collapse)", flush=True)

    rows = [report(probs, args.policy_temperature, t, args.rollouts_per_batch)
            for t in args.temperatures]
    print(f"\n{'T':>5} {'H (nats)':>9} {'p(argmax)':>10} {'P(K same/dec)':>14} "
          f"{'P(K same/batch)':>16}  mean route probs / argmax counts")
    for row in rows:
        print(f"{row['temperature']:>5.1f} {row['mean_entropy']:>9.4f} {row['mean_p_argmax']:>10.4f} "
              f"{row['p_all_k_identical_per_decision']:>14.4f} "
              f"{row['p_all_k_identical_whole_batch']:>16.6f}  "
              f"{row['mean_route_probs']} / {row['argmax_counts']}")

    out = {
        "router_model_path": args.router_model_path,
        "sampled_policy_temperature": args.policy_temperature,
        "rollouts_per_batch": args.rollouts_per_batch,
        "n_decisions": len(probs),
        "prompt_tokens": {"min": min(prompt_lengths), "max": max(prompt_lengths),
                          "median": statistics.median(prompt_lengths)},
        "per_decision_probs": [[round(float(x), 6) for x in p] for p in probs],
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "entropy_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output / 'entropy_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
