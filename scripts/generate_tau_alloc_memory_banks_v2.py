#!/usr/bin/env python3
"""Build tau-bench memory banks under allocation rubrics (v2).

Thin adapter: assembles tau-bench trajectories, then hands them to the
benchmark-agnostic engine in `trajectory_memory_lab.alloc_bank_builder`.
Record and summary formats are unchanged from the original run, so existing
records resume and rebuild byte-identical banks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_tau_memory_writer_candidates import _domain_data, _trajectory
from trajectory_memory_lab.alloc_bank_builder import BuilderConfig, build_banks


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-parallel-chains", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--budget-fraction", type=float, default=0.4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    train = {
        (domain, str(task_id))
        for domain in DOMAINS
        for task_id in manifest["rubric_train"][domain]["task_ids"]
    }
    forbidden = {
        (domain, str(task_id))
        for split in ("dev", "test")
        for domain in DOMAINS
        for task_id in manifest[split][domain]["task_ids"]
    }
    if train & forbidden:
        raise ValueError("train/dev/test leakage before memory generation")

    groups: dict[str, list[str]] = {}
    trajectories: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        data = _domain_data(ROOT / "third_party/tau2-bench/data/simulations", domain)
        task_ids = [str(task_id) for task_id in manifest["rubric_train"][domain]["task_ids"]]
        groups[domain] = task_ids
        trajectories[domain] = {
            task_id: _trajectory(
                domain,
                data["tasks"][task_id],
                data["simulations"][task_id],
                data["policy"],
            )
            for task_id in task_ids
        }

    config = BuilderConfig(
        output=args.output,
        record_protocol="tau_alloc_memory_decision_v2",
        summary_protocol="tau_alloc_memory_banks_v2",
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        seed=args.seed,
        budget_fraction=args.budget_fraction,
        max_parallel_chains=args.max_parallel_chains,
    )
    summary = build_banks(
        groups,
        trajectories,
        config,
        extra_summary={
            "source_split": "rubric_train",
            "source_tasks": len(train),
            "dev_overlap": 0,
            "test_overlap": 0,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
