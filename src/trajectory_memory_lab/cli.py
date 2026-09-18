from __future__ import annotations

import argparse
import os
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", type=Path, default=Path("tasks/public_readonly.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--initial-memory",
        type=Path,
        help="Seed every task with an existing memory_bank.json file.",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Run tasks without generating new context or SFT examples.",
    )
    parser.add_argument(
        "--no-tool-audit",
        action="store_true",
        help="Skip the second model pass that audits memory and SFT tool proposals.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run_experiment(
        ExperimentConfig(
            task_file=args.tasks,
            output_root=args.output_root,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            limit=args.limit,
            max_steps=args.max_steps,
            headless=not args.headed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            initial_memory_file=args.initial_memory,
            review_artifacts=not args.no_review,
            audit_retention_tools=not args.no_tool_audit,
        )
    )
    print(f"Run written to {run_dir}")


if __name__ == "__main__":
    main()
