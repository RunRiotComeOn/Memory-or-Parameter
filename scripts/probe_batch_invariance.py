#!/usr/bin/env python3
"""Does batching change the answer?

The determinism that matters for this project is not "same prompt twice on an
idle server" but "same prompt, whether it happens to share a decode batch with
other requests or not".  That is what varies between runs of a parallel rollout,
and it is what has to hold if we want determinism *and* throughput.

Protocol: generate the target prompt alone, then generate it again while N
unrelated filler prompts are in flight, and compare.  Run with and without
VLLM_BATCH_INVARIANT=1 on the server to see whether that flag buys the property.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys

import requests


TARGET = (
    "You are an autonomous agent that solves tasks by writing Python code.\n"
    "Task: I want to know which of my Spotify playlists has the most songs, and "
    "how many songs are in it. Write out your full reasoning step by step, then "
    "the code you would run, then explain what could go wrong and how you would "
    "recover from each failure. Be thorough and specific.\n"
)

FILLERS = [
    "Explain in detail how a B-tree index speeds up range queries.",
    "Write a long, careful explanation of the CAP theorem with examples.",
    "Describe step by step how to debug a memory leak in a Python service.",
    "Explain how HTTP/2 multiplexing differs from HTTP/1.1 pipelining.",
    "Walk through the derivation of the bias-variance decomposition.",
    "Describe the lifecycle of a Kubernetes pod from create to terminate.",
    "Explain how a Bloom filter works and when it gives false positives.",
    "Describe how git rebase rewrites history, with a worked example.",
]


def complete(base_url: str, model: str, prompt: str, max_tokens: int, seed: int) -> str:
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def first_divergence(a: str, b: str) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-tau")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--concurrency-schedule",
        default=None,
        help="comma-separated per-round concurrency, e.g. 8,4,8,6,2. Overrides "
        "--concurrency/--rounds. A real rollout's batch composition drifts with "
        "timing jitter, so a fixed concurrency under-tests invariance.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if args.concurrency_schedule:
        schedule = [int(value) for value in args.concurrency_schedule.split(",")]
    else:
        schedule = [args.concurrency] * args.rounds

    solo = complete(args.base_url, args.model, TARGET, args.max_tokens, args.seed)
    print(f"solo: {len(solo)} chars", flush=True)

    batched = []
    for round_index, concurrency in enumerate(schedule):
        prompts = [TARGET] + FILLERS[: concurrency - 1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(complete, args.base_url, args.model, prompt, args.max_tokens, args.seed)
                for prompt in prompts
            ]
            texts = [future.result() for future in futures]
        batched.append(texts[0])
        print(
            f"batched round {round_index + 1} (concurrency {concurrency}): "
            f"{len(texts[0])} chars",
            flush=True,
        )

    divergences = [first_divergence(solo, text) for text in batched]
    report = {
        "max_tokens": args.max_tokens,
        "concurrency_schedule": schedule,
        "solo_chars": len(solo),
        "batched_chars": [len(text) for text in batched],
        "first_divergence_chars": divergences,
        "distinct_batched_outputs": len(set(batched)),
        "batched_agree_with_each_other": len(set(batched)) == 1,
        "batch_invariant": all(value is None for value in divergences),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["batch_invariant"] else 1)


if __name__ == "__main__":
    main()
