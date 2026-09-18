#!/usr/bin/env python3
"""Ask whether the serving stack is reproducible at all before spending GPU-days on it.

Sends the same prompt N times, serially, and reports whether the completions are
byte-identical and where the first divergence is.  A long generation is used on
purpose: a single flipped token early in a rollout is what turns into a
different episode 40 steps later, so short probes under-report the problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import requests


PROMPT = (
    "You are an autonomous agent that solves tasks by writing Python code.\n"
    "Task: I want to know which of my Spotify playlists has the most songs, and "
    "how many songs are in it. Write out your full reasoning step by step, then "
    "the code you would run, then explain what could go wrong and how you would "
    "recover from each failure. Be thorough and specific.\n"
)


def complete(base_url: str, model: str, max_tokens: int, seed: int) -> str:
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
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
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    outputs = []
    for index in range(args.repeats):
        text = complete(args.base_url, args.model, args.max_tokens, args.seed)
        outputs.append(text)
        print(f"[{index + 1}/{args.repeats}] {len(text)} chars", flush=True)

    reference = outputs[0]
    divergences = [first_divergence(reference, text) for text in outputs[1:]]
    identical = sum(1 for value in divergences if value is None)
    report = {
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "identical_to_first": identical + 1,
        "distinct_outputs": len(set(outputs)),
        "reference_chars": len(reference),
        "first_divergence_chars": divergences,
        "length_histogram": dict(Counter(len(text) for text in outputs)),
        "deterministic": len(set(outputs)) == 1,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["deterministic"] else 1)


if __name__ == "__main__":
    main()
