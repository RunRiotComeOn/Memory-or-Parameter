"""Benchmark-agnostic external-memory retrieval.

Deliberately a byte-for-byte behavioural port of the BM25 retrieval baked into
`third_party/tau2-bench/src/tau2/agent/llm_agent.py`, so a memory bank scores
the same way whichever benchmark consumes it.  Only `scope` and `content` are
rendered to the agent; `conditions` and `exceptions` stay audit metadata.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


MEMORY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MEMORY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "customer", "do", "for", "from", "has", "have", "i", "if", "in",
    "is", "it", "me", "must", "my", "of", "on", "or", "please", "should",
    "that", "the", "their", "them", "they", "this", "to", "user", "want",
    "when", "with", "you",
}


def memory_tokens(text: str) -> list[str]:
    return [
        token
        for token in MEMORY_TOKEN_PATTERN.findall(text.lower())
        if token not in MEMORY_STOPWORDS
    ]


def load_bank(path: str | Path) -> list[dict[str, str]]:
    """Load active entries, keeping only the fields the agent ever sees."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [
        {key: entry[key] for key in ("id", "scope", "content") if entry.get(key)}
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("status", "active") == "active"
        and entry.get("content")
    ]


def retrieve(
    entries: list[dict[str, str]], query: str, top_k: int
) -> list[tuple[dict[str, str], float]]:
    """BM25 over scope+content, with scope counted twice."""
    if not entries or top_k <= 0:
        return []
    query_terms = Counter(memory_tokens(query))
    if not query_terms:
        return []
    documents = [
        memory_tokens(f"{entry.get('scope', '')} {entry.get('scope', '')} {entry.get('content', '')}")
        for entry in entries
    ]
    average_length = sum(map(len, documents)) / len(documents)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    scored = []
    for entry, document in zip(entries, documents):
        frequencies = Counter(document)
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies[term]
            if not term_frequency:
                continue
            frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - frequency + 0.5) / (frequency + 0.5)
            )
            length_normalization = 1.2 * (
                1 - 0.75 + 0.75 * len(document) / max(average_length, 1)
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (1.2 + 1)
                / (term_frequency + length_normalization)
                * min(query_frequency, 3)
            )
        if score > 0:
            scored.append((entry, score))
    scored.sort(key=lambda item: (-item[1], item[0].get("id", "")))
    return scored[:top_k]


def render_memory_block(entries: list[dict[str, str]]) -> str:
    """Render retrieved entries with the same wrapper the tau2 agent uses."""
    if not entries:
        return ""
    return (
        "\n<memory_from_prior_training_tasks>\n"
        "The following external memory was accumulated from earlier tasks. "
        "Use it when relevant, but prefer the current task, the API docs, "
        "and environment feedback if they conflict.\n"
        f"{json.dumps(entries, ensure_ascii=False, indent=2)}\n"
        "</memory_from_prior_training_tasks>"
    )


def retrieved_block(
    bank_path: str | Path | None, query: str, top_k: int
) -> tuple[str, list[dict[str, Any]]]:
    """Convenience: load, retrieve, render. Returns (block, selection log)."""
    if not bank_path:
        return "", []
    entries = load_bank(bank_path)
    ranked = retrieve(entries, query, top_k)
    selected = [entry for entry, _ in ranked]
    log = [
        {"id": entry.get("id"), "score": round(score, 3)} for entry, score in ranked
    ]
    return render_memory_block(selected), log
