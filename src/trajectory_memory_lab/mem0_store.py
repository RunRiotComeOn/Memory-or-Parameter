"""mem0-backed external memory, replacing the hand-rolled bank + BM25 stack.

Why: the hand-rolled path kept the write side (`router_bank_builder`'s forced
refine) and the read side (`memory_retrieval`'s BM25) as our own code, and both
turned out to be where the damage was -- a Jaccard-0.25 dedup silently
destroying complementary memories. mem0 is a maintained implementation of the
same job, so the allocation question this project actually studies is not
riding on our own memory plumbing.

Two properties this project cannot give up, and how they are preserved here:

* **Determinism.** Everything stays local and fixed: the fact-extraction and
  ADD/UPDATE/DELETE calls go to the same deterministic vLLM replica the agent
  uses (temperature 0, top_p 1), embeddings come from a pinned CPU ONNX model,
  and the vector store is an on-disk Qdrant local to one candidate. No hosted
  service is contacted.
* **No outbound telemetry.** mem0 ships posthog analytics on by default; this
  module disables it before mem0 is imported, which is when the flag is read.
"""

from __future__ import annotations

import os

# Must precede the mem0 import: mem0.memory.telemetry reads this at import time,
# and its default is "True" (posthog). Nothing about this project's data should
# leave the machine.
os.environ.setdefault("MEM0_TELEMETRY", "False")
# mem0's OpenAI client prefers OpenRouter whenever this is set; make sure a
# stray value in the environment cannot redirect our calls off-box.
os.environ.pop("OPENROUTER_API_KEY", None)

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from mem0 import Memory  # noqa: E402

# Small, CPU-friendly ONNX embedder. Pinned: changing it silently changes every
# retrieval, so it is part of the experiment's configuration, not a default.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384
COLLECTION = "appworld"
DEFAULT_USER_ID = "router"

# mem0 2.0's own extraction prompt (ADDITIVE_EXTRACTION_PROMPT) is written for a
# personal assistant: fed a procedural note about an AppWorld API it decides
# there is no personal fact present and stores nothing. `custom_instructions` is
# the supported override -- it is injected into the extraction user prompt at
# highest priority, and unlike `custom_fact_extraction_prompt` (a MemoryConfig
# field that 2.0.20 never reads) it actually reaches the model. The output
# schema stays mem0's, so this only redirects *what counts as* a memory.
CUSTOM_INSTRUCTIONS = """These memories are reusable task knowledge written by a coding agent that solves tasks against the AppWorld API sandbox. They are NOT personal facts about a human user; treat the content as engineering knowledge and store it.

Store a memory when the input contains any of:
- An exact API name with its arguments or return shape.
- An ordered procedure that accomplishes a goal.
- A task-semantic rule: what a domain term is defined to mean, or the exact condition under which something should be kept, removed, or skipped.
- A pitfall: something that looks correct but fails, and what to do instead.

Rules:
- Preserve exact identifiers (API names, field names, parameter names) verbatim.
- Keep a definition or condition rule as its own memory. Never drop such a rule merely because another memory concerns the same app or topic.
- Two memories about the same app but stating different things are both worth keeping; only genuinely restating the same claim should update an existing memory.
- Never invent API names or behaviour the input does not state.
"""


def _force_deterministic_calls(client: Any) -> None:
    """Make every mem0 LLM call non-thinking.

    The agent's replica runs `--reasoning-parser qwen3`. Left to think, Qwen puts
    the whole answer in reasoning_content and `content` comes back None, so every
    mem0 extraction fails with "Error parsing extraction response" and silently
    stores nothing. mem0 builds its request dict internally with no extra_body
    hook, so the flag is injected at the client instead.
    """
    completions = client.chat.completions
    original = completions.create

    def create(**kwargs: Any) -> Any:
        body = dict(kwargs.pop("extra_body", None) or {})
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        # mem0 asks for response_format={"type": "json_object"}. On this vLLM
        # build that schema-less guided-decoding mode emits broken JSON for the
        # extraction prompt -- measured: '{{"": "memory: 0",\n  \t}' -- while the
        # very same request without it returns a correctly-schemed object. Since
        # a malformed body is silently swallowed ("Error parsing extraction
        # response" -> nothing stored), drop the flag and let mem0's own
        # extract_json fallback handle parsing.
        if kwargs.get("response_format") == {"type": "json_object"}:
            kwargs.pop("response_format")
        return original(extra_body=body, **kwargs)

    completions.create = create


def build_memory(
    store_path: str | Path,
    base_url: str,
    model: str,
    *,
    embed_model: str = EMBED_MODEL,
    max_tokens: int = 2048,
) -> Memory:
    """A Memory whose LLM is our own deterministic vLLM replica."""
    store = Path(store_path)
    store.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "api_key": "EMPTY",
                "openai_base_url": base_url,
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {"model": embed_model, "embedding_dims": EMBED_DIMS},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": COLLECTION,
                "embedding_model_dims": EMBED_DIMS,
                "path": str(store / "qdrant"),
                "on_disk": True,
            },
        },
        "history_db_path": str(store / "history.db"),
        "version": "v1.1",
        "custom_instructions": CUSTOM_INSTRUCTIONS,
    }
    memory = Memory.from_config(config)
    _force_deterministic_calls(memory.llm.client)
    return memory


def add_entry(
    memory: Memory,
    content: str,
    *,
    scope: str = "",
    source_task_id: str = "",
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Hand one routed memory to mem0.

    mem0 -- not us -- then decides whether this becomes a new memory, updates an
    existing one, or is dropped. That is the whole point of the swap, but note
    it moves the merge decision out of the router's action space: see the v5
    note in DESIGN.md section 13.
    """
    payload = f"{scope}: {content}" if scope else content
    return memory.add(
        payload,
        user_id=user_id,
        metadata={"scope": scope, "source_task_id": source_task_id},
    )


def search_entries(
    memory: Memory, query: str, top_k: int, *, user_id: str = DEFAULT_USER_ID,
    threshold: float | None = None,
) -> list[tuple[dict[str, str], float]]:
    """Retrieve in the shape `memory_retrieval.retrieve` returns, so the agent
    prompt and the selection log stay byte-identical in structure to the BM25
    path and results remain comparable across the swap."""
    if top_k <= 0:
        return []
    # Without a threshold mem0 fills all k slots regardless of similarity, while
    # BM25 drops anything scoring zero. Measured on k1: mem0 injected 3.0
    # entries/task against BM25's 1.7, and scored 0.30 where BM25 scored 0.70 on
    # the same bank, replica and tasks -- 24 of its 30 retrieved slots sat below
    # 0.40 similarity. The threshold restores the relevance floor BM25 gets for
    # free.
    # NOTE: mem0's own `threshold=` argument does NOT filter on the score it
    # returns -- measured on this store, threshold=0.30 still returned entries
    # scoring 0.287 and 0.231, and threshold=0.50 kept 0.287 while dropping
    # 0.231. Whatever scale it applies to, it is not the one the caller sees, so
    # filtering happens below on the returned score instead: same number the
    # selection log records, and trivially checkable.
    found = memory.search(query, top_k=top_k, filters={"user_id": user_id}) or {}
    results = found.get("results", found) if isinstance(found, dict) else found
    ranked: list[tuple[dict[str, str], float]] = []
    for item in results or []:
        meta = item.get("metadata") or {}
        entry = {
            "id": str(item.get("id", "")),
            "scope": str(meta.get("scope") or ""),
            "content": str(item.get("memory") or ""),
        }
        score = float(item.get("score") or 0.0)
        if entry["content"] and (threshold is None or score >= threshold):
            ranked.append((entry, score))
    return ranked[:top_k]


def export_entries(
    memory: Memory, *, user_id: str = DEFAULT_USER_ID
) -> list[dict[str, Any]]:
    """Dump the store to the bank-shaped JSON the rest of the codebase reads,
    so existing inspection and analysis tooling keeps working."""
    found = memory.get_all(filters={"user_id": user_id}) or {}
    results = found.get("results", found) if isinstance(found, dict) else found
    entries = []
    for item in results or []:
        meta = item.get("metadata") or {}
        entries.append({
            "id": str(item.get("id", "")),
            "scope": str(meta.get("scope") or ""),
            "content": str(item.get("memory") or ""),
            "status": "active",
            "source_task_id": meta.get("source_task_id") or "",
            "backend": "mem0",
        })
    return entries
