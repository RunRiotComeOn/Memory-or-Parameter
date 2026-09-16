#!/usr/bin/env python3
"""Validate, length-filter, and convert retained tau episodes to parquet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from datasets import Dataset
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _token_ids(tokenizer, messages: list[dict], tools: list[dict]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return rendered["input_ids"] if isinstance(rendered, Mapping) else rendered


def _encode_nested_fields(record: dict) -> dict:
    """Keep sparse OpenAI messages/tool schemas opaque to Arrow's type merger."""
    encoded = dict(record)
    encoded["messages"] = json.dumps(
        record["messages"], ensure_ascii=False, separators=(",", ":")
    )
    encoded["tools"] = json.dumps(
        record.get("tools") or [], ensure_ascii=False, separators=(",", ":")
    )
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_parquet", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("no retained SFT episodes")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    accepted = []
    rejected = []
    lengths = []
    for record in records:
        if record.get("validation_status") != "accepted":
            raise ValueError("non-accepted record reached SFT preparation")
        token_ids = _token_ids(tokenizer, record["messages"], record.get("tools") or [])
        length = len(token_ids)
        lengths.append(length)
        if length <= args.max_length:
            accepted.append(record)
        else:
            rejected.append(
                {
                    "source_task_id": record.get("source_task_id"),
                    "domain": record.get("domain"),
                    "tokens": length,
                    "reason": "exceeds_max_length",
                }
            )
    if not accepted:
        raise ValueError("all retained episodes exceeded max length")
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([_encode_nested_fields(record) for record in accepted]).to_parquet(
        str(args.output_parquet)
    )

    # This is a hard regression guard.  Nested sparse dictionaries must survive
    # Parquet exactly; otherwise Arrow fills every tool call/schema with the
    # union of all keys and thousands of null-valued arguments.
    reloaded = pq.read_table(args.output_parquet).to_pylist()
    if len(reloaded) != len(accepted):
        raise ValueError("Parquet row count changed during serialization")
    verified_lengths = []
    for source, stored in zip(accepted, reloaded, strict=True):
        messages = json.loads(stored["messages"])
        tools = json.loads(stored["tools"])
        if messages != source["messages"] or tools != (source.get("tools") or []):
            raise ValueError(
                f"nested JSON changed for {source.get('source_task_id')}"
            )
        restored_ids = _token_ids(tokenizer, messages, tools)
        source_ids = _token_ids(
            tokenizer, source["messages"], source.get("tools") or []
        )
        if restored_ids != source_ids:
            raise ValueError(
                f"rendered tokens changed for {source.get('source_task_id')}"
            )
        verified_lengths.append(len(restored_ids))
    summary = {
        "input_records": len(records),
        "training_records": len(accepted),
        "rejected_records": len(rejected),
        "max_length": args.max_length,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "serialization": "json_strings_v2",
        "token_exact_rows": len(verified_lengths),
        "verified_min_tokens": min(verified_lengths),
        "verified_max_tokens": max(verified_lengths),
        "rejected": rejected,
        "output": str(args.output_parquet),
    }
    summary_path = args.output_parquet.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
