#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into Qwen3.5 and save a standard HF model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5MoeForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model")
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading base model on CPU...", flush=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("Loading and merging LoRA...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload(safe_merge=True, progressbar=True)

    print("Saving merged model...", flush=True)
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output)
    print(f"Merged model written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
