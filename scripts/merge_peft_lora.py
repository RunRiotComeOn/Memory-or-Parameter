#!/usr/bin/env python3
"""Merge a standard PEFT LoRA adapter into a Hugging Face causal LM on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model")
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    architecture = (config.architectures or [""])[0]
    model_class = AutoModelForImageTextToText if "ForConditionalGeneration" in architecture else AutoModelForCausalLM
    model = model_class.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True, progressbar=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        processor.save_pretrained(args.output)
    except Exception as exc:
        print(f"Processor copy skipped: {exc}")


if __name__ == "__main__":
    main()
