#!/usr/bin/env python3
"""Merge a standard PEFT LoRA adapter into a Hugging Face causal LM on CPU."""

from __future__ import annotations

import argparse
import shutil
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
    # Same as training (train_agent_sft_lora_peft.keep_peft_targets_as_given):
    # without this, loading re-derives fused-expert targets the adapter has
    # no weights for.
    from peft.utils import transformers_weight_conversion
    transformers_weight_conversion.convert_peft_config_for_transformers = lambda *a, **k: None
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

    # The server reads its config, stop tokens (generation_config.json), chat
    # template and tokenizer from the merged directory. Take them verbatim
    # from the base rather than from this re-save:
    # - config.json: a LoRA merge changes no shape, so the base config
    #   describes the merged weights exactly, while a re-save is written in
    #   THIS transformers' schema. Gemma 4 re-saved by 5.17 (`per_layer_config`,
    #   vision `rope_type: axial`) does not load in the server's 5.5.4.
    # - generation_config.json: GLM-4.7-Flash's assistant turns end only
    #   through its eos list; a re-save that dropped one would never stop.
    # verify_merged_model.py checks the merged weight names against the base.
    base_dir = Path(args.base_model)
    if base_dir.is_dir():
        for source in sorted(base_dir.iterdir()):
            if source.suffix == ".safetensors" or source.name == "model.safetensors.index.json":
                continue
            if source.is_file() and not source.name.startswith(".") and source.suffix in (".json", ".jinja", ".txt", ".model"):
                shutil.copy2(source.resolve(), args.output / source.name)
                print(f"copied {source.name} from base")


if __name__ == "__main__":
    main()
