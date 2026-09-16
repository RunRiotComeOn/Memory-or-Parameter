#!/usr/bin/env python3
"""Create a text-only PEFT adapter from a multimodal all-linear adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LANGUAGE_TARGETS = {
    "down_proj",
    "gate_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "shared_expert_gate",
    "up_proj",
    "v_proj",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    source_weights = args.source / "adapter_model.safetensors"
    source_config = args.source / "adapter_config.json"
    args.target.mkdir(parents=True, exist_ok=True)

    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {
            key: handle.get_tensor(key)
            for key in handle.keys()
            if ".language_model." in key
        }

    if not tensors:
        raise RuntimeError(f"No language-model tensors found in {source_weights}")
    save_file(
        tensors,
        args.target / "adapter_model.safetensors",
        metadata=metadata,
    )

    config = json.loads(source_config.read_text())
    config["target_modules"] = sorted(LANGUAGE_TARGETS)
    (args.target / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(f"Wrote {len(tensors)} text LoRA tensors to {args.target}")


if __name__ == "__main__":
    main()
