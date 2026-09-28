#!/usr/bin/env python3
"""Re-point a LoRA adapter trained on the language tower at the full model.

The SFT LoRA is trained through `AutoModelForCausalLM`, which for this
checkpoint builds `Qwen3_5MoeForCausalLM` -- the language tower only, the
transformers equivalent of the servers' `--language-model-only`. vLLM, on the
other hand, serves the checkpoint's own architecture,
`Qwen3_5MoeForConditionalGeneration`, where every one of those modules sits
one level deeper:

    trained on   model.layers.0.linear_attn.out_proj
    served as    model.language_model.layers.0.linear_attn.out_proj

Merging the adapter into the served architecture without fixing this matches
nothing. PEFT does not error on that -- it would just produce a "merged"
checkpoint identical to the base, and the replay would silently measure the
untrained model.

Two things are rewritten here:

- Tensor keys, by inserting `language_model.` at the one place it belongs.
- `target_modules`, from PEFT's 13 bare SUFFIXES (`down_proj`, `in_proj_qkv`,
  ...) to explicit full module paths. Suffixes are what makes this dangerous
  in the other direction: the full model also has `down_proj`/`gate_proj`
  inside its 27-block vision tower and its MTP head (461 nn.Linear vs the
  language tower's 351), so suffix matching would attach fresh adapters to
  modules the training never saw and the state dict has no weights for.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

OLD_PREFIX = "base_model.model.model."
NEW_PREFIX = "base_model.model.model.language_model."
PEFT_WRAPPER = "base_model.model."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path, help="adapter dir trained on AutoModelForCausalLM")
    parser.add_argument("output", type=Path, help="re-keyed adapter dir for the full architecture")
    parser.add_argument("--insert", default="language_model.",
                        help="path segment to insert after the base model prefix")
    args = parser.parse_args()

    tensors = load_file(str(args.adapter / "adapter_model.safetensors"))
    old_prefix = OLD_PREFIX
    new_prefix = OLD_PREFIX + args.insert
    if any(k.startswith(new_prefix) for k in tensors):
        raise SystemExit(f"adapter already targets {new_prefix!r}; nothing to do")
    missing = [k for k in tensors if not k.startswith(old_prefix)]
    if missing:
        raise SystemExit(f"{len(missing)} tensor(s) do not start with {old_prefix!r}, e.g. {missing[0]}")

    remapped = {new_prefix + k[len(old_prefix):]: v for k, v in tensors.items()}

    # Explicit module paths, derived from the tensors themselves so the config
    # can never disagree with the weights.
    modules = sorted({
        k[len(PEFT_WRAPPER):].rsplit(".lora_", 1)[0]
        for k in remapped if ".lora_" in k
    })

    config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    before = config.get("target_modules")
    config["target_modules"] = modules

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(remapped, str(args.output / "adapter_model.safetensors"))
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for extra in ("README.md",):
        source = args.adapter / extra
        if source.exists():
            shutil.copy2(source, args.output / extra)

    print(f"re-keyed {len(remapped)} tensors {old_prefix!r} -> {new_prefix!r}")
    print(f"target_modules: {len(before) if isinstance(before, list) else before} suffix(es) "
          f"-> {len(modules)} explicit paths, e.g. {modules[0]}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
