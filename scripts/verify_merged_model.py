#!/usr/bin/env python3
"""Check that a merged checkpoint really carries the LoRA, and only where trained.

Replaces the inline heredoc every grid script used, which compared two keys
hard-coded for Qwen3.5 (`linear_attn.out_proj`, `visual.blocks`) and only
printed the result. Here the keys come from the adapter itself:

  target     the first trained module: must have CHANGED (rel > 0) -- a
             merge that matched nothing yields a checkpoint identical to the
             base, and the replay would silently measure the untrained model;
  untouched  the input embedding, never a LoRA target: must be bit-identical,
             and so must the first vision-tower and MoE-expert weights when
             the checkpoint has them.

Exits non-zero on any violation.

  python scripts/verify_merged_model.py BASE MERGED ADAPTER_FULLMODEL
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open


def load(directory: Path, index: dict[str, str], key: str):
    with safe_open(str(directory / index[key]), framework="pt") as handle:
        return handle.get_tensor(key).float()


def main() -> None:
    base, merged, adapter = (Path(a) for a in sys.argv[1:4])
    base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    merged_index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
    modules = json.loads((adapter / "adapter_config.json").read_text())["target_modules"]
    if not isinstance(modules, list):
        raise SystemExit(f"{adapter}: target_modules is {modules!r}, not explicit paths; run rekey_lora_to_full_model.py")

    # merge_peft_lora.py serves the merged weights under the BASE config, so
    # every merged weight must be one the base has. The reverse is allowed:
    # Qwen3.5's re-save drops its 785 MTP-head weights, never served.
    foreign = sorted(set(merged_index) - set(base_index))
    if foreign:
        raise SystemExit(f"{len(foreign)} merged weight(s) not in the base checkpoint, e.g. {foreign[:3]}")
    dropped = len(set(base_index) - set(merged_index))
    print(f"  weights: {len(merged_index)} merged, all in base ({dropped} base-only, e.g. MTP head)")

    target = next((m + ".weight" for m in sorted(modules) if m + ".weight" in base_index), None)
    if target is None:
        raise SystemExit(f"no trained module of {adapter} has a weight in {base}'s index, e.g. {sorted(modules)[0]}")
    untouched = [k for k in base_index if k.endswith("embed_tokens.weight")][:1]
    untouched += [k for k in sorted(base_index) if ".visual." in k or ".vision_tower." in k][:1]
    # MoE experts are not a LoRA target in this recipe; a change here means
    # PEFT widened the adapter onto them (see keep_peft_targets_as_given).
    # Fused (`experts.down_proj`: Gemma 4, Qwen3.5) or per-expert
    # (`experts.0.down_proj.weight`: GLM); `shared_expert` IS a target.
    untouched += [k for k in sorted(base_index) if re.search(r"\.experts\.(\d+\.)?down_proj(\.weight)?$", k)][:1]

    ok = True
    for key, kind in [(target, "target")] + [(k, "untouched") for k in untouched]:
        if key not in merged_index:
            print(f"  [{kind}] {key}: MISSING from merged checkpoint")
            ok = False
            continue
        a, b = load(base, base_index, key), load(merged, merged_index, key)
        rel = float((a - b).norm() / a.norm())
        good = rel > 0 if kind == "target" else rel == 0
        ok &= good
        print(f"  [{kind}] {key} rel={rel:.2e} {'ok' if good else 'WRONG'}")
    if not ok:
        raise SystemExit("merged checkpoint failed verification")


if __name__ == "__main__":
    main()
