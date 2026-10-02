"""Backbone profiles: everything that differs between the task-agent models.

The ablation grid (`alfworld_summary.md` and its per-benchmark siblings) was
built on Qwen3.5-35B-A3B, and the model leaked into four places that each
broke differently when a second backbone was tried:

- serving: vLLM build, reasoning / tool-call parsers, served model name;
- SFT: how an assistant turn ends in the chat template (the trainer
  supervises through the terminator, so a wrong one teaches the model never
  to stop), and whether the template renders a turn the same way in history
  as at generation time;
- merge: which module path the language tower sits under in the served
  architecture (the LoRA is trained on the language tower only);
- bookkeeping: output directories, so a second backbone's grid does not
  "skip" every step because the first backbone's artifacts already exist.

A profile is selected by name (`MODEL_PROFILE=gemma4`) in the grid scripts,
or by checkpoint (`profile_for_path`, which reads `model_type` from its
config.json) in the serving script -- a merged checkpoint has the same
model_type as its base, so it is served with the same parsers.

Stdlib only: imported from .venv (3.13), router_venv and alfworld_venv310.

Shell use:
  eval "$(python -m trajectory_memory_lab.model_profiles shell gemma4)"
  eval "$(python -m trajectory_memory_lab.model_profiles serve-args /path/to/checkpoint)"
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

HF_HUB = Path("/nas04/yixuh/hf_cache/hub")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hf_id: str
    revision: str
    model_types: tuple[str, ...]
    served_name: str
    # Directory segment that namespaces every artifact of a non-default
    # backbone (`alfworld_experiment/gemma4/...`, `/nas04/yixuh/gemma4_*_merged`).
    # Empty for the original backbone, so its existing paths are unchanged.
    tag: str
    vllm_bin: str
    reasoning_parser: str
    tool_call_parser: str
    extra_serve_args: tuple[str, ...] = ()
    # What closes an assistant turn in the chat template; the SFT trainer
    # supervises through it, so it is what teaches the model to stop. When the
    # template has no closing token (GLM) a turn runs straight into the next
    # role header, so these are the headers that can follow it -- each one a
    # generation eos -- and `implicit_stop` is appended where nothing follows.
    turn_ends: tuple[str, ...] = ()
    implicit_stop: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_path(self) -> Path:
        org, repo = self.hf_id.split("/")
        return HF_HUB / f"models--{org}--{repo}" / "snapshots" / self.revision


PROFILES: dict[str, ModelProfile] = {
    profile.name: profile
    for profile in (
        ModelProfile(
            name="qwen35",
            hf_id="Qwen/Qwen3.5-35B-A3B",
            revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
            model_types=("qwen3_5_moe",),
            served_name="qwen35-tau",
            tag="",
            vllm_bin=str(REPO_ROOT / ".venv/bin/vllm"),
            reasoning_parser="qwen3",
            tool_call_parser="qwen3_xml",
            extra_serve_args=("--language-model-only",),
            turn_ends=("<|im_end|>",),
        ),
        ModelProfile(
            name="gemma4",
            hf_id="google/gemma-4-26B-A4B-it",
            revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            model_types=("gemma4",),
            served_name="gemma4-26b-a4b",
            tag="gemma4",
            # vLLM 0.17.1 (.venv) has no Gemma 4. 0.19.1 has it and pins the same
            # torch 2.10.0+cu128 that .venv runs on this CUDA 12.6 driver; 0.30's
            # torch 2.13+cu130 failed at worker init ("driver too old"). A separate venv keeps the
            # Qwen server bit-for-bit what every existing number was measured on.
            vllm_bin="/nas04/yixuh/vllm019_venv/bin/vllm",
            reasoning_parser="gemma4",
            tool_call_parser="gemma4",
            # vLLM auto-detects the "openai" (list-of-parts) content format for
            # this template, which renders a system turn ending in "." as
            # ". <turn|>": one token the HF rendering the SFT trainer uses
            # does not have. "string" makes the served prompt token-identical.
            extra_serve_args=("--language-model-only", "--chat-template-content-format", "string"),
            turn_ends=("<turn|>",),
            notes=(
                "thinking disabled: the generation prompt ends in an empty "
                "'<|channel>thought\\n<channel|>' that history turns do not carry",
                "final_logit_softcapping=30 (text_config): applied by the SFT loss",
            ),
        ),
        ModelProfile(
            name="glm47flash",
            hf_id="zai-org/GLM-4.7-Flash",
            revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",
            model_types=("glm4_moe_lite",),
            served_name="glm47-flash",
            tag="glm47flash",
            vllm_bin="/nas04/yixuh/vllm019_venv/bin/vllm",
            reasoning_parser="glm45",
            tool_call_parser="glm47",
            # Text-only checkpoint: no --language-model-only.
            turn_ends=("<|user|>", "<|observation|>"),
            implicit_stop="<|user|>",
            notes=("assistant turns have no closing token; <|user|> is a generation eos",),
        ),
    )
}
DEFAULT_PROFILE = "qwen35"


def get_profile(name: str | None) -> ModelProfile:
    key = name or DEFAULT_PROFILE
    if key not in PROFILES:
        raise SystemExit(f"unknown MODEL_PROFILE {key!r}; known: {', '.join(PROFILES)}")
    return PROFILES[key]


def profile_for_path(path: str | Path) -> ModelProfile:
    """The profile whose model_type matches a checkpoint (base or merged)."""
    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    for profile in PROFILES.values():
        if model_type in profile.model_types:
            return profile
    raise SystemExit(f"{path}: model_type {model_type!r} matches no profile in model_profiles.py")


def _shell_exports(profile: ModelProfile) -> str:
    values = {
        "MODEL_PROFILE": profile.name,
        "BASE": str(profile.base_path),
        "SERVED_NAME": profile.served_name,
        "PROFILE_TAG": profile.tag,
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def _serve_args(profile: ModelProfile) -> str:
    args = [
        "--served-model-name", profile.served_name,
        "--reasoning-parser", profile.reasoning_parser,
        "--enable-auto-tool-choice",
        "--tool-call-parser", profile.tool_call_parser,
        *profile.extra_serve_args,
    ]
    return (f"VLLM_BIN={shlex.quote(profile.vllm_bin)}\n"
            f"PROFILE_SERVE_ARGS=({' '.join(shlex.quote(a) for a in args)})")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] not in ("shell", "serve-args", "base-path"):
        raise SystemExit("usage: model_profiles {shell <profile> | serve-args <checkpoint> | base-path <profile>}")
    command, value = argv
    if command == "shell":
        print(_shell_exports(get_profile(value)))
    elif command == "serve-args":
        print(_serve_args(profile_for_path(value)))
    else:
        print(get_profile(value).base_path)


if __name__ == "__main__":
    main(sys.argv[1:])
