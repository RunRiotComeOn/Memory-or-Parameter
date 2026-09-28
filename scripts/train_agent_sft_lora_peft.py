#!/usr/bin/env python3
"""Train the task-agent SFT LoRA with plain transformers+peft across 2 GPUs.

Replaces the `swift sft` + deepspeed path in `router_sft_lora_update.sh`,
which never once completed on this box: running_log.md records six
consecutive real failures on this exact model (plain zero3 on 2 and 4 GPUs,
zero3_offload, a params-only offload config, `--experts_impl eager`, and bnb
int4), every one of them pinned at 46-47GiB/GPU no matter the GPU count,
offload target or quantization. Both deepspeed's per-parameter sharding and
bitsandbytes' `nn.Linear` replacement work by recognizing standard module
shapes, and this MoE keeps its 256 experts as fused 3D `nn.Parameter`s
(`mlp.experts.gate_up_proj`), which neither mechanism sees.

What works instead is the recipe the GRPO router already runs on
(router_llm_trainable.py): let accelerate's `device_map` split the model
across two cards as a plain pipeline, turn on gradient checkpointing, and
train only the LoRA adapter. No sharding of individual parameters is
attempted, so nothing has to understand the expert layout. Measured there:
32.3GiB of weights per card, 16k-token backward at 42.8GiB peak.

The one thing SFT needs that the router did not: a loss over EVERY target
token rather than four logits at one position. A full [1, seq, 248320]
logits tensor is 3.0GB in bf16 at 6k tokens, ~6GB more once cross-entropy
upcasts to float32, and as much again for its gradient -- on top of 32.3GiB
of weights on the lm_head's card. So the head and the loss are applied in
chunks over the sequence (`--loss-chunk`), which keeps that term at a few
hundred MB and is exactly equivalent: cross-entropy is a sum over positions,
so summing per-chunk losses and dividing once by the token count gives the
same value and the same gradient as computing it in one piece.

Run:
  CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/nas04/yixuh/hf_cache PYTHONPATH=src \
  /nas04/yixuh/router_venv/bin/python -u scripts/train_agent_sft_lora_peft.py \
      --pool alfworld_experiment/router_force_sft_v1/sft_pool.jsonl \
      --output alfworld_experiment/force_sft_lora/adapter

Every pool row -- text (ALFWorld/ScienceWorld/WebShop) or tool-calling
(tau2: rows carry `tools`) -- becomes one sequence per user-delimited segment
(`build_segment_examples`), so each supervised turn sees exactly the prompt
the server built for it; all of a row's sequences make ONE optimizer step
with the row's token-mean loss. `--legacy-single-sequence` restores the
earlier one-hand-rendered-sequence-per-row path (text rows only), which was
exact for the first assistant turn only -- kept to reproduce LoRAs trained
before this change. Measured dry run on two tau2 airline rows: 12 sequences,
5.1k-9.0k tokens, 36.3GiB peak.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_BASE_MODEL = (
    "/nas04/yixuh/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "59d61f3ce65a6d9863b86d2e96597125219dc754"
)


NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
IM_START = "<|im_start|>"
ASSISTANT_HEADER = IM_START + "assistant\n"
TURN_END = "<|im_end|>"


def build_example(tokenizer, messages: list[dict[str, str]], max_length: int) -> dict[str, Any] | None:
    """LEGACY (`--legacy-single-sequence`): one hand-rendered sequence per row.

    Superseded by `build_segment_examples`. Measured afterwards: only the FIRST
    assistant turn's context here matches the served prompt. The empty think
    prefix is injected into every assistant turn, but at inference the
    template renders history turns WITHOUT it, so from turn 2 on the history
    the model trained on (`assistant\n<think>\n\n</think>\n\ngo to dresser 1`)
    is not the history it is served (`assistant\ngo to dresser 1`) -- on an
    ALFWorld row, 1 of 3 turns exact. `_assert_matches_inference` below only
    ever checked the first turn, which is how it went unnoticed. Kept only to
    reproduce adapters trained before the fix.

    Original description:

    Two things this has to get right, both measured rather than assumed:

    1. The Qwen3.5 template emits `<think>\n\n</think>\n\n` ONLY for a
       trailing assistant message, never for assistant turns already in the
       history -- so a straight rendering of a finished conversation contains
       no thinking prefix anywhere, while at inference the generation prompt
       hands the model exactly that prefix and expects the action after it.
       Training on the bare rendering would teach "action directly after
       `<|im_start|>assistant\n`" and then serve a prompt that never looks
       like that. This is what `router_sft_lora_update.sh`'s
       `--add_non_thinking_prefix true` was for, so the prefix is injected
       into every assistant turn here and `_assert_matches_inference`
       verifies the result really is what the served prompt looks like.

    2. Spans are located against the SINGLE full rendering by scanning for
       the assistant header, not by re-rendering each prefix and diffing --
       point 1 is precisely why prefix-diffing is wrong here.
    """
    prepared = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant" and not content.startswith(NON_THINKING_PREFIX):
            content = NON_THINKING_PREFIX + content
        prepared.append({"role": message["role"], "content": content})

    # Rendered by hand rather than through apply_chat_template: the template
    # STRIPS the thinking block out of assistant messages that sit in the
    # history, so injecting the prefix into the content and rendering gives
    # it straight back with no prefix anywhere -- which is how the first
    # attempt silently produced training text the server never sends.
    # `_assert_matches_inference` proves this hand-rendering is right: it
    # comes out byte-identical to what the template produces for a live
    # generation prompt.
    full = "".join(
        f"{IM_START}{m['role']}\n{m['content']}{TURN_END}\n" for m in prepared
    )

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        header = full.find(ASSISTANT_HEADER, cursor)
        if header < 0:
            break
        start = header + len(ASSISTANT_HEADER)
        end = full.find(TURN_END, start)
        if end < 0:
            break
        spans.append((start, end + len(TURN_END)))
        cursor = end
    expected = sum(1 for m in prepared if m["role"] == "assistant")
    if len(spans) != expected or not spans:
        return None

    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    labels = [-100] * len(input_ids)
    for position, (start, end) in enumerate(encoded["offset_mapping"]):
        if end <= start:
            continue
        if any(s <= start and end <= e for s, e in spans):
            labels[position] = input_ids[position]

    if len(input_ids) > max_length:
        # Keep the system turn AND the tail. Dropping a plain prefix -- what
        # this did first -- removes the system turn itself, so the example no
        # longer starts the way every served prompt starts and
        # `_assert_matches_inference` would be describing a shape the training
        # data does not have. It never fired on ALFWorld (longest example
        # 8,462 tokens against a 6,144 cap only for a handful) but ScienceWorld
        # examples run 5.8k-8.1k, so more than half of them were being cut.
        head_ids: list[int] = []
        if prepared and prepared[0]["role"] == "system":
            head_ids = tokenizer(
                f"{IM_START}system\n{prepared[0]['content']}{TURN_END}\n",
                add_special_tokens=False,
            )["input_ids"]
        keep_tail = max(1, max_length - len(head_ids))
        input_ids = head_ids + input_ids[-keep_tail:]
        labels = [-100] * len(head_ids) + labels[-keep_tail:]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels, "text": full}


def _assert_matches_inference(tokenizer, messages: list[dict[str, str]], text: str) -> None:
    """The training text must contain, verbatim, the prompt the server builds.

    Takes the conversation up to the first assistant turn, renders it the way
    a live rollout does (`add_generation_prompt=True`, thinking disabled), and
    requires that string to be a prefix of the training text. If it is not,
    the model is being trained on something it will never be shown.
    """
    cut = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
    if cut is None:
        return
    served = tokenizer.apply_chat_template(
        messages[:cut], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # A prefix, not equality: the training text continues past the generation
    # prompt with the assistant turn that is the target.
    if not text.startswith(served):
        raise AssertionError(
            "training text does not start with the prompt a live rollout would send:\n"
            f"  served  ...{served[-160:]!r}\n  training...{text[:len(served)][-160:]!r}"
        )


def _parsed_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool-call arguments as dicts, the form the chat template iterates.

    The pool stores them the OpenAI way, as JSON strings, which is also how
    the agent sent them to vLLM; vLLM parses them back into dicts before
    applying the template (`arguments|items` needs a mapping). Rendering the
    string form would produce a different -- and wrong -- tool-call block.
    """
    prepared = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call = dict(call)
                function = dict(call.get("function") or {})
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"] or "{}")
                call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        else:
            message.pop("tool_calls", None)
        prepared.append(message)
    return prepared


def build_segment_examples(
    tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_length: int,
) -> tuple[list[dict[str, Any]], int]:
    """One pool row -> training sequences whose context for EVERY supervised
    assistant turn is byte-identical to the served prompt.

    Rendered through the chat template itself (with `tools` for tool-calling
    rows: schemas in the system turn, tool calls as `<tool_call><function=...>`
    XML, tool results folded into user turns). The template renders an
    assistant turn differently depending on where it sits: turns after the
    last real user message carry an empty `<think>\\n\\n</think>\\n\\n`,
    earlier ones do not. So a turn looks one way when it is generated and
    another once a later user message pushes it into history, and no single
    sequence can be exact for all turns. Measured: a single full rendering
    matched the served prompt at 1-2 of 8-12 model turns on tau2 airline
    conversations, and the legacy hand-rendering (`build_example`) at 1 of 3
    on an ALFWorld row.

    What is exact: split at real user messages. Every assistant turn in the
    segment that follows user message u is generated against a prompt that is
    a prefix of the rendering that runs to the end of that segment (the last
    query is u throughout). One sequence per segment, supervising only that
    segment's assistant turns, from the end of the served prompt (which
    already ends in the empty think block -- given, not generated) through
    `<|im_end|>`. Each supervised span's start is found by rendering the
    served prompt itself and requiring it to be a prefix, so a mismatch
    raises instead of training on a shape never served.

    For text rows every user message is a real query, so each assistant turn
    is its own segment; tool-calling rows group a turn with the tool calls
    that follow it. Cost: one sequence per segment instead of one per row
    (tau2 airline: 5-7 sequences of 5k-9k tokens, ~5k of it policy + tool
    schemas).

    Returns (sequences, dropped_over_max_length). A segment longer than
    `max_length` is dropped rather than truncated: cutting it would remove
    either the system turn or history the target depends on.
    """
    prepared = _parsed_tool_arguments(messages)

    def render(prefix: list[dict[str, Any]], generation: bool) -> str:
        return tokenizer.apply_chat_template(
            prefix, tools=tools or None, tokenize=False,
            add_generation_prompt=generation, enable_thinking=False,
        )

    user_positions = [i for i, m in enumerate(prepared) if m["role"] == "user"]
    segments: dict[int, list[int]] = {}
    for index, message in enumerate(prepared):
        # An assistant message before any user message (tau2's scripted
        # greeting) was never generated by the model: context only.
        if message["role"] != "assistant" or not any(u < index for u in user_positions):
            continue
        end = next((u for u in user_positions if u > index), len(prepared))
        segments.setdefault(end, []).append(index)

    sequences, dropped = [], 0
    for end, turns in sorted(segments.items()):
        text = render(prepared[:end], False)
        spans = []
        for turn in turns:
            served = render(prepared[:turn], True)
            if not text.startswith(served):
                raise AssertionError(
                    f"served prompt for assistant turn {turn} is not a prefix of its segment rendering:\n"
                    f"  served  ...{served[-160:]!r}\n  segment ...{text[:len(served)][-160:]!r}"
                )
            stop = text.find(TURN_END, len(served))
            if stop < 0:
                raise AssertionError(f"assistant turn {turn} has no {TURN_END} in its segment rendering")
            spans.append((len(served), stop + len(TURN_END)))
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = encoded["input_ids"]
        if len(input_ids) > max_length:
            dropped += 1
            continue
        labels = [-100] * len(input_ids)
        for position, (start, stop) in enumerate(encoded["offset_mapping"]):
            if stop > start and any(s <= start and stop <= e for s, e in spans):
                labels[position] = input_ids[position]
        if any(label != -100 for label in labels):
            sequences.append({"input_ids": input_ids, "labels": labels})
    return sequences, dropped


def chunked_causal_loss(
    causal_lm, hidden: torch.Tensor, labels: torch.Tensor, chunk: int,
) -> tuple[torch.Tensor, int]:
    """Sum of per-token cross-entropy, computed `chunk` positions at a time.

    Returns (summed_loss, supervised_token_count). The caller divides once,
    so this is the same number -- and the same gradient -- as one big
    cross-entropy over the full sequence, at a fraction of the peak memory:
    only `chunk x vocab` logits are alive at a time instead of `seq x vocab`.
    """
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    total = torch.zeros((), device=shift_hidden.device, dtype=torch.float32)
    supervised = int((shift_labels != -100).sum())
    if supervised == 0:
        return total, 0
    for start in range(0, shift_hidden.shape[1], chunk):
        piece_hidden = shift_hidden[:, start : start + chunk, :]
        piece_labels = shift_labels[:, start : start + chunk]
        if int((piece_labels != -100).sum()) == 0:
            continue
        logits = causal_lm.lm_head(piece_hidden).float()
        total = total + torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            piece_labels.reshape(-1).to(logits.device),
            ignore_index=-100,
            reduction="sum",
        )
    return total, supervised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True,
                        help="sft_pool.jsonl: {'messages': [...]} per line, plus 'tools' for tool-calling "
                             "pools (tau2), both rendered per segment through the chat template (build_segment_examples)")
    parser.add_argument("--output", type=Path, required=True, help="adapter output directory")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--epochs", type=int, default=3, help="router_sft_lora_update.sh's value")
    parser.add_argument("--lr", type=float, default=1e-4, help="this repo's SFT LoRA rate, not the router's 1e-5")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--max-length", type=int, default=12288,
                        help="the swift path used 3072 only because it was OOMing. Measured example "
                             "lengths: ALFWorld 0.9k-8.5k, ScienceWorld 5.8k-8.1k, so 12288 truncates "
                             "nothing in either. The memory ceiling for this model on two 49GB cards "
                             "with gradient checkpointing is 16,384 (24,576 OOMs)")
    parser.add_argument("--loss-chunk", type=int, default=512,
                        help="sequence positions per lm_head+cross-entropy chunk")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--legacy-single-sequence", action="store_true",
                        help="text rows only: the pre-fix one-sequence-per-row rendering (exact for the first "
                             "assistant turn only); for reproducing adapters trained before the fix")
    parser.add_argument("--dry-run-steps", type=int, default=0,
                        help=">0: run only this many optimizer steps and skip saving (memory/pace probe)")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    raw = [json.loads(line) for line in args.pool.open(encoding="utf-8")]
    # One group per pool row: the row's sequences are accumulated into a single
    # optimizer step, with the loss normalized over ALL of the row's supervised
    # tokens -- the same step count, schedule and per-token weighting as the
    # legacy one-sequence-per-row path, only with correct contexts.
    groups: list[list[dict[str, Any]]] = []
    skipped = dropped_segments = 0
    for row in raw:
        messages = row["messages"]
        if args.legacy_single_sequence:
            if row.get("tools"):
                raise SystemExit("--legacy-single-sequence cannot render tool-calling rows")
            built = build_example(tokenizer, messages, args.max_length)
            if built is None:
                skipped += 1
                continue
            if not groups:  # the legacy check, first row, first turn only
                _assert_matches_inference(tokenizer, messages, built["text"])
            built.pop("text", None)
            groups.append([built])
            continue
        sequences, dropped = build_segment_examples(tokenizer, messages, row.get("tools"), args.max_length)
        dropped_segments += dropped
        if sequences:
            groups.append(sequences)
        else:
            skipped += 1
    sequences_all = [sequence for group in groups for sequence in group]
    lengths = sorted(len(e["input_ids"]) for e in sequences_all)
    supervised_total = sum(sum(1 for x in e["labels"] if x != -100) for e in sequences_all)
    print(f"pool: {len(raw)} rows -> {len(groups)} usable ({skipped} unusable), "
          f"{len(sequences_all)} training sequences "
          f"({'legacy single-sequence' if args.legacy_single_sequence else 'per segment'}; "
          f"{dropped_segments} segment(s) over --max-length dropped)\n"
          f"  tokens per sequence: min={lengths[0]} p50={lengths[len(lengths)//2]} max={lengths[-1]}; "
          f"forward tokens per epoch: {sum(lengths):,}\n"
          f"  supervised (assistant) tokens: {supervised_total:,} "
          f"({100*supervised_total/sum(lengths):.1f}% of all tokens)", flush=True)

    print(f"loading {args.base_model} with device_map={args.device_map!r} ...", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map=args.device_map,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    input_device = model.get_input_embeddings().weight.device
    causal_lm = model.get_base_model()
    print(f"  loaded in {time.time()-t0:.1f}s; {sum(p.numel() for p in trainable):,} trainable LoRA params; "
          f"input_device={input_device} lm_head={causal_lm.lm_head.weight.device}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    steps_per_epoch = math.ceil(len(groups) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)),
    )
    print(f"schedule: {len(groups)} rows x {args.epochs} epochs = {total_steps} steps "
          f"(warmup {warmup}), lr={args.lr}, grad_accum={args.grad_accum}", flush=True)

    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(groups)))
        random.shuffle(order)
        optimizer.zero_grad()
        for position, index in enumerate(order):
            group = groups[index]
            group_supervised = sum(sum(1 for x in e["labels"] if x != -100) for e in group)
            t_step = time.time()
            group_loss = 0.0
            for example in group:
                input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=input_device)
                labels = torch.tensor([example["labels"]], dtype=torch.long, device=input_device)
                hidden = causal_lm.model(input_ids=input_ids, use_cache=False).last_hidden_state
                summed, _ = chunked_causal_loss(causal_lm, hidden, labels, args.loss_chunk)
                # Row-level token mean: each sequence contributes its summed
                # loss over the row's total supervised tokens, so the row's
                # gradient equals that of one sequence holding all its turns.
                (summed / max(1, group_supervised) / args.grad_accum).backward()
                group_loss += float(summed.detach())
                del hidden, summed
            loss = group_loss / max(1, group_supervised)
            if (position + 1) % args.grad_accum == 0 or position + 1 == len(order):
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                peak = max(torch.cuda.max_memory_allocated(i) / 2**30
                           for i in range(torch.cuda.device_count()))
                print(f"  epoch {epoch+1} step {step}/{total_steps} loss={loss:.4f} "
                      f"ppl={math.exp(min(20, loss)):.2f} grad_norm={grad_norm:.3f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} seqs={len(group)} "
                      f"tokens={sum(len(e['input_ids']) for e in group)} sup={group_supervised} "
                      f"peak={peak:.1f}GiB {time.time()-t_step:.1f}s", flush=True)
                if args.dry_run_steps and step >= args.dry_run_steps:
                    print(f"dry run: stopping after {step} step(s), not saving", flush=True)
                    return

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"saved adapter to {args.output}", flush=True)


if __name__ == "__main__":
    main()
